"""The repairer: the LLM side of the plan-verifier loop.

`_plan_repair_loop.verify_and_repair` drives a loop of "propose an op,
verify it, accept or reject" until no binding contract clause is
violated. `_plan_repair_ops.py` is the vocabulary of legal mutations and
`_plan_contract.py` is what "violated" means; neither calls a model. This
module is the other half: it prompts an LLM to propose exactly ONE op
against ONE named violation, and converts whatever tool call comes back
into the raw dict `_plan_repair_ops.parse_repair_op` expects.

This module owns none of the executor's rules -- it only DESCRIBES them,
in prose, for the model to read. The one substantive rule worth stating
here (`_ORDERING_RULE_TEXT`) is transcribed by hand from
`_planner_exec.build_dependency_graph`'s VP-projection branch on purpose:
that function is Python control flow the verifier calls directly
(`_plan_contract._ordering_violations` imports and calls it, never
re-derives it), but an LLM prompt can only ever be English, so there is no
"import the rule into the prompt" available here. If that branch's logic
ever changes, this text needs a matching update -- there is no test that
can catch drift between the two, because one is code and the other is a
paragraph a model reads.
"""
from __future__ import annotations

import json
import typing
from collections.abc import Callable
from typing import Any

from panini_dr.adapters.ours._plan_contract import Violation
from panini_dr.adapters.ours._plan_repair_ops import OP_NAMES, RepairOp
from panini_dr.adapters.ours._planner_exec import GSWPlan

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_ORDERING_RULE_TEXT = """\
How the executor orders blanks (the `ordering` clause):
For a verb phrase `S -phrase-> O` where both S and O are blanks, the
executor adds a dependency edge ONLY when:
  - S is the plan's target (the edge FLIPS: the target then depends on
    O, i.e. O must resolve first), OR
  - S.value_type == "entity".
If neither holds -- e.g. S is typed `date`, `number`, or `list` -- NO
edge is added in either direction. S and O then sit at the SAME
dependency level, so a `{S}` placeholder inside O's retrieval_hint is
not guaranteed to have resolved yet, and the executor drops it unrendered.
`ordering` is the most common violation. The two structural fixes are
`reverse_edge` (flip the verb phrase so the target depends on the other
blank) and `retype_node` (retype the referenced subject to `entity` so a
projection edge is created). `inline_placeholder` sidesteps ordering
entirely by copying the referenced blank's OWN hint text directly into
the placeholder's owner, so nothing needs to resolve first."""

REPAIRER_SYSTEM = f"""\
You are the repairer in a plan-verifier loop. You are shown ONE contract
violation on ONE plan, and you must propose exactly ONE repair op via a
tool call. You never emit a whole plan, and you never propose more than
one op per turn -- a deterministic verifier will check your op against
the plan and accept or reject it; if rejected, you will be shown the
reason and asked again.

Preference order when more than one op could plausibly fix the
violation: prefer RELOCATING text that already exists on the plan
(`inline_placeholder`) over AUTHORING new text (`set_hint`).
`inline_placeholder` copies another node's own retrieval_hint into the
node you are repairing; `set_hint` invents fresh prose. Existing text is
already grounded in the plan's own vocabulary -- prefer it whenever the
text you would need already lives on some other node.

{_ORDERING_RULE_TEXT}

Always call exactly one tool. If you cannot find an op that plausibly
fixes the violation, do not call a tool at all -- an empty response
means "give up on this violation," which is itself a valid outcome."""


# ---------------------------------------------------------------------------
# Tool schemas -- derived from the `_plan_repair_ops` pydantic models, one
# per member of `OP_NAMES`, so schema drift from a newly added op is
# impossible: `test_every_op_is_offered_as_a_tool` asserts set equality
# against `OP_NAMES` directly.
# ---------------------------------------------------------------------------

# Short, hand-written blurbs of WHAT each op does and WHEN to reach for it.
# The models carry no docstrings (see `_plan_repair_ops.py`), so this is the
# only source for the "description" half of a tool schema; the "parameters"
# half is derived from the model's own JSON schema below, never
# hand-transcribed.
_OP_DESCRIPTIONS: dict[str, str] = {
    "reverse_edge": (
        "Swap a verb phrase's subject and object (id and phrase unchanged). "
        "Use to fix `ordering` when flipping which side is the subject would "
        "create the missing dependency edge (see the ordering rule above)."
    ),
    "retype_node": (
        "Change a node's value_type. Use to fix `ordering` by retyping a "
        "non-target subject to `entity` so a projection edge is created, or "
        "to fix `target_type` by giving the target a type that matches what "
        "a constraint writes into it."
    ),
    "set_hint": (
        "Replace a node's retrieval_hint with new text you author. Last "
        "resort -- prefer `inline_placeholder` whenever the text you need "
        "already exists on another node."
    ),
    "inline_placeholder": (
        "Relocate another node's own retrieval_hint text into a `{ref}` "
        "placeholder inside this node's hint, replacing the placeholder. "
        "The preferred fix for `ordering`: it removes the unordered "
        "reference instead of trying to order it."
    ),
    "add_node": (
        "Add a brand-new entity plus at least one verb-phrase edge wiring "
        "it into the plan. Heaviest op -- only when no existing node can be "
        "repaired into place."
    ),
    "delete_node": (
        "Remove an entity and cascade-remove any verb phrase or constraint "
        "that referenced it. Never targets the plan's target blank -- "
        "deleting the target does not fix `target_producible`, it just "
        "turns the violation into a different one."
    ),
    "add_edge": (
        "Add a new verb phrase between two existing nodes without touching "
        "any other node."
    ),
    "delete_edge": (
        "Remove a verb phrase by id. Use when an edge is itself the cause "
        "of a cycle (`acyclic`) or a bad ordering that no reversal fixes."
    ),
    "set_target": (
        "Move `is_target` to a different existing node. Use only when the "
        "current target is unfixable in place and another node already "
        "carries the right shape to be the answer."
    ),
    "add_constraint": (
        "Add a new constraint (derived / argmax / argmin / equals / gt / "
        "lt / in_list) with its output_blank_id set. Use to fix "
        "`target_producible` by giving the target a producer."
    ),
    "delete_constraint": (
        "Remove a constraint by id. Use to fix `acyclic` when a constraint "
        "reads the very blank it writes and nothing else salvages it."
    ),
    "modify_constraint": (
        "Patch fields on an existing constraint (never its id). Use for a "
        "surgical fix to a constraint's operands instead of deleting and "
        "re-adding it."
    ),
}


def _repair_op_models() -> tuple[type, ...]:
    """The pydantic model classes backing `RepairOp`, in `OP_NAMES` order.

    `RepairOp` is `Annotated[Union[...], Field(discriminator="op")]`; this
    unwraps both layers via `typing.get_args` rather than re-listing the
    12 classes by hand, so a model added to the union in `_plan_repair_ops`
    is picked up here automatically.
    """
    union_type = typing.get_args(RepairOp)[0]
    return typing.get_args(union_type)


def _op_name(model: type) -> str:
    const = model.model_fields["op"].annotation
    # `Literal["some_name"]` -- pull the sole literal argument.
    return typing.get_args(const)[0]


def _tool_schema(model: type) -> dict[str, Any]:
    """One OpenAI function-tool schema for `model`.

    `parameters` is derived from `model.model_json_schema()` with the `op`
    discriminator field stripped (the tool NAME already carries it -- an
    LLM tool call has no reason to repeat it as an argument). `$defs` for
    a nested spec (`AddNodeOp`'s `_EdgeSpec`, `AddConstraintOp2`'s
    `_ConstraintSpec`) pass through unchanged.
    """
    name = _op_name(model)
    schema = model.model_json_schema()
    properties = {k: v for k, v in schema.get("properties", {}).items() if k != "op"}
    required = [r for r in schema.get("required", []) if r != "op"]
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    if "$defs" in schema:
        parameters["$defs"] = schema["$defs"]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": _OP_DESCRIPTIONS.get(name, f"Apply the {name!r} repair op."),
            "parameters": parameters,
        },
    }


def build_repair_tools() -> list[dict[str, Any]]:
    """One tool schema per member of `OP_NAMES` -- no more, no fewer.

    `test_every_op_is_offered_as_a_tool` checks this against `OP_NAMES`
    with a set-equality assertion, so an op added to
    `_plan_repair_ops.OP_NAMES` without a corresponding pydantic model
    here would fail loudly rather than silently going un-offered.
    """
    tools = [_tool_schema(m) for m in _repair_op_models()]
    names = {t["function"]["name"] for t in tools}
    assert names == set(OP_NAMES), (
        f"tool/OP_NAMES drift: {sorted(names ^ set(OP_NAMES))}"
    )
    return tools


# ---------------------------------------------------------------------------
# Proposer
# ---------------------------------------------------------------------------


def _render_user_message(
    plan: GSWPlan, violation: Violation, last_reject: str | None
) -> str:
    lines = [
        f"Violation clause: {violation.clause}",
        f"Blank: {violation.blank_id}",
        f"Evidence: {violation.evidence}",
        "",
        "Full plan (JSON):",
        json.dumps(plan.model_dump(), indent=2, default=str),
    ]
    if last_reject:
        lines += [
            "",
            f"Your previous proposal for this SAME violation was rejected: "
            f"{last_reject}",
            "Propose a different op that addresses the rejection reason.",
        ]
    return "\n".join(lines)


def _tool_call_name_and_args(tc: Any) -> tuple[str | None, str | None]:
    """Extract (name, arguments-json-string) from one tool-call entry.

    Two shapes reach this function. `LLMClient.chat()` (the real client,
    `src/panini_dr/models/llm_client.py`) returns `LLMResponse.tool_calls`
    as a list of FLAT DICTS -- `{"id":..., "name":..., "arguments":...}` --
    built that way by every one of its normalizers
    (`_normalize_openai_blocking`, `_consume_stream`,
    `_normalize_responses_blocking`, `_consume_responses_stream`). A raw
    OpenAI SDK tool-call object, by contrast, nests these under
    `.function.name` / `.function.arguments` -- the shape a test double
    that mimics the SDK directly (rather than `LLMResponse`) will use.
    Both are handled here so this module works against the real client
    AND against a double built either way.
    """
    if isinstance(tc, dict):
        if "name" in tc or "arguments" in tc:
            return tc.get("name"), tc.get("arguments")
        fn = tc.get("function") or {}
        return fn.get("name"), fn.get("arguments")
    fn = getattr(tc, "function", None)
    if fn is not None:
        return getattr(fn, "name", None), getattr(fn, "arguments", None)
    return getattr(tc, "name", None), getattr(tc, "arguments", None)


def make_proposer(
    llm: Any, max_tokens: int, seed: int | None
) -> Callable[[GSWPlan, Violation, str | None], dict | None]:
    """Build a `propose(plan, violation, last_reject) -> dict | None`.

    The returned callable is exactly the shape `_plan_repair_loop.
    verify_and_repair` calls: one violation in, one raw op dict (fed
    straight to `_plan_repair_ops.parse_repair_op`) or `None` out. `None`
    means the model declined to call a tool -- the loop reads that as
    "give up on this violation," which demotes its clause after the
    attempt cap (see `_plan_repair_loop.py`).
    """
    tools = build_repair_tools()

    def propose(plan: GSWPlan, violation: Violation, last_reject: str | None) -> dict | None:
        messages = [
            {"role": "system", "content": REPAIRER_SYSTEM},
            {"role": "user", "content": _render_user_message(plan, violation, last_reject)},
        ]
        resp = llm.chat(
            messages,
            tools=tools,
            tool_choice="required",
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed,
        )
        tool_calls = getattr(resp, "tool_calls", None) or []
        if not tool_calls:
            return None
        name, arguments = _tool_call_name_and_args(tool_calls[0])
        if not name:
            return None
        try:
            args = json.loads(arguments) if arguments else {}
        except (json.JSONDecodeError, TypeError):
            # Two distinct ways `arguments` can be unusable, both caught
            # here: malformed JSON text (`json.JSONDecodeError` -- Bedrock's
            # documented corrupted tool-use streaming, or a call truncated
            # mid-argument by a tight `max_tokens`), and `arguments` not
            # being a str/bytes at all (`TypeError` -- e.g. a raw-SDK-shaped
            # tool call whose `.function.arguments` was never JSON-encoded
            # in the first place). Neither may escape `propose` --
            # `verify_and_repair` (`_plan_repair_loop.py`) calls
            # `propose(...)` with no try/except of its own; it only wraps
            # the NEXT line, `parse_repair_op(raw)`. An exception raised
            # here would propagate out of the whole repair pass instead of
            # being handled as one rejected attempt. `None` is the same
            # signal an empty tool-call response already produces: it costs
            # one of `MAX_ATTEMPTS_PER_VIOLATION` and is recoverable (a
            # fresh call next attempt may not be truncated), whereas
            # silently retrying inside `propose` itself would hide the
            # failure from the transcript and never count against the cap.
            return None
        if not isinstance(args, dict):
            # `json.loads` can succeed on syntactically valid JSON that
            # isn't a JSON *object* -- `"null"` parses to `None`, `"[1,2]"`
            # parses to a list. Both crash the `**args` unpack below just as
            # hard as malformed JSON does, and both are reachable the same
            # way malformed JSON is: nothing about a confused model or
            # Bedrock's corrupted streaming (llm_client.py:707-709)
            # guarantees the parsed value is an object, only that
            # `json.loads` doesn't raise. Same give-up signal as above.
            return None
        return {"op": name, **args}

    return propose
