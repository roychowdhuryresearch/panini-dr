"""Plan-updater agent.

A one-shot LLM call that revises an existing ``GSWPlan`` in light of
retrieved evidence. Used by the LLM-orchestrator when it believes the
current plan is wrong or incomplete.

**Phase-3.6 contract**: instead of asking the LLM to emit a full
revised plan JSON (which gpt-oss-120b 100% no-op'd on the Phase-3.4
baseline), the LLM emits ONE structured :class:`PlanDiffOp` describing
a single edit (add a blank, change the target, etc.). The op is
applied deterministically via :func:`apply_diff`. A "do nothing"
response is impossible by construction — the diff vocabulary has no
"keep current plan" op.

Inputs:
- ``old_plan``: the current (possibly stale) GSWPlan.
- ``state``: mapping ``blank_id -> BlankResult`` showing what has
  already been resolved.
- ``question``: the original question.
- ``reason``: short note from the orchestrator on WHY an update is
  needed.
- ``evidence``: short string capturing the retrieved facts that
  contradict the current plan.

Output: a revised ``GSWPlan`` plus a ``PlanReconcileDiff`` describing
which blank_ids survived, which were added, and which were dropped.

The updater MUST preserve existing blank_ids wherever the underlying
semantics still apply, because the orchestrator carries resolved
values across the revision by blank_id. (This is enforced
structurally — the diff vocabulary only allows ``add_blank`` to
introduce new ids; existing blanks keep their ids unless explicitly
``remove_blank``-ed.)

**Rejection sampling**: ``update_plan`` retries up to
``MAX_REVISION_ATTEMPTS`` (default 3) times when the LLM emits a
trivial / structurally-no-op op. After the final attempt fails it
raises :class:`PlanEmitError`. The orchestrator catches this and
treats it as ``ok=False``, surfacing the error so the orchestrator
picks a different recovery (re-dispatch / submit_answer).
"""

from __future__ import annotations

import json
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import ValidationError

from panini_dr.adapters.ours._plan_contract import contract_violations
from panini_dr.adapters.ours._plan_diff import (
    DiffApplyError,
    PlanDiffOp,
    apply_diff,
    is_trivial_op,
    parse_diff_op,
    plan_signature,
)
from panini_dr.adapters.ours._plan_request import PlanRequest
from panini_dr.adapters.ours._plan_request_author import author_contract
from panini_dr.adapters.ours._plan_request_clauses import (
    Clause, binding_violations, satisfied_set,
)
from panini_dr.adapters.ours._planner_emit import PlanEmitError, PlanEmitMeta
from panini_dr.adapters.ours._planner_exec import BlankResult, GSWPlan

# Maximum number of LLM calls per ``update_plan`` request. Each
# attempt that returns a trivial op or applies to a no-op signature
# burns one attempt. Failures DO NOT count against
# ``MAX_PLAN_UPDATES_PER_RUN`` in the orchestrator (that cap is for
# orchestrator-level requests).
MAX_REVISION_ATTEMPTS = 3

# Ops that redefine or delete part of the question rather than elaborating it.
# The contract-exhaustion fail-open must never keep one of these over an
# explicit rejection -- see the comment at the fallback assignment in
# `update_plan`.
_DESTRUCTIVE_OPS = frozenset({"change_target", "remove_blank"})


# ---------------------------------------------------------------------------
# Reconcile diff (unchanged shape from Phase-3.4 — used by orchestrator)
# ---------------------------------------------------------------------------


@dataclass
class PlanReconcileDiff:
    """What changed between the old and new plan, by blank id."""

    preserved_ids: list[str] = field(default_factory=list)
    added_ids: list[str] = field(default_factory=list)
    dropped_ids: list[str] = field(default_factory=list)
    invalidated_ids: list[str] = field(default_factory=list)
    # Summary string the orchestrator can surface in its trajectory.
    summary: str = ""
    # Phase-3.6: the structured op the updater emitted, e.g.
    # "add_blank(b_year)". Surfaced in plan_updates trace for debugging.
    op_summary: str = ""
    # Number of LLM attempts used (1..MAX_REVISION_ATTEMPTS).
    attempts: int = 1
    # Plan-request contract (gsw_plan_request_contract). Empty when the flag is
    # off, when no PlanRequest was attached, or when the author failed.
    contract: list[dict] = field(default_factory=list)
    contract_satisfied: list[dict] = field(default_factory=list)
    # Final-review fix wave, Finding 2: explicit signal so an analyst can
    # tell "flag off" (False) apart from "contract active but the author
    # returned an empty contract" (True, with contract == []) -- `contract
    # == []` alone conflates the two. True iff contract_enabled AND a
    # PlanRequest was attached (the authored-clause machinery ran), regardless
    # of what it returned.
    contract_active: bool = False
    # Finding 1: True when every authored clause was already satisfied by the
    # IDENTITY transform (before == after == old_plan) -- e.g. an author that
    # emits only `preserved(x)` (always true of the identity) or
    # `retyped(x, <type already matching>)`. When true, the strict-superset
    # progress check is unsatisfiable by construction (there is nothing left
    # to add), so it is skipped for this update and only the BINDING
    # non-increase check applies. Recorded here rather than silently
    # discarding the clauses.
    contract_identity_satisfied: bool = False
    # True when every attempt was rejected ONLY by the authored-clause
    # progress check and the update was kept anyway (fail-open). Distinguishes
    # "the contract was honoured" from "the contract was not reachable and we
    # took the structurally-valid op rather than lose the replan".
    contract_unmet: bool = False


# ---------------------------------------------------------------------------
# Prompt — diff-op output contract.
# ---------------------------------------------------------------------------


UPDATER_SYSTEM = textwrap.dedent(
    """
    You are the **Plan-Updater** for a multi-agent question-answering
    system. The orchestrator has decided the current GSWPlan is wrong
    or incomplete and asked you to revise it.

    Your output is **one structured edit operation** in JSON, NOT a
    full revised plan. The edit is applied deterministically by the
    framework, the result is re-validated, and the orchestrator
    continues. Returning the current plan is impossible — the edit
    vocabulary has no "keep plan" op.

    ## Edit ops (pick exactly ONE)

    ```
    {"op": "add_blank",
     "blank": {"id": "b_<new>", "value_type": "<type>", "role": "<role>", "is_target": false},
     "wire_via_vp": {"id": "vp_<new>", "phrase": "<predicate>",
                     "subject_id": "<existing_id>", "object_id": "b_<new>"}}

    {"op": "remove_blank", "blank_id": "<existing_blank>"}

    {"op": "change_target", "from_id": "<current_target>", "to_id": "<other_blank>"}

    {"op": "add_vp",
     "vp": {"id": "vp_<new>", "phrase": "<predicate>",
            "subject_id": "<existing>", "object_id": "<existing>"}}

    {"op": "remove_vp", "vp_id": "<existing_vp>"}

    {"op": "add_constraint",
     "constraint": {"id": "c_<new>", "kind": "derived|argmax|argmin|equals|in_list|gt|lt",
                    "op": "diff|sum|avg|max|min|count|concat|mul|div|round_nearest",
                    "args_blanks": ["<blank>", ...],
                    "args_refs": ["<blank_or_filled_literal>", ...],
                    "candidate_entity_ids": ["<id>", ...],
                    "sort_by_blank_ids": ["<blank>", ...],
                    "left_ref": "<id>", "right_ref": "<id>",
                    "output_blank_id": "<blank>"}}

    {"op": "replace_constraint",
     "constraint_id": "<existing_constraint>",
     "constraint": {"id": "<same_or_fresh_constraint_id>",
                    "kind": "derived|argmax|argmin|equals|in_list|gt|lt",
                    "op": "diff|sum|avg|max|min|count|concat|mul|div|round_nearest",
                    "args_blanks": ["<blank>", ...],
                    "args_refs": ["<blank_or_filled_literal>", ...],
                    "candidate_entity_ids": ["<id>", ...],
                    "sort_by_blank_ids": ["<blank>", ...],
                    "left_ref": "<id>", "right_ref": "<id>",
                    "output_blank_id": "<blank>"}}

    {"op": "remove_constraint", "constraint_id": "<existing>"}

    {"op": "change_value_type", "blank_id": "<existing>",
     "to": "date|number|entity|attribute|list|text|bool"}
    ```

    ## Rules

    1. New blank ids MUST be fresh — not present in the current plan.
       Use names like `b_bridge1`, `b_year`, `b_intermediate_X`.
    2. New VP/constraint ids MUST also be fresh.
    3. `add_blank` MUST come with a `wire_via_vp` that connects the
       new blank to an existing entity. A blank with no VP is
       structurally dangling and will be rejected.
    4. `change_target.from_id` MUST be the current target.
       `change_target.to_id` MUST be an existing blank.
    5. `change_value_type.to` MUST differ from the current value_type.
    6. `change_target` and `add_vp`/`remove_vp` must not introduce a
       cycle in the dependency graph.
    7. Filled entities (`kind=filled`) are grounded in the question
       text — never rename or remove them.
    8. For derived constraints, prefer `args_refs` when an input is a
       filled literal/constraint-value entity; keep `args_blanks` for
       old blank-only constraints. For `round_nearest`, one arg rounds
       to the nearest whole number; two args use the second input as
       the interval. For "nearest ten", add/reference a filled
       constraint-value with `literal_value: 10`.
       For `in_list`, use either left_ref/right_ref or
       args_blanks=[member_blank, list_blank].
    9. Use `replace_constraint` when the correct edit is to change an
       existing constraint's inputs, input order, output, operator, or
       kind. Do NOT spend two plan updates on remove-then-add for the
       same constraint. For chronological/order fixes, replace the
       existing `concat` constraint with the same constraint id and the
       correctly ordered `args_blanks`.

    ## Worked examples

    **Example 1 — orchestrator says a bridge step is missing:**
    Current plan: `(e_movie) --[directed_by]--> (t)` where `t` is the
    target director. But the question is "what won the Oscar in the
    year that movie X was directed". Orchestrator's `reason`:
    "missing intermediate step: bridge year between movie X and the
    Oscar winner." Right move:
    ```
    {"op": "add_blank",
     "blank": {"id": "b_year", "value_type": "date", "role": "bridge-date"},
     "wire_via_vp": {"id": "vp_year", "phrase": "released_in",
                     "subject_id": "e_movie", "object_id": "b_year"}}
    ```

    **Example 2 — orchestrator says target value is wrong type:**
    Current plan target `t` has `value_type: number` but the question
    expects an entity name. Orchestrator's `reason`: "committed value
    is a count but question asks for the player name." Right move:
    ```
    {"op": "change_value_type", "blank_id": "t", "to": "entity"}
    ```

    **Example 3 — orchestrator says a constraint dead-ends the search:**
    The plan has `argmax` over a candidate list that the corpus
    doesn't contain. Orchestrator's `reason`: "constraint c1 expects
    candidates that aren't in the corpus." Right move:
    ```
    {"op": "remove_constraint", "constraint_id": "c1"}
    ```

    Return ONLY the JSON op object. No prose, no markdown fences.
    """
).strip()


_UPDATER_HINTS_INSTRUCTION = (
    "\n\n## Blank hints (when adding a blank)\n"
    "If your diff op adds a new blank (`add_blank`), include "
    "`retrieval_hint` and `verification_claim` on that blank "
    "inside the `blank` object:\n"
    "  - retrieval_hint: a human-style search question for the new blank.\n"
    "  - verification_claim: one precise sentence the value must satisfy.\n"
    "  - Use `{<upstream_blank_id>}` placeholders for values that depend on an "
    "earlier blank; never a raw id otherwise.\n"
    "  - Preserve existing hints when a blank's meaning is unchanged; update "
    "them when it changes."
)


def _state_summary_for_prompt(
    old_plan: GSWPlan,
    state: dict[str, BlankResult],
) -> str:
    """Render the current state for the updater prompt."""
    lines: list[str] = []
    ent_by_id = {e.id: e for e in old_plan.entities}
    for bid, br in state.items():
        ent = ent_by_id.get(bid)
        role = getattr(ent, "role", "—") or "—"
        vt = getattr(ent, "value_type", "text") or "text"
        if br.status == "resolved":
            val = br.value
            if isinstance(val, str) and len(val) > 80:
                val = val[:77] + "..."
            lines.append(
                f"- `{bid}` (role=`{role}`, value_type=`{vt}`, "
                f"RESOLVED): {val!r}"
            )
        else:
            lines.append(
                f"- `{bid}` (role=`{role}`, value_type=`{vt}`, "
                f"status=`{br.status}`)"
            )
    if not lines:
        return "_(no blanks yet)_"
    return "\n".join(lines)


def _render_plan_for_updater(plan: GSWPlan) -> str:
    """Compact textual rendering of the plan for the updater prompt.

    Includes ids, kinds, value_types, and the full VP / constraint
    graph so the updater knows what it's editing without us pasting
    the raw pydantic dump.
    """
    lines: list[str] = ["### Entities"]
    for e in plan.entities:
        flags = []
        if e.is_target:
            flags.append("TARGET")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        if e.kind == "filled":
            literal = (
                f", literal_value={e.literal_value!r}"
                if e.literal_value is not None
                else ""
            )
            lines.append(
                f"- `{e.id}` (filled, role=`{e.role or '—'}`): "
                f"name={e.name!r}{literal}{flag_str}"
            )
        else:
            lines.append(
                f"- `{e.id}` (blank, role=`{e.role or '—'}`, "
                f"value_type=`{e.value_type or '—'}`){flag_str}"
            )

    lines.append("\n### Verb-phrases")
    if plan.verb_phrases:
        for vp in plan.verb_phrases:
            lines.append(
                f"- `{vp.id}`: ({vp.subject_id}) --[{vp.phrase}]--> ({vp.object_id})"
            )
    else:
        lines.append("_(none)_")

    lines.append("\n### Constraints")
    if plan.constraints:
        for c in plan.constraints:
            args = []
            if c.op:
                args.append(f"op={c.op}")
            if c.args_blanks:
                args.append(f"args={c.args_blanks}")
            if c.args_refs:
                args.append(f"args_refs={c.args_refs}")
            if c.candidate_entity_ids:
                args.append(f"candidates={c.candidate_entity_ids}")
            if c.sort_by_blank_ids:
                args.append(f"sort_by={c.sort_by_blank_ids}")
            if c.output_blank_id:
                args.append(f"output={c.output_blank_id}")
            lines.append(f"- `{c.id}` (kind={c.kind}): " + ", ".join(args))
    else:
        lines.append("_(none)_")
    return "\n".join(lines)


def build_updater_messages(
    question: str,
    old_plan: GSWPlan,
    state: dict[str, BlankResult],
    reason: str,
    evidence: str,
    *,
    blank_hints: bool = False,
) -> list[dict[str, Any]]:
    """System + user message list for the diff-op updater call."""
    plan_render = _render_plan_for_updater(old_plan)
    state_block = _state_summary_for_prompt(old_plan, state)
    user = textwrap.dedent(
        f"""
        ## Question

        {question}

        ## Current plan

        {plan_render}

        ## State so far

        {state_block}

        ## Orchestrator's reason for asking for a revision

        {reason.strip() or "_(no reason supplied)_"}

        ## Evidence

        {evidence.strip() or "_(no evidence supplied)_"}

        ## Task

        Emit ONE diff op (JSON object) that reconciles the plan with the
        evidence. Pick the smallest meaningful change — `add_blank`
        for a missing intermediate, `change_target` for a wrong target
        blank, `add_constraint` for a new aggregation/filter, or
        `replace_constraint` when an existing constraint has the wrong
        args/order/operator. Return ONLY the JSON object.
        """
    ).strip()

    system = UPDATER_SYSTEM + (_UPDATER_HINTS_INSTRUCTION if blank_hints else "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_updater_retry_messages(
    question: str,
    old_plan: GSWPlan,
    state: dict[str, BlankResult],
    reason: str,
    evidence: str,
    bad_response: str,
    failure_detail: str,
    *,
    blank_hints: bool = False,
) -> list[dict[str, Any]]:
    """Sharpened retry — appends the failure detail and demands a
    different op."""
    msgs = build_updater_messages(
        question, old_plan, state, reason, evidence, blank_hints=blank_hints
    )
    msgs.append({"role": "assistant", "content": bad_response})
    msgs.append(
        {
            "role": "user",
            "content": textwrap.dedent(
                f"""
                Your previous response failed: **{failure_detail}**

                Emit a DIFFERENT diff op that:
                - Has fresh ids for any new blank/VP/constraint
                - Actually changes the plan structure (not a no-op shape)
                - Addresses the orchestrator's reason above

                If `add_blank` was rejected because the id collides,
                pick a different fresh id. If a constraint edit failed
                because remove/add left a dangling target or duplicate
                output, use `replace_constraint` with the corrected
                complete constraint. If `change_target` was rejected,
                you may instead want `change_value_type` or `add_blank`
                for a missing bridge step. Return ONLY the JSON op
                object.
                """
            ).strip(),
        }
    )
    return msgs


# ---------------------------------------------------------------------------
# Main entry point — rejection sampling over diff ops.
# ---------------------------------------------------------------------------


def _try_parse_op(text: str) -> tuple[Optional[PlanDiffOp], str]:
    """Try to extract a PlanDiffOp from raw LLM text.

    Returns ``(op, "")`` on success or ``(None, error_msg)``.
    """
    if not text:
        return None, "empty response"
    # Strip common markdown fences.
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the first line and the trailing fence.
        first_nl = stripped.find("\n")
        if first_nl >= 0:
            stripped = stripped[first_nl + 1 :]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3].rstrip()
    # Some models emit a JSON object embedded in prose; find the first {...}.
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError:
        # Last-ditch: scan for the first '{' through the last '}'.
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            return None, f"no JSON object found in response: {stripped[:120]!r}"
        try:
            raw = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            return None, f"JSON decode failed: {exc}"
    if not isinstance(raw, dict):
        return None, f"expected JSON object, got {type(raw).__name__}"
    try:
        op = parse_diff_op(raw)
    except ValidationError as exc:
        return None, f"diff op validation failed: {exc.errors()[0].get('msg', exc)}"
    return op, ""


def _op_summary(op: PlanDiffOp) -> str:
    """Short human-readable summary of the op for trace logs."""
    name = op.op
    if name == "add_blank":
        return f"add_blank({op.blank.id} via {op.wire_via_vp.id})"
    if name == "remove_blank":
        return f"remove_blank({op.blank_id})"
    if name == "change_target":
        return f"change_target({op.from_id} -> {op.to_id})"
    if name == "add_vp":
        return f"add_vp({op.vp.id})"
    if name == "remove_vp":
        return f"remove_vp({op.vp_id})"
    if name == "add_constraint":
        return f"add_constraint({op.constraint.id})"
    if name == "replace_constraint":
        return f"replace_constraint({op.constraint_id} -> {op.constraint.id})"
    if name == "remove_constraint":
        return f"remove_constraint({op.constraint_id})"
    if name == "change_value_type":
        return f"change_value_type({op.blank_id} -> {op.to})"
    return name


def _how_to_satisfy(clause) -> str:
    """The OP SHAPE that would satisfy `clause`, for the retry prompt.

    The rejection message used to name the clause and stop there -- "still
    unmet: ['depends_on(b_writer1)']" -- which tells the updater WHICH
    requirement failed but not HOW to meet it. Measured cost: `depends_on` was
    satisfied 4/33 in the 824 run and 0/5 after an author-side prompt fix, with
    the same error every time -- a verb phrase wired `b_X -> b_new` when the
    clause needs `b_new -> b_X`. Three prompt attempts aimed at the AUTHOR
    (which clause to choose) moved nothing; none had told the UPDATER, the
    component that must actually act, what shape of op complies.

    The updater follows concrete structural instructions faithfully -- it
    implements researcher hints verbatim -- so it is given one here.
    """
    kind = getattr(clause, "kind", None)
    b = getattr(clause, "blank_id", None)
    if kind == "blank_added":
        return "add_blank -- any new blank satisfies this"
    if kind == "depends_on":
        return (
            f"make {b!r} DEPEND ON something new. Either add_vp with "
            f"subject_id=<the new/other blank> and object_id={b!r}, or "
            f"add_constraint with output_blank_id={b!r}. DIRECTION MATTERS: a "
            f"verb phrase with {b!r} as the SUBJECT points the other way and "
            f"does NOT satisfy this -- {b!r} must be the OBJECT"
        )
    if kind == "preserved":
        return f"leave {b!r} exactly as it is -- do not edit, retype or remove it"
    if kind == "retyped":
        vt = getattr(clause, "value_type", None)
        return f"change_value_type with blank_id={b!r} and to={vt!r}"
    return "propose an op that changes the plan in a way relevant to the reason above"


def update_plan(
    *,
    old_plan: GSWPlan,
    state: dict[str, BlankResult],
    question: str,
    reason: str,
    evidence: str,
    llm_client: Any,
    max_tokens: int = 4096,
    enable_repair: bool = True,
    llm_seed: int | None = None,
    blank_hints: bool = False,
    plan_request: Optional[PlanRequest] = None,
    contract_enabled: bool = False,
    carried_contract: Optional[list[dict]] = None,
) -> tuple[GSWPlan, PlanReconcileDiff, PlanEmitMeta]:
    """Rejection-sampling loop over diff ops.

    Up to :data:`MAX_REVISION_ATTEMPTS` LLM calls. The first call
    builds the system/user prompt; each subsequent call appends a
    sharpened retry message describing why the previous attempt was
    rejected (parse failure / trivial op / DiffApplyError /
    no-op-after-apply).

    Returns ``(new_plan, diff, meta)`` on success. Raises
    :class:`PlanEmitError` on terminal failure — the orchestrator
    catches and surfaces ``{"ok": false, "error": ...}``.

    The ``enable_repair`` flag is kept for API compat with the
    Phase-3.4 caller; the new contract retries on any failure regardless,
    and the orchestrator's ``max_attempts`` is governed by
    ``MAX_REVISION_ATTEMPTS``.
    """
    meta = PlanEmitMeta()
    last_bad_response = ""
    last_failure_detail = ""
    old_signature = plan_signature(old_plan)

    # Author the contract ONCE, then freeze it. Semantic judgment happens here
    # and nowhere else; every accept/reject below is mechanical, which is what
    # bounds the loop. An empty contract means BINDING clauses only -- still
    # strictly stronger than the liveness test this replaces.
    #
    # CARRY-FORWARD (added after FRAMES q456). A contract used to bind only the
    # update that cited a `plan_request_id`; every LATER update on the same
    # question ran with no contract at all. Measured consequence on q456
    # ("how old will Heath Ledger's daughter be on the film's 50th
    # anniversary?", gold 52): update 1 was gated, and produced exactly the
    # right plan -- `sum(b_release_date, e_fifty) -> b_anniversary_date`, so
    # age = anniversary - birth = 52. Update 2 cited nothing, ran ungated, and
    # rewired the age constraint to `diff(b_release_date, b_child_birth_date)`
    # -- the child's age at RELEASE, not at the anniversary. Answer: 2. The
    # flag-off arm, which never replanned at all, answered 52.
    #
    # Replaying that edit against the SAME contract rejects it: the rewiring
    # drops `b_anniversary_date` from the dependency graph, which un-satisfies
    # `depends_on(b_anniversary_date)` -- the very clause the researcher's
    # escalation existed to establish -- so the satisfied set SHRINKS and the
    # strict-superset rule refuses it.
    #
    # So once a contract exists for a question it governs every subsequent
    # update on that question. The caller passes it back as `carried_contract`;
    # the baseline is re-derived against the CURRENT old_plan, so "progress" is
    # always measured from where the plan actually is, not from where it began.
    clauses: list[Clause] = []
    carried = False
    if contract_enabled and carried_contract:
        clauses = [Clause(c.get("kind"), c.get("blank_id"), c.get("value_type"))
                   for c in carried_contract if c.get("kind")]
        carried = bool(clauses)
    if contract_enabled and not clauses and plan_request is not None:
        try:
            clauses = author_contract(
                plan_request, old_plan, llm_client, max_tokens, llm_seed)
        except Exception:  # noqa: BLE001 — fail-open; replan is default-on
            clauses = []
    contract_active = bool(
        contract_enabled and (plan_request is not None or carried))
    base_satisfied = satisfied_set(clauses, old_plan, old_plan) if clauses else frozenset()
    # Final-review fix wave, Finding 1: two clause kinds are true of the
    # IDENTITY transform (before == after == old_plan) -- `preserved(x)`
    # always, and `retyped(x, <current type>)` whenever the type already
    # matches. The author is explicitly invited to emit `preserved` for a
    # blank "already carrying a resolved value"
    # (`_plan_request_author.py`), so a contract made ONLY of such clauses
    # is possible in production. When that happens, `base_satisfied` already
    # equals the full clause set before any op has run, so the strict-
    # superset progress check below (`now > base_satisfied`) can never pass
    # -- there is nothing left to add, no matter what the op does. Every
    # attempt would be rejected, all MAX_REVISION_ATTEMPTS burn, and
    # update_plan raises PlanEmitError, losing the replan entirely -- the
    # same failure class the non-increase binding fix (fix round 1) already
    # removed from a different code path. Detected once, up front (clauses
    # and base_satisfied are frozen for the whole call): when true, this
    # update falls back to BINDING-only checking -- the strict-superset
    # clause check is skipped, the non-increase binding check still applies
    # -- and the fact is recorded on the diff rather than the clauses being
    # silently discarded.
    identity_satisfied = bool(clauses) and base_satisfied == frozenset(clauses)
    # First op that cleared parse/trivial/apply/signature AND the BINDING
    # non-increase check but failed only the authored-clause progress check.
    # Returned on attempt exhaustion instead of raising -- see the comment at
    # the fallback assignment below. `None` means no such op was ever seen, in
    # which case exhaustion is a genuine failure and still raises.
    fallback: Optional[tuple[GSWPlan, Any, int, frozenset]] = None
    # Baseline BINDING-violation counts, computed ONCE -- old_plan does not
    # change between attempts. Fix round 1: the acceptance rule is
    # non-increase, not absolute-zero. Measured on FRAMES-824: 31 of 824
    # shipped plans already carry an `ordering` violation the emit-time
    # repair loop demoted rather than fixed (7 of those 31 also replanned).
    # An absolute "must be clean" rule rejects every candidate op on exactly
    # those plans, burns all attempts, and the orchestrator loses the
    # replan entirely -- on the plans most in need of one. A Counter
    # (per-clause-KIND multiset), not a set of names, so (a) a plan that
    # trades away one kind's violation while gaining a different kind's
    # doesn't net to "unchanged", and (b) a second violation of the SAME
    # kind -- which a plain set of clause names would collapse and miss
    # entirely -- still counts as an increase.
    # Final-review fix wave, Finding 3: gated on `contract_enabled` alone,
    # not `contract_active` -- this check costs no LLM call and is a strict
    # improvement, so it must run even when the orchestrator never attaches
    # a PlanRequest (measured: at least 71 of 134 replanning questions never
    # escalate at all, so `contract_active` was False for all of them and
    # the binding check never ran). Only the AUTHORED-clause machinery
    # (`clauses`, `base_satisfied`, the strict-superset check) still
    # requires a PlanRequest.
    old_binding_counts: "Counter[str]" = (
        Counter(binding_violations(old_plan, question)) if contract_enabled else Counter()
    )

    for attempt in range(1, MAX_REVISION_ATTEMPTS + 1):
        if attempt == 1:
            messages = build_updater_messages(
                question, old_plan, state, reason, evidence,
                blank_hints=blank_hints,
            )
        else:
            messages = build_updater_retry_messages(
                question, old_plan, state, reason, evidence,
                bad_response=last_bad_response,
                failure_detail=last_failure_detail,
                blank_hints=blank_hints,
            )

        try:
            resp = llm_client.chat(
                messages=messages,
                max_tokens=max_tokens,
                seed=llm_seed,
            )
        except Exception as exc:  # noqa: BLE001
            raise PlanEmitError(kind="llm_error", detail=str(exc)) from exc

        text = getattr(resp, "text", "") or ""
        if attempt == 1:
            meta.raw_response = text
            meta.raw_reasoning = getattr(resp, "reasoning_content", "") or ""
        else:
            meta.repair_used = True
            meta.repair_response = text
            meta.repair_reasoning = getattr(resp, "reasoning_content", "") or ""
        meta.prompt_tokens += int(getattr(resp, "prompt_tokens", 0) or 0)
        meta.completion_tokens += int(getattr(resp, "completion_tokens", 0) or 0)
        last_bad_response = text

        # Parse + validate + apply, in order. Any failure → retry.
        op, parse_err = _try_parse_op(text)
        if op is None:
            last_failure_detail = f"parse failure: {parse_err}"
            meta.parse_error = parse_err[:400]
            continue

        trivial, trivial_reason = is_trivial_op(op, old_plan)
        if trivial:
            last_failure_detail = f"trivial op rejected: {trivial_reason}"
            continue

        try:
            new_plan = apply_diff(old_plan, op)
        except DiffApplyError as exc:
            last_failure_detail = f"apply_diff failed: {exc}"
            continue

        # Belt-and-suspenders: even if we got here, double-check the
        # signature actually changed. This handles the (unlikely)
        # case of a diff that mutates fields the signature ignores.
        if plan_signature(new_plan) == old_signature:
            last_failure_detail = (
                "post-apply signature equals input plan (no-op)"
            )
            continue

        if contract_enabled:
            new_binding_counts = Counter(binding_violations(new_plan, question))
            increased = sorted(
                kind for kind, count in new_binding_counts.items()
                if count > old_binding_counts.get(kind, 0)
            )
            if increased:
                # The measured failure: 15 of 191 updates increased the binding
                # violation count, including 2 `acyclic` that emit-time never
                # produces. Reject and let the retry message name the clause
                # AND (fix round 1, Finding 3) the concrete evidence for it --
                # `binding_violations` only returns clause names (Task 2's
                # frozen return type, left untouched), so evidence comes from
                # calling `contract_violations` directly here, at the
                # rejection site, rather than widening that interface.
                old_total = sum(old_binding_counts.values())
                new_total = sum(new_binding_counts.values())
                evidence_by_kind: dict[str, str] = {}
                for v in contract_violations(new_plan, question):
                    if v.clause in increased and v.clause not in evidence_by_kind:
                        evidence_by_kind[v.clause] = v.evidence
                # NOT `evidence` -- that name is the function's `evidence`
                # PARAMETER (the researcher's retrieved evidence, threaded
                # into every subsequent retry prompt at
                # `build_updater_retry_messages` below). Reusing it here
                # reassigns the same local, permanently replacing the
                # researcher's evidence with violation text for the rest of
                # the loop from the first rejection onward (fix round 2,
                # caught in review).
                violation_evidence = "; ".join(evidence_by_kind[k] for k in increased)
                last_failure_detail = (
                    f"the revised plan raises binding violations from "
                    f"{old_total} to {new_total} ({', '.join(increased)})"
                    f"{f': {violation_evidence}' if violation_evidence else ''}; "
                    "propose a different op that does not make the plan's "
                    "existing structural problems worse"
                )
                continue
            # Finding 1: the strict-superset progress check is skipped when
            # every authored clause was already true of the identity -- see
            # the comment on `identity_satisfied` above. Only BINDING
            # non-increase (just checked) gates this update in that case.
            if clauses and not identity_satisfied:
                now = satisfied_set(clauses, old_plan, new_plan)
                if not (now > base_satisfied):
                    unmet = [c for c in clauses if c not in now]
                    unmet_desc = [
                        f"{c.kind}({c.blank_id})" if c.blank_id else c.kind
                        for c in unmet
                    ]
                    # Corollary to Finding 1: `unmet_desc` can only be empty
                    # when `now` already equals the full clause set, which --
                    # given `now` is always a subset of `base_satisfied ∪
                    # clauses` and rejection means `now` is NOT a strict
                    # superset of `base_satisfied` -- only happens when
                    # `base_satisfied` already equals the full clause set,
                    # i.e. exactly the `identity_satisfied` case this branch
                    # now excludes. Kept as a defensive fallback (not
                    # reachable today) rather than ever rendering the
                    # literal, unhelpful "still unmet: []" to the LLM.
                    if unmet_desc:
                        how = "; ".join(
                            f"{d} -> {_how_to_satisfy(c)}"
                            for d, c in zip(unmet_desc, unmet)
                        )
                        last_failure_detail = (
                            "the revised plan does not advance the contract. "
                            f"Unmet, and what would satisfy each: {how}"
                        )
                    else:
                        last_failure_detail = (
                            "the revised plan does not advance the contract: "
                            "every authored clause was already satisfied "
                            "before this op ran, so none remains to name as "
                            "unmet -- propose an op that changes the plan in "
                            "a way relevant to the researcher's reason above"
                        )
                    # FAIL-OPEN on contract exhaustion. Observed live on
                    # FRAMES q351: the author wrote
                    # `depends_on(b_other_astronaut)`, no op reached it in
                    # MAX_REVISION_ATTEMPTS, every attempt was rejected HERE,
                    # and update_plan raised -- losing the replan outright on
                    # a question whose researcher had escalated precisely
                    # because the plan was wrong.
                    #
                    # That is the same failure class as the identity-satisfied
                    # case above and as the absolute-zero binding rule before
                    # it, arriving through a third door: an authored clause
                    # that is satisfiable in principle but not reached inside
                    # the attempt budget. The branch's stated constraint is
                    # fail-open everywhere -- an authored clause is a
                    # PREFERENCE about how the plan should change, never a
                    # licence to discard a structurally-valid update.
                    #
                    # So remember the first op that cleared everything except
                    # this progress check -- it parsed, was non-trivial,
                    # applied, changed the signature, and did NOT increase the
                    # binding-violation count. On exhaustion it is returned
                    # instead of raising, flagged `contract_unmet` so a trace
                    # shows the contract was not honoured rather than the
                    # replan silently vanishing.
                    # ...but NEVER fail open on a DESTRUCTIVE op. Observed live
                    # on FRAMES q806: "what is greater, the combined 2011
                    # populations of four villages, or the 2022 population of
                    # London?" The contract rejected `change_target(t ->
                    # b_sum_2011)` on all three attempts; the fail-open kept it;
                    # the system stopped answering "which is greater" and
                    # answered "what is the sum" -- 34765 instead of "London".
                    # The flag-off arm got it right.
                    #
                    # `change_target` redefines WHICH QUESTION is being
                    # answered and `remove_blank` deletes a step; keeping either
                    # over a contract's explicit objection is strictly worse
                    # than losing the replan, which merely leaves the previous
                    # plan -- and the previous plan was answering the right
                    # question. Additive ops (add_blank, add_vp,
                    # add_constraint, change_value_type) can only elaborate the
                    # plan, so those still fail open.
                    if fallback is None and getattr(op, "op", None) not in _DESTRUCTIVE_OPS:
                        fallback = (new_plan, op, attempt, now)
                    continue

        # Success.
        diff = _reconcile_state(old_plan, new_plan, state)
        diff.op_summary = _op_summary(op)
        diff.attempts = attempt
        diff.contract_active = contract_active
        diff.contract_identity_satisfied = identity_satisfied
        if contract_active:
            diff.contract = [
                {"kind": c.kind, "blank_id": c.blank_id, "value_type": c.value_type}
                for c in clauses
            ]
            diff.contract_satisfied = [
                {"kind": c.kind, "blank_id": c.blank_id, "value_type": c.value_type}
                for c in satisfied_set(clauses, old_plan, new_plan)
            ]
        diff.summary = (
            f"{diff.summary} via {diff.op_summary}"
            f"{f' (after {attempt} attempts)' if attempt > 1 else ''}"
        )
        return new_plan, diff, meta

    # All attempts exhausted. If some op was structurally fine and merely
    # failed to advance the authored contract, take it rather than losing the
    # replan -- fail-open, per the comment at the fallback assignment.
    if fallback is not None:
        new_plan, op, attempt, now = fallback
        diff = _reconcile_state(old_plan, new_plan, state)
        diff.op_summary = _op_summary(op)
        diff.attempts = MAX_REVISION_ATTEMPTS
        diff.contract_active = contract_active
        diff.contract_identity_satisfied = identity_satisfied
        diff.contract_unmet = True
        if contract_active:
            diff.contract = [
                {"kind": c.kind, "blank_id": c.blank_id, "value_type": c.value_type}
                for c in clauses
            ]
            diff.contract_satisfied = [
                {"kind": c.kind, "blank_id": c.blank_id, "value_type": c.value_type}
                for c in now
            ]
        diff.summary = (
            f"{diff.summary} via {diff.op_summary} (contract unmet after "
            f"{MAX_REVISION_ATTEMPTS} attempts; kept the structurally-valid op)"
        )
        return new_plan, diff, meta

    raise PlanEmitError(
        kind="parse_failure",
        detail=(
            f"plan-updater could not produce a structural change after "
            f"{MAX_REVISION_ATTEMPTS} attempts. last failure: "
            f"{last_failure_detail}"
        ),
    )


# ---------------------------------------------------------------------------
# State reconciliation — mutates ``state`` in place
# ---------------------------------------------------------------------------


def _reconcile_state(
    old_plan: GSWPlan,
    new_plan: GSWPlan,
    state: dict[str, BlankResult],
) -> PlanReconcileDiff:
    """Preserve resolved blanks whose ids survive; drop obsolete ids;
    init ``unknown`` for newly-introduced blank ids. Mutates
    ``state`` in place and returns a summary."""
    old_ids = {e.id for e in old_plan.blank_entities()}
    new_ids = {e.id for e in new_plan.blank_entities()}

    preserved = sorted(old_ids & new_ids)
    added = sorted(new_ids - old_ids)
    dropped = sorted(old_ids - new_ids)
    invalidated = _constraint_outputs_to_invalidate(old_plan, new_plan)

    # Add unknown entries for newly-introduced ids.
    for bid in added:
        if bid not in state:
            state[bid] = BlankResult(blank_id=bid, status="unknown")

    # Drop state for ids no longer in the plan.
    for bid in dropped:
        state.pop(bid, None)

    # If a producing constraint changed, any old computed value for its
    # output is stale. Reset the output and downstream computed outputs
    # so the next cascade recomputes them from the revised plan.
    for bid in invalidated:
        if bid in state:
            state[bid] = BlankResult(blank_id=bid, status="unknown")

    summary = (
        f"preserved={len(preserved)}, added={len(added)}, dropped={len(dropped)}"
    )
    if invalidated:
        summary = f"{summary}, invalidated={len(invalidated)}"
    return PlanReconcileDiff(
        preserved_ids=preserved,
        added_ids=added,
        dropped_ids=dropped,
        invalidated_ids=invalidated,
        summary=summary,
    )


def _constraint_outputs_to_invalidate(
    old_plan: GSWPlan,
    new_plan: GSWPlan,
) -> list[str]:
    old_constraints = {c.id: c for c in old_plan.constraints}
    new_constraints = {c.id: c for c in new_plan.constraints}
    new_blank_ids = {e.id for e in new_plan.blank_entities()}
    invalidated: set[str] = set()

    for cid, old_c in old_constraints.items():
        new_c = new_constraints.get(cid)
        if new_c is None:
            if old_c.output_blank_id in new_blank_ids:
                invalidated.add(old_c.output_blank_id)
            continue
        if _constraint_fingerprint(old_c) != _constraint_fingerprint(new_c):
            for out_id in (old_c.output_blank_id, new_c.output_blank_id):
                if out_id in new_blank_ids:
                    invalidated.add(out_id)

    for cid, new_c in new_constraints.items():
        if cid not in old_constraints and new_c.output_blank_id in new_blank_ids:
            invalidated.add(new_c.output_blank_id)

    changed = True
    while changed:
        changed = False
        for c in new_plan.constraints:
            out_id = c.output_blank_id
            if not out_id or out_id not in new_blank_ids or out_id in invalidated:
                continue
            inputs = _constraint_input_refs(c)
            if any(ref in invalidated for ref in inputs):
                invalidated.add(out_id)
                changed = True

    return sorted(invalidated)


def _constraint_fingerprint(c) -> tuple:
    return (
        c.id,
        c.kind,
        c.op or "",
        tuple(c.args_blanks),
        tuple(c.args_refs),
        tuple(c.candidate_entity_ids),
        tuple(c.sort_by_blank_ids),
        c.left_ref or "",
        c.right_ref or "",
        c.output_blank_id or "",
    )


def _constraint_input_refs(c) -> list[str]:
    if c.kind == "derived":
        return [*c.args_refs, *c.args_blanks]
    if c.kind in {"argmax", "argmin"}:
        return list(c.sort_by_blank_ids or [])
    if c.kind in {"equals", "in_list", "gt", "lt"}:
        return [ref for ref in (c.left_ref, c.right_ref) if ref]
    return []
