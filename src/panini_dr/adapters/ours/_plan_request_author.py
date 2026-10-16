"""Turn a researcher's plan-change proposal into a frozen, checkable contract.

ONE LLM call. This is the only semantic judgment in the loop; everything after
it is mechanical. The output is constrained to the five-kind vocabulary in
`_plan_request_clauses`, and anything the model emits outside that is DROPPED
rather than passed along -- an unrecognised clause can never be satisfied, so
smuggling one through would stall the contract permanently.

Never raises. Returns [] when it cannot author a usable contract, which the
caller reads as "BINDING clauses only" -- still strictly stronger than the
liveness-only test it replaces.
"""
from __future__ import annotations

import json
import textwrap
from typing import Any, Optional

from panini_dr.adapters.ours._plan_repair_prompt import _tool_call_name_and_args
from panini_dr.adapters.ours._plan_request import PlanRequest
from panini_dr.adapters.ours._plan_request_clauses import CLAUSE_KINDS, Clause
from panini_dr.adapters.ours._planner_exec import GSWPlan

MAX_AUTHOR_ATTEMPTS = 2

AUTHOR_SYSTEM = textwrap.dedent(
    """
    You are the **Contract Author** for a multi-agent question-answering system.

    A researcher tried to fill one blank in a plan, found no evidence, and
    proposed a structural fix. Your job is to turn that proposal into a CONTRACT:
    a short list of requirements the revised plan must meet.

    You do not edit the plan. You state what a correct revision must achieve.
    A separate component proposes the edit and is checked against your contract
    mechanically, so every clause must be objectively decidable.

    Emit clauses ONLY from this vocabulary:

      blank_added                     the plan gains at least one new blank
      depends_on(blank_id)            that blank RESOLVES AFTER something new
      preserved(blank_id)             that blank survives UNCHANGED
      retyped(blank_id, value_type)   that blank ends with this value_type
                                      (one of: date number entity attribute
                                       list text bool)

    ### `depends_on` — name the DOWNSTREAM blank

    `depends_on(B)` is satisfied only when B gains an UPSTREAM dependency, i.e.
    B must now wait for something else. That happens when a new verb phrase
    runs `(new_blank) -> B`, or a new constraint OUTPUTS into B.

    A researcher usually describes the opposite direction: "add `b_new`, linked
    by (`b_X` -> `b_new`)" means the NEW blank is fed FROM `b_X`, so it is
    `b_new` that waits — not `b_X`. The clause is therefore
    `depends_on(b_new)`, NOT `depends_on(b_X)`.

    Ask yourself: after this change, which blank has to be resolved LAST? Name
    that one.

    This is the single most common error. In a measured 824-question run, 29 of
    33 `depends_on` clauses named the upstream blank the researcher happened to
    mention, and went unsatisfied — burning up to three repair attempts each.
    If you are unsure which side gains the dependency, emit `blank_added`
    instead: it is satisfied whenever any new blank appears, and cannot point
    the wrong way.

    Rules:
    - Two to four clauses. Fewer is better. A contract that demands everything
      is unsatisfiable; one that demands nothing is useless.
    - Only name blanks that exist in the plan, EXCEPT that `blank_added` names
      no blank at all.
    - Use `preserved` for blanks already carrying a resolved value that the
      revision must not disturb.
    - Do NOT restate structural rules like acyclicity or ordering. Those are
      always enforced and are not yours to write.

    Call `emit_contract` exactly once.
    """
).strip()

_TOOLS = [{
    "type": "function",
    "function": {
        "name": "emit_contract",
        "description": "Emit the contract the revised plan must satisfy.",
        "parameters": {
            "type": "object",
            "properties": {
                "clauses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string", "enum": sorted(CLAUSE_KINDS)},
                            "blank_id": {"type": "string"},
                            "value_type": {"type": "string"},
                        },
                        "required": ["kind"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["clauses"],
            "additionalProperties": False,
        },
    },
}]


def _render_plan(plan: GSWPlan) -> str:
    lines = []
    for e in plan.entities:
        mark = " [TARGET]" if getattr(e, "is_target", False) else ""
        lines.append(
            f"  {e.id} ({e.kind}, {e.value_type or '?'}){mark}"
            f" hint={e.retrieval_hint or '-'}"
        )
    for vp in plan.verb_phrases:
        lines.append(f"  EDGE {vp.subject_id} -{vp.phrase}-> {vp.object_id}")
    return "\n".join(lines)


def _messages(request: PlanRequest, plan: GSWPlan, retry: bool) -> list[dict[str, Any]]:
    user = textwrap.dedent(
        f"""
        ## Blank that could not be resolved
        {request.blank_id or "(unknown)"}

        ## Why the researcher gave up
        {request.reason}

        ## The researcher's proposed fix
        {request.hint}

        ## Current plan
        {_render_plan(plan)}

        Call `emit_contract` with 2-4 clauses.
        """
    ).strip()
    if retry:
        user += (
            "\n\nYour previous response was not a usable `emit_contract` call. "
            "Call the tool with a `clauses` array and nothing else."
        )
    return [{"role": "system", "content": AUTHOR_SYSTEM},
            {"role": "user", "content": user}]


def _extract(resp: Any) -> Optional[list[dict[str, Any]]]:
    """Pull the clause list out of a tool call.

    Shape-walking is delegated to `_tool_call_name_and_args`
    (`_plan_repair_prompt.py`) rather than re-derived here. The real
    `LLMClient.chat()` always returns `tool_calls` as FLAT DICTS --
    `{"id":..., "name":..., "arguments":...}` -- never nested `.function`
    objects; a bespoke walker that only recognised the nested shape returned
    None against every real response, silently making the whole feature a
    no-op in production while every test (built on the nested shape) stayed
    green. `_tool_call_name_and_args` already handles both shapes and is
    proven against the real client by `_plan_repair_prompt.make_proposer`.
    """
    for tc in (getattr(resp, "tool_calls", None) or []):
        name, raw = _tool_call_name_and_args(tc)
        if name != "emit_contract":
            continue
        if isinstance(raw, dict):
            args = raw
        else:
            try:
                args = json.loads(raw or "")
            except (json.JSONDecodeError, TypeError):
                continue
        if not isinstance(args, dict):
            continue
        clauses = args.get("clauses")
        if isinstance(clauses, list):
            return clauses
    return None


def author_contract(
    request: PlanRequest,
    plan: GSWPlan,
    llm_client: Any,
    max_tokens: int,
    llm_seed: Optional[int],
) -> list[Clause]:
    """Author the contract. Never raises; [] means BINDING clauses only.

    The try/except spans the WHOLE body -- computing `known`, the LLM call,
    response shape-walking, JSON parsing, and the clause filter -- not just
    the network call. Everything past the network call is untrusted model
    output (a `kind` the model emits as a list or dict is unhashable against
    `CLAUSE_KINDS` and raises `TypeError`; a non-str/non-list `tool_calls` or
    a `None` plan raise their own way) and this is a default-on path, so any
    of it failing must degrade to [] exactly like a dropped connection would,
    never propagate.
    """
    try:
        known = {e.id for e in plan.entities}
        for attempt in range(MAX_AUTHOR_ATTEMPTS):
            resp = llm_client.chat(
                messages=_messages(request, plan, retry=attempt > 0),
                tools=_TOOLS,
                tool_choice="required",
                max_tokens=max_tokens,
                temperature=0.0,
                seed=llm_seed,
            )
            raw = _extract(resp)
            if raw is None:
                continue
            out: list[Clause] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                kind = item.get("kind")
                if kind not in CLAUSE_KINDS:
                    continue
                blank_id = item.get("blank_id")
                blank_id = str(blank_id) if blank_id else None
                if kind == "blank_added":
                    # The one kind that names no blank -- normalize away any
                    # stray blank_id the model attaches, or two blank_added
                    # clauses differing only in that ignored field would both
                    # survive the `if c not in out` dedupe below.
                    blank_id = None
                elif blank_id is None or blank_id not in known:
                    # Every other clause naming an unknown id is unsatisfiable
                    # by construction and would leave the contract
                    # permanently incomplete.
                    continue
                vt = item.get("value_type")
                c = Clause(kind, blank_id, str(vt) if vt else None)
                if c not in out:
                    out.append(c)
            return out
        return []
    except Exception:  # noqa: BLE001 — fail-open; replan is default-on
        return []
