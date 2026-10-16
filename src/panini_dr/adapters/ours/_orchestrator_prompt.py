"""System-prompt construction for the LLM orchestrator.

The orchestrator does not retrieve directly; it decides which blanks
to dispatch to a per-blank researcher, when to ask the plan-updater
to revise the plan, and when to submit the final answer.

It is one live conversation. Before a plan exists its system prompt is
``build_planner_bootstrap_prompt``; once the plan is installed it is
``build_orchestrator_system_static`` (mission, plan schema legend, plan
brief, advisory plan-quality warnings, rules), which never changes
during the question. The state table arrives as a conversation turn
(``build_state_message``) and every later turn appends its tool result
plus a one-line status (``build_live_status_line``).
"""

from __future__ import annotations

import textwrap
from typing import Any

from panini_dr.adapters.ours._planner_exec import BlankResult
from panini_dr.adapters.ours._planner_react_prompt import (
    SYSTEM_PROMPT_LEGEND,
    render_plan_brief,
)

ORCHESTRATOR_MISSION = textwrap.dedent(
    """
    You are the **Orchestrator** of a multi-agent research team
    answering one question.

    You hold:
    - The question (in the user message).
    - The current **GSWPlan**: a typed DAG of entities, verb-phrases,
      and constraints. One blank entity is the target whose value is
      the final answer.
    - The **global state**: which blanks are resolved, their committed
      values, and their evidence chunk ids.

    Each turn you pick EXACTLY ONE of four tools:

    - `dispatch_subplan(blank_ids: list[str], hints?: str)` — ask a
      fan-out of researchers to retrieve and commit values for a
      specific set of blanks. Each blank in the list is resolved by
      its OWN researcher in parallel. Use this for the main
      retrieval work.
    - `request_plan_update(reason: str, evidence: str)` — the plan is
      wrong or missing something. A plan-updater LLM rewrites the
      plan from the evidence you provide; the revised plan replaces
      the current one and resolved blanks are carried over by id
      where possible.
    - `get_state()` — re-read the current state. Rarely needed;
      state is already in this prompt.
    - `submit_answer(answer: str)` — terminate with the final answer.
      Only allowed when the TARGET blank is resolved in state. The
      answer should match the committed target value.

    The run ends when you call `submit_answer` (success) or hit the
    turn budget (fail).
    """
).strip()


ORCHESTRATOR_RULES = textwrap.dedent(
    """
    ## Rules

    1. Every turn MUST be exactly one tool call. Never reply in prose.
    2. `dispatch_subplan` may include 1 to 8 blank_ids. Include
       multiple ids ONLY when they have no mutual dependency
       (different rows in the topological level chart below). When in
       doubt, dispatch fewer — the researchers are isolated and will
       not help each other solve siblings.
    3. Do not dispatch a blank that is already `status=resolved` in
       state. Use `request_plan_update` for a wrong value ONLY when
       the evidence shows a schema-supported graph change is needed.
    4. Only call `request_plan_update` when you have concrete
       evidence that the plan structure is wrong. It is for structural
       edits: adding/dropping blanks, changing relations, changing the
       target blank, or changing constraints supported by the current
       schema (including mul, div, round_nearest, equals, gt, lt, and
       in_list). A single weak retrieval is NOT enough; it just means
       "dispatch again with different blanks".
       Do NOT use `request_plan_update` for answer formatting only
       (for example semicolon vs space), for replacing a committed
       value while leaving the graph unchanged. If a plan update is
       rejected as a no-op, do not retry the same request.
    5. Constraint-output blanks fill themselves once their input
       blanks are resolved. Never dispatch a blank that is the output
       of a constraint; dispatch its INPUTS.
    6. `submit_answer(answer)` is the ONLY way to end. Pass the
       committed target value verbatim. If retrieval genuinely
       cannot find the target, you may still submit a best-guess
       answer — but prefer `request_plan_update` first if the plan
       itself looks wrong.
    7. If a `dispatch_subplan` result contains `revision_requests`
       (a researcher escalated because the plan is missing a step),
       your **default next move** is `request_plan_update` using the
       first request's `reason` as the update reason and `hint` as
       the evidence. Override this default ONLY if you have a concrete
       cheaper recovery (e.g. re-dispatching the same blank with
       targeted hints will plausibly succeed). Do NOT ignore an
       escalation silently.{PLAN_REQUEST_ID_NOTE}

    8. **Plan-update cap**. You may call `request_plan_update` at
       most TWICE per question. After the second call, further
       `request_plan_update` calls will be REJECTED. If two revisions
       have not unlocked the answer, the corpus likely lacks the
       required data — do NOT keep trying. Either:
         (a) `submit_answer("")` — accept the question is unanswerable
             from the corpus (the harness will treat this as
             `stopped_reason="give_up_unanswerable"`); or
         (b) dispatch one more researcher with explicit hints asking
             it to commit a best-guess from chunks already retrieved
             (use only if you have a plausible best-guess on hand).
    9. **Per-blank dispatch cap**. After a blank has triggered TWO
       researcher escalations (`suggest_plan_revision`), further
       `dispatch_subplan` calls including that blank will be REJECTED.
       The corpus genuinely lacks the data; permuting hints will not
       help. Your options:
         (a) call `request_plan_update` (if cap above not reached) to
             revise the plan with new intermediate blanks;
         (b) dispatch a DIFFERENT (still-unresolved) blank;
         (c) `submit_answer("")` (give up, acceptable when the corpus
             gap is structural — see rule 8 path (a)).
       Do NOT keep dispatching the same blank with new hints once it
       has hit this cap; that loop is what the cap is built to prevent.

    ## Answer format for `submit_answer`

    - Person / entity: just the name.
    - Year / date: just the date value.
    - Number: just the number.
    - Bool: `True` or `False`.
    - List: comma-separated values, no surrounding prose.
    - Never prefix with "Answer:", "Final:", etc.
    """
).strip()


# gsw_plan_request_contract, Finding 3(b) (final-review fix wave): rule 7's
# `{PLAN_REQUEST_ID_NOTE}` marker (in ORCHESTRATOR_RULES above) is spliced
# to this sentence only when the
# flag is on -- see `_rules_text` below. Nothing in rule 7 previously told
# the orchestrator to cite `plan_request_id`, so the contract's authored-
# clause machinery could never fire even when a researcher HAD escalated
# (it also requires the orchestrator to attach the PlanRequest by id).
#
# STRENGTHENED after a live 200Q run: the first phrasing (" When you do, also
# pass that request's id as the `plan_request_id` argument...") was ignored.
# On q806 the orchestrator called `request_plan_update` TWICE with three
# escalations available and cited none of them, while its own `reason`
# paraphrased the researcher's text -- so it had read the escalation and
# dropped the id. Trailing, softly-worded advice at the end of a long rule
# does not survive; the instruction is now imperative, says exactly where to
# read the value, and names the consequence of omitting it.
_PLAN_REQUEST_ID_NOTE = (
    " **You MUST pass `plan_request_id`** on that call: copy it verbatim from "
    "the `request_id` field of the `revision_requests` entry you are acting "
    "on (e.g. `\"806:b_pop_london:0\"`). Omitting it discards the "
    "researcher's concrete structural proposal and the updater sees only "
    "your paraphrase of it. If several researchers escalated, cite the one "
    "whose blank you are fixing."
)


def _rules_text(*, plan_request_contract: bool) -> str:
    """``ORCHESTRATOR_RULES``, with rule 7's `plan_request_id` note spliced
    in only when ``plan_request_contract`` is on. Flag off replaces the
    marker with an empty string, which deletes it from the text -- leaving
    the result BYTE-IDENTICAL to the pre-splice constant, per
    ``gsw_plan_request_contract``'s "flag off ⇒ default path unchanged"
    constraint (mirrors ``_orchestrator_tools``'s treatment of the tool
    schema, the other half of the same constraint)."""
    note = _PLAN_REQUEST_ID_NOTE if plan_request_contract else ""
    return ORCHESTRATOR_RULES.replace("{PLAN_REQUEST_ID_NOTE}", note)


def _state_table_for_prompt(
    plan_dict: dict[str, Any],
    state: dict[str, BlankResult],
) -> str:
    """Render every blank in the state as a row the orchestrator can read.

    Constraint-output blanks are marked so the orchestrator doesn't
    try to dispatch them.
    """
    ent_by_id = {e.get("id"): e for e in plan_dict.get("entities", [])}
    constraint_outputs = {
        c.get("output_blank_id")
        for c in plan_dict.get("constraints", [])
        if c.get("output_blank_id")
    }
    target_id = next(
        (
            e.get("id")
            for e in plan_dict.get("entities", [])
            if e.get("is_target")
        ),
        None,
    )
    rows: list[str] = []
    for bid, br in state.items():
        ent = ent_by_id.get(bid, {})
        role = ent.get("role", "—")
        vt = ent.get("value_type", "text")
        flags: list[str] = []
        if bid == target_id:
            flags.append("TARGET")
        if bid in constraint_outputs:
            flags.append("auto-computed (do not dispatch)")
        flag_str = "  [" + ", ".join(flags) + "]" if flags else ""
        if br.status == "resolved":
            val = br.value
            if isinstance(val, str) and len(val) > 80:
                val = val[:77] + "..."
            rows.append(
                f"- `{bid}` (role=`{role}`, value_type=`{vt}`, "
                f"**RESOLVED**): {val!r}{flag_str}"
            )
        else:
            rows.append(
                f"- `{bid}` (role=`{role}`, value_type=`{vt}`, "
                f"status=`{br.status}`){flag_str}"
            )
    if not rows:
        return "_(no blanks in this plan)_"
    return "\n".join(rows)


def build_orchestrator_system_static(
    plan_dict: dict[str, Any],
    order: list[str],
    levels: dict[str, int],
    *,
    plan_issue_warnings: list[dict[str, Any]] | None = None,
    plan_request_contract: bool = False,
) -> str:
    """The half of the orchestrator prompt that does not change during a run.

    ``--gsw-orchestrator-live`` sends this once and appends its own turns after
    it, so everything mutable — the state table, the recent-activity line, the
    turn counter, the cap-rejection example — is deliberately absent: those
    arrive as tool results instead. Keeping this byte-identical for the whole
    question is the point, since a prefix cache can then serve every token
    before the newest tool message.

    The plan brief IS included and is *not* mutable in the same sense: when
    ``request_plan_update`` swaps the plan, the loop announces the new plan in
    the conversation rather than rewriting this.
    """
    warnings_section = ""
    if plan_issue_warnings:
        lines = [
            "## Plan-quality warnings (advisory)",
            "The following are possible plan issues. They are hints, not instructions.",
            "You MAY call request_plan_update if a warning blocks progress or makes the current plan unreliable.",
        ]
        for w in plan_issue_warnings:
            bid = f"[{w['blank_id']}]" if w.get("blank_id") else ""
            lines.append(
                f"- ({w['issue_id']}) [{w['severity']}][{w['issue_type']}]{bid} {w['suggested_fix']}"
            )
        warnings_section = "\n".join(lines) + "\n"
    rules = _rules_text(plan_request_contract=plan_request_contract)
    return (
        f"{ORCHESTRATOR_MISSION}\n\n"
        f"{SYSTEM_PROMPT_LEGEND}\n\n"
        f"{render_plan_brief(plan_dict, order, levels)}\n\n"
        f"{warnings_section}"
        f"{rules}\n\n"
        "## Your working memory\n\n"
        "This is a conversation. Every dispatch you make and every result it "
        "returns stays above — read your own history before acting. If you have "
        "already dispatched a blank and it came back unresolved or partial, "
        "dispatching it again unchanged will fail the same way; change the hint, "
        "dispatch a different blank, or request a plan update. Call `get_state` "
        "whenever you want the current blank table."
    )


def build_state_message(plan_dict: dict[str, Any],
                        state: dict[str, BlankResult]) -> str:
    """The blank table as a conversation turn, for the live orchestrator."""
    return f"## Current state\n\n{_state_table_for_prompt(plan_dict, state)}"


def build_live_status_line(
    plan_dict: dict[str, Any],
    state: dict[str, BlankResult],
    *,
    turn: int,
    max_turns: int,
    target_id: str = "",
) -> str:
    """One appended line naming what is still open, for the live orchestrator.

    The champion prompt re-renders the state table and the turn counter every
    single turn, and that repetition is load-bearing: it keeps saying what is
    unresolved and how much budget is left. A static system prompt says it once,
    and on the 20-question smoke one question submitted an answer having
    dispatched nothing at all. This restores the nag as an APPENDED turn, so the
    cached prefix is untouched — the cost is one short line per turn.
    """
    resolved, unresolved = [], []
    for bid, br in (state or {}).items():
        (resolved if br is not None and br.status == "resolved"
         else unresolved).append(str(bid))
    parts = [f"TURN {turn}/{max_turns}."]
    parts.append("Unresolved: " + (", ".join(sorted(unresolved)) or "none"))
    parts.append("Resolved: " + (", ".join(sorted(resolved)) or "none"))
    if target_id:
        done = target_id in resolved
        # The resolved branch deliberately does NOT say "you may submit_answer".
        # It did, and it cost questions: on FRAMES-200, 7 of the 9 questions the
        # live arm lost against champion were ones where it stopped EARLIER and
        # committed to an under-qualified answer — Fincher's most recent film
        # instead of his most recent *as of* Britney Spears' first (20 turns to
        # 9), the Lakers instead of the Celtics, 22 minutes instead of 52. A
        # target resolves before its qualifiers are checked, so an invitation to
        # submit at that moment is an invitation to submit too soon. State the
        # fact; let the rules decide what to do about it.
        parts.append(
            f"Target {target_id} is {'resolved' if done else 'NOT resolved — submit_answer will be refused'}."
        )
    parts.append(
        "Do not re-dispatch a blank you already dispatched unless you change "
        "the hint; check the results above first."
    )
    return " ".join(parts)


#: Corrective turns spent on a model that answers in prose instead of calling a
#: tool. Observed live: with its own history in front of it the orchestrator
#: replied `{"answer": "..."}` as content and the loop stopped dead. Bounded so
#: a model that never emits a tool call cannot burn the whole turn budget.
MAX_PROSE_NUDGES = 2

PROSE_NUDGE = (
    "TOOL CALL REQUIRED. Your last reply was ordinary text, which does nothing "
    "— this loop only acts on tool calls. Even the final answer must go through "
    "`submit_answer`, and it is refused unless the target blank is resolved. "
    "Call exactly one of: dispatch_subplan, get_state, request_plan_update, "
    "submit_answer."
)


def build_planner_bootstrap_prompt() -> str:
    """The orchestrator's prompt before any plan exists (the planning phase).

    There is no brief and no state table to render, because nothing has been
    planned yet. The one useful move is `call_planner`, and saying so plainly
    beats letting the model discover it by having its other tool calls refused.

    The instruction about recalled entities is here rather than only in the tool
    description because it is the reason the audit comes back at all: a plan is
    written before any retrieval, so a name in it that the question never used
    was recalled from weights, and on real plans that happens 6% of the time.
    """
    return (
        f"{ORCHESTRATOR_MISSION}\n\n"
        "## You have no plan yet\n\n"
        "Nothing has been planned and nothing has been retrieved. Your first "
        "move must be `call_planner`, which returns a worksheet of typed blanks "
        "plus an AUDIT of that worksheet.\n\n"
        "`call_planner` takes NO arguments, deliberately. You have retrieved "
        "nothing, so anything you could tell the planner about this question "
        "would be recalled from memory rather than known — and on a 200-question "
        "sweep 75% of the invented entities the audit caught had been named by "
        "the orchestrator before the planner ever ran. Ask for the plan; judge "
        "what comes back.\n\n"
        "When the audit finds something, there is no tool to revise the plan "
        "yet — note what looks wrong and use `request_plan_update` once you "
        "have evidence to justify the change; never state a fact yourself, "
        "because you have retrieved nothing either.\n\n"
        "Read the audit before you act on the plan. `hallucinated` lists "
        "entities the planner named that your question never mentioned: those "
        "were recalled from memory, not retrieved, and any candidate the planner "
        "failed to recall cannot win a ranking. When the audit flags something, "
        "your lever is `request_plan_update` once evidence exists — describe the "
        "STRUCTURAL defect (a filled entity that should be a retrieved blank), "
        "not the fact you think it should hold.\n\n"
        "Until a plan exists, dispatch_subplan, get_state and submit_answer are "
        "unavailable.\n\n"
        "`call_planner` builds the INITIAL plan only. Once researchers have run, "
        "it is refused: re-emitting from the question would discard blanks they "
        "already resolved. From then on, revise with `request_plan_update`, "
        "which edits the current plan from your evidence and keeps resolved "
        "values."
    )
