"""The planner as a tool: the orchestrator calls it, and gets the audit back.

The plan comes into existence the way anything else does in an agent loop: the
orchestrator calls a tool. Its planning phase offers only `call_planner`; the tool
result carries the plan AND its audit. The audit is currently just the
deterministic `plan_audit_tags` from `_planner_emit.py` — the LLM auditor and the
deterministic hallucination check that used to fill `issues` were both removed.
`request_plan_update` is the separate, diff-op repair channel, reachable mid-run.

Nothing may dispatch or submit before a plan exists. The gate is mechanical, not
a prompt instruction.
"""
from __future__ import annotations

import json

import pytest

from tests.ours.orchestrator._helpers import (
    _RoutedLLM,
    _StubResponse,
    _mk_llm_adapter,
    _one_level_plan,
    _tc,
)

ORCH = "You are the **Orchestrator**"


def _orch_calls(llm):
    out = []
    for c in llm.calls:
        sys_msg = next((m for m in c["messages"] if m.get("role") == "system"), None)
        if sys_msg and ORCH in (sys_msg.get("content") or ""):
            out.append(c)
    return out


def _sys(call):
    return next(m["content"] for m in call["messages"] if m.get("role") == "system")


def _tool_results(llm):
    """Every tool message the orchestrator was sent (live mode only)."""
    out = []
    for c in _orch_calls(llm):
        out += [m for m in c["messages"] if m.get("role") == "tool"]
    return out


def _run(*, script=None, extra=None, plan_dict=None):
    routes = {
        ORCH: script or [
            _StubResponse(tool_calls=[_tc("o1", "call_planner", {})]),
            # reaching for another tool ends the planning phase (the accept
            # signal); the main loop then re-issues the dispatch itself
            _StubResponse(tool_calls=[_tc("o2", "dispatch_subplan",
                                          {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o3", "dispatch_subplan",
                                          {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o4", "submit_answer",
                                          {"answer": "Dark Side of the Moon"})]),
        ],
        "resolve exactly these blanks": [
            _StubResponse(tool_calls=[_tc("r1", "update_blank", {
                "blank_id": "t", "value": "Dark Side of the Moon",
                "evidence_chunk_ids": ["c"]})]),
        ],
    }
    cfg = dict(extra or {})
    # this module scripts the planning phase itself, starting at call_planner
    llm = _RoutedLLM(routes, auto_plan=False)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict or _one_level_plan(),
                              orchestrator_max_turns=8, extra=cfg)
    traj = adapter.run_question("Which album?", question_id="qpt")
    return llm, traj


# --- the flag ----------------------------------------------------------------

# --- bootstrap: no plan until the tool is called ------------------------------

def test_the_first_turn_has_no_plan_and_offers_the_planner():
    llm, _ = _run()
    first = _sys(_orch_calls(llm)[0])
    assert "call_planner" in first
    # the brief of the pre-emitted plan must NOT be there — there is no plan yet
    assert "b_intermediate" not in first
    tools = _orch_calls(llm)[0].get("tools") or []
    assert "call_planner" in [t.get("function", {}).get("name") for t in tools]


def test_dispatch_before_a_plan_is_refused():
    """Mechanical gate, not a prompt instruction: the loop cannot dispatch
    against a plan that does not exist."""
    script = [
        _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan", {"blank_ids": ["t"]})]),
        _StubResponse(tool_calls=[_tc("o2", "call_planner", {})]),
        _StubResponse(tool_calls=[_tc("o3", "dispatch_subplan", {"blank_ids": ["t"]})]),
        _StubResponse(tool_calls=[_tc("o4", "dispatch_subplan", {"blank_ids": ["t"]})]),
        _StubResponse(tool_calls=[_tc("o5", "submit_answer", {"answer": "Dark Side of the Moon"})]),
    ]
    llm, traj = _run(script=script)
    assert traj.extra["planner_tool_calls"] >= 1
    # the run still completes: the refusal is a tool error, not a dead end
    assert traj.final_answer == "Dark Side of the Moon"


def test_call_planner_installs_the_plan_and_the_run_completes():
    llm, traj = _run()
    assert traj.final_answer == "Dark Side of the Moon"
    assert traj.extra["stopped_reason"] == "finished"
    assert traj.extra["plan_json"]["entities"]          # recorded for the trace
    assert traj.extra["planner_tool_calls"] == 1
    assert len(traj.extra["dispatches"]) == 1


def test_the_tool_takes_no_arguments_at_all():
    """Directives were removed, not sanitised. Stripping them protected the
    PLANNER but not the orchestrator: in live mode its own tool call is appended
    to the conversation verbatim, so "(Metallica, Slayer, Megadeth, Anthrax)"
    stayed in its context for the rest of the question and could leak into a
    later request_plan_update or dispatch hint. 75% of plan hallucinations were
    named in a directive first; a channel that carries facts and cannot be
    unsaid is not worth its shape-hints."""
    llm, traj = _run()
    schema = None
    for c in _orch_calls(llm):
        for t in c.get("tools") or []:
            if t.get("function", {}).get("name") == "call_planner":
                schema = t["function"]["parameters"]
    assert schema is not None
    assert not (schema.get("properties") or {}), schema
    assert not (schema.get("required") or [])
    assert "directives" not in json.dumps(traj.extra["planner_tool_log"][0])


# --- the audit rides back -----------------------------------------------------

def test_live_gets_the_audit_in_the_tool_result():
    llm, traj = _run()
    tool_msgs = [m for m in _tool_results(llm) if "audit" in (m.get("content") or "")]
    assert tool_msgs, "no tool result carried an audit"
    assert '"tags"' in tool_msgs[-1]["content"]


def test_the_tool_plan_audit_carries_the_deterministic_tags():
    """The plan the tool returns carries `plan_audit_tags` (the deterministic,
    non-LLM tag mechanism in `_planner_emit.py`) back in its audit — the LLM
    auditor and the hallucination check were both removed, so `issues` is
    always empty today; a verifier will populate it next.
    """
    llm, traj = _run()
    log = traj.extra["planner_tool_log"]
    assert "audit" in log[0]
    assert log[0]["audit"] == {"issues": [], "tags": []}


# --- bounded ------------------------------------------------------------------

def test_planner_calls_are_capped():
    """An orchestrator that keeps re-planning must not burn the whole budget."""
    script = [_StubResponse(tool_calls=[_tc(f"o{i}", "call_planner", {})])
              for i in range(6)]
    llm, traj = _run(script=script)
    assert traj.extra["planner_tool_calls"] <= 3


# --- call_planner is for creating the plan, not repairing it -----------------

def test_call_planner_is_refused_once_evidence_exists():
    """`call_planner` re-emits from the QUESTION alone — no state, no resolved
    values, no evidence about what failed. Used mid-run it silently discards
    retrieval work: a blank it renames starts `unknown`, and the plan-negotiation
    arm measured 26 of 30 regeneration rounds deleting blanks nobody asked about.

    `request_plan_update` is the repair channel and already does this properly —
    diff ops against the current plan, evidence-driven, resolved blanks carried
    over by id. So after the first dispatch, the planner tool steps aside and
    says which tool to use instead.
    """
    script = [
        _StubResponse(tool_calls=[_tc("o1", "call_planner", {})]),
        _StubResponse(tool_calls=[_tc("o2", "dispatch_subplan", {"blank_ids": ["t"]})]),
        _StubResponse(tool_calls=[_tc("o3", "dispatch_subplan", {"blank_ids": ["t"]})]),
        _StubResponse(tool_calls=[_tc("o4", "call_planner", {})]),
        _StubResponse(tool_calls=[_tc("o5", "submit_answer",
                                      {"answer": "Dark Side of the Moon"})]),
    ]
    llm, traj = _run(script=script)
    refusals = [tc for tc in (traj.tool_calls or [])
                if tc.name == "call_planner" and "request_plan_update" in (tc.error or "")]
    assert refusals, "the mid-run call_planner should have been refused"
    # the refusal costs nothing but the turn: the plan and the answer survive
    assert traj.final_answer == "Dark Side of the Moon"
    assert traj.extra["planner_tool_calls"] == 1


def test_the_refusal_does_not_burn_the_planner_budget():
    """A refused call must not count against the cap — otherwise one mistaken
    mid-run call costs a legitimate re-plan the orchestrator may still need."""
    script = [
        _StubResponse(tool_calls=[_tc("o1", "call_planner", {})]),
        _StubResponse(tool_calls=[_tc("o2", "dispatch_subplan", {"blank_ids": ["t"]})]),
        _StubResponse(tool_calls=[_tc("o3", "call_planner", {})]),
        _StubResponse(tool_calls=[_tc("o4", "submit_answer",
                                      {"answer": "Dark Side of the Moon"})]),
    ]
    llm, traj = _run(script=script)
    assert traj.extra["planner_tool_calls"] == 1


def test_planner_tool_state_is_thread_local():
    """The harness shares ONE adapter across its question ThreadPool, so plain
    instance attributes leak between concurrent questions. Observed live at
    concurrency 16 on the first sweep: q111 made ONE planner call, and its log
    held 28 entries with the counter reading 3 — other questions were eating its
    3-call budget, so 27 of those entries were "cap reached" for calls it never
    made. Same failure the negotiation log had.
    """
    import threading

    from panini_dr.adapters.ours.gsw_planner_orchestrator_v1 import (
        OursGSWPlannerOrchestratorV1Adapter as _A,
    )

    ad = _A.__new__(_A)
    ad._checker_sink_tls = threading.local()
    seen = {}
    barrier = threading.Barrier(2)

    def worker(name):
        ad._planner_tool_calls = 0
        ad._planner_tool_log = []
        ad._planner_tool_calls += 1
        ad._planner_tool_log.append({"who": name})
        barrier.wait()
        seen[name] = (ad._planner_tool_calls, list(ad._planner_tool_log))

    ts = [threading.Thread(target=worker, args=(n,)) for n in ("A", "B")]
    for t in ts: t.start()
    for t in ts: t.join()
    for name, (calls, log) in seen.items():
        assert calls == 1, f"{name} saw another question's planner calls: {calls}"
        assert [e["who"] for e in log] == [name], f"{name} saw another question's log"


def test_each_question_starts_with_an_empty_planner_tool_record():
    """Thread-local is not enough: the SAME thread runs many questions in
    sequence, so the per-question logs must be cleared at the top of
    run_question. They were initialised in __init__ only, and the sweep showed
    it — q535 recorded 4 orchestrator turns for a 1-turn question, carrying the
    turns of two later questions ("average attendance for six games", "Bad
    Religion drummer") in its own trace.
    """
    def _question_script(n):
        # plan, then the accept signal (a dispatch the loop executes), submit
        return [
            _StubResponse(tool_calls=[_tc(f"o{n}1", "call_planner", {})]),
            _StubResponse(tool_calls=[_tc(f"o{n}2", "dispatch_subplan", {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc(f"o{n}3", "submit_answer", {"answer": ""})]),
        ]

    def _researcher_turn(n):
        return _StubResponse(tool_calls=[_tc(f"r{n}", "update_blank", {
            "blank_id": "t", "value": "Dark Side of the Moon", "evidence_chunk_ids": ["c"]})])

    routes = {
        ORCH: _question_script(1) + _question_script(2),
        "resolve exactly these blanks": [_researcher_turn(1), _researcher_turn(2)],
    }
    llm = _RoutedLLM(routes, auto_plan=False)
    adapter = _mk_llm_adapter(llm, plan_dict=_one_level_plan(),
                              orchestrator_max_turns=6)
    first = adapter.run_question("Q one", question_id="qA")
    second = adapter.run_question("Q two", question_id="qB")
    assert first.extra["planner_tool_calls"] == 1
    assert second.extra["planner_tool_calls"] == 1, "second question inherited the first's calls"
    assert len(second.extra["planner_tool_log"]) == 1
    # the turn log now spans two phases (planning + main loop), so it is no
    # longer equal to traj.turns; what matters is that it holds only THIS
    # question's turns
    assert second.extra["orchestrator_turns_log"], "no turns recorded"
    assert all(t.get("phase") == "planning" or t.get("turn", 0) >= 0
               for t in second.extra["orchestrator_turns_log"])


# --- what the tool actually hands back -------------------------------------

def test_the_tool_result_carries_the_topology_not_just_raw_json():
    """The orchestrator has to decide WHICH blanks to dispatch together, and that
    is a question about dependency levels. Champion renders
    `render_plan_brief(plan_dict, order, levels)` into its prompt every turn; the
    planner tool was handing back a bare entities/verb_phrases/constraints dict
    and leaving the orchestrator to infer the order itself."""
    llm, traj = _run()
    rec = traj.extra["planner_tool_log"][0]
    assert "brief" in rec, sorted(rec)
    brief = rec["brief"]
    assert "t" in brief
    assert "level" in brief.lower() or "order" in brief.lower()


def test_the_static_prompt_is_replaced_once_a_plan_exists():
    """The bootstrap prompt says "you have no plan yet ... dispatch_subplan,
    get_state and submit_answer are unavailable". It was never swapped, so that
    instruction contradicted the installed plan on every later turn — the arm ran
    26% more turns than live+fix and hit max_turns nearly twice as often."""
    llm, _ = _run()
    calls = _orch_calls(llm)
    assert len(calls) >= 2
    first, later = _sys(calls[0]), _sys(calls[-1])
    assert "no plan yet" in first.lower()
    assert "no plan yet" not in later.lower(), later[:300]
    assert "unavailable" not in later.lower()


def test_a_first_install_is_not_announced_as_a_replacement():
    """"The plan you were given at the start has been replaced" is false when
    there was no plan at the start."""
    llm, _ = _run()
    body = " ".join(m.get("content") or ""
                    for c in _orch_calls(llm) for m in c["messages"])
    assert "has been replaced" not in body


# --- the planning phase must hand back to the NORMAL pipeline ----------------

def test_the_run_keeps_the_whole_champion_pipeline():
    """The bootstrap entered the orchestrator loop with plan=None,
    qgsw_state=None and plan_issue_summary=[], then returned early from
    run_question — so the evidence-state init, the deterministic plan-quality
    pass, verify_graph, coverage and the failure taxonomy never ran. The arm was
    not champion-plus-a-flag, it was champion-minus-five-components, which is
    why its -6 could not be attributed to the design.

    The plan now comes from a bounded PLANNING PHASE, and everything downstream
    is the ordinary path.
    """
    ex = _run(extra={"gsw_evidence_state": True,
                                "gsw_plan_quality_checks": True})[1].extra
    assert ex["planner_tool_calls"] == 1
    assert ex["plan_json"]["entities"]
    assert ex["topological_order"], "topology must be recorded as usual"
    assert "topology_levels" in ex
    assert "gsw_plan_quality" in ex, "the deterministic plan-quality pass ran"
    assert "gsw_state" in ex, "evidence state was initialised and mirrored"


def test_the_planning_phase_still_refuses_to_dispatch_before_a_plan():
    """The gate moves into the planning phase; it must not disappear."""
    # NB the third response is the ACCEPT signal — reaching for another tool is
    # how the orchestrator says "this plan will do" — so the planning phase
    # consumes it and the main loop needs its own dispatch turn.
    script = [
        _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan", {"blank_ids": ["t"]})]),
        _StubResponse(tool_calls=[_tc("o2", "call_planner", {})]),
        _StubResponse(tool_calls=[_tc("o3", "dispatch_subplan", {"blank_ids": ["t"]})]),
        _StubResponse(tool_calls=[_tc("o4", "dispatch_subplan", {"blank_ids": ["t"]})]),
        _StubResponse(tool_calls=[_tc("o5", "submit_answer",
                                      {"answer": "Dark Side of the Moon"})]),
    ]
    llm, traj = _run(script=script)
    assert traj.extra["planner_tool_calls"] == 1
    assert traj.final_answer == "Dark Side of the Moon"


def test_the_planning_conversation_carries_into_the_main_loop():
    """Live mode: the orchestrator should not lose the planning turns when the
    run moves on — the plan it chose and the audit it read are the context for
    its first dispatch."""
    llm, traj = _run()
    later = _orch_calls(llm)[-1]["messages"]
    assert any(m.get("role") == "tool" and "audit" in (m.get("content") or "")
               for m in later), [m["role"] for m in later]


# --- the call that ends the planning phase must still happen ----------------

def test_the_accept_signal_call_is_executed_not_discarded():
    """Reaching for another tool ends the planning phase — but that call is a
    real decision the orchestrator already made, and dropping it lost it.

    Measured on 67 live questions: the discarded call was `request_plan_update`
    39 times (58%) and `dispatch_subplan` 28 times. The plan updates never came
    back (0.12/question, same as the arm without the planner tool), so the
    audit-driven revision this design exists to enable was being swallowed by
    the phase boundary, and dispatches ran 17% below the comparison arm.
    """
    script = [
        _StubResponse(tool_calls=[_tc("o1", "call_planner", {})]),
        # this dispatch ends planning AND must be carried into the main loop
        _StubResponse(tool_calls=[_tc("o2", "dispatch_subplan", {"blank_ids": ["t"]})]),
        _StubResponse(tool_calls=[_tc("o3", "submit_answer",
                                      {"answer": "Dark Side of the Moon"})]),
    ]
    llm, traj = _run(script=script)
    assert len(traj.extra["dispatches"]) == 1, "the accept-signal dispatch never ran"
    assert traj.final_answer == "Dark Side of the Moon"


def test_the_planning_conversation_has_no_dangling_tool_call():
    """An assistant tool_call with no tool response tells the model the call
    already happened — which is why it did not re-issue it."""
    llm, _ = _run()
    msgs = _orch_calls(llm)[-1]["messages"]
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            assert nxt and nxt.get("role") == "tool", (
                f"assistant tool_call at {i} has no tool response: "
                f"{[x['role'] for x in msgs]}")


def test_planning_hand_off_is_thread_local():
    """The conversation the planning phase held and the call that ended it are
    handed to the orchestrator loop through the adapter. The harness runs every
    question concurrently on ONE adapter, so the hand-off must be per question:
    as plain attributes, one question's loop could continue ANOTHER question's
    planning conversation and execute its pending call."""
    import threading

    from panini_dr.adapters.ours.gsw_planner_orchestrator_v1 import (
        OursGSWPlannerOrchestratorV1Adapter as _A,
    )

    ad = _A.__new__(_A)
    ad._checker_sink_tls = threading.local()
    seen = {}
    barrier = threading.Barrier(2)

    def worker(name):
        ad._planning_messages = [{"role": "user", "content": name}]
        ad._pending_call = {"name": "dispatch_subplan", "who": name}
        barrier.wait()
        seen[name] = (list(ad._planning_messages), dict(ad._pending_call))

    ts = [threading.Thread(target=worker, args=(n,)) for n in ("A", "B")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for name, (msgs, pending) in seen.items():
        assert msgs == [{"role": "user", "content": name}], f"{name} read another question's planning"
        assert pending["who"] == name, f"{name} would execute another question's pending call"
