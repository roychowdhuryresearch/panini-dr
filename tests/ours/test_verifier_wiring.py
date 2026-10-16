"""The plan-verifier repair loop, wired into the planner tool behind a flag.

`_emit_plan_with_optional_planner_route` -- the funnel `_tool_call_planner` calls
to emit a plan -- runs `_plan_repair_loop.verify_and_repair` (with
`_plan_repair_prompt.make_proposer` as the repairer) immediately after a plan is
emitted, gated on `self.gsw_plan_verifier`, and records the transcript on
`PlanEmitMeta.extras["plan_repair"]`, which `_tool_call_planner` copies onto its
planner-tool log record (`rec["plan_repair"]`).

`revise_plan` and the whole advisory-auditor system it belonged to were
already removed in 02d5ba7e -- measured over eight 200-question sweeps: a
one-shot planner scored 0.400 defects per plan, the fully audited arm 0.41;
the advisory loop revised 12 of 200 questions for a net gain of 2 defects,
and at high reasoning effort revised 21 for a net of exactly zero. The test
below only confirms it stayed deleted.
"""
from __future__ import annotations

from tests.ours.orchestrator._helpers import (
    _RoutedLLM, _StubResponse, _mk_llm_adapter, _one_level_plan, _tc,
)


class _LLM:
    def chat(self, *a, **k):
        raise AssertionError("not used")


def test_the_flag_defaults_off_and_the_attribute_exists():
    a = _mk_llm_adapter(_LLM(), plan_dict=_one_level_plan(), extra={})
    assert a.gsw_plan_verifier is False


def test_the_flag_turns_it_on():
    a = _mk_llm_adapter(_LLM(), plan_dict=_one_level_plan(),
                        extra={"gsw_plan_verifier": True})
    assert a.gsw_plan_verifier is True


def test_revise_plan_is_no_longer_offered_as_a_tool():
    """revise_plan and the advisory-auditor system it belonged to were
    removed in 02d5ba7e, before this task started. Nothing to delete here --
    only to confirm it stayed deleted."""
    import panini_dr.adapters.ours.gsw_planner_orchestrator_v1 as m
    assert not hasattr(m, "_REVISE_PLAN_TOOL")
    src = open(m.__file__).read()
    assert "revise_plan" not in src


# ---------------------------------------------------------------------------
# End-to-end wiring: flag off is byte-identical, flag on runs the loop and
# records the transcript on the planner-tool log record.
# ---------------------------------------------------------------------------


def _clean_plan():
    """A plan with zero binding contract violations: the target carries its
    own retrieval_hint, so `target_producible` is satisfied and there is
    nothing for the repair loop to fix."""
    plan = _one_level_plan()
    plan["entities"][1]["retrieval_hint"] = "The album released by Pink Floyd"
    return plan


def _run(*, extra):
    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "call_planner", {})]),
            _StubResponse(tool_calls=[_tc("o2", "submit_answer", {"answer": "x"})]),
        ],
    }
    llm = _RoutedLLM(routes, auto_plan=False)
    cfg = dict(extra)
    adapter = _mk_llm_adapter(llm, plan_dict=_one_level_plan(),
                              orchestrator_max_turns=8, extra=cfg)
    traj = adapter.run_question("Which album?", question_id="q0")
    return llm, traj


def test_off_by_default_no_transcript_is_recorded():
    """Byte-identical-by-default: the flag off means verify_and_repair never
    runs, so the planner-tool log record carries no `plan_repair` key at
    all -- not even an empty one."""
    _llm, traj = _run(extra={})
    rec = traj.extra["planner_tool_log"][0]
    assert "plan_repair" not in rec


def test_on_a_clean_plan_the_loop_runs_and_records_an_empty_transcript():
    """The flag on and a plan with no binding violations: the loop is wired
    in (it runs), finds nothing to fix, and records a transcript with zero
    accepted/rejected/demoted ops."""
    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "call_planner", {})]),
            _StubResponse(tool_calls=[_tc("o2", "submit_answer", {"answer": "x"})]),
        ],
    }
    llm = _RoutedLLM(routes, auto_plan=False)
    adapter = _mk_llm_adapter(llm, plan_dict=_clean_plan(),
                              orchestrator_max_turns=8,
                              extra={"gsw_plan_verifier": True})
    traj = adapter.run_question("Which album?", question_id="q0")
    rec = traj.extra["planner_tool_log"][0]
    assert rec["plan_repair"] == {
        "accepted": [], "rejected": [], "demoted": [], "before": 0, "after": 0,
    }


def test_on_a_violated_plan_an_accepted_op_is_recorded_and_applied():
    """The flag on and a plan missing the target's retrieval_hint
    (`target_producible`): the repairer LLM proposes `set_hint`, the gate
    accepts it (strictly fewer binding violations, no new clause), and the
    REPAIRED plan -- not the original -- is what lands in `rec["plan"]`.

    Also guards against the funnel move accidentally running the loop
    twice on this path (once from the old in-`_tool_call_planner` copy,
    once from the new shared funnel): the repairer route is scripted with
    exactly one response, so a second call would consume the "(no route)"
    fallback and the accepted-op count would drop to 0 or crash the
    orchestrator turn -- either way the assertions below would catch it.
    """
    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "call_planner", {})]),
            _StubResponse(tool_calls=[_tc("o2", "submit_answer", {"answer": "x"})]),
        ],
        "You are the repairer in a plan-verifier loop": [
            _StubResponse(tool_calls=[_tc("r1", "set_hint", {
                "node_id": "t", "text": "The album released by Pink Floyd",
            })]),
        ],
    }
    llm = _RoutedLLM(routes, auto_plan=False)
    adapter = _mk_llm_adapter(llm, plan_dict=_one_level_plan(),
                              orchestrator_max_turns=8,
                              extra={"gsw_plan_verifier": True})
    traj = adapter.run_question("Which album?", question_id="q0")
    rec = traj.extra["planner_tool_log"][0]
    repair = rec["plan_repair"]
    assert repair["before"] == 1
    assert repair["after"] == 0
    assert repair["demoted"] == []
    assert repair["rejected"] == []
    assert len(repair["accepted"]) == 1
    assert repair["accepted"][0]["op"] == {
        "op": "set_hint", "node_id": "t",
        "text": "The album released by Pink Floyd",
    }
    target = next(e for e in rec["plan"]["entities"] if e["id"] == "t")
    assert target["retrieval_hint"] == "The album released by Pink Floyd"

    repairer_calls = [
        c for c in llm.calls
        if any(m.get("role") == "system"
               and "You are the repairer in a plan-verifier loop" in (m.get("content") or "")
               for m in c["messages"])
    ]
    assert len(repairer_calls) == 1


class _RaisingOnRepairLLM:
    """Every route works exactly as `_RoutedLLM` (delegated to an inner
    instance) EXCEPT the repairer's, which raises instead of returning a
    scripted response -- simulating a failure (rate limit, timeout, auth,
    or the Bedrock corrupted-tool-use-streaming failure
    `_plan_repair_prompt.py` documents) that happens INSIDE `llm.chat()`
    itself. `make_proposer`'s own JSONDecodeError/TypeError guard only wraps
    the RETURN VALUE of `chat()`, so this is a distinct failure mode it
    cannot catch -- exercising it requires a stub that raises from within
    the call, not one that returns something unparseable.
    """

    def __init__(self, routes, *, auto_plan=True):
        self._inner = _RoutedLLM(routes, auto_plan=auto_plan)

    @property
    def calls(self):
        return self._inner.calls

    def chat(self, messages, *, tools=None, max_tokens=None, **kwargs):
        sys_content = next(
            (m.get("content", "") for m in messages if m.get("role") == "system"), ""
        )
        if "You are the repairer in a plan-verifier loop" in sys_content:
            self._inner.calls.append(
                {"messages": list(messages), "tools": tools, **kwargs}
            )
            raise RuntimeError("simulated rate limit")
        return self._inner.chat(messages, tools=tools, max_tokens=max_tokens, **kwargs)


def test_a_repairer_llm_failure_is_fail_open_not_a_lost_question():
    """`llm.chat()` raising inside the repair loop must never turn a working
    plan into a lost question. `run_question`'s own try/except only catches
    PlanEmitError -- an unguarded repair-loop exception would otherwise
    propagate all the way up and force a hard empty-answer miss on a
    question the ORIGINAL (unrepaired) plan could have answered. Fail-open:
    the failure lands on `rec["plan_repair"]["error"]` and the run proceeds
    to a real answer with the plan exactly as the planner emitted it --
    the same call/response shape `test_planner_tool.py::_run` uses for a
    genuine champion-mode completion (planning-phase-then-dispatch), so a
    `raise` at the repairer never derails the orchestrator's own turns."""
    orch_marker = "You are the **Orchestrator**"
    routes = {
        orch_marker: [
            _StubResponse(tool_calls=[_tc("o1", "call_planner", {})]),
            _StubResponse(tool_calls=[_tc("o2", "dispatch_subplan", {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o3", "dispatch_subplan", {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o4", "submit_answer",
                                          {"answer": "Dark Side of the Moon"})]),
        ],
        "resolve exactly these blanks": [
            _StubResponse(tool_calls=[_tc("r1", "update_blank", {
                "blank_id": "t", "value": "Dark Side of the Moon",
                "evidence_chunk_ids": ["c"]})]),
        ],
    }
    llm = _RaisingOnRepairLLM(routes, auto_plan=False)
    adapter = _mk_llm_adapter(llm, plan_dict=_one_level_plan(),
                              orchestrator_max_turns=8,
                              extra={"gsw_plan_verifier": True})
    traj = adapter.run_question("Which album?", question_id="q0")

    assert traj.final_answer == "Dark Side of the Moon"
    rec = traj.extra["planner_tool_log"][0]
    assert "error" in rec["plan_repair"]
    assert "repair loop failed" in rec["plan_repair"]["error"]
    # The original, unrepaired plan reached the researcher: `t` never got a
    # retrieval_hint, because the repair that would have set one never
    # completed.
    target = next(e for e in rec["plan"]["entities"] if e["id"] == "t")
    assert target.get("retrieval_hint") is None
