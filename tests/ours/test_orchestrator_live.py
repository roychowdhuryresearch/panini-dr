"""The orchestrator as a live tool-calling agent.

The orchestrator keeps one message list for the whole question, the way a
researcher does: its planning phase, every tool call and every tool result are
appended to it, followed by a one-line status turn.

The measured motive, on the champion 200Q FRAMES run (whose orchestrator prompt
was rebuilt from scratch every turn, with the last result paraphrased into one
line): 47/200 questions (24%) re-dispatched a blank they had already dispatched,
one of them nine times.

The load-bearing property asserted here: once the plan is installed, the system
prompt must be **byte-identical across turns**, because everything mutable lives
in appended messages — that is what lets a prefix cache serve the whole
conversation up to the newest tool message.
"""
from __future__ import annotations

from tests.ours.orchestrator._helpers import (
    _RoutedLLM,
    _StubResponse,
    _mk_llm_adapter,
    _one_level_plan,
    _tc,
)

ORCH = "You are the **Orchestrator**"


def _orchestrator_calls(llm: _RoutedLLM) -> list[dict]:
    out = []
    for c in llm.calls:
        sys_msg = next((m for m in c["messages"] if m.get("role") == "system"), None)
        if sys_msg and ORCH in (sys_msg.get("content") or ""):
            out.append(c)
    return out


def _loop_calls(llm: _RoutedLLM) -> list[dict]:
    """Orchestrator calls of the main loop. The planning phase that precedes it
    (``_plan_via_tool``) offers only ``call_planner``; its last call decides
    the first dispatch, which the loop then executes without another call."""
    return [
        c for c in _orchestrator_calls(llm)
        if any(t["function"]["name"] == "dispatch_subplan" for t in (c.get("tools") or []))
    ]


def _planning_calls(llm: _RoutedLLM) -> list[dict]:
    return [c for c in _orchestrator_calls(llm) if c not in _loop_calls(llm)]


def _system_of(call: dict) -> str:
    return next(m["content"] for m in call["messages"] if m.get("role") == "system")


def _two_dispatch_run():
    """Dispatch t, look at the state, then submit: three orchestrator turns with
    tool results in between, which is the whole point of the live loop."""
    routes = {
        ORCH: [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                          {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "get_state", {})]),
            _StubResponse(tool_calls=[_tc("o3", "submit_answer",
                                          {"answer": "Dark Side of the Moon"})]),
        ],
        "resolve exactly these blanks": [
            _StubResponse(tool_calls=[_tc("r1", "update_blank", {
                "blank_id": "t",
                "value": "Dark Side of the Moon",
                "evidence_chunk_ids": ["c"],
            })]),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=_one_level_plan())
    traj = adapter.run_question("Q", question_id="qlive")
    return llm, traj


# --- the flag ---------------------------------------------------------------

# --- champion mode is untouched ---------------------------------------------

# --- live mode ---------------------------------------------------------------

def test_live_keeps_one_conversation_and_appends_the_tool_result():
    llm, traj = _two_dispatch_run()
    calls = _loop_calls(llm)
    assert len(calls) == 2
    roles = [m["role"] for m in calls[0]["messages"]]
    # The planning phase (question, call_planner, its result) continues as the
    # same conversation; the plan is then announced, the first dispatch and its
    # result are appended, and the trailing "user" is the per-turn status line,
    # appended not re-rendered.
    assert roles == ["system", "user", "assistant", "tool", "user",
                     "assistant", "tool", "user"], roles

    assistant = calls[0]["messages"][5]
    assert assistant["tool_calls"][0]["function"]["name"] == "dispatch_subplan"
    tool_msg = calls[0]["messages"][6]
    assert tool_msg["tool_call_id"] == "o1"
    # The result itself, not a one-line paraphrase of it.
    assert "Dark Side of the Moon" in tool_msg["content"]
    # The next turn extends the same list.
    later = calls[1]["messages"]
    assert later[: len(calls[0]["messages"])][1:] == calls[0]["messages"][1:]


def test_live_system_prompt_is_byte_identical_across_turns():
    """The prefix-cache property. Everything that changes turn to turn has to
    live in appended messages, or the cache is invalidated at token 0."""
    llm, _ = _two_dispatch_run()
    calls = _loop_calls(llm)
    assert _system_of(calls[0]) == _system_of(calls[1])


def test_live_system_prompt_carries_the_static_half_only():
    llm, _ = _two_dispatch_run()
    sys_prompt = _system_of(_loop_calls(llm)[0])
    assert ORCH in sys_prompt                    # mission
    assert "dispatch_subplan" in sys_prompt      # rules
    assert "Recent activity" not in sys_prompt   # the overwritten one-liner
    assert "(Turn " not in sys_prompt            # per-turn counter


def test_live_seeds_the_state_once_in_the_opening_user_message():
    """The state table is not gone — it arrives once, with the plan, and from
    there the orchestrator tracks resolution from its own tool results."""
    llm, _ = _two_dispatch_run()
    messages = _loop_calls(llm)[0]["messages"]
    assert messages[1] == {"role": "user", "content": "Q"}
    installed = messages[4]
    assert installed["role"] == "user"
    assert installed["content"].startswith("PLAN INSTALLED")
    assert "## Current state" in installed["content"]
    assert "`t`" in installed["content"]


def test_live_run_completes():
    _, live = _two_dispatch_run()
    assert live.final_answer == "Dark Side of the Moon"
    assert live.extra["stopped_reason"] == "finished"
    assert len(live.extra["dispatches"]) == 1


# --- the per-turn nag, which the static prompt gave up ------------------------

def test_live_appends_a_status_line_after_every_tool_result():
    """Champion re-renders the state table and turn counter into the system
    prompt every turn, and that repetition is doing work — it keeps naming what
    is still unresolved. Live mode says it once at the start, and on the 20Q
    smoke one question answered with ZERO dispatches. The status line puts the
    nag back as an APPENDED turn, which keeps the prefix stable."""
    llm, _ = _two_dispatch_run()
    second = _loop_calls(llm)[0]["messages"]
    status = [m for m in second
              if m.get("role") == "user" and "TURN " in (m.get("content") or "")]
    assert status, [m["role"] for m in second]
    body = status[-1]["content"]
    assert "TURN 1/" in body
    assert "t" in body                      # names the blanks by id


def test_the_status_line_does_not_disturb_the_static_prompt():
    llm, _ = _two_dispatch_run()
    calls = _loop_calls(llm)
    assert _system_of(calls[0]) == _system_of(calls[1])


# --- prose instead of a tool call ---------------------------------------------

def _prose_then(*follow_ups: _StubResponse, prose_turns: int = 1):
    routes = {
        ORCH: ([_StubResponse(text='{"answer": "Boston Celtics"}', tool_calls=[])]
               * prose_turns) + list(follow_ups),
        "resolve exactly these blanks": [
            _StubResponse(tool_calls=[_tc("r1", "update_blank", {
                "blank_id": "t", "value": "Dark Side of the Moon",
                "evidence_chunk_ids": ["c"]})]),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=_one_level_plan(), orchestrator_max_turns=8)
    return llm, adapter.run_question("Q", question_id="qprose")


def test_live_nudges_a_prose_reply_instead_of_giving_up():
    """Observed on the smoke run (q767): with its own history in front of it the
    model answered in prose despite tool_choice='required', and the loop stopped
    dead on `no_tool_call`. One corrective turn is far cheaper than losing the
    question."""
    llm, traj = _prose_then(
        _StubResponse(tool_calls=[_tc("o2", "dispatch_subplan", {"blank_ids": ["t"]})]),
        _StubResponse(tool_calls=[_tc("o3", "submit_answer",
                                      {"answer": "Dark Side of the Moon"})]))
    assert traj.extra["stopped_reason"] == "finished"
    assert traj.final_answer == "Dark Side of the Moon"
    calls = _orchestrator_calls(llm)
    nudge = [m for c in calls for m in c["messages"]
             if m.get("role") == "user" and "TOOL CALL REQUIRED" in (m.get("content") or "")]
    assert nudge, "no corrective turn was appended"


def test_the_prose_nudge_is_bounded():
    """A model that only ever emits prose must not burn the whole turn budget:
    the planning phase stops at its call cap with the plan it has, and the loop
    stops after MAX_PROSE_NUDGES corrective turns."""
    from panini_dr.adapters.ours._orchestrator_prompt import MAX_PROSE_NUDGES
    from panini_dr.adapters.ours.gsw_planner_orchestrator_v1 import (
        OursGSWPlannerOrchestratorV1Adapter as A,
    )

    llm, traj = _prose_then(
        _StubResponse(text="still prose", tool_calls=[]), prose_turns=8)
    assert traj.extra["stopped_reason"] == "no_tool_call"
    assert len(_planning_calls(llm)) <= A.MAX_PLANNER_TOOL_CALLS
    assert len(_loop_calls(llm)) == 1 + MAX_PROSE_NUDGES


# --- a plan swap must be announced, not silently re-rendered -----------------

def _add_blank_op(blank_id: str, vp_id: str) -> str:
    from tests.ours.orchestrator._helpers import _add_blank_op as _op
    return _op(blank_id, vp_id)


def test_live_announces_a_plan_update_in_the_conversation():
    """`request_plan_update` swaps the plan mid-run. With a live history the OLD
    plan stays in the transcript, so the new one has to arrive as a message —
    otherwise the orchestrator keeps planning against a plan that no longer
    exists."""
    routes = {
        ORCH: [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                          {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "request_plan_update", {
                "reason": "researcher escalated for t",
                "evidence": "need intermediate",
            })]),
            _StubResponse(tool_calls=[_tc("o3", "dispatch_subplan",
                                          {"blank_ids": ["b_intermediate"]})]),
            _StubResponse(tool_calls=[_tc("o4", "submit_answer",
                                          {"answer": "FINAL"})]),
        ],
        "`t` (role=`target`, value_type=`entity`": [
            _StubResponse(tool_calls=[_tc("rA1", "suggest_plan_revision", {
                "reason": "no chunk surfaces t directly",
                "hint": "add b_intermediate; then derive t",
            })]),
        ],
        "`b_intermediate` (role=`bridge-attribute`": [
            _StubResponse(tool_calls=[_tc("rB1", "update_blank", {
                "blank_id": "b_intermediate",
                "value": "associated thing",
                "evidence_chunk_ids": ["c2"],
            })]),
        ],
        "You are the **Plan-Updater**": [
            _StubResponse(text=_add_blank_op("b_intermediate", "vp_int")),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=_one_level_plan(),
                              orchestrator_max_turns=10)
    adapter.run_question("Q", question_id="qliveplan")

    calls = _loop_calls(llm)
    assert len(calls) >= 3
    # The turn after the accepted update carries a plain-language announcement
    # naming the blank the update introduced.
    after = calls[1]["messages"]
    announced = [m for m in after
                 if m.get("role") == "user" and "PLAN UPDATED" in (m.get("content") or "")]
    assert announced, [m["role"] for m in after]
    assert "b_intermediate" in announced[-1]["content"]
    # ...and the system prompt still never changed.
    assert _system_of(calls[0]) == _system_of(calls[1]) == _system_of(calls[2])
