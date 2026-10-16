"""The last hop: escalation -> orchestrator -> update_plan, unmodified.

The researcher's structural proposal already reaches the orchestrator. What it
does NOT do today is reach `update_plan`, because the orchestrator writes its own
prose `reason`/`evidence` and the structured hint is dropped. These tests pin the
plumbing that closes that gap, and the telemetry that makes the 13 dropped
escalations visible.

Fix round 1: three findings from review -- (1) `_plan_requests`/
`_consumed_request_ids` must be THREAD-LOCAL, not plain instance attributes,
because eval/harness.py fans `run_question` over a ThreadPoolExecutor against
ONE adapter instance; (2) the tool-schema `plan_request_id` property and the
`request_id`/`pending` keys stamped onto a `revision_requests` entry must be
gated on `gsw_plan_request_contract`, or a flag-off run's prompt is no longer
byte-identical; (3) the field stamped at registration was misleadingly named
`consumed` (it is only ever true AT registration) -- renamed to `pending` --
and `_plan_request_summary()` needed a caller (`traj.extra["plan_requests"]`).
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

from panini_dr.adapters.ours._planner_exec import BlankResult, GSWPlan

from tests.ours.orchestrator._helpers import (
    _RoutedLLM, _StubResponse, _add_blank_op, _concat_plan, _mk_llm_adapter,
    _one_level_plan, _planning_phase_opening, _tc,
)

ORCH = "You are the **Orchestrator**"


def test_the_flag_defaults_off():
    """An unset run must be byte-identical to today."""
    ad = _mk_llm_adapter(_RoutedLLM({}), plan_dict=_one_level_plan(),
                         extra={})
    assert ad.gsw_plan_request_contract is False


def test_the_flag_can_be_turned_on():
    ad = _mk_llm_adapter(_RoutedLLM({}), plan_dict=_one_level_plan(),
                         extra={"gsw_plan_request_contract": True})
    assert ad.gsw_plan_request_contract is True


def test_the_tool_schema_advertises_plan_request_id_as_optional_when_on():
    """Optional, so the 71 questions that update with no escalation behind them
    keep working unchanged."""
    ad = _mk_llm_adapter(_RoutedLLM({}), plan_dict=_one_level_plan(),
                         extra={"gsw_plan_request_contract": True})
    spec = next(t for t in ad._orchestrator_tools
                if t["function"]["name"] == "request_plan_update")
    props = spec["function"]["parameters"]["properties"]
    assert "plan_request_id" in props
    assert "plan_request_id" not in spec["function"]["parameters"]["required"]


def test_the_tool_schema_has_no_plan_request_id_when_the_flag_is_off():
    """Finding 2: the base `_ORCHESTRATOR_TOOLS` module constant, and the tool
    list an adapter with the flag off actually sends the LLM, must both stay
    byte-identical to the pre-contract schema."""
    from panini_dr.adapters.ours.gsw_planner_orchestrator_v1 import (
        _ORCHESTRATOR_TOOLS,
    )
    for spec in _ORCHESTRATOR_TOOLS:
        if spec["function"]["name"] == "request_plan_update":
            assert "plan_request_id" not in spec["function"]["parameters"]["properties"]

    from panini_dr.adapters.ours.gsw_planner_orchestrator_v1 import (
        _CALL_PLANNER_TOOL,
    )

    ad = _mk_llm_adapter(_RoutedLLM({}), plan_dict=_one_level_plan(), extra={})
    assert ad.gsw_plan_request_contract is False
    assert ad._orchestrator_tools == _ORCHESTRATOR_TOOLS + [_CALL_PLANNER_TOOL]
    spec = next(t for t in ad._orchestrator_tools
                if t["function"]["name"] == "request_plan_update")
    assert "plan_request_id" not in spec["function"]["parameters"]["properties"]


def test_escalation_records_carry_no_request_id_or_pending_when_flag_is_off():
    """Finding 2: with the flag off, a malformed-or-not escalation dict must
    pass through `_register_revision_request` completely untouched, and
    nothing gets registered -- an id the orchestrator could never even see
    (the schema doesn't expose `plan_request_id` off-path either) must not be
    minted."""
    ad = _mk_llm_adapter(_RoutedLLM({}), plan_dict=_one_level_plan(), extra={})
    assert ad.gsw_plan_request_contract is False
    rr = {"blank_id": "b_x", "assigned_blank_ids": ["b_x"],
          "reason": "nothing in the corpus", "hint": "add b_bridge"}
    out = ad._register_revision_request(dict(rr), question_id="q9")
    assert out == rr
    assert "request_id" not in out
    assert "pending" not in out
    assert ad._plan_requests == {}


def test_an_escalation_is_registered_and_gets_a_stable_id():
    """The id is what the orchestrator passes back, so it must appear in the
    dispatch result the orchestrator reads."""
    ad = _mk_llm_adapter(_RoutedLLM({}), plan_dict=_one_level_plan(),
                         extra={"gsw_plan_request_contract": True})
    ad._plan_requests = {}
    rr = {"blank_id": "b_x", "assigned_blank_ids": ["b_x"],
          "reason": "nothing in the corpus", "hint": "add b_bridge"}
    out = ad._register_revision_request(rr, question_id="q9")
    assert out["request_id"] == "q9:b_x:0"
    # "pending", not "consumed": this dict is stamped once, at registration,
    # and never revisited -- the up-to-date consumed/not split lives only in
    # `_plan_request_summary()`.
    assert out["pending"] is True
    assert ad._plan_requests["q9:b_x:0"].hint == "add b_bridge"


def test_a_malformed_escalation_is_passed_through_without_an_id():
    """A researcher that escalates with an empty hint must not break dispatch,
    even with the flag on."""
    ad = _mk_llm_adapter(_RoutedLLM({}), plan_dict=_one_level_plan(),
                         extra={"gsw_plan_request_contract": True})
    ad._plan_requests = {}
    out = ad._register_revision_request({"reason": "x", "hint": "  "}, question_id="q9")
    assert "request_id" not in out
    assert ad._plan_requests == {}


def test_consumed_is_recorded_when_an_update_cites_the_request():
    ad = _mk_llm_adapter(_RoutedLLM({}), plan_dict=_one_level_plan(),
                         extra={"gsw_plan_request_contract": True})
    ad._plan_requests = {}
    ad._consumed_request_ids = set()
    ad._register_revision_request(
        {"blank_id": "b_x", "assigned_blank_ids": ["b_x"],
         "reason": "r", "hint": "h"}, question_id="q9")
    ad._consumed_request_ids.add("q9:b_x:0")
    summary = ad._plan_request_summary()
    assert summary == [{"request_id": "q9:b_x:0", "blank_id": "b_x", "consumed": True}]


def test_an_unconsumed_request_is_reported_as_such():
    """This is the measurement: 13 escalations were dropped and nothing said so."""
    ad = _mk_llm_adapter(_RoutedLLM({}), plan_dict=_one_level_plan(),
                         extra={"gsw_plan_request_contract": True})
    ad._plan_requests = {}
    ad._consumed_request_ids = set()
    ad._register_revision_request(
        {"blank_id": "b_x", "assigned_blank_ids": ["b_x"],
         "reason": "r", "hint": "h"}, question_id="q9")
    assert ad._plan_request_summary() == [
        {"request_id": "q9:b_x:0", "blank_id": "b_x", "consumed": False}]


def test_the_real_update_call_site_records_consumption_and_tags_plan_updates():
    """End-to-end through the real call site, not a direct call to the
    helpers.

    Every direct-call test above exercises `_register_revision_request` /
    `_plan_request_summary` directly -- none of them would notice if the wire
    at the actual `update_plan(...)` call site (where `_consumed_request_ids`
    is really updated, and where a `plan_updates` record is really built) went
    missing. This one runs the full dispatch -> escalate -> request_plan_update
    (citing plan_request_id) -> commit path and checks the real state,
    including `traj.extra["plan_requests"]`, which only `run_question` writes.
    """
    plan_dict = _one_level_plan()
    routes = {
        ORCH: [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "request_plan_update", {
                "reason": "researcher escalated for t",
                "evidence": "need intermediate",
                "plan_request_id": "qplumb:t:0",
            })]),
            _StubResponse(tool_calls=[_tc("o3", "dispatch_subplan",
                                         {"blank_ids": ["b_intermediate"]})]),
            _StubResponse(tool_calls=[_tc("o4", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o5", "submit_answer",
                                         {"answer": "FINAL"})]),
        ],
        # Researcher for t (original plan) escalates first, then commits after
        # the plan revision installs b_intermediate.
        "`t` (role=`target`, value_type=`entity`": [
            _StubResponse(tool_calls=[_tc("rA1", "suggest_plan_revision", {
                "reason": "no chunk surfaces t directly",
                "hint": "add b_intermediate; then derive t",
            })]),
            _StubResponse(tool_calls=[_tc("rA2", "update_blank", {
                "blank_id": "t",
                "value": "FINAL",
                "evidence_chunk_ids": ["c1"],
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
    adapter = _mk_llm_adapter(
        llm, plan_dict=plan_dict, orchestrator_max_turns=10,
        extra={"gsw_plan_request_contract": True},
    )
    traj = adapter.run_question("Q", question_id="qplumb")

    assert traj.final_answer == "FINAL"
    # blank_id "t", first escalation this question -> seq 0.
    assert "qplumb:t:0" in adapter._plan_requests
    # Recorded by the real call site, not by a test manually poking the set.
    assert adapter._consumed_request_ids == {"qplumb:t:0"}
    assert adapter._plan_request_summary() == [
        {"request_id": "qplumb:t:0", "blank_id": "t", "consumed": True}]
    plan_updates = traj.extra["plan_updates"]
    assert len(plan_updates) == 1
    assert plan_updates[0]["plan_request_id"] == "qplumb:t:0"
    # Finding 3: `_plan_request_summary()` must actually be reachable off the
    # trajectory, not just off the live adapter object.
    assert traj.extra["plan_requests"] == adapter._plan_request_summary()


def test_a_plan_added_blank_reaches_the_serialised_workspace_state():
    """2026-09-04. `qgsw_state` is initialised from the INITIAL plan, and the
    mirror used to skip result ids it did not know, so a blank added by
    `request_plan_update` never appeared in `traj.extra["gsw_state"]` -- on
    the BCP / FRAMES / MoNaCo runs of record, none of the 50 / 113 / 502
    revision-added blanks did, and their verifier flags were dropped with
    them. Same route table as the consumption test above; this one checks the
    serialised workspace, which only `run_question` writes."""
    plan_dict = _one_level_plan()
    routes = {
        ORCH: [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "request_plan_update", {
                "reason": "researcher escalated for t",
                "evidence": "need intermediate",
                "plan_request_id": "qplumb2:t:0",
            })]),
            _StubResponse(tool_calls=[_tc("o3", "dispatch_subplan",
                                         {"blank_ids": ["b_intermediate"]})]),
            _StubResponse(tool_calls=[_tc("o4", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o5", "submit_answer",
                                         {"answer": "FINAL"})]),
        ],
        "`t` (role=`target`, value_type=`entity`": [
            _StubResponse(tool_calls=[_tc("rA1", "suggest_plan_revision", {
                "reason": "no chunk surfaces t directly",
                "hint": "add b_intermediate; then derive t",
            })]),
            _StubResponse(tool_calls=[_tc("rA2", "update_blank", {
                "blank_id": "t",
                "value": "FINAL",
                "evidence_chunk_ids": ["c1"],
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
    adapter = _mk_llm_adapter(
        _RoutedLLM(routes), plan_dict=plan_dict, orchestrator_max_turns=10,
        extra={"gsw_plan_request_contract": True,
               "gsw_evidence_state": True, "gsw_evidence_gate": "off"},
    )
    traj = adapter.run_question("Q", question_id="qplumb2")
    assert traj.final_answer == "FINAL"
    assert traj.extra["plan_updates"][0]["added_ids"] == ["b_intermediate"]

    blanks = traj.extra["gsw_state"]["blanks"]
    assert "b_intermediate" in blanks, sorted(blanks)
    bi = blanks["b_intermediate"]
    assert bi["status"] == "resolved"
    assert bi["value"] == "associated thing"
    assert bi["evidence_chunk_ids"] == ["c2"]
    assert bi["is_target"] is False
    assert traj.extra["gsw_resolution_provenance"]["b_intermediate"] == "react"
    # the ORIGINAL blanks are untouched by the resync
    assert blanks["t"]["is_target"] is True


# ---------------------------------------------------------------------------
# Final-review fix wave (2026-08-09): Finding 2 (contract fields thrown away),
# Finding 3(b) (prompt never told the orchestrator to cite plan_request_id),
# and Finding 4 (a failed id lookup indistinguishable from no id).
# ---------------------------------------------------------------------------


def test_plan_updates_record_carries_contract_and_contract_active():
    """Finding 2: `diff.contract` / `diff.contract_satisfied` were computed
    by `update_plan` and thrown away at the orchestrator call site -- they
    reached neither the `plan_updates` record nor `traj.extra`, making the
    spec's "contract satisfaction rate logged per update" success criterion
    unmeasurable from a run. `contract_active` is also carried explicitly
    (not inferred from `contract == []`), so an analyst can tell "flag off"
    apart from "contract active but the author returned an empty contract".

    Adds a real "You are the **Contract Author**" route (the other plumbing
    tests in this file leave it unrouted, which fails open to an empty
    contract and would let a "never actually copied" bug hide behind
    `.get(..., [])` defaults) so the assertions below check REAL non-default
    content, not just the presence of the keys.
    """
    plan_dict = _one_level_plan()
    routes = {
        ORCH: [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "request_plan_update", {
                "reason": "researcher escalated for t",
                "evidence": "need intermediate",
                "plan_request_id": "qcontract:t:0",
            })]),
            _StubResponse(tool_calls=[_tc("o3", "dispatch_subplan",
                                         {"blank_ids": ["b_intermediate"]})]),
            _StubResponse(tool_calls=[_tc("o4", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o5", "submit_answer",
                                         {"answer": "FINAL"})]),
        ],
        "`t` (role=`target`, value_type=`entity`": [
            _StubResponse(tool_calls=[_tc("rA1", "suggest_plan_revision", {
                "reason": "no chunk surfaces t directly",
                "hint": "add b_intermediate; then derive t",
            })]),
            _StubResponse(tool_calls=[_tc("rA2", "update_blank", {
                "blank_id": "t",
                "value": "FINAL",
                "evidence_chunk_ids": ["c1"],
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
        "You are the **Contract Author**": [
            _StubResponse(tool_calls=[_tc("a1", "emit_contract",
                                          {"clauses": [{"kind": "blank_added"}]})]),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(
        llm, plan_dict=plan_dict, orchestrator_max_turns=10,
        extra={"gsw_plan_request_contract": True},
    )
    traj = adapter.run_question("Q", question_id="qcontract")

    assert traj.final_answer == "FINAL"
    plan_updates = traj.extra["plan_updates"]
    assert len(plan_updates) == 1
    rec = plan_updates[0]
    assert rec["plan_request_id"] == "qcontract:t:0"
    assert rec["contract_active"] is True
    assert rec["contract"] == [
        {"kind": "blank_added", "blank_id": None, "value_type": None}]
    assert rec["contract_satisfied"] == [
        {"kind": "blank_added", "blank_id": None, "value_type": None}]


def test_plan_request_id_raw_is_recorded_separately_from_the_resolved_id():
    """Finding 4: citing an id this question never registered (hallucinated,
    or crossed from another question mid-flight) must not be
    indistinguishable from omitting `plan_request_id` altogether -- both
    used to collapse to the same `plan_request_id: None` on the record. The
    RAW argument is now carried alongside the resolved one, so a lookup MISS
    (raw truthy, resolved None) is visible.

    No `suggest_plan_revision` fires in this script at all -- `t` resolves
    directly -- so `bogus-id-never-registered` cannot possibly be a real,
    just-stale id; it is cited out of nowhere, the case this finding is
    about.
    """
    plan_dict = _one_level_plan()
    routes = {
        ORCH: [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "request_plan_update", {
                "reason": "orchestrator's own read",
                "evidence": "state looks wrong",
                "plan_request_id": "bogus-id-never-registered",
            })]),
            _StubResponse(tool_calls=[_tc("o3", "submit_answer",
                                         {"answer": "FINAL"})]),
        ],
        "`t` (role=`target`, value_type=`entity`": [
            _StubResponse(tool_calls=[_tc("r1", "update_blank", {
                "blank_id": "t", "value": "FINAL", "evidence_chunk_ids": ["c1"],
            })]),
        ],
        "You are the **Plan-Updater**": [
            _StubResponse(text=_add_blank_op("b_intermediate", "vp_int")),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(
        llm, plan_dict=plan_dict, orchestrator_max_turns=10,
        extra={"gsw_plan_request_contract": True},
    )
    traj = adapter.run_question("Q", question_id="qmiss")

    assert traj.final_answer == "FINAL"
    plan_updates = traj.extra["plan_updates"]
    assert len(plan_updates) == 1
    rec = plan_updates[0]
    assert rec["plan_request_id"] is None, "unregistered id -- lookup must miss"
    assert rec["plan_request_id_raw"] == "bogus-id-never-registered"


def test_flag_off_prompt_is_byte_identical_and_flag_on_adds_one_sentence():
    """Finding 3(b): rule 7 previously never told the orchestrator to cite
    `plan_request_id`, so the authored-clause machinery could never fire even
    when a researcher HAD escalated. The fix splices one sentence into rule 7
    -- but ONLY when `gsw_plan_request_contract` is on, exactly like the tool
    schema is already gated (`test_the_tool_schema_has_no_plan_request_id_
    when_the_flag_is_off` above). This is the prompt half of that same "flag
    off -> byte-identical" constraint.

    Checked directly against the builder functions (not through an adapter
    run) so the assertion is exact: omitting `plan_request_contract` entirely
    (today's every other call site, pre-feature) must render identically to
    passing it explicitly as `False`, for BOTH the static half
    (`build_orchestrator_system_static`, cached across a whole question) and
    (the only orchestrator prompt once the plan is installed).
    """
    from panini_dr.adapters.ours._orchestrator_prompt import (
        build_orchestrator_system_static,
    )
    plan_dict = _one_level_plan()
    state = {"t": BlankResult(blank_id="t", status="unknown")}

    omitted_static = build_orchestrator_system_static(
        plan_dict, ["t"], {"t": 0})
    off_static = build_orchestrator_system_static(
        plan_dict, ["t"], {"t": 0}, plan_request_contract=False)
    on_static = build_orchestrator_system_static(
        plan_dict, ["t"], {"t": 0}, plan_request_contract=True)
    assert off_static == omitted_static, "flag-off must match omitted"
    assert "plan_request_id" not in off_static
    assert "{PLAN_REQUEST_ID_NOTE}" not in off_static
    assert "{PLAN_REQUEST_ID_NOTE}" not in on_static
    assert on_static != off_static
    assert "plan_request_id" in on_static


def test_orchestrator_calls_build_prompt_with_the_flag_when_live():
    """The flag must actually reach the prompt builder from a real adapter,
    not just be plumbable in principle. With the flag on, the FIRST live
    orchestrator system message (built before any tool call) must carry the
    `plan_request_id` sentence; with it off, it must not."""
    plan_dict = _one_level_plan()

    llm_on = _RoutedLLM({ORCH: [
        _StubResponse(tool_calls=[_tc("o1", "submit_answer", {"answer": "FINAL"})]),
    ]})
    ad_on = _mk_llm_adapter(
        llm_on, plan_dict=plan_dict, orchestrator_max_turns=5,
        extra={"gsw_plan_request_contract": True},
    )
    ad_on.run_question("Q", question_id="qpromptflagon")
    on_system = _first_loop_system(llm_on)
    assert "plan_request_id" in on_system

    llm_off = _RoutedLLM({ORCH: [
        _StubResponse(tool_calls=[_tc("o1", "submit_answer", {"answer": "FINAL"})]),
    ]})
    ad_off = _mk_llm_adapter(
        llm_off, plan_dict=plan_dict, orchestrator_max_turns=5,
    )
    ad_off.run_question("Q", question_id="qpromptflagoff")
    off_system = _first_loop_system(llm_off)
    assert "plan_request_id" not in off_system


def _first_loop_system(llm: _RoutedLLM) -> str:
    """System message of the first main-loop orchestrator call (the planning
    phase before it offers only call_planner and has its own prompt)."""
    call = next(c for c in llm.calls
                if any(t["function"]["name"] == "dispatch_subplan"
                       for t in (c.get("tools") or [])))
    return call["messages"][0]["content"]


def test_run_question_records_the_consumed_and_dropped_split():
    """Finding 3's own success criterion: 'the 13 dropped escalations become
    measurable'. Two escalations in one question, only one cited by an update
    -- the other must show up as `consumed: False` in
    `traj.extra["plan_requests"]`, not be silently absent from it."""
    plan_dict = _concat_plan()
    routes = {
        ORCH: [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan", {"blank_ids": ["b_a"]})]),
            _StubResponse(tool_calls=[_tc("o2", "request_plan_update", {
                "reason": "b_a needs a bridge", "evidence": "e",
                "plan_request_id": "qsplit:b_a:0",
            })]),
            _StubResponse(tool_calls=[_tc("o3", "dispatch_subplan", {"blank_ids": ["b_b"]})]),
        ],
        "`b_a` (role=`bridge-attribute`, value_type=`text`": [
            _StubResponse(tool_calls=[_tc("ra1", "suggest_plan_revision", {
                "reason": "no chunk for b_a", "hint": "add a bridge blank",
            })]),
        ],
        "`b_b` (role=`bridge-attribute`, value_type=`text`": [
            _StubResponse(tool_calls=[_tc("rb1", "suggest_plan_revision", {
                "reason": "no chunk for b_b either", "hint": "add another bridge",
            })]),
        ],
        "You are the **Plan-Updater**": [
            _StubResponse(text=_add_blank_op("b_bridge", "vp_bridge", subject_id="e1")),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(
        llm, plan_dict=plan_dict, orchestrator_max_turns=10,
        extra={"gsw_plan_request_contract": True},
    )
    traj = adapter.run_question("Q", question_id="qsplit")

    # b_a registered first (seq 0, dispatched alone) and was cited and
    # consumed; b_b registered second (seq 1, dispatched alone, after the
    # update) and was never cited by anything.
    assert traj.extra["plan_requests"] == [
        {"request_id": "qsplit:b_a:0", "blank_id": "b_a", "consumed": True},
        {"request_id": "qsplit:b_b:1", "blank_id": "b_b", "consumed": False},
    ]


def test_the_validator_guided_repair_escalation_site_is_wired():
    """Pin: `_try_validator_guided_repair` has its OWN escalation-collection
    site, separate from `dispatch_subplan`'s. As committed, unwiring just
    this second site left every other test in this file (and the full
    391-test selection) green -- nothing pinned it."""
    plan_dict = _one_level_plan()
    plan = GSWPlan.model_validate(plan_dict)
    state = {"t": BlankResult(blank_id="t", status="resolved", value="Wrong Answer",
                              evidence_chunk_ids=["c0"])}
    routes = {
        "`t` (role=`target`, value_type=`entity`": [
            _StubResponse(tool_calls=[_tc("r1", "suggest_plan_revision", {
                "reason": "no chunk supports t", "hint": "add a bridging blank",
            })]),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict,
                              extra={"gsw_plan_request_contract": True})
    adapter._active_question_id = "qvgr"
    adapter._plan_requests = {}
    adapter._consumed_request_ids = set()

    verdict = SimpleNamespace(
        error_type="wrong_entity", confidence="high", flagged_blank="t",
        reason="wrong entity", corrected_answer=None, suggested_correction=None,
        evidence_support="",
    )
    result, action = adapter._try_validator_guided_repair(
        verdict=verdict, proposed_answer="Wrong Answer",
        plan=plan, plan_dict=plan_dict, state=state,
        order=["t"], levels={"t": 0}, question="Q", dispatch_idx=0,
        constraint_outputs=set(), guided_repairs_used=0,
    )
    assert result is not None
    rrs = action.get("revision_requests") or []
    assert len(rrs) == 1
    assert rrs[0].get("request_id") == "qvgr:t:0"
    assert adapter._plan_requests  # registered, not just passed through


def test_the_per_question_reset_prevents_seq_numbering_from_leaking_across_questions():
    """Pin: as committed, removing the per-question reset in `run_question`
    (while keeping the thread-local backing from Finding 1) currently passes
    every other test in this file. One thread runs two questions in
    sequence -- exactly the ThreadPoolExecutor-worker-reuse pattern real runs
    use -- and the second question's escalation must start its own numbering
    at seq 0, not continue the first question's count."""
    plan_dict = _one_level_plan()
    routes = {
        ORCH: [_StubResponse(tool_calls=[_tc("o1", "dispatch_subplan", {"blank_ids": ["t"]})])],
        "`t` (role=`target`, value_type=`entity`": [
            _StubResponse(tool_calls=[_tc("r1", "suggest_plan_revision", {
                "reason": "no chunk for t", "hint": "add a bridge",
            })]),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(llm, plan_dict=plan_dict, orchestrator_max_turns=5,
                              extra={"gsw_plan_request_contract": True})
    traj1 = adapter.run_question("Q1", question_id="q1")
    assert traj1.extra["plan_requests"] == [
        {"request_id": "q1:t:0", "blank_id": "t", "consumed": False}]

    # Re-arm the SAME llm object with a fresh script for question 2 -- a real
    # pooled worker thread runs many questions in sequence against the one
    # adapter it was given, not just one.
    adapter.llm.routes = {k: list(v) for k, v in routes.items()}
    traj2 = adapter.run_question("Q2", question_id="q2")
    assert traj2.extra["plan_requests"] == [
        {"request_id": "q2:t:0", "blank_id": "t", "consumed": False}]


class _WaitThen:
    """Sentinel for `_GatedRoutedLLM`: when popped from a route's script, it
    announces (if given) that this thread has reached the gate, then blocks
    until `release` is set before the NEXT scripted response for that route
    is used. Lets a test force one question to run to completion strictly
    BETWEEN another question's escalate turn and its request_plan_update
    turn, instead of hoping a race shows up under timing luck."""

    def __init__(self, release: threading.Event, *,
                 announce: threading.Event | None = None, timeout: float = 10.0):
        self.release = release
        self.announce = announce
        self.timeout = timeout


class _GatedRoutedLLM(_RoutedLLM):
    """`_RoutedLLM`, but a `_WaitThen` in a route's script list blocks that
    thread -- without holding the routing lock, so the other question's
    `.chat()` calls are never blocked by it -- until released."""

    def chat(self, messages, *, tools=None, max_tokens=None, **kwargs):
        opening = _planning_phase_opening(messages, tools)
        if opening is not None:
            return opening
        sys_content = ""
        for m in messages:
            if m.get("role") == "system":
                sys_content = m.get("content", "") or ""
                break
        with self._lock:
            self.calls.append({"messages": list(messages), "tools": tools, **kwargs})
            chosen_key = None
            for key in self.routes:
                if key in sys_content:
                    chosen_key = key
                    break
            if chosen_key is None or not self.routes[chosen_key]:
                return _StubResponse(text="(no route)", tool_calls=[])
            item = self.routes[chosen_key].pop(0)
        if isinstance(item, _WaitThen):
            if item.announce is not None:
                item.announce.set()
            if not item.release.wait(timeout=item.timeout):
                raise TimeoutError("test gate was never released")
            with self._lock:
                item = self.routes[chosen_key].pop(0)
        return item


def _named_one_level_plan(entity_name: str, target_id: str) -> dict:
    """`_one_level_plan()`, but with a distinctive entity name and target id
    so two of these can share one `_GatedRoutedLLM`'s routing without either
    question's script leaking into the other's."""
    return {
        "entities": [
            {"id": "e1", "kind": "filled", "name": entity_name, "role": "subject"},
            {"id": target_id, "kind": "blank", "role": "target",
             "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "released", "subject_id": "e1", "object_id": target_id},
        ],
        "constraints": [],
    }


def test_concurrent_questions_on_one_adapter_do_not_corrupt_each_others_registry():
    """Finding 1 (CRITICAL): `eval/harness.py` fans `run_question` over a
    ThreadPoolExecutor against ONE adapter instance (concurrency 32-64 in
    real runs). Before the thread-local fix, one question's per-question
    reset (top of `run_question`) wiped another in-flight question's
    registry -- reproduced originally with A's cited `plan_request_id`
    resolving to None mid-flight and the contract silently not firing.

    This forces that exact interleaving with a gate instead of hoping a race
    shows up under timing luck: question A registers its escalation, then
    blocks (mid tool-call, holding no lock) while question B runs an entire
    escalate -> update -> resolve cycle to completion -- including B's OWN
    per-question reset -- on the SAME adapter, before A is allowed to finish
    citing its own `plan_request_id`.
    """
    plan_a = _named_one_level_plan("Pink Floyd", "tA")
    plan_b = _named_one_level_plan("Radiohead", "tB")

    reached_gate = threading.Event()
    release_gate = threading.Event()

    routes = {
        # Blank-id markers first: they are more specific and must win over
        # the entity-name markers below, which appear in BOTH the
        # orchestrator's prompt and the researcher's (upstream anchor).
        "`tA` (role=`target`, value_type=`entity`": [
            _StubResponse(tool_calls=[_tc("rA1", "suggest_plan_revision", {
                "reason": "no chunk surfaces tA directly", "hint": "add a bridge for A",
            })]),
            _StubResponse(tool_calls=[_tc("rA2", "update_blank", {
                "blank_id": "tA", "value": "FINAL_A", "evidence_chunk_ids": ["c1"],
            })]),
        ],
        "`tB` (role=`target`, value_type=`entity`": [
            _StubResponse(tool_calls=[_tc("rB1", "suggest_plan_revision", {
                "reason": "no chunk surfaces tB directly", "hint": "add a bridge for B",
            })]),
            _StubResponse(tool_calls=[_tc("rB2", "update_blank", {
                "blank_id": "tB", "value": "FINAL_B", "evidence_chunk_ids": ["c2"],
            })]),
        ],
        "Pink Floyd": [
            _StubResponse(tool_calls=[_tc("oA1", "dispatch_subplan", {"blank_ids": ["tA"]})]),
            _WaitThen(release_gate, announce=reached_gate),
            _StubResponse(tool_calls=[_tc("oA2", "request_plan_update", {
                "reason": "A escalated for tA", "evidence": "need bridge A",
                "plan_request_id": "qA:tA:0",
            })]),
            _StubResponse(tool_calls=[_tc("oA3", "dispatch_subplan", {"blank_ids": ["tA"]})]),
            _StubResponse(tool_calls=[_tc("oA4", "submit_answer", {"answer": "FINAL_A"})]),
        ],
        "Radiohead": [
            _StubResponse(tool_calls=[_tc("oB1", "dispatch_subplan", {"blank_ids": ["tB"]})]),
            _StubResponse(tool_calls=[_tc("oB2", "request_plan_update", {
                "reason": "B escalated for tB", "evidence": "need bridge B",
                "plan_request_id": "qB:tB:0",
            })]),
            _StubResponse(tool_calls=[_tc("oB3", "dispatch_subplan", {"blank_ids": ["tB"]})]),
            _StubResponse(tool_calls=[_tc("oB4", "submit_answer", {"answer": "FINAL_B"})]),
        ],
        # Shared marker (the updater's SYSTEM message never varies by plan) --
        # two IDENTICAL, order-independent entries so it doesn't matter which
        # question's call consumes which list slot.
        "You are the **Plan-Updater**": [
            _StubResponse(text=_add_blank_op("b_intermediate", "vp_int", subject_id="e1")),
            _StubResponse(text=_add_blank_op("b_intermediate", "vp_int", subject_id="e1")),
        ],
    }
    llm = _GatedRoutedLLM(routes)
    adapter = _mk_llm_adapter(
        llm, plan_dict=plan_a, orchestrator_max_turns=10,
        # The live orchestrator's system half is static (no state table; state
        # arrives as separate messages `_RoutedLLM` doesn't inspect), so
        # "Pink Floyd"/"Radiohead" cleanly distinguish the two questions'
        # orchestrator turns from their researcher turns.
        extra={"gsw_plan_request_contract": True},
    )
    # `_mk_llm_adapter` fakes `emit_plan` to always return ONE fixed plan;
    # this test needs A's question to plan as plan_a and B's as plan_b, so
    # route on the question text (which the real `emit_plan` would do too).
    import panini_dr.adapters.ours.gsw_planner_orchestrator_v1 as gmod
    from panini_dr.adapters.ours._planner_emit import PlanEmitMeta

    def _multi_plan_emit(question, llm_client, **kwargs):
        chosen = plan_a if "Pink Floyd" in question else plan_b
        return GSWPlan.model_validate(chosen), PlanEmitMeta(
            prompt_tokens=1, completion_tokens=1, raw_response="{}")

    gmod.emit_plan = _multi_plan_emit

    results: dict[str, "object"] = {}
    errors: list[BaseException] = []

    def run_a():
        try:
            results["A"] = adapter.run_question("Q about Pink Floyd", question_id="qA")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def run_b():
        try:
            assert reached_gate.wait(timeout=10.0), "question A never reached the gate"
            results["B"] = adapter.run_question("Q about Radiohead", question_id="qB")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            release_gate.set()

    ta = threading.Thread(target=run_a)
    tb = threading.Thread(target=run_b)
    ta.start()
    tb.start()
    ta.join(timeout=20.0)
    tb.join(timeout=20.0)

    assert not ta.is_alive() and not tb.is_alive(), "a thread hung"
    assert not errors, errors

    traj_a, traj_b = results["A"], results["B"]
    assert traj_a.final_answer == "FINAL_A"
    assert traj_b.final_answer == "FINAL_B"

    # The load-bearing assertions: each question's own request resolved to
    # its OWN citation, and neither registry was emptied by the other's
    # in-flight reset.
    assert traj_a.extra["plan_updates"][0]["plan_request_id"] == "qA:tA:0"
    assert traj_b.extra["plan_updates"][0]["plan_request_id"] == "qB:tB:0"
    assert traj_a.extra["plan_requests"] == [
        {"request_id": "qA:tA:0", "blank_id": "tA", "consumed": True}]
    assert traj_b.extra["plan_requests"] == [
        {"request_id": "qB:tB:0", "blank_id": "tB", "consumed": True}]
