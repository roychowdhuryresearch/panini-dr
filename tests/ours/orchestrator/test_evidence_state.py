"""Evidence state (QueryGSWState), the evidence gate in measure mode, Stage-B verify_graph, hard type filter, coverage and cascade diagnostics.

Split from the research repo's test_planner_orchestrator.py."""

from __future__ import annotations

from tests.ours.orchestrator._helpers import *  # noqa: F401,F403


def test_gsw_evidence_state_off_is_byte_identical():
    adapter = _mk_adapter(
        _orchestrated_llm(_one_level_script()), plan_dict=_one_level_plan()
    )  # flag defaults off
    traj = adapter.run_question(
        "What album did Pink Floyd release?", question_id="q1"
    )
    assert "gsw_state" not in traj.extra
    assert traj.final_answer == "Wish You Were Here"


def test_gsw_evidence_state_on_logs_without_changing_answer():
    adapter = _mk_adapter(
        _orchestrated_llm(_one_level_script()),
        plan_dict=_one_level_plan(),
        extra={"gsw_evidence_state": True},
    )
    traj = adapter.run_question(
        "What album did Pink Floyd release?", question_id="q1"
    )
    # Answer is identical to the flag-off path.
    assert traj.final_answer == "Wish You Were Here"
    # Structured state is logged.
    gsw = traj.extra["gsw_state"]
    assert gsw["question"] == "What album did Pink Floyd release?"
    assert set(gsw["blanks"]) == {"t"}
    t = gsw["blanks"]["t"]
    assert t["is_target"] is True
    assert t["status"] == "resolved"
    assert t["value"] == "Wish You Were Here"
    assert t["evidence_chunk_ids"] == ["chunk_1"]
    # Projection: t is the object of vp1 "released" with subject "Pink Floyd".
    assert t["subject"] == "Pink Floyd"
    assert t["relation"] == "released"
    assert t["direction"] == "forward"
    assert gsw["final_answer"] == "Wish You Were Here"


def test_evidence_gate_measure_commits_regardless():
    plan = _GSWPlan.model_validate(_one_level_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    script = [
        _StubResponse(tool_calls=[_tc("c1", "update_blank", {
            "blank_id": "t", "value": "Animals",
            "evidence_chunk_ids": ["chunk_1"],
        })]),
    ]
    r = ReActResearcher(
        llm=_ScriptedLLM(script), corpus=_grounding_corpus(),
        retriever=_StubRetriever({}), plan=plan, state=state,
        allowed_blank_ids=["t"], system_prompt="SYS", question="Q",
        max_turns=3,
    )
    r.solve()
    assert state["t"].value == "Animals"
    assert state["t"].status == "resolved"


# ---------------------------------------------------------------------------
# Phase 2 — edges collected into QueryGSWState + metrics on traj.extra
# ---------------------------------------------------------------------------


def test_evidence_gate_logged_into_gsw_state_and_metrics():
    script = [
        _StubResponse(tool_calls=[_tc("c1", "update_blank", {
            "blank_id": "t", "value": "The Wall",
            "evidence_chunk_ids": ["chunk_1"],
        })]),
    ]
    adapter = _mk_adapter(
        _orchestrated_llm(script), plan_dict=_one_level_plan(),
        corpus=_grounding_corpus(),
        extra={"gsw_evidence_gate": "measure"},
    )
    traj = adapter.run_question("What album?", question_id="q1")

    gsw = traj.extra["gsw_state"]
    edges = gsw["evidence_edges"]
    assert len(edges) == 1
    (edge,) = edges.values()
    assert edge["subject"] == "Pink Floyd"
    assert edge["relation"] == "released"
    assert edge["object"] == "The Wall"
    assert edge["status"] == "accepted"
    m = traj.extra["gsw_evidence_metrics"]
    assert m["n_grounded"] == 1 and m["n_missing_dependency"] == 0 and m["pct_grounded"] == 1.0


def test_evidence_gate_off_default_no_metrics_key():
    adapter = _mk_adapter(
        _orchestrated_llm(_one_level_script()), plan_dict=_one_level_plan(),
    )
    traj = adapter.run_question("What album?", question_id="q1")
    assert "gsw_evidence_metrics" not in traj.extra
    assert "gsw_state" not in traj.extra  # gate off AND state off
    assert traj.final_answer == "Wish You Were Here"


# ---------------------------------------------------------------------------
# Phase 2 — llm-orchestrator-mode parity for the gate
# ---------------------------------------------------------------------------


def test_evidence_gate_llm_mode_collects_edges():
    plan_dict = _one_level_plan()
    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "submit_answer",
                                         {"answer": "The Wall"})]),
        ],
        "resolve exactly these blanks": [
            _StubResponse(tool_calls=[_tc("r1", "update_blank", {
                "blank_id": "t", "value": "The Wall",
                "evidence_chunk_ids": ["chunk_1"],
            })]),
        ],
    }
    adapter = _mk_llm_adapter(
        _RoutedLLM(routes), plan_dict=plan_dict, corpus=_grounding_corpus(),
        extra={"gsw_evidence_gate": "measure"},
    )
    traj = adapter.run_question("What album?", question_id="qllm1")
    assert traj.final_answer == "The Wall"
    gsw = traj.extra["gsw_state"]
    assert len(gsw["evidence_edges"]) == 1
    (edge,) = gsw["evidence_edges"].values()
    assert edge["subject"] == "Pink Floyd"
    assert edge["relation"] == "released"
    assert edge["object"] == "The Wall"
    assert edge["status"] == "accepted"
    assert traj.extra["gsw_evidence_metrics"]["n_grounded"] == 1


# ---------------------------------------------------------------------------
# Task 3 — Stage B verify_graph wired into run_question (both modes)
# ---------------------------------------------------------------------------


def test_stage_b_grounds_multihop_relation_after_merge():
    # Two dependent blanks are resolved across LLM dispatches. Stage B must
    # ground "t co_founded b_group" using b_group's RESOLVED value.
    # The Stage B verify_graph edge uses edge_id = "{vp_id}:rel" = "vp1:rel",
    # which is distinct from the per-worker edges ("t:e0", "b_group:e0").
    plan_dict = {
        "entities": [
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True},
            {"id": "b_group", "kind": "blank", "value_type": "entity"},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "co_founded", "subject_id": "t", "object_id": "b_group"},
        ],
        "constraints": [],
    }
    corpus = _StubCorpus({
        "c1": _StubChunk(chunk_id="c1", title="Yvon Chouinard",
                         text="Yvon Chouinard co-founded One Percent for the Planet."),
    })
    # Route order matters: the target researcher sees b_group in prior values,
    # so the target route must be checked before the b_group route.
    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["b_group"]})]),
            _StubResponse(tool_calls=[_tc("o2", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o3", "submit_answer",
                                         {"answer": "Yvon Chouinard"})]),
        ],
        "**TARGET**": [_StubResponse(tool_calls=[_tc("r2", "update_blank", {
            "blank_id": "t", "value": "Yvon Chouinard",
            "evidence_chunk_ids": ["c1"]})])],
        "`b_group`": [_StubResponse(tool_calls=[_tc("r1", "update_blank", {
            "blank_id": "b_group", "value": "One Percent for the Planet",
            "evidence_chunk_ids": ["c1"]})])],
    }
    adapter = _mk_llm_adapter(
        _RoutedLLM(routes), plan_dict=plan_dict, corpus=corpus,
        extra={"gsw_evidence_gate": "measure"},
    )
    traj = adapter.run_question("who?", question_id="qm")
    edges = traj.extra["gsw_state"]["evidence_edges"]
    # Stage B verify_graph produces an edge with edge_id = "{vp_id}:rel".
    # This is the ONLY way to get a relation-level edge with the fully-resolved
    # object value ("One Percent for the Planet" rather than raw blank id "b_group").
    stage_b_edge = edges.get("vp1:rel")
    assert stage_b_edge is not None, (
        f"Stage B did not build the vp1:rel relation edge; got edge ids: "
        f"{list(edges.keys())}"
    )
    assert stage_b_edge["relation"] == "co_founded"
    assert stage_b_edge["object"] == "One Percent for the Planet"   # resolved, not 'b_group'
    assert stage_b_edge["status"] == "accepted"


# ---------------------------------------------------------------------------
# Phase 3-lite: Stage A hard type filter (value-local admission)
# ---------------------------------------------------------------------------


def test_hard_type_filter_rejects_type_mismatch_then_accepts_match():
    # `t` is a DATE blank. The type filter must reject "Orhan Pamuk" (not a
    # date) and admit "2006". Value-local, so it needs no other blank.
    plan = _GSWPlan.model_validate({
        "entities": [
            {"id": "e", "kind": "filled", "name": "Serbia"},
            {"id": "t", "kind": "blank", "value_type": "date", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "became_independent_in_year",
             "subject_id": "e", "object_id": "t"},
        ],
        "constraints": [],
    })
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    script = [
        _StubResponse(tool_calls=[_tc("c1", "update_blank", {
            "blank_id": "t", "value": "Orhan Pamuk",   # not a date → rejected
            "evidence_chunk_ids": ["chunk_1"]})]),
        _StubResponse(tool_calls=[_tc("c2", "update_blank", {
            "blank_id": "t", "value": "2006",           # a date → admitted
            "evidence_chunk_ids": ["chunk_1"]})]),
    ]
    r = ReActResearcher(
        llm=_ScriptedLLM(script), corpus=_grounding_corpus(),
        retriever=_StubRetriever({}), plan=plan, state=state,
        allowed_blank_ids=["t"], system_prompt="SYS", question="Q",
        max_turns=5, hard_type_filter=True,
    )
    trace = r.solve()
    assert state["t"].value == "2006"
    assert state["t"].status == "resolved"
    assert any(
        tc.name == "update_blank" and "Type filter" in (tc.error or "")
        for tc in trace["tool_calls"]
    )


def test_hard_type_filter_off_is_permissive():
    plan = _GSWPlan.model_validate({
        "entities": [
            {"id": "e", "kind": "filled", "name": "Serbia"},
            {"id": "t", "kind": "blank", "value_type": "date", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "x", "subject_id": "e", "object_id": "t"},
        ],
        "constraints": [],
    })
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    script = [_StubResponse(tool_calls=[_tc("c1", "update_blank", {
        "blank_id": "t", "value": "Orhan Pamuk",
        "evidence_chunk_ids": ["chunk_1"]})])]
    r = ReActResearcher(
        llm=_ScriptedLLM(script), corpus=_grounding_corpus(),
        retriever=_StubRetriever({}), plan=plan, state=state,
        allowed_blank_ids=["t"], system_prompt="SYS", question="Q",
        max_turns=3,  # hard_type_filter defaults False
    )
    r.solve()
    assert state["t"].value == "Orhan Pamuk"   # off → committed as-is


def test_gsw_cascade_diag_zero_when_flag_off():
    """Flag OFF (default) -> the cascade fires as today, no blocks are
    recorded, and ``gsw_cascade_diag`` is present but all-zero. The final
    answer is the concatenated value (cascade unaffected)."""
    adapter = _mk_cascade_block_adapter({"gsw_block_unverified_cascade": False})
    traj = adapter.run_question("Concat alice's a and b?", question_id="qcasc_off")

    diag = traj.extra["gsw_cascade_diag"]
    assert diag["cascade_blocked_count"] == 0
    assert diag["partial_cascade_blocked"] == 0
    assert diag["unverified_cascade_blocked"] == 0
    assert diag["blocked_input_blank_ids"] == []
    assert diag["blocked_and_unresolved"] is False
    # Cascade fired: target resolved from concat(alpha, beta).
    assert diag["target_resolved"] is True
    assert "alpha" in traj.final_answer and "beta" in traj.final_answer


# ---------------------------------------------------------------------------
# Phase 4 Task 8 (Part A) — combined behavior-neutrality test
# ---------------------------------------------------------------------------


def test_phase4_logging_non_destructive():
    """Coverage log_only (default) + cascade flag off: Phase 4 is pure logging —
    final answer + committed values are byte-identical to a run with the Phase-4
    extras absent-by-effect (log_only writes only traj.extra).

    Non-vacuous: both runs use the same fixture, and the Phase-4 extras ARE
    present on the explicit run (gsw_coverage + gsw_cascade_diag in traj.extra)
    yet must not alter the final answer or committed blank values.
    """
    base = _run_fixture_orchestrator({})
    explicit = _run_fixture_orchestrator({
        "gsw_coverage_policy": "log_only",
        "gsw_block_unverified_cascade": False,
    })

    # Core contract: Phase-4 logging must not alter results.
    assert explicit.final_answer == base.final_answer
    assert _committed_values(explicit) == _committed_values(base)

    # Non-vacuity: the Phase-4 extras are written (not silently skipped).
    assert "gsw_coverage" in explicit.extra
    assert "gsw_cascade_diag" in explicit.extra
