"""Investigation-memory dispatch: prior-investigation dossier and scoped researcher budgets.

Split from the research repo's test_planner_orchestrator.py."""

from __future__ import annotations

from tests.ours.orchestrator._helpers import *  # noqa: F401,F403


def test_dispatch_dossier_wiring_renders_main_thread_and_passes_prior(monkeypatch):
    plan_dict = _one_level_plan()
    plan = GSWPlan.model_validate(plan_dict)
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    chunk = _StubChunk(
        chunk_id="chunk_pf_0",
        title="Pink Floyd",
        text="Pink Floyd released The Dark Side of the Moon in 1973.",
    )

    adapter = OursGSWPlannerOrchestratorV1Adapter(
        _routing_ctx({
            "gsw_investigation_memory": True,
            "gsw_memory_dossier": True,
            "corpus": _StubCorpus({"chunk_pf_0": chunk}),
        })
    )
    assert adapter.gsw_memory_dossier is True
    _seed_open_dossier_memory(adapter, plan_dict)

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(gmod, "ReActResearcher", _mk_researcher_spy(calls))

    adapter._dispatch_subplan_parallel(
        plan=plan, plan_dict=plan_dict, state=state,
        order=["t"], levels={"t": 0}, question="q?",
        blank_ids=["t"], hints="", level_tag=0, qgsw_state=None,
    )

    dossier = calls[0]["prior_investigation"]
    assert dossier is not None
    assert "Prior investigation of this blank" in dossier
    assert "Wish You Were Here" in dossier
    assert adapter._dossier_log == [
        {"blank": "t", "round": 2, "chars": len(dossier)},
    ]


def test_dispatch_dossier_flag_off_never_passes_prior_or_logs(monkeypatch):
    plan_dict = _one_level_plan()
    plan = GSWPlan.model_validate(plan_dict)
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    chunk = _StubChunk(
        chunk_id="chunk_pf_0",
        title="Pink Floyd",
        text="Pink Floyd released The Dark Side of the Moon in 1973.",
    )

    adapter = OursGSWPlannerOrchestratorV1Adapter(
        _routing_ctx({
            "gsw_investigation_memory": True,
            "corpus": _StubCorpus({"chunk_pf_0": chunk}),
        })
    )
    assert adapter.gsw_memory_dossier is False
    _seed_open_dossier_memory(adapter, plan_dict)

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(gmod, "ReActResearcher", _mk_researcher_spy(calls))

    adapter._dispatch_subplan_parallel(
        plan=plan, plan_dict=plan_dict, state=state,
        order=["t"], levels={"t": 0}, question="q?",
        blank_ids=["t"], hints="", level_tag=0, qgsw_state=None,
    )

    assert calls[0]["prior_investigation"] is None
    assert adapter._dossier_log == []


def test_memory_scoped_dispatch_flag_off_uses_full_researcher_budget(monkeypatch):
    from panini_dr.adapters.ours._investigation_memory import (
        InvestigationMemory,
    )

    plan_dict = _one_level_plan()
    plan = GSWPlan.model_validate(plan_dict)
    state = {"t": BlankResult(blank_id="t", status="unknown")}

    adapter = OursGSWPlannerOrchestratorV1Adapter(
        _routing_ctx({"gsw_investigation_memory": True})
    )
    assert adapter.gsw_memory_scoped_dispatch is False
    im = InvestigationMemory(plan_dict)
    im.mem.dry_hit_at["t"] = 2
    adapter._invmem = im
    adapter._invmem_round = 0

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(gmod, "ReActResearcher", _mk_researcher_spy(calls))

    adapter._dispatch_subplan_parallel(
        plan=plan, plan_dict=plan_dict, state=state,
        order=["t"], levels={"t": 0}, question="q?",
        blank_ids=["t"], hints="", level_tag=0, qgsw_state=None,
    )

    assert calls[0]["max_turns"] == adapter.level_max_turns
    assert adapter._scoped_log == []


def test_memory_scoped_dispatch_exhausted_blank_uses_small_researcher_budget(monkeypatch):
    from panini_dr.adapters.ours._investigation_memory import (
        InvestigationMemory,
    )

    plan_dict = _one_level_plan()
    plan = GSWPlan.model_validate(plan_dict)
    state = {"t": BlankResult(blank_id="t", status="unknown")}

    adapter = OursGSWPlannerOrchestratorV1Adapter(
        _routing_ctx({
            "gsw_investigation_memory": True,
            "gsw_memory_scoped_dispatch": True,
        })
    )
    assert adapter.gsw_memory_scoped_dispatch is True
    im = InvestigationMemory(plan_dict)
    im.mem.dry_hit_at["t"] = 2
    adapter._invmem = im
    adapter._invmem_round = 0

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(gmod, "ReActResearcher", _mk_researcher_spy(calls))

    adapter._dispatch_subplan_parallel(
        plan=plan, plan_dict=plan_dict, state=state,
        order=["t"], levels={"t": 0}, question="q?",
        blank_ids=["t"], hints="", level_tag=0, qgsw_state=None,
    )

    assert calls[0]["max_turns"] == 3
    assert adapter._scoped_log == [{"blank": "t", "budget": 3}]


def test_run_question_traj_extra_dossier_present_only_when_flag_on():
    plan_dict = _one_level_plan()
    chunk = _StubChunk(
        chunk_id="chunk_pf_0",
        title="Pink Floyd",
        text="Pink Floyd released The Dark Side of the Moon in 1973.",
    )
    corpus = _StubCorpus({"chunk_pf_0": chunk})
    retriever = _StubRetriever({"pink": [chunk]})

    routes_on = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o3", "submit_answer",
                                         {"answer": "Dark Side of the Moon"})]),
        ],
        "**Level 1** — resolve exactly these blanks": [
            _StubResponse(tool_calls=[_tc("r1", "search", {
                "query": "pink floyd released album",
                "top_k": 1,
            })]),
        ],
        "**Level 2** — resolve exactly these blanks": [
            _StubResponse(tool_calls=[_tc("r2", "update_blank", {
                "blank_id": "t",
                "value": "Dark Side of the Moon",
                "evidence_chunk_ids": ["chunk_pf_0"],
            })]),
        ],
    }
    adapter_on = _mk_llm_adapter(
        _RoutedLLM(routes_on), plan_dict=plan_dict, corpus=corpus,
        retriever=retriever,
        extra={"gsw_investigation_memory": True, "gsw_memory_dossier": True},
    )
    traj_on = adapter_on.run_question("Q", question_id="q_dossier_on")
    assert traj_on.final_answer == "Dark Side of the Moon"
    assert traj_on.extra["dossier"][0]["blank"] == "t"
    assert traj_on.extra["dossier"][0]["round"] == 2
    assert traj_on.extra["dossier"][0]["chars"] > 0

    routes_off = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o3", "submit_answer",
                                         {"answer": "Dark Side of the Moon"})]),
        ],
        "**Level 1** — resolve exactly these blanks": [
            _StubResponse(tool_calls=[_tc("r1", "search", {
                "query": "pink floyd released album",
                "top_k": 1,
            })]),
        ],
        "**Level 2** — resolve exactly these blanks": [
            _StubResponse(tool_calls=[_tc("r2", "update_blank", {
                "blank_id": "t",
                "value": "Dark Side of the Moon",
                "evidence_chunk_ids": ["chunk_pf_0"],
            })]),
        ],
    }
    adapter_off = _mk_llm_adapter(
        _RoutedLLM(routes_off), plan_dict=plan_dict, corpus=corpus,
        retriever=retriever,
        extra={"gsw_investigation_memory": True},
    )
    traj_off = adapter_off.run_question("Q", question_id="q_dossier_off")
    assert traj_off.final_answer == "Dark Side of the Moon"
    assert "dossier" not in traj_off.extra
