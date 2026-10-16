"""TypedBlankResolver pre-pass in LLM-orchestrator mode, verifier enforce hand-off, verifier mode resolution, dossier blocks.

Split from the research repo's test_planner_orchestrator.py."""

from __future__ import annotations

from tests.ours.orchestrator._helpers import *  # noqa: F401,F403


# ---------------------------------------------------------------------------
# Task 5 — TypedBlankResolver pre-pass in LLM-orchestrator mode + provenance
# ---------------------------------------------------------------------------


def test_resolver_prepass_llm_mode_resolves_and_grounds():
    """LLM-mode: resolver pre-pass fires inside _dispatch_subplan_parallel,
    resolves the dispatched blank before the researcher pool, records
    resolved_by='resolver', and verify_graph still grounds the relation
    edge post-merge. Provenance map is written to traj.extra.
    """
    # Use Pink Floyd / one_level_plan + grounding corpus so verify_graph
    # can ground the released-relation from the resolver's committed value.
    plan_dict = _one_level_plan()
    corpus = _grounding_corpus()  # chunk_1: "Pink Floyd released The Wall"
    chunk = corpus.get_chunk("chunk_1")
    retriever = _AlwaysHitRetriever(chunk)

    resolver_json = json.dumps({
        "value": "The Wall",
        "object_type": "entity",
        "source_span": "Pink Floyd released The Wall",
        "supported": True,
        "confidence": 0.95,
        "evidence_chunk_ids": ["chunk_1"],
    })

    routes = {
        # Orchestrator: dispatch t first, then submit after seeing it resolved.
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "submit_answer",
                                         {"answer": "The Wall"})]),
        ],
        # Resolver's system prompt uniquely starts with this string.
        "You resolve exactly ONE typed slot": [
            _StubResponse(text=resolver_json),
        ],
        # The verifier checks every resolver candidate.
        "You verify whether a proposed VALUE": [_StubResponse(text=_ACCEPT)],
        # No researcher route needed: gsw_react_fallback=False skips the pool.
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(
        llm,
        plan_dict=plan_dict,
        corpus=corpus,
        retriever=retriever,
        extra={
            "gsw_typed_blank_resolver": True,
            "gsw_react_fallback": False,
            "gsw_evidence_gate": "measure",
        },
    )
    traj = adapter.run_question(
        "What album did Pink Floyd release?", question_id="q_resolver_llm"
    )

    # 1. Correct final answer.
    assert traj.final_answer == "The Wall"
    assert traj.extra["stopped_reason"] == "finished"

    # 2. blank resolved_by="resolver" in gsw_state.
    gsw = traj.extra["gsw_state"]
    t_blank = gsw["blanks"]["t"]
    assert t_blank["status"] == "resolved"
    assert t_blank["value"] == "The Wall"
    assert t_blank["resolved_by"] == "resolver"

    # 3. verify_graph grounded the relation edge post-merge.
    edges = gsw["evidence_edges"]
    assert len(edges) >= 1, f"no evidence edges; edges={edges}"
    edge_values = list(edges.values())
    relation_edge = next(
        (e for e in edge_values if e.get("relation") == "released"), None
    )
    assert relation_edge is not None, (
        f"no 'released' edge in evidence_edges; got: {list(edges.keys())}"
    )
    assert relation_edge["object"] == "The Wall"
    assert relation_edge["status"] == "accepted"

    # 4. Provenance map written to traj.extra.
    prov = traj.extra.get("gsw_resolution_provenance")
    assert prov is not None, "traj.extra['gsw_resolution_provenance'] missing"
    assert prov["t"] == "resolver"

    # 5. No researcher was spawned (resolver-only mode).
    assert traj.extra.get("researcher_traces", []) == [], (
        f"researcher_traces should be empty; got {traj.extra['researcher_traces']}"
    )


def test_phase3full_flags_default_off():
    a = _mk_adapter(_ScriptedLLM([]), plan_dict=_one_level_plan())
    assert a.gsw_blank_qualifier is False


# ---------------------------------------------------------------------------
# Task 10 — verify + handoff in LLM-mode resolver pre-pass
# ---------------------------------------------------------------------------


def test_llm_prepass_verifier_reject_blocks_commit_and_handoff():
    plan_dict = _one_level_plan()
    corpus = _grounding_corpus()
    chunk = corpus.get_chunk("chunk_1")
    retriever = _AlwaysHitRetriever(chunk)
    resolver_json = json.dumps({
        "value": "Yvon Chouinard", "object_type": "entity", "source_span": "s",
        "supported": True, "confidence": 0.9, "evidence_chunk_ids": ["chunk_1"]})
    verifier_json = json.dumps({
        "satisfies_all": False, "unique": True,
        "missing_constraints": ["cofounder_of Netflix"], "deferred_constraints": [],
        "ambiguous_candidates": [], "reason": "not a Netflix cofounder"})
    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan", {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "submit_answer", {"answer": ""})]),
        ],
        "You resolve exactly ONE typed slot": [_StubResponse(text=resolver_json)],
        "You verify whether a proposed VALUE": [_StubResponse(text=verifier_json)],
    }
    adapter = _mk_llm_adapter(
        _RoutedLLM(routes), plan_dict=plan_dict, corpus=corpus, retriever=retriever,
        extra={"gsw_typed_blank_resolver": True,
               "gsw_react_fallback": False, "gsw_evidence_state": True})
    traj = adapter.run_question("q?", question_id="q_llm_verify")
    prov = traj.extra.get("gsw_resolution_provenance", {})
    assert prov.get("t") != "resolver"
    t_blank = traj.extra["gsw_state"]["blanks"]["t"]
    assert t_blank["resolver_attempts"]
    assert t_blank["resolver_attempts"][-1]["verifier"]["satisfies_all"] is False


def test_llm_prepass_resolver_failure_hands_off_to_react():
    plan_dict = _one_level_plan()
    corpus = _grounding_corpus()
    chunk = corpus.get_chunk("chunk_1")
    retriever = _AlwaysHitRetriever(chunk)
    resolver_unsupported = json.dumps({
        "value": None, "object_type": "entity", "source_span": "",
        "supported": False, "confidence": 0.0, "evidence_chunk_ids": ["chunk_1"]})
    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan", {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "submit_answer", {"answer": "The Wall"})]),
        ],
        "You resolve exactly ONE typed slot": [_StubResponse(text=resolver_unsupported)],
        "## Prior resolved values": [
            _StubResponse(tool_calls=[_tc("r1", "update_blank",
                {"blank_id": "t", "value": "The Wall", "evidence_chunk_ids": ["chunk_1"]})]),
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(
        llm, plan_dict=plan_dict, corpus=corpus, retriever=retriever,
        extra={"gsw_typed_blank_resolver": True, "gsw_react_fallback": True,
               "gsw_evidence_state": True})
    traj = adapter.run_question("q?", question_id="q_llm_fail_handoff")
    researcher_sys = [
        m.get("content") or "" for c in llm.calls for m in c["messages"]
        if m.get("role") == "system" and "## Resolver handoff" in (m.get("content") or "")
    ]
    assert researcher_sys, "no researcher prompt carried a handoff block"
    text = researcher_sys[0]
    assert "## Resolver handoff for t" in text
    assert "chunk_1" in text
    t_blank = traj.extra["gsw_state"]["blanks"]["t"]
    assert any(a.get("candidate") is None for a in t_blank["resolver_attempts"])


def test_react_read_cited_chunks_filters_placeholders():
    from panini_dr.adapters.ours.gsw_planner_orchestrator_v1 import ReActResearcher
    chunk = _StubChunk(chunk_id="c1", title="T", text="body")
    corpus = _StubCorpus({"c1": chunk})
    r = ReActResearcher(llm=_ScriptedLLM([]), corpus=corpus, retriever=_StubRetriever({}),
                        plan=GSWPlan.model_validate(_one_level_plan()),
                        state={}, allowed_blank_ids=["t"], system_prompt="", question="q")
    got = r._read_cited_chunks(["c1", "insufficient_evidence", "missing__9"])
    assert [c["chunk_id"] for c in got] == ["c1"]
    assert got[0]["text"] == "body"


def test_fresh_researcher_appends_dossier_block():
    r = _make_min_researcher(prior_investigation="OPEN — verify vp2 ...")
    msgs = r._build_initial_messages()
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert r.question in user and "## Prior investigation" in user and "verify vp2" in user


def test_fresh_researcher_without_dossier_byte_identical():
    r = _make_min_researcher(prior_investigation=None)
    msgs = r._build_initial_messages()
    assert msgs == [
        {"role": "system", "content": r.system_prompt},
        {"role": "user", "content": r.question},
    ]


def test_blankresult_reliability_fields_default_none():
    from panini_dr.adapters.ours._planner_exec import BlankResult
    r = BlankResult(blank_id="t")
    assert r.verifier_status is None
    assert r.completeness_status is None


def test_block_unverified_cascade_flag_defaults_false():
    # Orchestrator reads gsw_block_unverified_cascade from extra; default off.
    adapter = _mk_adapter(_ScriptedLLM([]), plan_dict=_one_level_plan())
    assert adapter.block_unverified_cascade is False


def test_block_unverified_cascade_off_equals_today():
    """CONTRACT (Plan 2, Task 6): with ``gsw_block_unverified_cascade``
    absent (today) vs explicitly False, the orchestrator must produce a
    BYTE-IDENTICAL final answer + committed blank values.

    The behavior change introduced by Plan 2 must be isolated to the flag
    being ON. If this fails, a default leaked somewhere (a cascade call
    site not defaulting to False) -- a real production bug, not a test bug.
    """
    base = _run_fixture_orchestrator({})  # flag absent -> today
    off = _run_fixture_orchestrator({"gsw_block_unverified_cascade": False})

    assert off.final_answer == base.final_answer
    assert _committed_values(off) == _committed_values(base)


def test_coverage_and_bucket_logged_in_measure():
    adapter = _mk_adapter(
        _orchestrated_llm([_StubResponse(tool_calls=[_tc("c1", "update_blank", {
            "blank_id": "t", "value": "The Wall",
            "evidence_chunk_ids": ["chunk_1"],
        })])]),
        plan_dict=_one_level_plan(), corpus=_grounding_corpus(),
        extra={"gsw_evidence_gate": "measure"},
    )
    traj = adapter.run_question("What album?", question_id="q1")
    cov = traj.extra["gsw_coverage"]
    assert cov["final_answer_supported"] is True
    assert cov["unsupported_final_value"] is False
    assert traj.extra["gsw_failure_bucket"] == "ok"
    assert traj.extra["gsw_final_supporting_edge_ids"]  # non-empty
