"""ReActResearcher loop: slices, placeholders, empty-list terminals, budget warning, turn caps, checker sink.

Split from the research repo's test_planner_orchestrator.py."""

from __future__ import annotations

from tests.ours.orchestrator._helpers import *  # noqa: F401,F403


def test_checker_sink_is_thread_local_per_question():
    adapter = _mk_adapter(_ScriptedLLM([]), plan_dict=_one_level_plan())
    barrier = threading.Barrier(2)
    seen: list[tuple[str, list[dict[str, str]]]] = []
    lock = threading.Lock()

    def _run_question_thread(qid: str) -> None:
        adapter._checker_sink = []
        adapter._active_question_id = qid
        adapter._checker_sink.append({"qid": qid})
        barrier.wait(timeout=5)
        with lock:
            seen.append((adapter._active_question_id or "", list(adapter._checker_sink)))

    threads = [
        threading.Thread(target=_run_question_thread, args=("q1",)),
        threading.Thread(target=_run_question_thread, args=("q2",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert sorted(seen) == [
        ("q1", [{"qid": "q1"}]),
        ("q2", [{"qid": "q2"}]),
    ]


# ---------------------------------------------------------------------------
# Researcher-level tests
# ---------------------------------------------------------------------------


def test_researcher_rejects_update_outside_slice():
    plan = GSWPlan.model_validate(_two_level_plan())
    state = {
        "b_author": BlankResult(blank_id="b_author", status="unknown"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    # Researcher assigned only to `t`; try to update `b_author`.
    script = [
        _StubResponse(
            tool_calls=[_tc("c1", "update_blank", {
                "blank_id": "b_author",
                "value": "Alice",
                "evidence_chunk_ids": ["x"],
            })]
        ),
        _StubResponse(
            tool_calls=[_tc("c2", "update_blank", {
                "blank_id": "t",
                "value": "Pulitzer Prize",
                "evidence_chunk_ids": ["y"],
            })]
        ),
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_AnyIdCorpus(),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=5,
    )
    trace = r.solve()
    # b_author stayed unknown; t resolved.
    assert state["b_author"].status == "unknown"
    assert state["t"].status == "resolved"
    assert state["t"].value == "Pulitzer Prize"
    assert trace["stopped"] == "all_resolved"
    assert "t" in trace["resolved"]
    # The first tool call must have produced an error message.
    assert any(
        tc.name == "update_blank" and "not in your assigned slice" in (tc.error or "")
        for tc in trace["tool_calls"]
    )


def test_researcher_rejects_placeholder_values():
    plan = GSWPlan.model_validate(_one_level_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    script = [
        _StubResponse(
            tool_calls=[_tc("c1", "update_blank", {
                "blank_id": "t",
                "value": "t",
                "evidence_chunk_ids": ["chunk_1"],
            })]
        ),
        _StubResponse(
            tool_calls=[_tc("c2", "update_blank", {
                "blank_id": "t",
                "value": "Wish You Were Here",
                "evidence_chunk_ids": ["chunk_1"],
            })]
        ),
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_AnyIdCorpus(),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=5,
    )
    trace = r.solve()
    assert state["t"].value == "Wish You Were Here"
    assert trace["stopped"] == "all_resolved"
    assert any(
        tc.name == "update_blank" and "blank id" in (tc.error or "")
        for tc in trace["tool_calls"]
    )


def test_researcher_accepts_well_evidenced_empty_list_as_terminal():
    plan = GSWPlan.model_validate(_one_level_list_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    script = [
        _StubResponse(
            tool_calls=[_tc("c1", "update_blank", {
                "blank_id": "t",
                "value": [],
                "evidence_chunk_ids": ["chunk_costar_lost"],
            })]
        ),
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_AnyIdCorpus(),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=5,
    )
    trace = r.solve()
    assert state["t"].status == "resolved"
    assert state["t"].value == []
    assert trace["stopped"] == "all_resolved"
    assert not any(
        tc.name == "update_blank" and "empty list" in (tc.error or "")
        for tc in trace["tool_calls"]
    )


def test_researcher_still_rejects_empty_list_with_only_sentinel_evidence():
    """`insufficient_evidence` is a metadata marker, not real evidence --
    an empty list backed by ONLY that sentinel must still escalate."""
    plan = GSWPlan.model_validate(_one_level_list_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    script = [
        _StubResponse(
            tool_calls=[_tc("c1", "update_blank", {
                "blank_id": "t",
                "value": [],
                "evidence_chunk_ids": ["insufficient_evidence"],
            })]
        ),
        _StubResponse(
            tool_calls=[_tc("c2", "update_blank", {
                "blank_id": "t",
                "value": ["Lin Karaczynski"],
                "evidence_chunk_ids": ["chunk_1"],
            })]
        ),
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_AnyIdCorpus(),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=5,
    )
    trace = r.solve()
    assert state["t"].status == "resolved"
    assert state["t"].value == ["Lin Karaczynski"]
    assert any(
        tc.name == "update_blank" and "empty list" in (tc.error or "")
        for tc in trace["tool_calls"]
    )


def test_researcher_still_rejects_empty_list_with_no_evidence_at_all():
    plan = GSWPlan.model_validate(_one_level_list_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    script = [
        _StubResponse(
            tool_calls=[_tc("c1", "update_blank", {
                "blank_id": "t",
                "value": [],
                "evidence_chunk_ids": [],
            })]
        ),
        _StubResponse(
            tool_calls=[_tc("c2", "update_blank", {
                "blank_id": "t",
                "value": ["Lin Karaczynski"],
                "evidence_chunk_ids": ["chunk_1"],
            })]
        ),
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_AnyIdCorpus(),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=5,
    )
    trace = r.solve()
    assert state["t"].status == "resolved"
    assert state["t"].value == ["Lin Karaczynski"]
    assert any(
        tc.name == "update_blank" and "empty list" in (tc.error or "")
        for tc in trace["tool_calls"]
    )


def test_researcher_still_rejects_empty_list_when_claim_requires_nonempty():
    """A verification_claim that explicitly requires non-emptiness must
    still escalate an empty-list commit, even with real evidence."""
    plan = GSWPlan.model_validate(_one_level_list_plan(
        verification_claim="The list must not be empty; at least one costar must be present."
    ))
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    script = [
        _StubResponse(
            tool_calls=[_tc("c1", "update_blank", {
                "blank_id": "t",
                "value": [],
                "evidence_chunk_ids": ["chunk_costar_lost"],
            })]
        ),
        _StubResponse(
            tool_calls=[_tc("c2", "update_blank", {
                "blank_id": "t",
                "value": ["Lin Karaczynski"],
                "evidence_chunk_ids": ["chunk_1"],
            })]
        ),
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_AnyIdCorpus(),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=5,
    )
    trace = r.solve()
    assert state["t"].status == "resolved"
    assert state["t"].value == ["Lin Karaczynski"]
    assert any(
        tc.name == "update_blank" and "empty list" in (tc.error or "")
        for tc in trace["tool_calls"]
    )


def test_researcher_still_rejects_empty_list_for_non_list_blank():
    """The valid-terminal exception is scoped to value_type=='list'; a
    non-list blank must keep the original unconditional rejection."""
    plan = GSWPlan.model_validate(_one_level_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    script = [
        _StubResponse(
            tool_calls=[_tc("c1", "update_blank", {
                "blank_id": "t",
                "value": [],
                "evidence_chunk_ids": ["chunk_1"],
            })]
        ),
        _StubResponse(
            tool_calls=[_tc("c2", "update_blank", {
                "blank_id": "t",
                "value": "Wish You Were Here",
                "evidence_chunk_ids": ["chunk_1"],
            })]
        ),
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_AnyIdCorpus(),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=5,
    )
    trace = r.solve()
    assert state["t"].value == "Wish You Were Here"
    assert any(
        tc.name == "update_blank" and "empty list" in (tc.error or "")
        for tc in trace["tool_calls"]
    )


def test_researcher_adds_final_budget_warning():
    plan = GSWPlan.model_validate(_one_level_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    script = [
        _StubResponse(tool_calls=[_tc("c1", "search", {"query": "x"})]),
        _StubResponse(tool_calls=[_tc("c2", "search", {"query": "x"})]),
        _StubResponse(
            tool_calls=[_tc("c3", "update_blank", {
                "blank_id": "t",
                "value": "Wish You Were Here",
                "evidence_chunk_ids": ["chunk_1"],
            })]
        ),
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_AnyIdCorpus(),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=3,
    )
    trace = r.solve()
    assert trace["stopped"] == "all_resolved"
    third_call_messages = llm.calls[2]["messages"]
    assert any(
        "FINAL BUDGET WARNING" in (m.get("content") or "")
        for m in third_call_messages
        if m.get("role") == "user"
    )


def test_researcher_stops_when_all_resolved_before_max_turns():
    plan = GSWPlan.model_validate(_one_level_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    script = [
        _StubResponse(
            tool_calls=[_tc("c1", "update_blank", {
                "blank_id": "t",
                "value": "Wish You Were Here",
                "evidence_chunk_ids": ["chunk_1"],
            })]
        ),
        # Even if scripted further, researcher must stop after resolution.
        _StubResponse(text="extra, should not be reached", tool_calls=[]),
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_AnyIdCorpus(),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=5,
    )
    trace = r.solve()
    assert trace["stopped"] == "all_resolved"
    assert trace["turns"] == 1  # single LLM turn → resolved → stop
    assert state["t"].status == "resolved"


def test_researcher_records_exact_post_evidence_token_burn():
    plan = GSWPlan.model_validate(_one_level_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    chunk = _StubChunk("chunk_1", "Pink Floyd", "Wish You Were Here was released.")
    script = [
        _StubResponse(tool_calls=[_tc("c1", "search", {"query": "album"})]),
        _StubResponse(tool_calls=[_tc("c2", "search", {"query": "album"})]),
        _StubResponse(tool_calls=[_tc("c3", "search", {"query": "album"})]),
        _StubResponse(
            tool_calls=[
                _tc(
                    "c4",
                    "update_blank",
                    {
                        "blank_id": "t",
                        "value": "Wish You Were Here",
                        "evidence_chunk_ids": ["chunk_1"],
                    },
                )
            ]
        ),
    ]
    researcher = ReActResearcher(
        llm=_ScriptedLLM(script),
        corpus=_StubCorpus({"chunk_1": chunk}),
        retriever=_StubRetriever({"album": [chunk]}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=5,
    )

    trace = researcher.solve()

    assert trace["turn_usage"] == [
        {"turn": turn, "prompt_tokens": 10, "completion_tokens": 20}
        for turn in range(1, 5)
    ]
    assert trace["readiness"]["t"] == {
        "evidence_chunk_ids": ["chunk_1"],
        "evidence_ready_turn": 1,
        "post_ready_prompt_tokens": 30,
        "post_ready_actions": 3,
        "post_ready_searches": 2,
        "post_ready_zero_new_searches": 2,
    }


def test_researcher_hits_max_turns_without_resolving():
    plan = GSWPlan.model_validate(_one_level_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    # LLM only ever calls search — never commits.
    script = [
        _StubResponse(
            tool_calls=[_tc(f"c{i}", "search", {"query": "x"})]
        )
        for i in range(5)
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_AnyIdCorpus(),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=3,
    )
    trace = r.solve()
    assert trace["stopped"] == "max_turns"
    assert trace["turns"] == 3
    assert state["t"].status == "unknown"
    assert trace["unresolved"] == ["t"]


def test_researcher_get_state_flags_writable():
    plan = GSWPlan.model_validate(_two_level_plan())
    state = {
        "b_author": BlankResult(blank_id="b_author", value="Alice", status="resolved", evidence_chunk_ids=["x"]),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    script = [
        _StubResponse(tool_calls=[_tc("c1", "get_state", {})]),
        _StubResponse(tool_calls=[_tc("c2", "update_blank", {
            "blank_id": "t",
            "value": "Pulitzer Prize",
            "evidence_chunk_ids": ["y"],
        })]),
    ]
    llm = _ScriptedLLM(script)
    r = ReActResearcher(
        llm=llm,
        corpus=_AnyIdCorpus(),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=5,
    )
    trace = r.solve()
    # Find the get_state tool call's result.
    gs_call = next(tc for tc in trace["tool_calls"] if tc.name == "get_state")
    result = json.loads(gs_call.result_full)
    ids_writable = {b["blank_id"]: b["writable"] for b in result["blanks"]}
    assert ids_writable == {"b_author": False, "t": True}
