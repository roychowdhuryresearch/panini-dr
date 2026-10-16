"""Unit tests for the Phase-1 GSW evidence state (state + logging only)."""

from __future__ import annotations

from panini_dr.adapters.ours._gsw_state import (
    BlankState,
    EvidenceEdge,
    QueryGSWState,
)


def test_query_state_resolved_values():
    state = QueryGSWState(question="Q")
    state.blanks["b1"] = BlankState(
        blank_id="b1", subject="Pink Floyd", relation="released",
        value_type="entity", status="resolved", value="Animals",
    )
    state.blanks["b2"] = BlankState(
        blank_id="b2", subject="", relation="", value_type="date",
        status="pending",
    )
    assert state.resolved_values() == {"b1": "Animals"}


def test_query_state_to_dict_round_trips_blanks_and_edges():
    state = QueryGSWState(question="Q", final_answer="Animals")
    state.blanks["b1"] = BlankState(
        blank_id="b1", subject="Pink Floyd", relation="released",
        value_type="entity", is_target=True, direction="forward",
        status="resolved", value="Animals", evidence_chunk_ids=["c1"],
    )
    state.evidence_edges["e1"] = EvidenceEdge(
        edge_id="e1", subject="Pink Floyd", relation="released",
        object="Animals", object_type="entity", doc_id="c1", blank_id="b1",
        status="accepted",
    )
    d = state.to_dict()
    assert d["question"] == "Q"
    assert d["final_answer"] == "Animals"
    assert d["blanks"]["b1"]["value"] == "Animals"
    assert d["blanks"]["b1"]["is_target"] is True
    assert d["blanks"]["b1"]["direction"] == "forward"
    assert d["blanks"]["b1"]["evidence_chunk_ids"] == ["c1"]
    assert d["evidence_edges"]["e1"]["object"] == "Animals"
    assert d["evidence_edges"]["e1"]["status"] == "accepted"
    assert d["evidence_edges"]["e1"]["blank_id"] == "b1"


from panini_dr.adapters.ours._gsw_state import blank_projection, blank_projection_edges
from panini_dr.adapters.ours._planner_exec import GSWPlan


def _projection_plan() -> GSWPlan:
    # b_obj is the OBJECT of vp1 (forward): Pink Floyd --released--> b_obj
    # b_subj is the SUBJECT of vp2 (reverse): b_subj --born_in--> NYC
    # b_derived is referenced only by a constraint (none).
    return GSWPlan.model_validate({
        "entities": [
            {"id": "e_band", "kind": "filled", "name": "Pink Floyd"},
            {"id": "e_city", "kind": "filled", "name": "NYC"},
            {"id": "b_obj", "kind": "blank", "value_type": "entity"},
            {"id": "b_subj", "kind": "blank", "value_type": "entity"},
            {"id": "b_a", "kind": "blank", "value_type": "number"},
            {"id": "b_b", "kind": "blank", "value_type": "number"},
            {"id": "b_derived", "kind": "blank", "value_type": "number",
             "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "released", "subject_id": "e_band",
             "object_id": "b_obj"},
            {"id": "vp2", "phrase": "born_in", "subject_id": "b_subj",
             "object_id": "e_city"},
            {"id": "vp3", "phrase": "has_a", "subject_id": "e_band",
             "object_id": "b_a"},
            {"id": "vp4", "phrase": "has_b", "subject_id": "e_band",
             "object_id": "b_b"},
        ],
        "constraints": [
            {"id": "c1", "kind": "derived", "op": "diff",
             "args_blanks": ["b_a", "b_b"], "output_blank_id": "b_derived"},
        ],
    })


def test_blank_projection_forward_object():
    plan = _projection_plan()
    assert blank_projection(plan, "b_obj") == ("Pink Floyd", "released", "forward")


def test_blank_projection_reverse_subject():
    plan = _projection_plan()
    assert blank_projection(plan, "b_subj") == ("NYC", "born_in", "reverse")


def test_blank_projection_constraint_only_is_none():
    plan = _projection_plan()
    assert blank_projection(plan, "b_derived") == ("", "", "none")


def test_blank_projection_edges_preserves_all_incident_vps():
    from panini_dr.adapters.ours._gsw_state import blank_projection_edges
    plan = _projection_plan()
    fwd = blank_projection_edges(plan, "b_obj")
    assert len(fwd) == 1
    assert fwd[0]["blank_role"] == "object"
    assert fwd[0]["other_label"] == "Pink Floyd"
    assert fwd[0]["phrase"] == "released"
    assert fwd[0]["direction"] == "forward"
    # Constraint-only blank has no incident VP edges.
    assert blank_projection_edges(plan, "b_derived") == []


from panini_dr.adapters.ours._gsw_state import initialize_query_gsw_state
from panini_dr.adapters.ours._planner_exec import build_dependency_graph


def test_initialize_state_one_blank_per_plan_blank():
    plan = _projection_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("My question", plan, deps)

    assert state.question == "My question"
    # One BlankState per blank entity; no filled entities.
    assert set(state.blanks) == {"b_obj", "b_subj", "b_a", "b_b", "b_derived"}

    b_obj = state.blanks["b_obj"]
    assert b_obj.subject == "Pink Floyd"
    assert b_obj.relation == "released"
    assert b_obj.direction == "forward"
    assert b_obj.value_type == "entity"
    assert b_obj.is_target is False
    assert b_obj.status == "pending"

    # Target flag is carried through.
    assert state.blanks["b_derived"].is_target is True

    # Deps come from the dependency graph: the derived output depends on
    # its two input blanks.
    assert set(state.blanks["b_derived"].deps) == {"b_a", "b_b"}

    # No edges and no answer yet.
    assert state.evidence_edges == {}
    assert state.final_answer is None


from panini_dr.adapters.ours._gsw_state import mirror_results_into_state
from panini_dr.adapters.ours._planner_exec import BlankResult


def test_mirror_results_maps_status_value_and_evidence():
    plan = _projection_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)

    results = {
        "b_obj": BlankResult(
            blank_id="b_obj", value="Animals", status="resolved",
            evidence_chunk_ids=["c1", "c2"],
        ),
        "b_subj": BlankResult(blank_id="b_subj", status="unknown"),
        "b_a": BlankResult(
            blank_id="b_a", status="error", error="boom",
        ),
    }
    mirror_results_into_state(state, results, final_answer="Animals")

    b_obj = state.blanks["b_obj"]
    assert b_obj.status == "resolved"
    assert b_obj.value == "Animals"
    assert b_obj.evidence_chunk_ids == ["c1", "c2"]

    # "unknown" maps to "unresolved"; "error" maps to "failed".
    assert state.blanks["b_subj"].status == "unresolved"
    assert state.blanks["b_a"].status == "failed"
    assert state.blanks["b_a"].last_error == "boom"

    assert state.final_answer == "Animals"
    assert state.resolved_values() == {"b_obj": "Animals"}


def test_mirror_ignores_unknown_blank_ids():
    plan = _projection_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)
    # A result for a blank id not in the plan must be skipped, not crash.
    mirror_results_into_state(
        state,
        {"ghost": BlankResult(blank_id="ghost", value="x", status="resolved")},
        final_answer=None,
    )
    assert "ghost" not in state.blanks
    assert state.final_answer is None


def test_blank_projection_edges_multiple_vps():
    # b_person is the subject of TWO verb-phrases — both must be returned.
    plan = GSWPlan.model_validate({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "NYC"},
            {"id": "e2", "kind": "filled", "name": "Catwoman"},
            {"id": "b_person", "kind": "blank", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "born_in", "subject_id": "b_person", "object_id": "e1"},
            {"id": "vp2", "phrase": "created", "subject_id": "b_person", "object_id": "e2"},
        ],
        "constraints": [],
    })
    edges = blank_projection_edges(plan, "b_person")
    assert len(edges) == 2
    assert {e["phrase"] for e in edges} == {"born_in", "created"}
    assert all(e["blank_role"] == "subject" and e["direction"] == "reverse" for e in edges)


# ---------------------------------------------------------------------------
# Stub corpus for the grounding tests below
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class _Chunk:
    chunk_id: str
    title: str
    text: str


class _Corpus:
    def __init__(self, chunks):
        self._by_id = {c.chunk_id: c for c in chunks}
        self._by_title = {c.title: c.text for c in chunks}

    def get_chunk(self, chunk_id):
        return self._by_id.get(chunk_id)

    def article_text(self, title):
        return self._by_title.get(title, "")


# ---------------------------------------------------------------------------
# Phase 2 — add_evidence_edge linkage + grounding metrics
# ---------------------------------------------------------------------------

from panini_dr.adapters.ours._gsw_state import evidence_metrics


def test_add_evidence_edge_links_accepted_to_blank():
    plan = _projection_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)

    accepted = EvidenceEdge(
        edge_id="b_obj:e0", subject="Pink Floyd", relation="released",
        object="Animals", object_type="entity", doc_id="c1", blank_id="b_obj",
        status="accepted",
    )
    low = EvidenceEdge(
        edge_id="b_subj:e0", subject="x", relation="born_in", object="NYC",
        object_type="entity", doc_id="c2", blank_id="b_subj",
        status="low_confidence",
    )
    state.add_evidence_edge(accepted)
    state.add_evidence_edge(low)

    # Both edges stored.
    assert set(state.evidence_edges) == {"b_obj:e0", "b_subj:e0"}
    # Only the accepted edge links onto its blank.
    assert state.blanks["b_obj"].accepted_edge_ids == ["b_obj:e0"]
    assert state.blanks["b_subj"].accepted_edge_ids == []


def test_evidence_metrics_counts_grounded():
    plan = _projection_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)
    state.add_evidence_edge(EvidenceEdge(
        edge_id="e0", subject="a", relation="r", object="v0",
        object_type="entity", doc_id="c1", blank_id="b_obj", status="accepted",
    ))
    state.add_evidence_edge(EvidenceEdge(
        edge_id="e1", subject="a", relation="r", object="v1",
        object_type="entity", doc_id="c2", blank_id="b_subj",
        status="low_confidence",
    ))
    m = evidence_metrics(state)
    assert m == {
        "n_committed": 2,
        "n_grounded": 1,
        "n_ungrounded": 1,
        "n_missing_dependency": 0,
        "pct_grounded": 0.5,
    }


# --- Phase 2.5: Stage B graph verification ---

from panini_dr.adapters.ours._gsw_state import (
    has_real_citation,
    verify_graph,
)


def _multihop_plan():
    return GSWPlan.model_validate({
        "entities": [
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True},
            {"id": "b_group", "kind": "blank", "value_type": "entity"},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "co_founded", "subject_id": "t", "object_id": "b_group"},
        ],
        "constraints": [],
    })


def _elev_plan():
    return GSWPlan.model_validate({
        "entities": [
            {"id": "peak", "kind": "filled", "name": "Eggstock"},
            {"id": "b_elev", "kind": "blank", "value_type": "number", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_elevation", "subject_id": "peak", "object_id": "b_elev"},
        ],
        "constraints": [],
    })


def _resolve(state, bid, value, chunks):
    state.blanks[bid].value = value
    state.blanks[bid].status = "resolved"
    state.blanks[bid].evidence_chunk_ids = list(chunks)


def test_has_real_citation():
    corpus = _Corpus([_Chunk("c1", "T", "body")])
    assert has_real_citation(corpus, ["c1"]) is True
    assert has_real_citation(corpus, ["insufficient_evidence"]) is False
    assert has_real_citation(corpus, ["nonexistent"]) is False
    assert has_real_citation(corpus, []) is False


def test_verify_graph_unresolved_endpoint_is_missing_dependency_not_ungrounded():
    plan = _multihop_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)
    _resolve(state, "t", "Yvon Chouinard", ["c1"])
    verify_graph(plan, state, _Corpus([]))
    (edge,) = state.evidence_edges.values()
    assert edge.status == "missing_dependency"
    assert edge.status != "low_confidence"
    assert edge.raw_payload.get("needs_slots") == ["b_group"]


def test_verify_graph_reevaluates_missing_dependency_on_rerun():
    plan = _multihop_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)
    corpus = _Corpus([_Chunk("c1", "Yvon Chouinard",
                             "Yvon Chouinard co-founded One Percent for the Planet.")])
    _resolve(state, "t", "Yvon Chouinard", ["c1"])
    verify_graph(plan, state, corpus)
    assert list(state.evidence_edges.values())[0].status == "missing_dependency"
    _resolve(state, "b_group", "One Percent for the Planet", ["c1"])
    verify_graph(plan, state, corpus)
    edge = list(state.evidence_edges.values())[0]
    assert edge.status == "accepted"
    assert edge.object == "One Percent for the Planet"
    assert edge.subject == "Yvon Chouinard"


def test_verify_graph_substring_numeric_tolerant_with_filled_anchor():
    plan = _elev_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)
    _resolve(state, "b_elev", 3554.0, ["c1"])
    corpus = _Corpus([_Chunk("c1", "Eggstock", "Its elevation is 3554 m.")])
    verify_graph(plan, state, corpus)
    (edge,) = state.evidence_edges.values()
    assert edge.subject == "Eggstock"
    assert edge.relation == "has_elevation"
    assert edge.object == "3554.0"
    assert edge.status == "accepted"


def test_verify_graph_ungrounded_when_value_absent():
    plan = _elev_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)
    _resolve(state, "b_elev", 9999, ["c1"])
    corpus = _Corpus([_Chunk("c1", "Eggstock", "Its elevation is 3554 m.")])
    verify_graph(plan, state, corpus)
    (edge,) = state.evidence_edges.values()
    assert edge.status == "low_confidence"
    assert edge.rejection_reason == "relation_not_supported_by_cited_evidence"


def test_verify_graph_filled_entity_without_name_falls_back_to_id():
    plan = GSWPlan.model_validate({
        "entities": [
            {"id": "anchor", "kind": "filled"},  # no name → must fall back to id
            {"id": "b_x", "kind": "blank", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "rel", "subject_id": "anchor", "object_id": "b_x"},
        ],
        "constraints": [],
    })
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)
    _resolve(state, "b_x", "Val", ["c1"])
    corpus = _Corpus([_Chunk("c1", "T", "Val is here.")])
    verify_graph(plan, state, corpus)
    (edge,) = state.evidence_edges.values()
    assert edge.subject == "anchor"   # id fallback, NOT None / missing_dependency
    assert edge.status == "accepted"


# ---------------------------------------------------------------------------
# Phase 2 — LLM grounding judge for Stage B
# ---------------------------------------------------------------------------

from panini_dr.adapters.ours._gsw_state import judge_relation_llm


class _FakeLLM:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        class _R:
            pass
        r = _R()
        r.text = self._text
        return r


def test_judge_relation_llm_uses_chunk_text_and_parses():
    corpus = _Corpus([
        _Chunk("Goku__5", "Goku",
               "Goku gains the Super Saiyan transformation in the Frieza arc."),
    ])
    llm = _FakeLLM('{"grounded": true, "span": "Super Saiyan"}')
    judge = judge_relation_llm(llm, corpus)
    grounded, span = judge("Goku", "has_transformation", "Super Saiyan", ["Goku__5"])
    assert grounded is True
    sent = llm.calls[-1][-1]["content"]
    assert "Super Saiyan transformation in the Frieza arc" in sent
    assert "Goku has transformation Super Saiyan" in sent


def test_judge_relation_llm_no_evidence_is_ungrounded():
    judge = judge_relation_llm(_FakeLLM("{}"), _Corpus([]))
    grounded, _ = judge("Goku", "has_transformation", "Super Saiyan",
                        ["insufficient_evidence"])
    assert grounded is False


def test_judge_relation_llm_tier2_escalation_upgrades_from_article():
    class _SeqLLM:  # returns scripted responses in order
        def __init__(self, texts):
            self._texts = list(texts)
            self.calls = []
        def chat(self, messages, **kw):
            self.calls.append(messages)
            class _R: pass
            r = _R(); r.text = self._texts.pop(0); return r
    # chunk.text lacks "10"; the full article has it.
    class _C2:
        def get_chunk(self, cid):
            return _Chunk("Drogba__5", "Didier Drogba", "He signed for Chelsea in 2004.")
        def article_text(self, title):
            return "x" * 5000 + " He scored 10 league goals that debut season. " + "y" * 5000
    llm = _SeqLLM(['{"grounded": false}', '{"grounded": true, "span": "10 league goals"}'])
    judge = judge_relation_llm(llm, _C2())
    grounded, span = judge("Didier Drogba", "debut_goals", "10", ["Drogba__5"])
    assert grounded is True            # Tier 2 rescued it
    assert len(llm.calls) == 2         # Tier 1 then Tier 2


def test_judge_relation_llm_fails_open_on_llm_error():
    class _BoomLLM:
        def chat(self, messages, **kw):
            raise RuntimeError("backend down")
    corpus = _Corpus([_Chunk("c1", "T", "some text")])
    judge = judge_relation_llm(_BoomLLM(), corpus)
    grounded, span = judge("A", "rel", "B", ["c1"])
    assert grounded is True            # fail-open: never block on infra
    assert span == ""


def test_mirror_copies_resolved_by():
    plan = _projection_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)
    mirror_results_into_state(
        state,
        {"b_obj": BlankResult(blank_id="b_obj", value="Animals", status="resolved",
                              evidence_chunk_ids=["c1"], resolved_by="resolver")},
        final_answer=None,
    )
    assert state.blanks["b_obj"].resolved_by == "resolver"


def test_judge_relation_llm_sink_records_each_call():
    corpus = _Corpus([
        _Chunk("c1", "Goku", "Goku gains the Super Saiyan transformation."),
    ])
    sink: list = []
    judge = judge_relation_llm(
        _FakeLLM('{"grounded": true, "span": "x"}'), corpus, sink=sink
    )
    judge("Goku", "has_transformation", "Super Saiyan", ["c1"])
    assert len(sink) == 1
    rec = sink[0]
    assert rec["kind"] == "grounding"
    assert rec["claim"] == "Goku has transformation Super Saiyan"
    assert rec["grounded"] is True
    assert rec["tier"] == "tier1"
    assert rec["passages_shown"]   # the chunk text the judge actually saw


def test_blankstate_resolver_attempts_default_and_serialised():
    bs = BlankState(blank_id="b1", subject="", relation="", value_type="entity")
    assert bs.resolver_attempts == []
    bs.resolver_attempts.append({"query": "q", "candidate": "X",
                                 "verifier": {"unique": False}})
    d = bs.to_dict()
    assert d["resolver_attempts"][0]["candidate"] == "X"


def test_grounding_sink_captures_prompt_and_output():
    corpus = _Corpus([
        _Chunk("c1", "Goku", "Goku gains the Super Saiyan transformation."),
    ])

    class _FakeLLMWithReasoning:
        def __init__(self, text, reasoning_content="thinking..."):
            self._text = text
            self._reasoning_content = reasoning_content
            self.calls = []

        def chat(self, messages, **kw):
            self.calls.append(messages)
            class _R:
                pass
            r = _R()
            r.text = self._text
            r.reasoning_content = self._reasoning_content
            return r

    llm = _FakeLLMWithReasoning(
        '{"grounded": true, "span": "Super Saiyan"}',
        reasoning_content="The chunk text clearly mentions Super Saiyan transformation."
    )
    sink: list = []
    judge = judge_relation_llm(llm, corpus, sink=sink)
    judge("Goku", "has_transformation", "Super Saiyan", ["c1"])

    rec = [r for r in sink if r.get("kind") == "grounding"][0]
    assert "prompt" in rec and "raw_output" in rec and "reasoning" in rec
    assert rec["prompt"] != ""        # the rendered judge prompt (post-LLM path)
    assert rec["raw_output"] != ""    # the LLM's raw text response
    assert "Super Saiyan" in rec["raw_output"]
    assert rec["reasoning"] != ""     # the reasoning_content from the LLM


def test_blankstate_verifier_status_default_and_serialised():
    bs = BlankState(blank_id="b1", subject="", relation="", value_type="entity")
    assert bs.verifier_status is None
    bs.verifier_status = "unverified_capped"
    assert bs.to_dict()["verifier_status"] == "unverified_capped"


def test_blankstate_carries_connection_evidence_and_serialises():
    bs = BlankState(blank_id="b1", subject="", relation="", value_type="entity")
    assert bs.connection_evidence == {}          # default empty -> off = byte-identical
    bs.connection_evidence = {"vp1": ["C1"], "vp2": ["C2", "C3"]}
    d = bs.to_dict()
    assert d["connection_evidence"] == {"vp1": ["C1"], "vp2": ["C2", "C3"]}
    assert d["connection_evidence"] is not bs.connection_evidence


from panini_dr.adapters.ours._gsw_state import collect_supporting_edges


def test_collect_supporting_edges_walks_target_and_deps():
    plan = _projection_plan()  # b_derived (target) depends on b_a, b_b
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)
    # Give b_a and b_b accepted edges; b_derived has its own (constraint) edge.
    for bid, eid in [("b_a", "ea"), ("b_b", "eb"), ("b_derived", "ed")]:
        state.evidence_edges[eid] = EvidenceEdge(
            edge_id=eid, subject="s", relation="r", object="o",
            object_type="number", doc_id="c", blank_id=bid, status="accepted",
        )
        state.blanks[bid].accepted_edge_ids.append(eid)

    edges = collect_supporting_edges(state, "b_derived")
    # Target's own edge plus its transitive dep edges, no duplicates.
    assert set(edges) == {"ed", "ea", "eb"}


def test_collect_supporting_edges_handles_missing_blank():
    plan = _projection_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)
    assert collect_supporting_edges(state, "nonexistent") == []


from panini_dr.adapters.ours._gsw_state import check_evidence_coverage


def _one_blank_target_plan():
    return GSWPlan.model_validate({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pink Floyd"},
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "released", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    })


def test_coverage_supported_when_target_has_accepted_edge():
    plan = _one_blank_target_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)
    state.blanks["t"].status = "resolved"
    state.blanks["t"].value = "Animals"
    state.evidence_edges["e0"] = EvidenceEdge(
        edge_id="e0", subject="Pink Floyd", relation="released", object="Animals",
        object_type="entity", doc_id="c1", blank_id="t", status="accepted",
    )
    state.blanks["t"].accepted_edge_ids.append("e0")

    rep = check_evidence_coverage(state, plan, "Animals")
    assert rep.final_answer_supported is True
    assert rep.unsupported_final_value is False
    assert rep.target_resolved is True


def test_coverage_flags_unsupported_when_no_edge():
    plan = _one_blank_target_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)
    state.blanks["t"].status = "resolved"
    state.blanks["t"].value = "Animals"   # resolved but NO accepted edge

    rep = check_evidence_coverage(state, plan, "Animals")
    assert rep.final_answer_supported is False
    assert rep.unsupported_final_value is True
    assert rep.resolved_without_edge == 1


def test_coverage_empty_answer_is_not_unsupported():
    plan = _one_blank_target_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)
    rep = check_evidence_coverage(state, plan, "")
    assert rep.unsupported_final_value is False
    assert rep.target_resolved is False


from panini_dr.adapters.ours._gsw_state import classify_failure


def test_classify_retrieval_miss_when_target_unresolved():
    plan = _one_blank_target_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)  # nothing resolved
    assert classify_failure(state, plan, "") == "retrieval_miss"


def test_classify_unsupported_final_value():
    plan = _one_blank_target_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)
    state.blanks["t"].status = "resolved"
    state.blanks["t"].value = "Animals"  # resolved, no edge
    assert classify_failure(state, plan, "Animals") == "unsupported_final_value"


def test_classify_ok_when_supported():
    plan = _one_blank_target_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)
    state.blanks["t"].status = "resolved"
    state.blanks["t"].value = "Animals"
    state.evidence_edges["e0"] = EvidenceEdge(
        edge_id="e0", subject="Pink Floyd", relation="released", object="Animals",
        object_type="entity", doc_id="c1", blank_id="t", status="accepted",
    )
    state.blanks["t"].accepted_edge_ids.append("e0")
    assert classify_failure(state, plan, "Animals") == "ok"


def test_classify_blank_wrong_value_with_gold():
    plan = _one_blank_target_plan()
    deps = build_dependency_graph(plan)
    state = initialize_query_gsw_state("Q", plan, deps)
    state.blanks["t"].status = "resolved"
    state.blanks["t"].value = "Animals"
    state.evidence_edges["e0"] = EvidenceEdge(
        edge_id="e0", subject="Pink Floyd", relation="released", object="Animals",
        object_type="entity", doc_id="c1", blank_id="t", status="accepted",
    )
    state.blanks["t"].accepted_edge_ids.append("e0")
    assert classify_failure(state, plan, "Animals", gold="The Wall") == "blank_wrong_value"


# ---------------------------------------------------------------------------
# Phase 4 — cascade_diagnostics pure helper
# ---------------------------------------------------------------------------


def _state_for_diag() -> QueryGSWState:
    """Build a QueryGSWState for cascade_diagnostics tests.

    Blanks:
      t  — target, unresolved, deps=[b]
      b  — resolved, no deps
      x  — resolved, no deps
      y  — resolved, no deps
    """
    state = QueryGSWState(question="Q")
    state.blanks["t"] = BlankState(
        blank_id="t", subject="", relation="", value_type="entity",
        is_target=True, deps=["b"], status="unresolved",
    )
    state.blanks["b"] = BlankState(
        blank_id="b", subject="", relation="", value_type="entity",
        status="resolved",
    )
    state.blanks["x"] = BlankState(
        blank_id="x", subject="", relation="", value_type="entity",
        status="resolved",
    )
    state.blanks["y"] = BlankState(
        blank_id="y", subject="", relation="", value_type="entity",
        status="resolved",
    )
    return state


def test_cascade_diagnostics_fields():
    from panini_dr.adapters.ours._gsw_state import cascade_diagnostics
    records = [
        {"output_blank_id": "t", "constraint_id": "c1", "blocked_input": "b", "kind": "unverified"},
        {"output_blank_id": "x", "constraint_id": "c2", "blocked_input": "y", "kind": "partial"},
    ]
    # state: target t depends on b (so b in t's dep closure); t unresolved; x resolved
    state = _state_for_diag()
    diag = cascade_diagnostics(records, state, target_id="t")
    assert diag["cascade_blocked_count"] == 2
    assert set(diag["blocked_input_blank_ids"]) == {"b", "y"}
    assert set(diag["blocked_output_blank_ids"]) == {"t", "x"}
    assert set(diag["blocked_constraint_ids"]) == {"c1", "c2"}
    assert diag["unverified_cascade_blocked"] == 1
    assert diag["partial_cascade_blocked"] == 1
    assert diag["target_depends_on_blocked_output"] is True    # t IS a blocked output
    assert diag["target_depends_on_blocked_input"] is True     # b (blocked input) in t's dep closure
    assert diag["target_resolved"] is False
    assert diag["blocked_and_unresolved"] is True
    assert diag["blocked_but_recovered"] is False


# --- 2026-09-04: plan-added blanks must reach the serialised state ------------
#
# MEASURED on the BCP/FRAMES/MoNaCo runs of record: every blank added by a plan
# revision (50 / 113 / 502 of them) was missing from ``gsw_state.blanks``,
# because the mirror skipped result ids the initial state never had. Their
# verifier flags were then dropped at the merge (``if _bs is not None``), and
# Stage-B grounding saw their relations as ``missing_dependency``. Given the
# FINAL plan, the mirror now creates the missing entries and re-syncs
# ``is_target`` (a ``change_target`` op moves the target to a new blank).


def _projection_plan_v2() -> GSWPlan:
    """The v1 plan after ``add_blank(b_added via vp5)`` + ``change_target``."""
    d = _projection_plan().model_dump()
    d["entities"].append({"id": "b_added", "kind": "blank", "value_type": "entity"})
    d["verb_phrases"].append({"id": "vp5", "phrase": "produced",
                              "subject_id": "e_band", "object_id": "b_added"})
    for e in d["entities"]:
        e["is_target"] = (e["id"] == "b_added")
    return GSWPlan.model_validate(d)


def test_mirror_creates_entries_for_blanks_added_by_a_plan_revision():
    plan_v1 = _projection_plan()
    state = initialize_query_gsw_state("Q", plan_v1, build_dependency_graph(plan_v1))
    assert "b_added" not in state.blanks
    plan_v2 = _projection_plan_v2()
    mirror_results_into_state(
        state,
        {"b_added": BlankResult(blank_id="b_added", value="Dark Side", status="resolved",
                                evidence_chunk_ids=["c9"], resolved_by="react")},
        final_answer="Dark Side",
        plan=plan_v2,
    )
    bs = state.blanks["b_added"]
    assert bs.status == "resolved" and bs.value == "Dark Side"
    assert bs.evidence_chunk_ids == ["c9"] and bs.resolved_by == "react"
    # projected from the FINAL plan exactly as initialize_query_gsw_state would
    assert (bs.subject, bs.relation, bs.direction) == blank_projection(plan_v2, "b_added")
    assert bs.projection_edges == blank_projection_edges(plan_v2, "b_added")
    assert bs.value_type == "entity"


def test_mirror_resyncs_is_target_from_the_final_plan():
    plan_v1 = _projection_plan()
    state = initialize_query_gsw_state("Q", plan_v1, build_dependency_graph(plan_v1))
    assert state.blanks["b_derived"].is_target is True
    mirror_results_into_state(
        state,
        {"b_added": BlankResult(blank_id="b_added", value="x", status="resolved")},
        final_answer=None,
        plan=_projection_plan_v2(),
    )
    assert state.blanks["b_added"].is_target is True
    assert state.blanks["b_derived"].is_target is False


def test_mirror_without_a_plan_still_skips_unknown_ids():
    # Backward-compatible: callers that cannot supply the final plan keep the
    # old behaviour (see test_mirror_ignores_unknown_blank_ids).
    plan = _projection_plan()
    state = initialize_query_gsw_state("Q", plan, build_dependency_graph(plan))
    mirror_results_into_state(
        state, {"ghost": BlankResult(blank_id="ghost", value="x", status="resolved")},
        final_answer=None)
    assert "ghost" not in state.blanks
