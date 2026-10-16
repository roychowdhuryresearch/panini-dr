"""Unit tests for typed blank resolution helpers (Phase 3-lite)."""

from __future__ import annotations

from panini_dr.adapters.ours._evidence_ranker import (
    object_matches_value_type,
)
from panini_dr.adapters.ours._planner_exec import GSWPlan


def _q492_plan():
    return GSWPlan.model_validate({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Netflix", "role": "subject"},
            {"id": "e4", "kind": "filled", "name": "environmental group",
             "category": True, "role": "scope-filter"},
            {"id": "b1", "kind": "blank", "value_type": "entity", "role": "bridge-entity"},
            {"id": "b_environmental_group", "kind": "blank", "value_type": "entity",
             "role": "bridge-entity"},
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
             "role": "target"},
        ],
        "verb_phrases": [
            {"id": "vp1", "subject_id": "b1", "phrase": "cofounder_of", "object_id": "e1"},
            {"id": "vp2", "subject_id": "b1", "phrase": "joined",
             "object_id": "b_environmental_group"},
            {"id": "vpq", "subject_id": "b_environmental_group", "phrase": "instance_of",
             "object_id": "e4"},
            {"id": "vp3", "subject_id": "t", "phrase": "co_founded",
             "object_id": "b_environmental_group"},
        ],
        "constraints": [],
    })


def test_object_matches_value_type_date():
    assert object_matches_value_type("2006", "date") is True
    assert object_matches_value_type("7 February 1812", "date") is True
    assert object_matches_value_type("1847-03-03", "date") is True
    assert object_matches_value_type("Orhan Pamuk", "date") is False
    assert object_matches_value_type("", "date") is False
    assert object_matches_value_type(None, "date") is False


def test_object_matches_value_type_number():
    assert object_matches_value_type("3554", "number") is True
    assert object_matches_value_type("10", "number") is True
    assert object_matches_value_type("1,234.5", "number") is True
    assert object_matches_value_type("five", "number") is False


def test_object_matches_value_type_bool():
    assert object_matches_value_type("true", "bool") is True
    assert object_matches_value_type("No", "bool") is True
    assert object_matches_value_type("maybe", "bool") is False


def test_object_matches_value_type_entity_is_permissive():
    # No cheap shape check for entity/text/attribute/list — always allow.
    assert object_matches_value_type("Orhan Pamuk", "entity") is True
    assert object_matches_value_type("2006", "entity") is True
    assert object_matches_value_type("anything", "text") is True
    assert object_matches_value_type("x", "unknown_type") is True


from panini_dr.adapters.ours._evidence_prompts import (
    RESOLVER_SYSTEM, RESOLVER_USER, parse_resolver_output,
)


def test_parse_resolver_output_valid():
    txt = ('{"value":"2006","object_type":"date","source_span":"...2006...",'
           '"supported":true,"confidence":0.9,"evidence_chunk_ids":["c1"]}')
    d = parse_resolver_output(txt)
    assert d["value"] == "2006" and d["supported"] is True
    assert d["evidence_chunk_ids"] == ["c1"] and d["confidence"] == 0.9


def test_parse_resolver_output_garbage_is_unsupported():
    d = parse_resolver_output("not json")
    assert d["supported"] is False and d["value"] is None
    assert d["evidence_chunk_ids"] == []


def test_resolver_user_template_formats():
    s = RESOLVER_USER.format(subject="Serbia", relation="independent_in",
                             value_type="date", question="Q", context="(none)",
                             chunks="[c1] body", unit_instruction="")
    assert "Serbia" in s and "date" in s and "[c1] body" in s
    assert isinstance(RESOLVER_SYSTEM, str) and "JSON" in RESOLVER_SYSTEM


# ---------------------------------------------------------------------------
# Task 2 — TypedBlankResolver (value + citations, NO edge building)
# ---------------------------------------------------------------------------

from panini_dr.adapters.ours._blank_resolver import (  # noqa: E402
    TypedBlankResolver, BlankResolutionResult,
)
from panini_dr.adapters.ours._planner_exec import GSWPlan as _GP  # noqa: E402


class _StubRetr:
    def __init__(self, hits):
        self._hits = hits
        self.queries: list[str] = []

    def search(self, q, top_k=5):
        self.queries.append(q)
        return self._hits


class _LLM:
    def __init__(self, text):
        self._t = text
        self.calls = []
    def chat(self, messages, **kw):
        self.calls.append({"messages": messages, **kw})
        r = type("R", (), {})(); r.text = self._t; return r


def _date_plan():
    return _GP.model_validate({
        "entities": [
            {"id": "e", "kind": "filled", "name": "Serbia"},
            {"id": "t", "kind": "blank", "value_type": "date", "is_target": True},
        ],
        "verb_phrases": [{"id": "vp1", "phrase": "independent_in", "subject_id": "e", "object_id": "t"}],
        "constraints": [],
    })


# _retrieve_chunks wraps each raw hit via getattr(h, "chunk", h) and reads
# .chunk_id / .title / .text from the result — so a simple namespace object
# with those attributes directly on it is the minimal correct stub.
def _hit(cid, text):
    h = type("H", (), {})()
    h.chunk_id = cid
    h.title = cid.split("__", 1)[0]
    h.text = text
    h.score = 1.0
    return h


def test_resolver_produces_value_and_citations_no_edges():
    plan = _date_plan()
    retr = _StubRetr([_hit("c1", "Serbia became independent in 2006.")])
    llm = _LLM('{"value":"2006","object_type":"date","source_span":"2006",'
               '"supported":true,"confidence":0.9,"evidence_chunk_ids":["c1"]}')
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=llm)
    res = r.resolve_blank(plan, "t", {}, "Q")
    assert res.status == "resolved"
    assert res.value == "2006"
    assert res.evidence_chunk_ids == ["c1"]
    assert not hasattr(res, "edges")   # NO edge field at all


def test_resolver_unsupported_stays_unresolved():
    plan = _date_plan()
    retr = _StubRetr([_hit("c1", "unrelated text")])
    llm = _LLM('{"value":null,"supported":false,"evidence_chunk_ids":[]}')
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=llm)
    assert r.resolve_blank(plan, "t", {}, "Q").status == "unresolved"


def test_resolver_type_mismatch_stays_unresolved():
    plan = _date_plan()
    retr = _StubRetr([_hit("c1", "text")])
    llm = _LLM('{"value":"Orhan Pamuk","object_type":"entity","supported":true,'
               '"confidence":0.8,"evidence_chunk_ids":["c1"]}')
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=llm, hard_type_filter=True)
    assert r.resolve_blank(plan, "t", {}, "Q").status == "unresolved"


def test_resolver_falls_back_to_chunk_ids_when_llm_omits_them():
    plan = _date_plan()
    retr = _StubRetr([_hit("c1", "Serbia became independent in 2006.")])
    llm = _LLM('{"value":"2006","object_type":"date","supported":true,'
               '"confidence":0.9,"evidence_chunk_ids":[]}')   # LLM gave no ids
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=llm)
    res = r.resolve_blank(plan, "t", {}, "Q")
    assert res.status == "resolved" and res.evidence_chunk_ids == ["c1"]  # fell back to retrieved id


def test_resolver_no_chunks_stays_unresolved():
    plan = _date_plan()
    r = TypedBlankResolver(retriever=_StubRetr([]), corpus=None, llm=_LLM("{}"))
    res = r.resolve_blank(plan, "t", {}, "Q")
    assert res.status == "unresolved"
    assert res.message == "no_chunks"


def test_resolver_contains_llm_error():
    class _BoomLLM:
        def chat(self, messages, **kw):
            raise RuntimeError("backend down")
    plan = _date_plan()
    r = TypedBlankResolver(retriever=_StubRetr([_hit("c1", "Serbia independent 2006.")]),
                           corpus=None, llm=_BoomLLM())
    res = r.resolve_blank(plan, "t", {}, "Q")
    assert res.status == "unresolved"            # contained, not raised
    assert res.message.startswith("llm_error")


def test_resolver_sink_records_each_call():
    plan = _date_plan()
    retr = _StubRetr([_hit("c1", "Serbia became independent in 2006.")])
    llm = _LLM('{"value":"2006","object_type":"date","supported":true,'
               '"confidence":0.9,"evidence_chunk_ids":["c1"]}')
    sink: list = []
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=llm, sink=sink)
    r.resolve_blank(plan, "t", {}, "Q")
    assert len(sink) == 1
    assert sink[0]["kind"] == "resolver"
    assert sink[0]["status"] == "resolved"
    assert sink[0]["value"] == "2006"
    assert "query" in sink[0] and sink[0]["query"]


def test_resolver_sink_records_retrieved_chunks():
    plan = _date_plan()
    retr = _StubRetr([
        _hit("c1", "Serbia became independent in 2006."),
        _hit("c2", "Unrelated chunk."),
    ])
    llm = _LLM('{"value":"2006","object_type":"date","supported":true,'
               '"confidence":0.9,"evidence_chunk_ids":["c1"]}')
    sink: list = []
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=llm, sink=sink)

    r.resolve_blank(plan, "t", {}, "Q")

    assert sink[0]["retrieved_chunks"] == [
        {
            "chunk_id": "c1",
            "title": "c1",
            "text": "Serbia became independent in 2006.",
        },
        {
            "chunk_id": "c2",
            "title": "c2",
            "text": "Unrelated chunk.",
        },
    ]


# ---------------------------------------------------------------------------
# Task 1 — normalize_relation_phrase (pure helper)
# ---------------------------------------------------------------------------

from panini_dr.adapters.ours._blank_resolver import normalize_relation_phrase


def test_normalize_relation_phrase_overrides_and_default():
    assert normalize_relation_phrase("cofounder_of") == "co-founder of"
    assert normalize_relation_phrase("instance_of") == ""        # dropped
    assert normalize_relation_phrase("has_occupation") == "occupation"
    # default: underscores -> spaces
    assert normalize_relation_phrase("joined") == "joined"
    assert normalize_relation_phrase("won_award_at") == "won award at"
    assert normalize_relation_phrase("") == ""


# ---------------------------------------------------------------------------
# Task 2 — build_resolver_query (direction-aware, partner-qualifier traversal)
# ---------------------------------------------------------------------------

from panini_dr.adapters.ours._blank_resolver import build_resolver_query  # noqa: E402


def test_build_query_b1_pulls_unresolved_partner_qualifier():
    plan = _q492_plan()
    q = build_resolver_query(plan, "b1", dep_values={})
    assert "co-founder of Netflix" in q
    assert "environmental group" in q            # pulled via partner instance_of
    assert "joined environmental group" in q
    assert "b_environmental_group" not in q       # never leak the raw blank id
    assert "b1" not in q


def test_build_query_substitutes_resolved_partner_value():
    plan = _q492_plan()
    q = build_resolver_query(plan, "b1",
                             dep_values={"b_environmental_group": "Sierra Club"})
    assert "joined Sierra Club" in q
    assert "environmental group" not in q


def test_build_query_omits_unqualified_unresolved_partner():
    plan = _q492_plan()
    plan.verb_phrases.append(
        type(plan.verb_phrases[0])(id="vpx", subject_id="t",
                                   phrase="mentored", object_id="b1"))
    q = build_resolver_query(plan, "t", dep_values={})
    assert "co-founded environmental group" in q
    assert "b1" not in q                          # bare unresolved partner omitted


def test_build_query_own_qualifier_renders_as_label():
    plan = _q492_plan()
    q = build_resolver_query(plan, "b_environmental_group", dep_values={})
    assert "environmental group" in q             # own instance_of edge -> bare label
    assert "instance_of" not in q


# ---------------------------------------------------------------------------
# Task 3 — resolve_blank result carries retrieved_chunks
# ---------------------------------------------------------------------------

def test_resolve_blank_result_carries_retrieved_chunks():
    class _Hit:
        def __init__(self): self.chunk_id, self.title, self.text = "C1", "T", "body"
    class _Retr:
        def search(self, q, top_k=5): return [_Hit()]
    class _LLM:
        def chat(self, **k):
            class R: text = ('{"value":"X","object_type":"entity",'
                             '"source_span":"s","supported":true,'
                             '"confidence":0.9,"evidence_chunk_ids":["C1"]}')
            return R()
    plan = _q492_plan()
    r = TypedBlankResolver(retriever=_Retr(), corpus=None, llm=_LLM(),
                           hard_type_filter=False)
    res = r.resolve_blank(plan, "b1", {}, "q")
    assert res.status == "resolved"
    assert res.retrieved_chunks and res.retrieved_chunks[0]["chunk_id"] == "C1"
    assert res.retrieved_chunks[0]["text"] == "body"


def test_resolver_sink_captures_prompt_output():
    class _Hit:
        def __init__(self): self.chunk_id, self.title, self.text = "C1", "T", "body"
    class _Retr:
        def search(self, q, top_k=5): return [_Hit()]
    class _LLM:
        def chat(self, **k):
            class R:
                text = ('{"value":"X","object_type":"entity","source_span":"s",'
                        '"supported":true,"confidence":0.9,"evidence_chunk_ids":["C1"]}')
                reasoning_content = "found X"
            return R()
    sink = []
    TypedBlankResolver(retriever=_Retr(), corpus=None, llm=_LLM(),
                       hard_type_filter=False, sink=sink).resolve_blank(
        _q492_plan(), "b1", {}, "q")
    rec = [r for r in sink if r["kind"] == "resolver"][0]
    assert rec["prompt"] and rec["raw_output"].startswith("{")
    assert rec["reasoning"] == "found X"


def _hint_plan_resolver():
    return _GP.model_validate({
        "entities": [
            {"id": "e", "kind": "filled", "name": "Democracy Index"},
            {"id": "b_country", "kind": "blank", "value_type": "entity",
             "retrieval_hint": "Which country ranked highest in the 2023 Democracy Index?"},
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
             "retrieval_hint": "What is the largest religion in {b_country} as of August 3, 2024?"},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "best_democracy_in", "subject_id": "b_country", "object_id": "e"},
            {"id": "vp2", "phrase": "largest_religion_in", "subject_id": "t", "object_id": "b_country"},
        ],
        "constraints": [],
    })


def test_resolver_uses_hint_query_when_enabled():
    retr = _StubRetr([])
    sink = []
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=_LLM("{}"), use_hints=True, sink=sink)
    r.resolve_blank(_hint_plan_resolver(), "b_country", {}, "orig question")
    assert retr.queries[0] == "Which country ranked highest in the 2023 Democracy Index?"
    assert sink[-1]["query_source"] == "hint"


def test_resolver_substitutes_dep_in_hint():
    retr = _StubRetr([])
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=_LLM("{}"), use_hints=True)
    r.resolve_blank(_hint_plan_resolver(), "t", {"b_country": "Norway"}, "orig question")
    assert retr.queries[0] == "What is the largest religion in Norway as of August 3, 2024?"


def test_resolver_unresolved_hint_does_not_search_or_fall_back():
    retr = _StubRetr([])
    sink = []
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=_LLM("{}"), use_hints=True, sink=sink)
    res = r.resolve_blank(_hint_plan_resolver(), "t", {}, "orig question")
    assert res.status == "unresolved"
    assert retr.queries == []
    assert sink[-1]["query_source"] == "unresolved_hint"


def test_resolver_edge_fallback_when_no_hint():
    plan = _GP.model_validate({
        "entities": [
            {"id": "e", "kind": "filled", "name": "Serbia"},
            {"id": "t", "kind": "blank", "value_type": "date", "is_target": True},
        ],
        "verb_phrases": [{"id": "vp1", "phrase": "independent_in", "subject_id": "e", "object_id": "t"}],
        "constraints": [],
    })
    retr = _StubRetr([])
    sink = []
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=_LLM("{}"), use_hints=True, sink=sink)
    r.resolve_blank(plan, "t", {}, "orig question")
    assert retr.queries and "{" not in retr.queries[0]
    assert sink[-1]["query_source"] == "edge"


def test_resolver_ignores_hint_when_disabled():
    retr = _StubRetr([])
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=_LLM("{}"), use_hints=False)
    r.resolve_blank(_hint_plan_resolver(), "b_country", {}, "orig question")
    assert retr.queries[0] != "Which country ranked highest in the 2023 Democracy Index?"


# ---------------------------------------------------------------------------
# Task 6 — resolver emits per-connection evidence map (connection_evidence)
# ---------------------------------------------------------------------------

def _conn_resolved_llm():
    # supported resolve over an entity-typed blank, citing chunk c1.
    return _LLM('{"value":"Reed Hastings","object_type":"entity","source_span":"s",'
                '"supported":true,"confidence":0.9,"evidence_chunk_ids":["c1"]}')


# ---------------------------------------------------------------------------
# Task 7 — resolver span-grounding auto-repair (connection_evidence)
# ---------------------------------------------------------------------------

def _span_resolver(*, connection_evidence, chunks, llm_payload, sink=None):
    return TypedBlankResolver(
        retriever=_StubRetr([_hit(cid, text) for cid, text in chunks]),
        corpus=None,
        llm=_LLM(llm_payload),
        hard_type_filter=False,
        connection_evidence=connection_evidence,
        sink=sink,
    )


# ---------------------------------------------------------------------------
# Task 8 — feedback-forward refutation history
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Task 9 — Hard rule 7 / q16 class: declared-unit propagation into the
# resolver call (Step 1) + deterministic dual-unit span reconciliation
# (Step 2). See _blank_resolver.reconcile_declared_unit_value for the exact
# conservative firing conditions.
# ---------------------------------------------------------------------------

from panini_dr.adapters.ours._blank_resolver import (  # noqa: E402
    reconcile_declared_unit_value,
)
from panini_dr.adapters.ours._evidence_prompts import RESOLVER_USER  # noqa: E402
from panini_dr.adapters.ours._planner_exec import (  # noqa: E402
    _format_chunks_for_prompt,
    _retrieve_chunks,
)


def _unit_hint_plan(
    hint: str = "What is the height of the tallest building in San Francisco, in meters?",
):
    """q16-shape: one numeric bridge blank whose retrieval_hint declares a
    required unit."""
    return _GP.model_validate({
        "entities": [
            {"id": "e_building", "kind": "filled",
             "name": "tallest building in San Francisco",
             "category": True, "role": "scope-filter"},
            {"id": "b_height", "kind": "blank", "value_type": "number",
             "role": "bridge-number", "retrieval_hint": hint},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_height", "subject_id": "e_building",
             "object_id": "b_height"},
        ],
        "constraints": [],
    })


# --- Step 1: unit_instruction reaches the resolver prompt -----------------

def test_resolver_prompt_carries_unit_instruction_when_hint_declares_unit():
    retr = _StubRetr([_hit("c1", "Tallest building height: 1,070 ft (326.1 m)")])
    llm = _LLM('{"value": 1070, "object_type": "number", '
               '"source_span": "Tallest building height: 1,070 ft (326.1 m)", '
               '"supported": true, "confidence": 1.0, "evidence_chunk_ids": ["c1"]}')
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=llm,
                           use_hints=True, hard_type_filter=False)
    r.resolve_blank(_unit_hint_plan(), "b_height", {}, "orig question")
    last_user = llm.calls[-1]["messages"][1]["content"]
    assert "REQUIRED UNIT: metres" in last_user
    assert "convert" in last_user.lower()


def test_resolver_prompt_byte_identical_when_no_unit_declared():
    """GUARD RAIL: for a blank with no declared unit, the resolver's actual
    LLM prompt must equal RESOLVER_USER rendered with unit_instruction=""
    -- i.e. byte-identical to before this feature existed."""
    plan = _date_plan()
    retr = _StubRetr([_hit("c1", "Serbia became independent in 2006.")])
    llm = _LLM('{"value":"2006","object_type":"date","source_span":"2006",'
               '"supported":true,"confidence":0.9,"evidence_chunk_ids":["c1"]}')
    sink: list = []
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=llm, sink=sink)
    r.resolve_blank(plan, "t", {}, "Q")

    rec = sink[-1]
    actual_prompt = rec["prompt"]
    expected_chunks_text = _format_chunks_for_prompt(_retrieve_chunks(retr, "x", top_k=5))
    expected_prompt = RESOLVER_USER.format(
        subject=rec["subject"] or "(unknown)",
        relation=rec["relation"] or "(unknown)",
        value_type=rec["value_type"],
        question="Q",
        context="(none)",
        chunks=expected_chunks_text,
        unit_instruction="",
    )
    assert actual_prompt == expected_prompt
    assert "REQUIRED UNIT" not in actual_prompt


def test_resolver_ignores_declared_unit_when_hints_disabled():
    """use_hints=False -> eq.source is never "hint" -> declared_unit is
    never computed, regardless of what the (unused) retrieval_hint says."""
    plan = _unit_hint_plan()
    retr = _StubRetr([_hit("c1", "Tallest building height: 1,070 ft (326.1 m)")])
    llm = _LLM('{"value": 1070, "object_type": "number", '
               '"source_span": "Tallest building height: 1,070 ft (326.1 m)", '
               '"supported": true, "confidence": 1.0, "evidence_chunk_ids": ["c1"]}')
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=llm,
                           use_hints=False, hard_type_filter=False)
    res = r.resolve_blank(plan, "b_height", {}, "Q")
    last_user = llm.calls[-1]["messages"][1]["content"]
    assert "REQUIRED UNIT" not in last_user
    assert res.value == 1070   # untouched -- no reconciliation without hints


def test_resolver_ignores_unit_hint_for_non_numeric_blank():
    """value_type != "number" -> declared_unit is never computed, even if
    the (non-numeric) blank's hint happens to contain a unit word."""
    plan = _GP.model_validate({
        "entities": [
            {"id": "e", "kind": "filled", "name": "X"},
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
             "retrieval_hint": "What entity is associated with X, in meters?"},
        ],
        "verb_phrases": [{"id": "vp1", "phrase": "rel", "subject_id": "e", "object_id": "t"}],
        "constraints": [],
    })
    retr = _StubRetr([_hit("c1", "1,070 ft (326.1 m)")])
    llm = _LLM('{"value": "Y", "object_type": "entity", '
               '"source_span": "1,070 ft (326.1 m)", "supported": true, '
               '"confidence": 0.9, "evidence_chunk_ids": ["c1"]}')
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=llm,
                           use_hints=True, hard_type_filter=False)
    res = r.resolve_blank(plan, "t", {}, "Q")
    last_user = llm.calls[-1]["messages"][1]["content"]
    assert "REQUIRED UNIT" not in last_user
    assert res.value == "Y"    # untouched


def test_resolver_hint_without_unit_word_is_noop():
    plan = _unit_hint_plan(hint="What is the population of X?")
    retr = _StubRetr([_hit("c1", "The population is 1,070 (326.1 estimate)")])
    llm = _LLM('{"value": 1070, "object_type": "number", '
               '"source_span": "The population is 1,070 (326.1 estimate)", '
               '"supported": true, "confidence": 1.0, "evidence_chunk_ids": ["c1"]}')
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=llm,
                           use_hints=True, hard_type_filter=False)
    res = r.resolve_blank(plan, "b_height", {}, "Q")
    last_user = llm.calls[-1]["messages"][1]["content"]
    assert "REQUIRED UNIT" not in last_user
    assert res.value == 1070


# --- Step 2: deterministic dual-unit reconciliation (pure function) -------

def test_reconcile_declared_unit_value_q16_real_span_prefers_metres():
    """REAL span from logs/_xunit_smoke_221023/.../traces/q_16.json,
    extra.gsw_checker_calls[2] (resolver call for blank b_building_height).
    Declared unit (from the hint "...in meters?") is metres; the model
    answered 1070 (feet). Reconciliation must return the metres bracket."""
    span = "Tallest building height: 1,070 ft (326.1 m)"
    assert reconcile_declared_unit_value(span, "metres") == 326.1


def test_reconcile_declared_unit_value_prefers_feet_when_declared():
    span = "Tallest building height: 1,070 ft (326.1 m)"
    assert reconcile_declared_unit_value(span, "feet") == 1070


def test_reconcile_declared_unit_value_dual_metres_first_order():
    span = "788 metres (2,585 ft)"
    assert reconcile_declared_unit_value(span, "feet") == 2585
    assert reconcile_declared_unit_value(span, "metres") == 788


def test_reconcile_declared_unit_value_no_declared_unit_is_noop():
    span = "Tallest building height: 1,070 ft (326.1 m)"
    assert reconcile_declared_unit_value(span, None) is None
    assert reconcile_declared_unit_value(span, "") is None


def test_reconcile_declared_unit_value_no_dual_pattern_is_noop():
    assert reconcile_declared_unit_value(
        "The building is 1,070 feet tall.", "metres") is None


def test_reconcile_declared_unit_value_empty_span_is_noop():
    assert reconcile_declared_unit_value("", "metres") is None


def test_reconcile_declared_unit_value_unrecognized_units_is_noop():
    # "million" isn't in the closed unit vocabulary -> both sides unknown.
    assert reconcile_declared_unit_value(
        "Revenue was 1,070 million (326 million last year)", "metres") is None


def test_reconcile_declared_unit_value_declared_unit_absent_is_noop():
    # Both brackets recognized, but neither is the declared unit.
    assert reconcile_declared_unit_value("1,070 ft (326.1 km)", "metres") is None


def test_reconcile_declared_unit_value_ambiguous_same_unit_is_noop():
    assert reconcile_declared_unit_value("5 m (5 m)", "metres") is None


# --- Step 2: wired into resolve_blank end-to-end ---------------------------

def test_resolver_reconciles_q16_building_height_end_to_end():
    """Reconstructs the real q16 failure from the trace and proves the fix
    flips it: same hint, same retrieved chunk, same (wrong) LLM value/span
    -- resolve_blank must now return 326.1 (metres), not 1070 (feet)."""
    plan = _unit_hint_plan()
    retr = _StubRetr([_hit(
        "List of tallest buildings in San Francisco__0",
        "Tallest building height: 1,070 ft (326.1 m)",
    )])
    llm = _LLM(
        '{\n  "value": 1070,\n  "object_type": "number",\n  '
        '"source_span": "Tallest building height: 1,070 ft (326.1 m)",\n  '
        '"supported": true,\n  "confidence": 1.0,\n  '
        '"evidence_chunk_ids": ["List of tallest buildings in San Francisco__0"]\n}'
    )
    sink: list = []
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=llm,
                           use_hints=True, hard_type_filter=False, sink=sink)
    res = r.resolve_blank(
        plan, "b_height", {},
        "How many times would the tallest building in San Francisco fit "
        "end to end from the bottom of the New Britain Trench to the "
        "ocean surface, rounded to the nearest whole number?",
    )
    assert res.status == "resolved"
    assert res.value == 326.1          # was 1070 (feet) before the fix
    reconcile_events = [e for e in sink if e.get("kind") == "unit_reconcile"]
    assert len(reconcile_events) == 1
    assert reconcile_events[0]["from"] == 1070
    assert reconcile_events[0]["to"] == 326.1
    assert reconcile_events[0]["declared_unit"] == "metres"


def test_resolver_reconcile_is_noop_when_llm_already_matches_declared_unit():
    """q771 class: the model happened to already pick the CORRECT bracketed
    number. Reconciliation must be an observable no-op (same value) and
    must not append a spurious sink entry."""
    plan = _unit_hint_plan()
    retr = _StubRetr([_hit("c1", "Depth: 1,070 ft (326.1 m)")])
    llm = _LLM('{"value": 326.1, "object_type": "number", '
               '"source_span": "Depth: 1,070 ft (326.1 m)", '
               '"supported": true, "confidence": 1.0, "evidence_chunk_ids": ["c1"]}')
    sink: list = []
    r = TypedBlankResolver(retriever=retr, corpus=None, llm=llm,
                           use_hints=True, hard_type_filter=False, sink=sink)
    res = r.resolve_blank(plan, "b_height", {}, "Q")
    assert res.value == 326.1
    assert not [e for e in sink if e.get("kind") == "unit_reconcile"]
