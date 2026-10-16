from panini_dr.adapters.ours._blank_verifier import BlankVerifier, VerifierResult
from panini_dr.adapters.ours._evidence_prompts import (
    VERIFIER_SYSTEM,
    VERIFIER_SYSTEM_SCOPED,
    parse_verifier_output,
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


class _StubLLM:
    def __init__(self, payload):
        self.payload = payload
        self.last_system = None
        self.last_user = None

    def chat(self, *, messages, **k):
        self.last_system = messages[0]["content"]
        self.last_user = messages[-1]["content"]
        class R: text = self.payload
        return R()


_CHUNKS = [{"chunk_id": "C1", "title": "T", "text": "body"}]


def test_verifier_rejects_wrong_value():
    llm = _StubLLM('{"satisfies_all": false, "unique": true,'
                   ' "missing_constraints": ["cofounder_of Netflix"],'
                   ' "deferred_constraints": [], "ambiguous_candidates": [],'
                   ' "reason": "not a Netflix cofounder"}')
    v = BlankVerifier(llm=llm).verify(_q492_plan(), "b1", "Yvon Chouinard",
                                      _CHUNKS, dep_values={})
    assert isinstance(v, VerifierResult)
    assert v.satisfies_all is False
    assert (v.satisfies_all and v.unique) is False  # rejected


def test_verifier_rejects_ambiguous():
    llm = _StubLLM('{"satisfies_all": true, "unique": false,'
                   ' "missing_constraints": [], "deferred_constraints": [],'
                   ' "ambiguous_candidates": ["Reed Hastings","Marc Randolph"],'
                   ' "reason": "two cofounders"}')
    v = BlankVerifier(llm=llm).verify(_q492_plan(), "b1", "Reed Hastings",
                                      _CHUNKS, dep_values={})
    assert v.unique is False
    assert v.ambiguous_candidates == ["Reed Hastings", "Marc Randolph"]


def test_verifier_constraint_block_uses_qualifier_not_raw_blank_id():
    llm = _StubLLM('{"satisfies_all": true, "unique": true,'
                   ' "missing_constraints": [], "deferred_constraints": [],'
                   ' "ambiguous_candidates": [], "reason": "ok"}')
    BlankVerifier(llm=llm).verify(_q492_plan(), "b1", "Reed Hastings",
                                  _CHUNKS, dep_values={})
    # the constraints text sent to the LLM must NOT leak the raw partner id,
    # and MUST surface its category qualifier instead.
    assert "b_environmental_group" not in llm.last_user
    assert "environmental group" in llm.last_user


def test_verifier_records_to_sink():
    sink = []
    llm = _StubLLM('{"satisfies_all": true, "unique": true,'
                   ' "missing_constraints": [], "deferred_constraints": [],'
                   ' "ambiguous_candidates": [], "reason": "ok"}')
    BlankVerifier(llm=llm, sink=sink).verify(_q492_plan(), "b1", "Reed Hastings",
                                             _CHUNKS, dep_values={})
    assert sink and sink[0]["kind"] == "verifier" and sink[0]["blank_id"] == "b1"


def test_verifier_sink_captures_prompt_output_reasoning():
    class _LLM:
        def chat(self, *, messages, **k):
            class R:
                text = ('{"satisfies_all": true, "unique": true,'
                        ' "missing_constraints": [], "deferred_constraints": [],'
                        ' "ambiguous_candidates": [], "reason": "ok"}')
                reasoning_content = "checked the cited chunk"
            return R()
    sink = []
    BlankVerifier(llm=_LLM(), sink=sink).verify(_q492_plan(), "b1", "Reed Hastings",
                                                _CHUNKS, dep_values={})
    rec = sink[0]
    assert rec["prompt"] and "Reed Hastings" in rec["prompt"]   # rendered user prompt
    assert rec["raw_output"].startswith("{")                    # verbatim model text
    assert rec["reasoning"] == "checked the cited chunk"        # optional, here present
    assert rec["verdict"]["satisfies_all"] is True


def test_verifier_result_has_completeness_defaults():
    r = VerifierResult(satisfies_all=True, unique=True)
    assert r.complete is True          # scalar default = complete
    assert r.partial is False
    assert r.missing_items == []
    assert r.accepted is True          # unchanged: satisfies_all and unique


def test_verifier_result_partial_list():
    r = VerifierResult(satisfies_all=True, unique=True, complete=False,
                       missing_items=["second founder birthdate"])
    assert r.partial is True           # partial derived from complete
    assert r.missing_items == ["second founder birthdate"]
    assert r.accepted is True          # partial is logged, NOT wired into accepted


def test_parse_verifier_output_reads_completeness():
    raw = ('{"satisfies_all": true, "unique": true, "complete": false, '
           '"missing_items": ["second founder"], "reason": "one of two"}')
    d = parse_verifier_output(raw)
    assert d["complete"] is False
    assert d["missing_items"] == ["second founder"]


def test_parse_verifier_output_completeness_defaults_true():
    raw = '{"satisfies_all": true, "unique": true, "reason": "ok"}'
    d = parse_verifier_output(raw)
    assert d["complete"] is True
    assert d["missing_items"] == []


def test_scoped_verifier_selects_scoped_system_prompt_and_rules():
    assert "Snapshot/as-of date qualifiers" in VERIFIER_SYSTEM_SCOPED
    assert "deferred_constraints" in VERIFIER_SYSTEM_SCOPED
    assert "missing_constraints" in VERIFIER_SYSTEM_SCOPED

    llm = _StubLLM(_ACCEPT)
    BlankVerifier(llm=llm, scoped=True).verify(
        _q492_plan(), "b1", "Reed Hastings", _CHUNKS, dep_values={}
    )
    assert llm.last_system == VERIFIER_SYSTEM_SCOPED

    llm_off = _StubLLM(_ACCEPT)
    BlankVerifier(llm=llm_off, scoped=False).verify(
        _q492_plan(), "b1", "Reed Hastings", _CHUNKS, dep_values={}
    )
    assert llm_off.last_system == VERIFIER_SYSTEM


def test_verifier_reasoning_guidance_is_prompted_and_guarded_only_when_present():
    reasoning = "Chunk C1 names Reed Hastings and explains the Netflix connection."
    llm = _StubLLM(_ACCEPT)
    BlankVerifier(llm=llm).verify(
        _q492_plan(), "b1", "Reed Hastings", _CHUNKS, dep_values={},
        reasoning=reasoning,
    )
    assert "Researcher's reasoning (GUIDANCE ONLY)" in llm.last_user
    assert reasoning in llm.last_user
    assert "verdict MUST be grounded in the cited EVIDENCE" in llm.last_system

    llm_off = _StubLLM(_ACCEPT)
    BlankVerifier(llm=llm_off).verify(
        _q492_plan(), "b1", "Reed Hastings", _CHUNKS, dep_values={}
    )
    assert "Researcher's reasoning" not in llm_off.last_user
    assert llm_off.last_system == VERIFIER_SYSTEM


def test_verifier_long_reasoning_needs_full_budget_to_avoid_silent_reject():
    """Regression for the max_completion_tokens truncation bug: a real chat
    completions API emits reasoning tokens before content tokens, so when
    the verifier's chain-of-thought is long and max_tokens is too small, the
    budget is exhausted mid-reasoning and NO content (JSON) is emitted at
    all. parse_verifier_output is documented to fail safe on empty/malformed
    output, so the caller silently gets satisfies_all=False with reason=""
    — the real verdict is discarded, indistinguishable from an intentional
    reject.

    BlankVerifier.__init__ hardcodes max_completion_tokens=400. All 5
    construction sites in gsw_planner_orchestrator_v1.py used to omit any
    override (0 of 5 passed it), so every verifier call ran capped at 400
    regardless of the orchestrator's configured budget
    (self.ctx.max_completion_tokens, default 50000). This test proves the
    mechanism end-to-end: the SAME long-reasoning response silently rejects
    at the old hardcoded default (400) and parses correctly once the caller
    supplies the real budget — exactly what the fix now threads through at
    every construction site."""
    full_json = ('{"satisfies_all": true, "unique": true,'
                 ' "missing_constraints": [], "deferred_constraints": [],'
                 ' "ambiguous_candidates": [], "reason": "matches all constraints"}')
    long_reasoning = "Considering each constraint in turn. " * 100  # ~600 "tokens" by our proxy, > 400

    class _BudgetLimitedLLM:
        """Crude proxy for a real reasoning-model completions API: content
        is only emitted if max_tokens leaves room after the reasoning
        tokens are spent."""
        def chat(self, *, messages, max_tokens, **k):
            reasoning_tokens = len(long_reasoning.split())
            class R:
                text = "" if max_tokens <= reasoning_tokens else full_json
                reasoning_content = long_reasoning
            return R()

    # Old hardcoded default (400 < reasoning_tokens): budget exhausted before
    # any JSON is emitted -> safe reject, diagnosis silently discarded.
    v_truncated = BlankVerifier(llm=_BudgetLimitedLLM(), max_completion_tokens=400).verify(
        _q492_plan(), "b1", "Reed Hastings", _CHUNKS, dep_values={})
    assert v_truncated.satisfies_all is False
    assert v_truncated.reason == ""

    # Orchestrator's real budget (self.ctx.max_completion_tokens, default
    # 50000) leaves room for the full reasoning chain AND the JSON -> the
    # real verdict parses instead of being thrown away.
    v_full = BlankVerifier(llm=_BudgetLimitedLLM(), max_completion_tokens=50000).verify(
        _q492_plan(), "b1", "Reed Hastings", _CHUNKS, dep_values={})
    assert v_full.satisfies_all is True
    assert v_full.reason == "matches all constraints"


from panini_dr.adapters.ours._planner_exec import GSWPlan as _GP
from panini_dr.adapters.ours._blank_verifier import (
    render_blank_claim, get_verification_claim,
)

_ACCEPT = ('{"satisfies_all": true, "unique": true, "missing_constraints": [],'
           ' "deferred_constraints": [], "ambiguous_candidates": [], "reason": "ok"}')


def _claim_plan():
    # NOTE: verb_phrases wire every entity (GSWPlan._check_no_dangling requires it).
    return _GP.model_validate({
        "entities": [
            {"id": "e", "kind": "filled", "name": "Democracy Index"},
            {"id": "b_country", "kind": "blank", "value_type": "entity"},
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
             "verification_claim": "The candidate must be the largest religion in {b_country} as of August 3, 2024."},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "best_democracy_in", "subject_id": "b_country", "object_id": "e"},
            {"id": "vp2", "phrase": "largest_religion_in", "subject_id": "t", "object_id": "b_country"},
        ],
        "constraints": [],
    })


def test_get_verification_claim_or_none():
    assert get_verification_claim(_claim_plan(), "t").startswith("The candidate must be")
    assert get_verification_claim(_claim_plan(), "e") is None


def test_render_blank_claim_substitutes_and_none_when_unresolved():
    assert render_blank_claim(_claim_plan(), "t", {"b_country": "Norway"}) == \
        "The candidate must be the largest religion in Norway as of August 3, 2024."
    assert render_blank_claim(_claim_plan(), "t", {}) is None


def test_verifier_uses_claim_when_enabled():
    llm = _StubLLM(_ACCEPT); sink = []
    BlankVerifier(llm=llm, sink=sink, use_hints=True).verify(
        _claim_plan(), "t", "Christianity", _CHUNKS, {"b_country": "Norway"})
    assert "largest religion in Norway as of August 3, 2024" in llm.last_user
    assert sink[-1]["claim_source"] == "hint"
    assert sink[-1]["rendered_verification_claim"] == \
        "The candidate must be the largest religion in Norway as of August 3, 2024."
    assert sink[-1]["verification_claim_raw"] == \
        "The candidate must be the largest religion in {b_country} as of August 3, 2024."


def test_verifier_falls_back_when_disabled():
    llm = _StubLLM(_ACCEPT); sink = []
    BlankVerifier(llm=llm, sink=sink, use_hints=False).verify(
        _claim_plan(), "t", "Christianity", _CHUNKS, {"b_country": "Norway"})
    assert "The candidate must be the largest religion" not in llm.last_user
    assert sink[-1]["claim_source"] == "projection_edges"


def _scaffold_plan():
    from panini_dr.adapters.ours._planner_exec import GSWPlan
    return GSWPlan.model_validate({
        "entities": [
            {"id": "e_show", "kind": "filled", "name": "Hot Ones"},
            {"id": "e_date", "kind": "filled", "name": "August 2024", "role": "as-of-date"},
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp_fact", "subject_id": "t", "phrase": "appeared_on", "object_id": "e_show"},
            {"id": "vp_scaf", "subject_id": "t", "phrase": "as_of_date", "object_id": "e_date"},
        ],
        "constraints": [],
    })


def _gillis_plan():
    return GSWPlan.model_validate({
        "entities": [
            {"id": "e_show", "kind": "filled", "name": "Hot Ones"},
            {"id": "e_snl", "kind": "filled", "name": "SNL"},
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "subject_id": "t", "phrase": "fired_from", "object_id": "e_snl"},
            {"id": "vp2", "subject_id": "t", "phrase": "appeared_on", "object_id": "e_show"},
        ],
        "constraints": [],
    })


class _NeverCalledLLM:
    def chat(self, *, messages, **k):
        raise AssertionError("verifier LLM must not be called on a mechanical block")


def _reject_llm():
    class _L:
        def chat(self, *, messages, **k):
            class R:
                text = '{"satisfies_all": false, "unique": true, "reason": "evidence contradicts"}'
            return R()
    return _L()


def _accept_llm():
    class _L:
        def chat(self, *, messages, **k):
            class R:
                text = '{"satisfies_all": true, "unique": true, "reason": "ok"}'
            return R()
    return _L()


def _read_chunks(ids):
    return [{"chunk_id": i, "title": "", "text": f"text of {i}"} for i in ids]


