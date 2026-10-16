"""Unit tests for planner-emitted blank hints (gsw_blank_hints)."""
from __future__ import annotations

from panini_dr.adapters.ours._planner_exec import Entity, GSWPlan
from panini_dr.adapters.ours._blank_resolver import (
    substitute_placeholders, has_unresolved_placeholders,
    get_retrieval_hint, effective_query, EffectiveQuery,
)


def test_entity_accepts_hint_fields():
    e = Entity.model_validate({
        "id": "b_country", "kind": "blank", "value_type": "entity", "role": "bridge-entity",
        "retrieval_hint": "Which country ranked highest in the 2023 Democracy Index?",
        "verification_claim": "The candidate must be the country ranked highest in the 2023 Democracy Index.",
    })
    assert e.retrieval_hint.startswith("Which country")
    assert e.verification_claim.startswith("The candidate must be")


def test_entity_hint_fields_default_none():
    e = Entity.model_validate({"id": "t", "kind": "blank", "value_type": "entity"})
    assert e.retrieval_hint is None
    assert e.verification_claim is None


def test_old_plan_without_hints_still_validates():
    plan = GSWPlan.model_validate({
        "entities": [
            {"id": "e", "kind": "filled", "name": "Serbia"},
            {"id": "t", "kind": "blank", "value_type": "date", "is_target": True},
        ],
        "verb_phrases": [{"id": "vp1", "phrase": "independent_in", "subject_id": "e", "object_id": "t"}],
        "constraints": [],
    })
    assert plan.entity_by_id("t").retrieval_hint is None


def test_hint_fields_do_not_leak_into_plan_brief():
    from panini_dr.adapters.ours._planner_react_prompt import render_plan_brief
    plan = GSWPlan.model_validate({
        "entities": [
            {"id": "e", "kind": "filled", "name": "Democracy Index"},
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
             "retrieval_hint": "SECRET_HINT_TOKEN",
             "verification_claim": "SECRET_CLAIM_TOKEN"},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "ranked_highest_in", "subject_id": "t", "object_id": "e"},
        ],
        "constraints": [],
    })
    order = ["t"]
    levels = {"t": 0}
    brief = render_plan_brief(plan.model_dump(), order, levels)
    assert "SECRET_HINT_TOKEN" not in brief
    assert "SECRET_CLAIM_TOKEN" not in brief
    assert "retrieval_hint" not in brief
    assert "verification_claim" not in brief


def _hint_plan():
    return GSWPlan.model_validate({
        "entities": [
            {"id": "e", "kind": "filled", "name": "Democracy Index"},
            {"id": "b_country", "kind": "blank", "value_type": "entity",
             "retrieval_hint": "Which country ranked highest in the 2023 Democracy Index?",
             "verification_claim": "The candidate must be the country ranked highest in the 2023 Democracy Index."},
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
             "retrieval_hint": "What is the largest religion in {b_country} as of August 3, 2024?",
             "verification_claim": "The candidate must be the largest religion in {b_country} as of August 3, 2024."},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "best_democracy_in", "subject_id": "b_country", "object_id": "e"},
            {"id": "vp2", "phrase": "largest_religion_in", "subject_id": "t", "object_id": "b_country"},
        ],
        "constraints": [],
    })


def test_substitute_fills_known_placeholder():
    assert substitute_placeholders(
        "largest religion in {b_country} as of 2024", {"b_country": "Norway"}
    ) == "largest religion in Norway as of 2024"


def test_substitute_leaves_unknown_intact():
    out = substitute_placeholders("religion in {b_country}", {})
    assert out == "religion in {b_country}"
    assert has_unresolved_placeholders(out) is True


def test_has_unresolved_false_when_clean():
    assert has_unresolved_placeholders("largest religion in Norway") is False


def test_get_retrieval_hint_or_none():
    plan = _hint_plan()
    assert get_retrieval_hint(plan, "b_country").startswith("Which country")
    assert get_retrieval_hint(plan, "e") is None


def test_get_retrieval_hint_unknown_id_returns_none():
    # broad except in get_retrieval_hint: a missing blank id must yield None, not raise
    assert get_retrieval_hint(_hint_plan(), "does_not_exist") is None


def test_effective_query_source_hint():
    eq = effective_query(_hint_plan(), "b_country", {}, "Q?", use_hints=True)
    assert isinstance(eq, EffectiveQuery)
    assert eq.source == "hint"
    assert eq.query == "Which country ranked highest in the 2023 Democracy Index?"
    assert eq.rendered_hint == eq.query


def test_effective_query_source_unresolved_hint():
    eq = effective_query(_hint_plan(), "t", {}, "Q?", use_hints=True)
    assert eq.source == "unresolved_hint"
    assert eq.query is None
    assert eq.raw_hint.startswith("What is the largest religion in {b_country}")


def test_effective_query_source_edge_when_no_hint():
    plan = GSWPlan.model_validate({
        "entities": [
            {"id": "e", "kind": "filled", "name": "Serbia"},
            {"id": "t", "kind": "blank", "value_type": "date", "is_target": True},
        ],
        "verb_phrases": [{"id": "vp1", "phrase": "independent_in", "subject_id": "e", "object_id": "t"}],
        "constraints": [],
    })
    eq = effective_query(plan, "t", {}, "Q fallback?", use_hints=True)
    assert eq.source == "edge"
    assert eq.query and "{" not in eq.query


def test_effective_query_off_ignores_hint():
    eq = effective_query(_hint_plan(), "b_country", {}, "Q?", use_hints=False)
    assert eq.source in ("edge", "question_fallback")
    assert eq.query != "Which country ranked highest in the 2023 Democracy Index?"


from panini_dr.adapters.ours._planner_prompts import (
    build_planner_messages, build_repair_messages,
)


def test_planner_messages_include_hint_instruction_when_on():
    system = build_planner_messages("Q?", blank_hints=True)[0]["content"]
    assert "retrieval_hint" in system and "verification_claim" in system
    assert "{b_country}" in system or "placeholder" in system.lower()
    assert "specific" in system.lower()


def test_planner_messages_omit_hint_instruction_when_off():
    system = build_planner_messages("Q?", blank_hints=False)[0]["content"]
    assert "verification_claim" not in system


def test_repair_messages_include_hint_instruction_when_on():
    msgs = build_repair_messages("Q?", "bad", "err", blank_hints=True)
    blob = " ".join(m["content"] for m in msgs)
    assert "retrieval_hint" in blob and "verification_claim" in blob


def test_repair_messages_omit_hint_instruction_when_off():
    msgs = build_repair_messages("Q?", "bad", "err", blank_hints=False)
    blob = " ".join(m["content"] for m in msgs)
    assert "verification_claim" not in blob


from panini_dr.adapters.ours._plan_updater import build_updater_messages
from panini_dr.adapters.ours._planner_exec import GSWPlan as _GP2


def _simple_plan():
    return _GP2.model_validate({
        "entities": [
            {"id": "e_movie", "kind": "filled", "name": "Movie X"},
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [{"id": "vp1", "phrase": "directed_by", "subject_id": "e_movie", "object_id": "t"}],
        "constraints": [],
    })


def test_updater_messages_include_hint_instruction_when_on():
    msgs = build_updater_messages("Q?", _simple_plan(), {}, "reason", "evidence", blank_hints=True)
    blob = " ".join(m["content"] for m in msgs)
    assert "retrieval_hint" in blob and "verification_claim" in blob


def test_updater_messages_omit_hint_instruction_when_off():
    msgs = build_updater_messages("Q?", _simple_plan(), {}, "reason", "evidence", blank_hints=False)
    blob = " ".join(m["content"] for m in msgs)
    assert "verification_claim" not in blob


from panini_dr.adapters.ours._plan_updater import build_updater_retry_messages


def test_updater_retry_messages_include_hint_instruction_when_on():
    msgs = build_updater_retry_messages(
        "Q?", _simple_plan(), {}, "reason", "evidence",
        bad_response="oops", failure_detail="parse_failure", blank_hints=True,
    )
    blob = " ".join(m["content"] for m in msgs)
    assert "retrieval_hint" in blob and "verification_claim" in blob


def test_updater_retry_messages_omit_hint_instruction_when_off():
    msgs = build_updater_retry_messages(
        "Q?", _simple_plan(), {}, "reason", "evidence",
        bad_response="oops", failure_detail="parse_failure", blank_hints=False,
    )
    blob = " ".join(m["content"] for m in msgs)
    assert "verification_claim" not in blob


from panini_dr.adapters.ours._plan_diff import apply_diff, parse_diff_op


def test_apply_diff_add_blank_preserves_hints():
    base = _GP2.model_validate({
        "entities": [
            {"id": "e_movie", "kind": "filled", "name": "Movie X"},
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "directed_by", "subject_id": "e_movie", "object_id": "t"},
        ],
        "constraints": [],
    })
    op = parse_diff_op({
        "op": "add_blank",
        "blank": {
            "id": "b_year",
            "value_type": "date",
            "role": "bridge-date",
            "retrieval_hint": "What year did Movie X release?",
            "verification_claim": "The candidate must be the release year of Movie X.",
        },
        "wire_via_vp": {
            "id": "vp_year",
            "phrase": "released_in",
            "subject_id": "e_movie",
            "object_id": "b_year",
        },
    })
    new_plan = apply_diff(base, op)
    new_blank = new_plan.entity_by_id("b_year")
    assert new_blank.retrieval_hint == "What year did Movie X release?"
    assert new_blank.verification_claim == "The candidate must be the release year of Movie X."
