from panini_dr.adapters.ours._planner_exec import GSWPlan
from panini_dr.adapters.ours._edge_taxonomy import (
    classify_edge, FACTUAL, SCAFFOLD_TEMPORAL, SCAFFOLD_TYPING,
)
from panini_dr.adapters.ours._gsw_state import blank_projection_edges


def _plan():
    return GSWPlan.model_validate({
        "entities": [
            {"id": "e_date", "kind": "filled", "name": "August 2024", "role": "as-of-date"},
            {"id": "e_cat", "kind": "filled", "name": "comedian", "category": True},
            {"id": "e_show", "kind": "filled", "name": "Hot Ones", "role": "scope-filter"},
            {"id": "e_year", "kind": "filled", "name": "1999"},
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True, "role": "target"},
        ],
        "verb_phrases": [
            {"id": "vp_t", "subject_id": "t", "phrase": "as_of_date", "object_id": "e_year"},
            {"id": "vp_role", "subject_id": "t", "phrase": "active_during", "object_id": "e_date"},
            {"id": "vp_inst", "subject_id": "t", "phrase": "instance_of", "object_id": "e_show"},
            {"id": "vp_cat", "subject_id": "t", "phrase": "works_as", "object_id": "e_cat"},
            {"id": "vp_fact", "subject_id": "t", "phrase": "appeared_on", "object_id": "e_show"},
            {"id": "vp_year", "subject_id": "t", "phrase": "published_in_year", "object_id": "e_year"},
        ],
        "constraints": [],
    })


def _edges_by_vp(plan):
    return {e["vp_id"]: e for e in blank_projection_edges(plan, "t")}


def test_as_of_date_phrase_is_temporal_scaffold():
    plan = _plan()
    assert classify_edge(plan, _edges_by_vp(plan)["vp_t"]) == SCAFFOLD_TEMPORAL


def test_as_of_date_partner_role_is_temporal_scaffold():
    plan = _plan()
    assert classify_edge(plan, _edges_by_vp(plan)["vp_role"]) == SCAFFOLD_TEMPORAL


def test_instance_of_phrase_is_typing_scaffold():
    plan = _plan()
    assert classify_edge(plan, _edges_by_vp(plan)["vp_inst"]) == SCAFFOLD_TYPING


def test_category_partner_is_typing_scaffold():
    plan = _plan()
    assert classify_edge(plan, _edges_by_vp(plan)["vp_cat"]) == SCAFFOLD_TYPING


def test_ordinary_and_in_year_edges_stay_factual():
    plan = _plan()
    assert classify_edge(plan, _edges_by_vp(plan)["vp_fact"]) == FACTUAL
    assert classify_edge(plan, _edges_by_vp(plan)["vp_year"]) == FACTUAL


def test_missing_partner_fails_closed_to_factual():
    plan = _plan()
    edge = dict(_edges_by_vp(plan)["vp_fact"], other_entity_id="nope")
    assert classify_edge(plan, edge) == FACTUAL
