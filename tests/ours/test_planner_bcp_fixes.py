"""BCP q7 planner fixes: range constraints must be expressible, and repair
rounds must see every defect at once.

MEASURED 2026-08-18 (BCP-200, q7): the planner modeled "published after 1990
but before 2004" as gt+lt constraints both pointing at b_pub_year via
output_blank_id and was rejected ("at most one producing constraint"); three
repair rounds then discovered the six structural validators ONE AT A TIME
(first-failure-only), exhausted, and the question died 0-turn with pred=''.
"""

import pytest

from panini_dr.adapters.ours._planner_exec import GSWPlan


def _plan(constraints):
    return {
        "entities": [
            {"id": "e_book", "kind": "filled", "name": "The Book", "role": "subject"},
            {"id": "e_1990", "kind": "filled", "name": "1990", "role": "constraint-value"},
            {"id": "e_2004", "kind": "filled", "name": "2004", "role": "constraint-value"},
            {"id": "b_pub_year", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "entity",
             "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "published_in", "subject_id": "e_book",
             "object_id": "b_pub_year"},
            {"id": "vp2", "phrase": "written_by", "subject_id": "e_book",
             "object_id": "t"},
        ],
        "constraints": constraints,
    }


def test_a_range_as_two_check_constraints_is_legal():
    """gt+lt bounds on one blank are a RANGE, not a producer conflict."""
    plan = GSWPlan.model_validate(_plan([
        {"id": "c_gt", "kind": "gt", "left_ref": "b_pub_year",
         "right_ref": "e_1990", "output_blank_id": "b_pub_year"},
        {"id": "c_lt", "kind": "lt", "left_ref": "b_pub_year",
         "right_ref": "e_2004", "output_blank_id": "b_pub_year"},
    ]))
    # normalization: the blank keeps AT MOST one owner; the colliding CHECK
    # constraint's spurious producer claim is stripped. (A SOLO checker's
    # output_blank_id stays -- the executor's Case-1 index reads it, and
    # historical FRAMES plans carry that shape; only the collision is new.)
    owners = [c.id for c in plan.constraints if c.output_blank_id]
    assert len(owners) <= 1, f"blank still has {len(owners)} producers: {owners}"


def test_two_real_producers_are_still_rejected():
    with pytest.raises(ValueError, match="at most one producing constraint"):
        GSWPlan.model_validate(_plan([
            {"id": "c_a", "kind": "derived", "op": "sum",
             "args_refs": ["e_1990", "e_2004"], "output_blank_id": "b_pub_year"},
            {"id": "c_b", "kind": "derived", "op": "diff",
             "args_refs": ["e_1990", "e_2004"], "output_blank_id": "b_pub_year"},
        ]))


def test_one_check_plus_one_producer_keeps_the_producer():
    plan = GSWPlan.model_validate(_plan([
        {"id": "c_prod", "kind": "derived", "op": "sum",
         "args_refs": ["e_1990", "e_2004"], "output_blank_id": "b_pub_year"},
        {"id": "c_gt", "kind": "gt", "left_ref": "b_pub_year",
         "right_ref": "e_1990", "output_blank_id": "b_pub_year"},
    ]))
    by_id = {c.id: c for c in plan.constraints}
    assert by_id["c_prod"].output_blank_id == "b_pub_year"
    assert by_id["c_gt"].output_blank_id in (None, "")


def test_quality_validators_report_all_defects_in_one_round():
    """First-failure-only turned 3 repair rounds into a one-mole-per-round
    discovery walk. All defects must surface together."""
    from panini_dr.adapters.ours import _planner_emit as pe

    calls = []
    real = {}
    for name in ("_reject_literal_constant_blanks",
                 "_reject_numeric_literal_refs_without_values",
                 "_reject_arg_selection_over_blank_candidates",
                 "_reject_date_granularity_missing",
                 "_reject_unit_mismatch_constraints",
                 "_reject_dropped_qualifier_hints"):
        real[name] = getattr(pe, name)

    plan = GSWPlan.model_validate(_plan([
        {"id": "c_gt", "kind": "gt", "left_ref": "b_pub_year",
         "right_ref": "e_1990", "output_blank_id": None},
        {"id": "c_lt", "kind": "lt", "left_ref": "b_pub_year",
         "right_ref": "e_2004", "output_blank_id": None},
    ]))
    import unittest.mock as m
    def boom_a(*a, **k): raise ValueError("defect_alpha: first")
    def boom_b(*a, **k): raise ValueError("defect_beta: second")
    with m.patch.object(pe, "_reject_literal_constant_blanks", boom_a), \
         m.patch.object(pe, "_reject_date_granularity_missing", boom_b):
        with pytest.raises(ValueError) as ei:
            pe._validate_plan_quality("q?", plan)
    msg = str(ei.value)
    assert "defect_alpha" in msg and "defect_beta" in msg, (
        "both defects must be reported in the same repair round"
    )
