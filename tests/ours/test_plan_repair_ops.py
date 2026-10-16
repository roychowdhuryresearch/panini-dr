import pytest

from panini_dr.adapters.ours._plan_repair_ops import (
    RepairApplyError, apply_op, is_destructive, parse_repair_op,
)
from panini_dr.adapters.ours._planner_exec import GSWPlan


def _plan() -> GSWPlan:
    return GSWPlan.model_validate({
        "entities": [
            {"id": "b_year", "kind": "blank", "role": "attribute",
             "value_type": "date", "retrieval_hint": "When was it admitted?"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "text",
             "is_target": True, "retrieval_hint": "British monarch on {b_year}?"},
        ],
        "verb_phrases": [{"id": "vp1", "subject_id": "b_year",
                          "phrase": "monarch_on", "object_id": "t"}],
        "constraints": [],
    })


def test_reverse_edge_swaps_subject_and_object():
    """q539, verbatim: a date-typed subject creates no ordering edge, so the
    hint is dropped. Reversing makes the target the subject, which flips the
    edge in build_dependency_graph and orders b_year first."""
    new = apply_op(_plan(), parse_repair_op({"op": "reverse_edge", "vp_id": "vp1"}))
    vp = new.verb_phrases[0]
    assert (vp.subject_id, vp.object_id) == ("t", "b_year")
    assert vp.phrase == "monarch_on"          # prose is preserved, not rewritten


def test_retype_node_changes_only_the_value_type():
    new = apply_op(_plan(), parse_repair_op(
        {"op": "retype_node", "node_id": "b_year", "value_type": "entity"}))
    n = next(e for e in new.entities if e.id == "b_year")
    assert n.value_type == "entity"
    assert n.retrieval_hint == "When was it admitted?"


def test_set_hint_replaces_the_text():
    new = apply_op(_plan(), parse_repair_op(
        {"op": "set_hint", "node_id": "t", "text": "Who was monarch in 1959?"}))
    assert next(e for e in new.entities if e.id == "t").retrieval_hint == (
        "Who was monarch in 1959?")


def test_inline_placeholder_substitutes_the_referenced_hint():
    """Relocation, not authorship: {b_year} becomes b_year's own hint text."""
    new = apply_op(_plan(), parse_repair_op(
        {"op": "inline_placeholder", "node_id": "t", "ref": "b_year"}))
    hint = next(e for e in new.entities if e.id == "t").retrieval_hint
    assert "{b_year}" not in hint
    assert "when was it admitted" in hint.lower()


def test_an_op_naming_an_unknown_id_raises():
    with pytest.raises(RepairApplyError):
        apply_op(_plan(), parse_repair_op({"op": "reverse_edge", "vp_id": "nope"}))


def test_only_content_removing_ops_are_destructive():
    assert not is_destructive(parse_repair_op({"op": "reverse_edge", "vp_id": "vp1"}))
    assert not is_destructive(parse_repair_op(
        {"op": "retype_node", "node_id": "b_year", "value_type": "entity"}))
    assert is_destructive(parse_repair_op(
        {"op": "set_hint", "node_id": "t", "text": ""}))


def test_retype_node_that_closes_a_cycle_raises():
    """build_dependency_graph keys a VP's edge direction off the subject's
    value_type: subject.value_type == "entity" makes the object depend on
    the subject. Here a constraint already makes x depend on y, and a VP
    x->y contributes NO edge while x is date-typed (matching the reverse_edge
    fixture's own q539 rule) — so the starting plan is acyclic. Retyping x to
    "entity" turns that VP into a live edge (y now depends on x too), closing
    a 2-cycle that GSWPlan.model_validate's schema checks can't see; only the
    same topology re-check apply_diff runs catches it. apply_op must apply
    that check to every op, not only reverse_edge."""
    plan = GSWPlan.model_validate({
        "entities": [
            {"id": "y", "kind": "blank", "value_type": "text",
             "retrieval_hint": "find y"},
            {"id": "x", "kind": "blank", "value_type": "date",
             "retrieval_hint": "find x"},
        ],
        "verb_phrases": [{"id": "vp1", "subject_id": "x", "phrase": "rel",
                          "object_id": "y"}],
        "constraints": [{"id": "c1", "kind": "derived", "op": "concat",
                         "args_refs": ["y"], "output_blank_id": "x"}],
    })
    with pytest.raises(RepairApplyError):
        apply_op(plan, parse_repair_op(
            {"op": "retype_node", "node_id": "x", "value_type": "entity"}))


def _plan_for_node_ops() -> GSWPlan:
    """A richer fixture (target + two extra blanks, one redundantly-wired
    "disposable" blank) so every Task-3 op — including the deletes, which
    need something removable without stranding a reference — has a
    successful application to run the aliasing check against."""
    return GSWPlan.model_validate({
        "entities": [
            {"id": "b1", "kind": "blank", "value_type": "text", "retrieval_hint": "h1"},
            {"id": "b2", "kind": "blank", "value_type": "text", "retrieval_hint": "h2"},
            {"id": "b3", "kind": "blank", "value_type": "text", "retrieval_hint": "h3",
             "is_target": True},
            {"id": "b5", "kind": "blank", "value_type": "text", "retrieval_hint": "h5"},
        ],
        "verb_phrases": [
            {"id": "vp1", "subject_id": "b1", "phrase": "rel", "object_id": "b2"},
            {"id": "vp2", "subject_id": "b5", "phrase": "extra", "object_id": "b3"},
        ],
        "constraints": [{"id": "c1", "kind": "equals", "left_ref": "b1",
                         "right_ref": "b2", "output_blank_id": "b3"}],
    })


def test_apply_op_does_not_mutate_or_alias_the_input_plan():
    """Load-bearing per the brief: a caller applies an op, inspects the
    result, and discards it if the plan didn't improve — aliasing shared
    sub-objects would let mutating (or discarding) the result corrupt the
    original the caller is still holding. Value-equality alone would miss a
    shared reference, so this also mutates the RETURNED plan's nested
    Entity/VerbPhrase objects and checks the ORIGINAL's are untouched."""
    raw_ops = [
        (_plan, {"op": "reverse_edge", "vp_id": "vp1"}),
        (_plan, {"op": "retype_node", "node_id": "b_year", "value_type": "entity"}),
        (_plan, {"op": "set_hint", "node_id": "t", "text": "changed"}),
        (_plan, {"op": "inline_placeholder", "node_id": "t", "ref": "b_year"}),
        # Task 3: node, edge, target and constraint ops, against a fixture
        # where each of these succeeds (see _plan_for_node_ops docstring).
        (_plan_for_node_ops, {"op": "add_node", "node_id": "b4", "kind": "blank",
                              "value_type": "text",
                              "out_edges": [{"phrase": "rel2", "object_id": "b3"}],
                              "in_edges": []}),
        (_plan_for_node_ops, {"op": "delete_node", "node_id": "b5"}),
        (_plan_for_node_ops, {"op": "add_edge", "subject_id": "b1",
                              "phrase": "another_rel", "object_id": "b3"}),
        (_plan_for_node_ops, {"op": "delete_edge", "vp_id": "vp1"}),
        (_plan_for_node_ops, {"op": "set_target", "node_id": "b1"}),
        (_plan_for_node_ops, {"op": "add_constraint", "constraint": {
            "id": "c2", "kind": "equals", "left_ref": "b1", "right_ref": "b2",
            "output_blank_id": "b5"}}),
        (_plan_for_node_ops, {"op": "delete_constraint", "constraint_id": "c1"}),
        (_plan_for_node_ops, {"op": "modify_constraint", "constraint_id": "c1",
                              "changes": {"right_ref": "b1"}}),
    ]
    for plan_factory, raw in raw_ops:
        original = plan_factory()
        before = original.model_dump()

        new = apply_op(original, parse_repair_op(raw))
        assert original.model_dump() == before, f"{raw['op']} mutated its input"

        for e in new.entities:
            e.retrieval_hint = "MUTATED"
            e.value_type = "bool"
        for vp in new.verb_phrases:
            vp.phrase = "MUTATED"
        for c in new.constraints:
            c.output_blank_id = "MUTATED"
        assert original.model_dump() == before, (
            f"{raw['op']}: apply_op aliased entities/verb_phrases/constraints "
            "with the input (mutating the result changed the original)")
