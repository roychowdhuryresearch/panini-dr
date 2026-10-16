import pytest

from panini_dr.adapters.ours._plan_repair_ops import (
    RepairApplyError, apply_op, is_destructive, parse_repair_op,
)
from panini_dr.adapters.ours._planner_exec import GSWPlan


def _plan() -> GSWPlan:
    return GSWPlan.model_validate({
        "entities": [
            {"id": "b_a", "kind": "blank", "role": "attribute",
             "value_type": "text", "retrieval_hint": "Which A?"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "bool",
             "is_target": True},
        ],
        # t -> b_a: keeps `t` referenced independent of c1, so redirecting
        # c1's output away from `t` (test_modify_constraint_can_redirect_
        # its_output) doesn't strand it as a dangling entity -- the brief's
        # fixture had only c1 wiring `t` in, which a full-revalidate redirect
        # correctly rejects as a stranded reference. Subject is `t`, not
        # `b_a`: build_dependency_graph makes the TARGET a sink of its VP
        # partner regardless of value_type (target-as-subject flips the
        # edge), so with `t` still the target this contributes the same
        # t-depends-on-b_a edge c1 already does (no new cycle risk); the
        # other direction would, once test_set_target_moves_the_flag
        # retargets to b_a, close a 2-cycle against c1's existing edge.
        "verb_phrases": [{"id": "vp0", "subject_id": "t", "phrase": "rel",
                          "object_id": "b_a"}],
        "constraints": [{"id": "c1", "kind": "equals", "left_ref": "b_a",
                         "right_ref": "b_a", "output_blank_id": "t"}],
    })


def test_add_node_requires_at_least_one_edge():
    """GSWPlan raises `dangling entities` for a node nothing references, so an
    edgeless add can never validate -- reject it at parse time instead."""
    with pytest.raises(Exception):
        parse_repair_op({"op": "add_node", "node_id": "b_new", "kind": "blank",
                         "value_type": "text", "in_edges": [], "out_edges": []})


def test_add_node_wires_its_edges():
    new = apply_op(_plan(), parse_repair_op({
        "op": "add_node", "node_id": "b_new", "kind": "blank",
        "value_type": "text", "hint": "Which B?",
        "in_edges": [], "out_edges": [{"phrase": "relates_to", "object_id": "t"}]}))
    assert any(e.id == "b_new" for e in new.entities)
    assert any(v.subject_id == "b_new" and v.object_id == "t"
               for v in new.verb_phrases)


def test_set_target_moves_the_flag_and_leaves_exactly_one():
    new = apply_op(_plan(), parse_repair_op(
        {"op": "set_target", "node_id": "b_a"}))
    targets = [e.id for e in new.entities if getattr(e, "is_target", False)]
    assert targets == ["b_a"]


def test_modify_constraint_can_redirect_its_output():
    new = apply_op(_plan(), parse_repair_op({
        "op": "modify_constraint", "constraint_id": "c1",
        "changes": {"output_blank_id": "b_a"}}))
    assert new.constraints[0].output_blank_id == "b_a"


def test_delete_node_and_delete_constraint_are_destructive():
    assert is_destructive(parse_repair_op(
        {"op": "delete_node", "node_id": "b_a"}))
    assert is_destructive(parse_repair_op(
        {"op": "delete_constraint", "constraint_id": "c1"}))


def test_deleting_a_node_a_constraint_still_references_raises():
    with pytest.raises(RepairApplyError):
        apply_op(_plan(), parse_repair_op({"op": "delete_node", "node_id": "b_a"}))


# ---------------------------------------------------------------------------
# Review round 1, finding 1 (Critical): set_target / delete_node could leave
# a plan with zero target blanks, and nothing caught it -- GSWPlan.model_
# validate never reads is_target, so schema+topology revalidation sailed
# through. A verifier loop that accepts an op whenever binding violations
# strictly decrease would then accept "delete the target" as a fix for
# target_producible, on a plan it just made unanswerable (contract_
# violations goes 1 -> 0 because plan.target() itself starts raising, and
# the target_* checks catch that ExecutionError and return []). Finding
# 2/3: add_node/add_edge/add_constraint/modify_constraint never checked id
# collisions, so a later _entity/_vp/_constraint lookup could silently
# resolve to the wrong (first) instance.
# ---------------------------------------------------------------------------


def test_set_target_at_a_filled_entity_is_rejected():
    """SetTargetOp doesn't restrict node_id to a blank -- pointing it at a
    filled entity clears the real target's flag and sets one GSWPlan.
    target() never counts (it only looks at blank entities), leaving zero
    targets. _revalidate's target check must catch this."""
    plan = GSWPlan.model_validate({
        "entities": [
            {"id": "f1", "kind": "filled", "name": "Foo"},
            {"id": "b_a", "kind": "blank", "value_type": "text",
             "retrieval_hint": "Which A?"},
            {"id": "t", "kind": "blank", "value_type": "bool", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "subject_id": "f1", "phrase": "rel", "object_id": "b_a"},
            {"id": "vp2", "subject_id": "b_a", "phrase": "rel2", "object_id": "t"},
        ],
        "constraints": [],
    })
    with pytest.raises(RepairApplyError):
        apply_op(plan, parse_repair_op({"op": "set_target", "node_id": "f1"}))


def test_delete_node_of_the_target_is_rejected():
    """`_delete_node` routes through `_plan_diff.RemoveBlankOp` + `apply_
    diff`, which validates schema + topology but never target count --
    deleting the target blank passes both when nothing else strands, and
    used to return a target-less plan. `_delete_node` must route its
    result back through `_revalidate` (not just `apply_diff`'s own checks)
    to catch it. b_b keeps b_a referenced after t (and the only VP naming
    it) is gone, so this exercises the target check specifically, not the
    pre-existing dangling-entity check."""
    plan = GSWPlan.model_validate({
        "entities": [
            {"id": "b_a", "kind": "blank", "value_type": "text", "retrieval_hint": "A?"},
            {"id": "b_b", "kind": "blank", "value_type": "text", "retrieval_hint": "B?"},
            {"id": "t", "kind": "blank", "value_type": "bool", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "subject_id": "b_a", "phrase": "rel", "object_id": "b_b"},
            {"id": "vp2", "subject_id": "b_b", "phrase": "rel2", "object_id": "t"},
        ],
        "constraints": [],
    })
    with pytest.raises(RepairApplyError):
        apply_op(plan, parse_repair_op({"op": "delete_node", "node_id": "t"}))


def test_add_node_with_a_colliding_id_is_rejected():
    """GSWPlan doesn't enforce id uniqueness itself; a duplicate would let
    later ops resolve to whichever of the two entities comes first."""
    with pytest.raises(RepairApplyError):
        apply_op(_plan(), parse_repair_op({
            "op": "add_node", "node_id": "b_a", "kind": "blank",
            "value_type": "text", "in_edges": [],
            "out_edges": [{"phrase": "relates_to", "object_id": "t"}]}))


def test_modify_constraint_cannot_rename_the_id():
    """Renaming isn't a repair, and it's exactly how a duplicate constraint
    id could sneak in -- id is excluded from modify_constraint's allowed
    keys outright, not merely checked for collisions."""
    with pytest.raises(RepairApplyError):
        apply_op(_plan(), parse_repair_op({
            "op": "modify_constraint", "constraint_id": "c1",
            "changes": {"id": "c2"}}))
