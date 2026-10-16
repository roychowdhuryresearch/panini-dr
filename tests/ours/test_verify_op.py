from panini_dr.adapters.ours._plan_repair_loop import verify_op
from panini_dr.adapters.ours._plan_repair_ops import parse_repair_op
from panini_dr.adapters.ours._planner_exec import GSWPlan


def _broken() -> GSWPlan:
    return GSWPlan.model_validate({
        "entities": [
            {"id": "b_year", "kind": "blank", "role": "attribute",
             "value_type": "date", "retrieval_hint": "When was it admitted?"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "text",
             "is_target": True, "retrieval_hint": "Monarch on {b_year}?"},
        ],
        "verb_phrases": [{"id": "vp1", "subject_id": "b_year",
                          "phrase": "monarch_on", "object_id": "t"}],
        "constraints": [],
    })


def test_an_op_that_clears_the_violation_is_accepted():
    v = verify_op(_broken(), parse_repair_op(
        {"op": "reverse_edge", "vp_id": "vp1"}), question="")
    assert v.accept is True
    assert v.plan is not None


def test_an_op_that_changes_nothing_is_rejected():
    """Not strictly decreasing is a rejection: it is how the loop terminates."""
    v = verify_op(_broken(), parse_repair_op(
        {"op": "set_hint", "node_id": "b_year", "text": "When was it admitted?"}),
        question="")
    assert v.accept is False
    assert "decrease" in v.reason.lower()


def test_an_op_that_breaks_validation_is_rejected():
    """The op parses and applies mechanically, but the result fails the
    post-apply schema check (`_check_known_refs`): `add_edge` is not
    id-checked by `apply_op` itself, so an edge to a nonexistent node makes
    it all the way to `_revalidate`'s `GSWPlan.model_validate`, which is
    where it actually fails.

    (The brief's original op here was `{"op": "reverse_edge",
    "vp_id": "missing"}`, selected via a dead `... if False else ...`
    conditional — that op fails at the pre-apply `_vp` lookup, i.e. "op does
    not apply", not "application fails validation", and duplicates
    `test_the_reason_is_always_populated_on_reject` below. Swapped for an op
    that actually exercises the named scenario.)
    """
    v = verify_op(_broken(), parse_repair_op(
        {"op": "add_edge", "subject_id": "b_year", "phrase": "extra_edge",
         "object_id": "nonexistent"}), question="")
    assert v.accept is False
    assert v.plan is None


def test_a_destructive_op_whose_text_lives_nowhere_else_is_rejected():
    v = verify_op(_broken(), parse_repair_op(
        {"op": "set_hint", "node_id": "b_year", "text": ""}), question="")
    assert v.accept is False
    assert "destructive" in v.reason.lower()


def test_the_reason_is_always_populated_on_reject():
    v = verify_op(_broken(), parse_repair_op(
        {"op": "reverse_edge", "vp_id": "missing"}), question="")
    assert v.accept is False and v.reason.strip()


def _broken_with_decoy() -> GSWPlan:
    """Two blanks, `b_year` and `decoy`, carrying the IDENTICAL hint text.

    Both reference `t` from the same dependency level, so both trip
    `ordering`; `t`'s own hint references `b_year` from the same level too
    -- 3 binding violations total. Built to reproduce review finding 1
    (task 4, round 2): a plan-wide text pool let clearing `b_year`'s hint
    look survivable because `decoy` happened to carry the same string.
    """
    return GSWPlan.model_validate({
        "entities": [
            {"id": "b_year", "kind": "blank", "role": "attribute",
             "value_type": "date", "retrieval_hint": "Who ruled after {t}?"},
            {"id": "decoy", "kind": "blank", "role": "attribute",
             "value_type": "date", "retrieval_hint": "Who ruled after {t}?"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "text",
             "is_target": True, "retrieval_hint": "Monarch on {b_year}?"},
        ],
        "verb_phrases": [
            {"id": "vp1", "subject_id": "b_year", "phrase": "monarch_on", "object_id": "t"},
            {"id": "vp2", "subject_id": "decoy", "phrase": "monarch_on2", "object_id": "t"},
        ],
        "constraints": [],
    })


def test_a_destructive_op_is_rejected_even_when_a_different_node_shares_its_text():
    """Review finding 1 (task 4, round 2): the deletion test must be
    entity-scoped. Clearing `b_year`'s hint drops the binding count 3 -> 2
    (its own `ordering` violation disappears) with no new clause -- on the
    old plan-wide `_text_pool` check this was wrongly ACCEPTED, because
    `decoy` happens to carry the identical hint string and the pool never
    asked *which* node the survivor lived on. `b_year` itself ends up with
    no retrieval guidance of its own; that must be rejected regardless of
    what `decoy` carries.
    """
    v = verify_op(_broken_with_decoy(), parse_repair_op(
        {"op": "set_hint", "node_id": "b_year", "text": ""}), question="")
    assert v.accept is False
    assert "destructive" in v.reason.lower()
    assert "b_year" in v.reason


def _retarget_plan() -> GSWPlan:
    """`set_target` moves the target from `t` to `t2`.

    Before: `t2`'s hint interpolates `{b_link}` and `{c_link}`, both at
    `t2`'s own dependency level (0) -- 2 `ordering` violations, both on
    `t2`. Constraint `c2` (kind=`equals`) already writes into `t2`, but
    `target_type` only inspects constraints that output into the CURRENT
    target, so it is invisible while `t` is target.

    After `set_target` to `t2`: the target-is-a-sink rule in
    `build_dependency_graph` makes `t2` depend on `b_link`/`c_link` (it is
    now the VP subject on the target side), which pushes `t2` to level 1
    and clears both `ordering` violations. But `c2` now writes a boolean
    into `t2`, whose `value_type` is `text` -- a fresh `target_type`
    violation that did not exist before. Net: 2 binding violations -> 1
    (strictly decreasing) while the clause SET gains `target_type`.
    """
    return GSWPlan.model_validate({
        "entities": [
            {"id": "t", "kind": "blank", "role": "target", "value_type": "bool",
             "is_target": True, "retrieval_hint": "Some fact?"},
            {"id": "t2", "kind": "blank", "role": "attribute", "value_type": "text",
             "is_target": False,
             "retrieval_hint": "Combine {b_link} and {c_link}?"},
            {"id": "b_link", "kind": "blank", "role": "attribute", "value_type": "date"},
            {"id": "c_link", "kind": "blank", "role": "attribute", "value_type": "date"},
            {"id": "lit1", "kind": "filled", "name": "one", "value_type": "number",
             "literal_value": 1},
            {"id": "lit2", "kind": "filled", "name": "two", "value_type": "number",
             "literal_value": 2},
        ],
        "verb_phrases": [
            {"id": "vp_t2_blink", "subject_id": "t2", "phrase": "rel1", "object_id": "b_link"},
            {"id": "vp_t2_clink", "subject_id": "t2", "phrase": "rel2", "object_id": "c_link"},
            {"id": "vp_t_blink", "subject_id": "b_link", "phrase": "rel3", "object_id": "t"},
        ],
        "constraints": [
            {"id": "c2", "kind": "equals", "output_blank_id": "t2",
             "left_ref": "lit1", "right_ref": "lit2"},
        ],
    })


def test_a_decrease_that_introduces_a_new_binding_clause_is_rejected():
    """Review finding 2 (task 4, round 2): the "no new binding clause"
    condition had no regression test -- disabling it left all five
    originally-shipped tests green. `set_target` here strictly decreases
    the binding count (2 `ordering` -> 1 `target_type`) yet swaps which
    clause is failing; a gate that only checks the count would wrongly
    ACCEPT a plan that traded two ordering violations for a target it can
    no longer produce the right type for.
    """
    v = verify_op(_retarget_plan(), parse_repair_op(
        {"op": "set_target", "node_id": "t2"}), question="")
    assert v.accept is False
    assert "target_type" in v.reason
