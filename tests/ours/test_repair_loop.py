from panini_dr.adapters.ours._plan_repair_loop import verify_and_repair
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


def test_a_good_proposal_clears_the_contract():
    def propose(plan, violation, last_reject):
        return {"op": "reverse_edge", "vp_id": "vp1"}
    plan, residual, tr = verify_and_repair(_broken(), "", propose)
    assert [v for v in residual if v.binding] == []
    assert tr.violations_before == 1 and tr.violations_after == 0
    assert len(tr.ops_accepted) == 1


def test_the_reject_reason_is_fed_back_to_the_next_proposal():
    seen = []
    def propose(plan, violation, last_reject):
        seen.append(last_reject)
        return ({"op": "reverse_edge", "vp_id": "missing"} if len(seen) == 1
                else {"op": "reverse_edge", "vp_id": "vp1"})
    verify_and_repair(_broken(), "", propose)
    assert seen[0] is None
    assert seen[1] and "does not apply" in seen[1]


def test_three_rejections_demote_the_clause_and_the_loop_exits():
    def propose(plan, violation, last_reject):
        return {"op": "reverse_edge", "vp_id": "missing"}     # always rejected
    plan, residual, tr = verify_and_repair(_broken(), "", propose)
    assert tr.demoted == ["ordering"]
    assert len(tr.ops_rejected) == 3
    assert [v for v in residual if v.binding] == []   # demoted, so not blocking
    assert any(v.clause == "ordering" for v in residual)  # but still reported


def test_a_proposer_that_gives_up_demotes_immediately():
    def propose(plan, violation, last_reject):
        return None
    _, residual, tr = verify_and_repair(_broken(), "", propose)
    assert tr.demoted == ["ordering"] and tr.ops_rejected == []


def _two_violations() -> GSWPlan:
    """ordering (t's hint interpolates {b_year} at the same level) AND
    acyclic (c1 writes into x while also reading x) at once, unrelated to
    each other -- one blank apiece, no shared node or edge."""
    return GSWPlan.model_validate({
        "entities": [
            {"id": "b_year", "kind": "blank", "role": "attribute",
             "value_type": "date", "retrieval_hint": "When was it admitted?"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "text",
             "is_target": True, "retrieval_hint": "Monarch on {b_year}?"},
            {"id": "x", "kind": "blank", "role": "attribute", "value_type": "text"},
        ],
        "verb_phrases": [{"id": "vp1", "subject_id": "b_year",
                          "phrase": "monarch_on", "object_id": "t"}],
        "constraints": [{"id": "c1", "kind": "derived", "op": "concat",
                          "args_blanks": ["x"], "output_blank_id": "x"}],
    })


def test_ops_accepted_attributes_the_clause_the_op_actually_resolved():
    """A proposer that answers each violation with a fix for the OTHER one:
    asked about `ordering`, it repairs `acyclic` instead (dropping x's
    self-reference by pointing c1 at b_year). verify_op accepts that --
    2 binding violations -> 1 is a strict decrease and introduces nothing
    new -- even though it never touched `ordering`. The transcript must say
    the op resolved `acyclic`, not `ordering`, regardless of which
    violation prompted the attempt.
    """
    def propose(plan, violation, last_reject):
        if violation.clause == "ordering":
            return {"op": "modify_constraint", "constraint_id": "c1",
                    "changes": {"args_blanks": ["b_year"]}}
        return {"op": "reverse_edge", "vp_id": "vp1"}

    _, residual, tr = verify_and_repair(_two_violations(), "", propose)

    assert len(tr.ops_accepted) == 1
    accepted = tr.ops_accepted[0]
    assert accepted["prompted_by"] == "ordering"
    assert accepted["clauses_resolved"] == ["acyclic"]
    # `ordering` was never actually fixed -- it exhausts its 3 attempts
    # (the proposer only ever offers the acyclic-fix in response to it) and
    # demotes, exactly as the standalone demotion tests above exercise.
    assert tr.demoted == ["ordering"]
    assert [v for v in residual if v.binding] == []
    assert any(v.clause == "ordering" for v in residual)


def test_a_clean_plan_never_calls_the_proposer():
    calls = []
    def propose(plan, violation, last_reject):
        calls.append(1)
        return None
    clean = GSWPlan.model_validate({
        "entities": [
            {"id": "b_v", "kind": "blank", "role": "attribute",
             "value_type": "entity", "retrieval_hint": "Which venue?"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "text",
             "is_target": True, "retrieval_hint": "Who led {b_v}?"},
        ],
        "verb_phrases": [{"id": "vp1", "subject_id": "b_v", "phrase": "led_by",
                          "object_id": "t"}],
        "constraints": [],
    })
    verify_and_repair(clean, "", propose)
    assert calls == []
