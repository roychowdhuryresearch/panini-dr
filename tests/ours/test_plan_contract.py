# tests/ours/test_plan_contract.py
"""The contract: what a plan must satisfy before it may be dispatched."""
from panini_dr.adapters.ours._plan_contract import (
    ADVISORY, BINDING, contract_violations,
)
from panini_dr.adapters.ours._planner_exec import GSWPlan


def _plan(entities, constraints=None, verb_phrases=None) -> GSWPlan:
    return GSWPlan.model_validate({
        "entities": entities, "verb_phrases": verb_phrases or [],
        "constraints": constraints or []})


def _b(i, vt="text", target=False, hint=None):
    d = {"id": i, "kind": "blank", "role": "target" if target else "attribute",
         "value_type": vt}
    if target:
        d["is_target"] = True
    if hint:
        d["retrieval_hint"] = hint
    return d


def _f(i, name=None, literal_value=None):
    d = {"id": i, "kind": "filled"}
    if name is not None:
        d["name"] = name
    if literal_value is not None:
        d["literal_value"] = literal_value
    return d


def _diff(out="t"):
    return {"id": "c1", "kind": "derived", "op": "diff",
            "args_blanks": ["b_a", "b_b"], "output_blank_id": out}


def test_the_tiers_are_disjoint_and_declared():
    assert not (BINDING & ADVISORY)
    assert BINDING == {"ordering", "acyclic", "target_type", "target_producible"}
    assert ADVISORY == {"numeric_operands", "date_diff_granularity"}


def test_an_unordered_placeholder_violates_ordering():
    """b_pub is entity-typed and the subject of an edge to b_town, so the
    executor orders b_town AFTER b_pub -- yet b_pub's hint needs {b_town}."""
    plan = _plan([_b("b_pub", "entity", hint="Which pub is in {b_town}?"),
                  _b("b_town", "text", hint="Which town?"),
                  _b("t", "entity", True, hint="Who ran it?")],
                 verb_phrases=[
                     {"id": "vp1", "subject_id": "b_pub", "phrase": "located_in",
                      "object_id": "b_town"},
                     {"id": "vp2", "subject_id": "b_pub", "phrase": "run_by",
                      "object_id": "t"}])
    vs = contract_violations(plan)
    assert [v.clause for v in vs] == ["ordering"]
    assert vs[0].binding is True and vs[0].blank_id == "b_pub"
    assert "b_town" in vs[0].evidence


def test_the_TARGET_as_subject_is_not_a_violation():
    """`if sub.id == target_id: deps[sub.id].add(obj.id)` -- the target depends
    on its partner, never the reverse, so {b_nick} in the target's hint renders.
    Getting this backwards was the single largest false class in the deleted
    implementation."""
    plan = _plan([_b("b_nick", "text", hint="What was the nickname?"),
                  _b("t", "entity", True, hint="Who was known as {b_nick}?")],
                 verb_phrases=[{"id": "vp1", "subject_id": "t",
                                "phrase": "has_nickname", "object_id": "b_nick"}])
    assert contract_violations(plan) == []


def test_a_non_entity_subject_creates_no_edge_so_the_pair_is_unordered():
    """`elif sub.value_type == "entity"` -- a list-typed subject yields no
    ordering at all, so both sit at level 0 and the placeholder can be
    dispatched unrendered."""
    plan = _plan([_b("b_years", "list", hint="Which years?"),
                  _b("b_role", "text", hint="Which role in {b_years}?"),
                  _b("t", "entity", True, hint="Who?")],
                 verb_phrases=[
                     {"id": "vp1", "subject_id": "b_years", "phrase": "covers",
                      "object_id": "b_role"},
                     {"id": "vp2", "subject_id": "b_role", "phrase": "held_by",
                      "object_id": "t"}])
    assert [v.clause for v in contract_violations(plan)] == ["ordering"]


def test_a_constraint_writing_into_its_own_operand_violates_acyclic():
    plan = _plan([_b("b_a", "number"), _b("t", "number", True)],
                 constraints=[{"id": "c1", "kind": "derived", "op": "sum",
                               "args_blanks": ["b_a", "t"], "output_blank_id": "t"}])
    assert "acyclic" in [v.clause for v in contract_violations(plan)]


def test_a_boolean_written_into_a_text_target_violates_target_type():
    plan = _plan([_b("b_a", "text"), _b("b_b", "text"), _b("t", "text", True)],
                 constraints=[{"id": "c1", "kind": "equals", "left_ref": "b_a",
                               "right_ref": "b_b", "output_blank_id": "t"}])
    assert "target_type" in [v.clause for v in contract_violations(plan)]


def test_a_bool_target_is_fine():
    plan = _plan([_b("b_a", "text"), _b("b_b", "text"), _b("t", "bool", True)],
                 constraints=[{"id": "c1", "kind": "gt", "left_ref": "b_a",
                               "right_ref": "b_b", "output_blank_id": "t"}])
    assert contract_violations(plan) == []


def test_a_boolean_written_into_a_NON_target_blank_is_not_a_target_type_violation():
    """target_type is about THE target's declared type -- its repair (route
    the comparison into a fresh blank so the target keeps its type) is
    meaningless when the boolean-producing constraint writes somewhere else."""
    plan = _plan([_b("b_a", "text"), _b("b_b", "text"), _b("b_flag", "text"),
                  _b("t", "text", True, hint="who?")],
                 constraints=[{"id": "c1", "kind": "equals", "left_ref": "b_a",
                               "right_ref": "b_b", "output_blank_id": "b_flag"}],
                 verb_phrases=[{"id": "vp1", "subject_id": "b_flag",
                                "phrase": "rel", "object_id": "t"}])
    assert "target_type" not in [v.clause for v in contract_violations(plan)]


def test_a_target_with_no_producer_and_no_hint_violates_target_producible():
    plan = _plan([_b("b_a", "text", hint="something"), _b("t", "number", True)],
                 verb_phrases=[{"id": "vp1", "subject_id": "t", "phrase": "rel",
                                "object_id": "b_a"}])
    assert "target_producible" in [v.clause for v in contract_violations(plan)]


def test_a_target_with_only_a_hint_is_fine():
    """Most blanks are simply retrieved. Reporting these was the largest false
    class in one 50-question sample of the deleted implementation."""
    plan = _plan([_b("b_a", "entity", hint="x"),
                  _b("t", "entity", True, hint="who won?")],
                 verb_phrases=[{"id": "vp1", "subject_id": "b_a", "phrase": "rel",
                                "object_id": "t"}])
    assert contract_violations(plan) == []


def test_a_demoted_clause_is_still_reported_but_stops_binding():
    plan = _plan([_b("b_pub", "entity", hint="Which pub is in {b_town}?"),
                  _b("b_town", "text", hint="Which town?"),
                  _b("t", "entity", True, hint="Who ran it?")],
                 verb_phrases=[
                     {"id": "vp1", "subject_id": "b_pub", "phrase": "located_in",
                      "object_id": "b_town"},
                     {"id": "vp2", "subject_id": "b_pub", "phrase": "run_by",
                      "object_id": "t"}])
    vs = contract_violations(plan, demoted=frozenset({"ordering"}))
    assert [v.clause for v in vs] == ["ordering"]
    assert vs[0].binding is False


def test_a_clean_plan_has_nothing_to_say():
    plan = _plan([_b("b_v", "entity", hint="Which venue?"),
                  _b("t", "text", True, hint="Who led {b_v}?")],
                 verb_phrases=[{"id": "vp1", "subject_id": "b_v",
                                "phrase": "led_by", "object_id": "t"}])
    assert contract_violations(plan) == []


# --- date_diff_granularity (ADVISORY) -----------------------------------------
#
# `_date_diff_constraint_value` (_planner_exec.py:1111) picks exactly one of
# three outcomes for a `diff` over two date-typed blanks:
#   1. both operands carry a BIRTH cue -> a "N months and M days" STRING;
#   2. else a year cue (question or output hint/claim) AND the output typed
#      `number` -> completed years;
#   3. else -> FALLS THROUGH to raw year subtraction (month/day discarded);
#      it does NOT resolve unknown.
# Both sub-cases below report under the single ADVISORY clause name
# "date_diff_granularity" -- ADVISORY only declares that one string.

def test_a_date_diff_with_no_birth_cue_and_no_year_cue_degrades_to_year_subtraction():
    """q413's shape: nothing routes it, so it silently becomes year
    subtraction -- reported, but ADVISORY, so it never blocks dispatch."""
    plan = _plan([_b("b_a", "date", hint="When was the treaty signed?"),
                  _b("b_b", "date", hint="When was it ratified?"),
                  _b("t", "number", target=True)],
                 constraints=[_diff()])
    vs = contract_violations(plan, question="How long between the two events?")
    clauses = [v.clause for v in vs]
    assert "date_diff_granularity" in clauses
    v = next(v for v in vs if v.clause == "date_diff_granularity")
    assert v.binding is False
    assert "year" in v.evidence and "0" in v.evidence


def test_a_year_cue_in_the_question_routes_it_cleanly():
    plan = _plan([_b("b_a", "date", hint="When was A born?"),
                  _b("b_b", "date", hint="When was B born?"),
                  _b("t", "number", target=True)],
                 constraints=[_diff()])
    assert contract_violations(
        plan, question="How old was A when B was born?") == []


def test_a_year_cue_in_the_output_hint_routes_it_cleanly():
    plan = _plan([_b("b_a", "date", hint="When did it open?"),
                  _b("b_b", "date", hint="When did it close?"),
                  _b("t", "number", target=True,
                     hint="What was the age of the building at closure?")],
                 constraints=[_diff()])
    assert contract_violations(plan, question="How long did it stand?") == []


def test_a_year_cue_but_a_non_numeric_output_still_degrades():
    """Branch 2 requires the output typed `number`; a `text` target falls
    through to raw year subtraction even with a perfect year cue."""
    plan = _plan([_b("b_a", "date", hint="When was A born?"),
                  _b("b_b", "date", hint="When was B born?"),
                  _b("t", "text", target=True)],
                 constraints=[_diff()])
    vs = contract_violations(plan, question="How old was A when B was born?")
    assert [v.clause for v in vs] == ["date_diff_granularity"]
    assert vs[0].binding is False


def test_a_birth_cue_on_both_operands_routes_it_cleanly():
    """`_ref_has_birth_cue` reads the blank's ID (and incident verb-phrase
    text), NOT the retrieval_hint -- these ids carry "birth" directly."""
    plan = _plan([_b("b_birth_a", "date", hint="When was A born?"),
                  _b("b_birth_b", "date", hint="When was B born?"),
                  _b("t", "text", target=True)],
                 constraints=[{"id": "c1", "kind": "derived", "op": "diff",
                               "args_blanks": ["b_birth_a", "b_birth_b"],
                               "output_blank_id": "t"}])
    assert contract_violations(plan, question="What is the age gap?") == []


def test_a_question_asking_for_days_is_flagged_even_though_the_diff_routes_fine():
    """q125 shape: gold is "4149 days". The plan ROUTES fine (birth ids on
    both operands, so no degrade), but neither branch returns a day count."""
    plan = _plan([_b("b_birth_a", "date", hint="When was A born?"),
                  _b("b_birth_b", "date", hint="When was B born?"),
                  _b("t", "number", target=True)],
                 constraints=[{"id": "c1", "kind": "derived", "op": "diff",
                               "args_blanks": ["b_birth_a", "b_birth_b"],
                               "output_blank_id": "t"}])
    vs = contract_violations(
        plan, question="How many days elapsed between A and B?")
    assert [v.clause for v in vs] == ["date_diff_granularity"]
    assert vs[0].binding is False


def test_a_question_asking_for_weeks_is_flagged_too():
    plan = _plan([_b("b_a", "date", hint="When did it start?"),
                  _b("b_b", "date", hint="When did it end?"),
                  _b("t", "number", target=True)],
                 constraints=[_diff()])
    vs = contract_violations(
        plan, question="How many weeks was the age difference in years "
                       "between them?")
    assert [v.clause for v in vs] == ["date_diff_granularity"]


def test_a_question_asking_for_years_is_silent():
    plan = _plan([_b("b_a", "date", hint="When was A born?"),
                  _b("b_b", "date", hint="When was B born?"),
                  _b("t", "number", target=True)],
                 constraints=[_diff()])
    assert contract_violations(plan, question="How old was A in that year?") == []


def test_a_non_date_diff_is_untouched():
    """`diff` over numbers is ordinary subtraction and has no granularity."""
    plan = _plan([_b("b_a", "number", hint="How many days?"),
                  _b("b_b", "number", hint="How many more?"),
                  _b("t", "number", target=True)],
                 constraints=[_diff()])
    assert contract_violations(plan, question="How many days more?") == []


def test_the_granularity_subcase_stays_silent_without_a_question():
    """The DEGRADE sub-case needs no question (it is a plan-only fact); the
    GRANULARITY sub-case does. This plan routes cleanly (birth cue on both
    operands), so with no question given, neither sub-case has anything to
    report."""
    plan = _plan([_b("b_birth_a", "date", hint="When was A born?"),
                  _b("b_birth_b", "date", hint="When was B born?"),
                  _b("t", "number", target=True)],
                 constraints=[{"id": "c1", "kind": "derived", "op": "diff",
                               "args_blanks": ["b_birth_a", "b_birth_b"],
                               "output_blank_id": "t"}])
    assert contract_violations(plan) == []


# --- numeric_operands (ADVISORY) -----------------------------------------------
#
# `_coerce_number` (_planner_exec.py:723) hard-fails on a non-parseable
# string operand and silently kills the arithmetic chain. Operand-VERSUS-
# operand unit mismatches are already hard-rejected by
# `_planner_emit._reject_unit_mismatch_constraints`; this clause covers what
# that check does not: a single filled operand with no numeric value at all.

def test_a_filled_operand_with_no_numeric_value_violates_numeric_operands():
    plan = _plan([_b("b_year", "number", hint="Which year?"),
                  _f("e_tenth", name="nearest tenth"),
                  _b("t", "number", target=True)],
                 constraints=[{"id": "c1", "kind": "derived", "op": "sum",
                               "args_refs": ["b_year", "e_tenth"],
                               "output_blank_id": "t"}])
    vs = contract_violations(plan)
    assert [v.clause for v in vs] == ["numeric_operands"]
    assert vs[0].binding is False
    assert "e_tenth" in vs[0].evidence and "nearest tenth" in vs[0].evidence


def test_a_literal_value_operand_is_fine_even_when_its_name_is_words():
    """`literal_value` is checked FIRST -- it is what `_coerce_number`
    actually receives, regardless of how the entity is named."""
    plan = _plan([_b("b_year", "number", hint="Which year?"),
                  _f("e_twelve", name="twelve", literal_value=12),
                  _b("t", "number", target=True)],
                 constraints=[{"id": "c1", "kind": "derived", "op": "sum",
                               "args_refs": ["b_year", "e_twelve"],
                               "output_blank_id": "t"}])
    assert contract_violations(plan) == []


def test_a_digit_shaped_name_is_fine_without_a_literal_value():
    plan = _plan([_b("b_year", "number", hint="Which year?"),
                  _f("e_num", name="12"),
                  _b("t", "number", target=True)],
                 constraints=[{"id": "c1", "kind": "derived", "op": "sum",
                               "args_refs": ["b_year", "e_num"],
                               "output_blank_id": "t"}])
    assert contract_violations(plan) == []


def test_a_blank_operand_is_never_flagged_by_numeric_operands():
    """Only FILLED operands are checked -- a blank has no value yet to
    coerce; that is retrieval's job, not this clause's."""
    plan = _plan([_b("b_a", "number", hint="Which year?"),
                  _b("b_b", "number", hint="Which other year?"),
                  _b("t", "number", target=True)],
                 constraints=[{"id": "c1", "kind": "derived", "op": "sum",
                               "args_blanks": ["b_a", "b_b"],
                               "output_blank_id": "t"}])
    assert contract_violations(plan) == []


def test_a_non_arithmetic_constraint_kind_is_untouched():
    plan = _plan([_b("b_a", "text"), _f("e_words", name="not a number"),
                  _b("t", "bool", target=True)],
                 constraints=[{"id": "c1", "kind": "equals", "left_ref": "b_a",
                               "right_ref": "e_words", "output_blank_id": "t"}])
    assert contract_violations(plan) == []


def test_multiple_bad_operands_are_both_named_in_one_violation():
    plan = _plan([_f("e_a", name="nearest tenth"), _f("e_b", name="a dozen or so"),
                  _b("t", "number", target=True)],
                 constraints=[{"id": "c1", "kind": "derived", "op": "sum",
                               "args_refs": ["e_a", "e_b"],
                               "output_blank_id": "t"}])
    vs = contract_violations(plan)
    assert [v.clause for v in vs] == ["numeric_operands"]
    assert "e_a" in vs[0].evidence and "e_b" in vs[0].evidence
