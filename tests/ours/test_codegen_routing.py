"""Task 4: route _compute_constraint through the skill library behind the flag.

Flag OFF (codegen=None) -> byte-identical (year-collapse). Flag ON -> the
compute step is served by the library: day-granular date diff (q413-like) and
birthday-aware age (q209/q25-like).
"""

from panini_dr.adapters.ours._planner_exec import (
    BlankResult,
    Constraint,
    Entity,
    GSWPlan,
    VerbPhrase,
    _compute_constraint,
)
from panini_dr.adapters.ours.codegen_library import registry
from panini_dr.adapters.ours.codegen_library.runtime import CodegenRuntime


def _runtime(question=""):
    return CodegenRuntime(
        enabled=True,
        lib=registry.Library.load(),
        question=question,
    )


def _date_diff_plan():
    return GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="album_a"),
            Entity(id="e2", kind="filled", name="album_b"),
            Entity(id="b1", kind="blank", value_type="date"),
            Entity(id="b2", kind="blank", value_type="date"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="released_on", subject_id="e1", object_id="b1"),
            VerbPhrase(id="vp2", phrase="released_on", subject_id="e2", object_id="b2"),
        ],
        constraints=[
            Constraint(id="c1", kind="derived", op="diff",
                       args_refs=["b1", "b2"], output_blank_id="t"),
        ],
    )


def _state(b1, b2):
    return {
        "b1": BlankResult(blank_id="b1", value=b1, status="resolved"),
        "b2": BlankResult(blank_id="b2", value=b2, status="resolved"),
    }


def test_date_diff_off_path_year_collapse():
    # Flag OFF: both dates collapse to year 2015 -> abs(2015-2015)=0 (the bug).
    plan = _date_diff_plan()
    res = _compute_constraint(plan.constraints[0], plan, _state("2015-03-15", "2015-03-20"))
    assert res.status == "resolved"
    assert res.value == 0


def test_date_diff_on_path_day_granular():
    # Flag ON: day-granular diff -> 5 (fixes q413's collapse-to-0 ties).
    plan = _date_diff_plan()
    rt = _runtime()
    res = _compute_constraint(plan.constraints[0], plan, _state("2015-03-15", "2015-03-20"), codegen=rt)
    assert res.status == "resolved"
    assert res.value == 5


def test_date_diff_on_path_distinct_gaps():
    # q413: two same-year candidate gaps must be DISTINCT on-path (5 vs 139),
    # whereas off-path both collapse to 0 (ties -> wrong pick).
    plan = _date_diff_plan()
    rt = _runtime()
    a = _compute_constraint(plan.constraints[0], plan, _state("2015-03-15", "2015-03-20"), codegen=rt)
    b = _compute_constraint(plan.constraints[0], plan, _state("2015-03-15", "2015-08-01"), codegen=rt)
    assert a.value == 5
    assert b.value == 139
    assert a.value != b.value
    # off-path both collapse to 0
    off_a = _compute_constraint(plan.constraints[0], plan, _state("2015-03-15", "2015-03-20"))
    off_b = _compute_constraint(plan.constraints[0], plan, _state("2015-03-15", "2015-08-01"))
    assert off_a.value == off_b.value == 0


def _age_plan():
    return GSWPlan(
        entities=[
            Entity(id="p", kind="filled", name="person"),
            Entity(id="b_birth", kind="blank", value_type="date"),
            Entity(id="b_event", kind="blank", value_type="date"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="born_in", subject_id="p", object_id="b_birth"),
            VerbPhrase(id="vp2", phrase="assassinated_on", subject_id="p", object_id="b_event"),
        ],
        constraints=[
            Constraint(id="c1", kind="derived", op="diff",
                       args_refs=["b_birth", "b_event"], output_blank_id="t"),
        ],
    )


def _age_state(birth, event):
    return {
        "b_birth": BlankResult(blank_id="b_birth", value=birth, status="resolved"),
        "b_event": BlankResult(blank_id="b_event", value=event, status="resolved"),
    }


def test_age_on_path_birthday_aware_before():
    # As-of date before the birthday -> 54 completed years (not year-diff 55).
    plan = _age_plan()
    res = _compute_constraint(plan.constraints[0], plan, _age_state("1965-06-30", "2020-02-22"),
                              codegen=_runtime())
    assert res.status == "resolved"
    assert res.value == 54


def test_age_on_path_birthday_aware_after():
    # As-of date after the birthday -> 55.
    plan = _age_plan()
    res = _compute_constraint(plan.constraints[0], plan, _age_state("1965-06-30", "2020-07-01"),
                              codegen=_runtime())
    assert res.value == 55


def test_age_arg_order_independent():
    # Constraint may present the dates in either order; result is stable.
    plan = _age_plan()
    res = _compute_constraint(plan.constraints[0], plan, _age_state("2020-02-22", "1965-06-30"),
                              codegen=_runtime())
    assert res.value == 54


def test_codegen_disabled_runtime_is_off_path():
    # A CodegenRuntime with enabled=False must behave exactly like flag-off.
    plan = _date_diff_plan()
    rt = CodegenRuntime(enabled=False, lib=registry.Library.load())
    res = _compute_constraint(plan.constraints[0], plan, _state("2015-03-15", "2015-03-20"), codegen=rt)
    assert res.value == 0


def _age_via_output_hint_plan(out_hint):
    """diff(event_date, birth_date) with the AGE intent expressed only in the
    output blank's retrieval_hint (not via input event verbs)."""
    return GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="person"),
            Entity(id="b_event", kind="blank", value_type="date",
                   retrieval_hint="On what date did they lose?"),
            Entity(id="b_birth", kind="blank", value_type="date",
                   retrieval_hint="When was the person born?"),
            Entity(id="t", kind="blank", value_type="number", is_target=True,
                   retrieval_hint=out_hint),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="lost_on", subject_id="e1", object_id="b_event"),
            VerbPhrase(id="vp2", phrase="born_on", subject_id="e1", object_id="b_birth"),
        ],
        constraints=[
            Constraint(id="c1", kind="derived", op="diff",
                       args_refs=["b_event", "b_birth"], output_blank_id="t"),
        ],
    )


def test_diff_routes_to_years_via_output_how_old_hint():
    # q25-like: event verb "lost" isn't in the input-cue list, but the output
    # hint says "how old" -> must compute completed YEARS, not days.
    plan = _age_via_output_hint_plan("How old was the person on the loss date?")
    state = {
        "b_event": BlankResult(blank_id="b_event", value="1990-02-11", status="resolved"),
        "b_birth": BlankResult(blank_id="b_birth", value="1966-06-30", status="resolved"),
    }
    res = _compute_constraint(plan.constraints[0], plan, state, _runtime("How old..."))
    assert res.value == 23  # 1966-06-30 -> 1990-02-11 = 23 completed years (not ~8626 days)


def test_diff_routes_to_years_via_output_number_of_years_hint():
    # q630-like: org founding -> event, output hint "number of years between".
    plan = _age_via_output_hint_plan("Calculate the number of years between the two dates")
    state = {
        "b_event": BlankResult(blank_id="b_event", value="1969-12-31", status="resolved"),
        "b_birth": BlankResult(blank_id="b_birth", value="1947-09-18", status="resolved"),
    }
    res = _compute_constraint(plan.constraints[0], plan, state, _runtime("years"))
    assert res.value == 22  # ~22 years, not ~8140 days
