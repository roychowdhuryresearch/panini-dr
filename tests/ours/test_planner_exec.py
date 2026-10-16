"""Unit tests for the GSW-fragment plan schema, dependency DAG and
constraint computation (the pure-Python layers; no LLM or retriever).
"""

from __future__ import annotations

from typing import Any

import pytest

from panini_dr.adapters.ours._planner_exec import (
    BlankResult,
    Constraint,
    Entity,
    ExecutionError,
    GSWPlan,
    VerbPhrase,
    _compute_constraint,
    _coerce_value,
    _stringify,
    build_dependency_graph,
    topological_sort_blanks,
    unresolved_upstream_deps,
)
from panini_dr.adapters.ours.gsw_planner_react_v1 import _cascade_auto_compute


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------


def test_topo_sort_simple_chain():
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="Pablo_Picasso"),
            Entity(id="e2", kind="filled", name="Pink_Floyd"),
            Entity(id="b1", kind="blank", value_type="date"),
            Entity(id="b2", kind="blank", value_type="entity", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="died_in", subject_id="e1", object_id="b1"),
            VerbPhrase(id="vp2", phrase="released_in_year", subject_id="e2", object_id="b1"),
            VerbPhrase(id="vp3", phrase="released", subject_id="e2", object_id="b2"),
        ],
        constraints=[],
    )
    order = topological_sort_blanks(plan)
    # b1 has no constraint deps; b2 has no constraint deps either (VP-only).
    # Both end up at indeg 0 here; whatever order is fine.
    assert set(order) == {"b1", "b2"}


def test_topo_sort_diff_chain():
    # Example B: Portland ME/OR population diff.
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="Portland_Maine"),
            Entity(id="e2", kind="filled", name="Portland_Oregon"),
            Entity(id="b1", kind="blank", value_type="number"),
            Entity(id="b2", kind="blank", value_type="number"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="has_population", subject_id="e1", object_id="b1"),
            VerbPhrase(id="vp2", phrase="has_population", subject_id="e2", object_id="b2"),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="derived",
                op="diff",
                args_blanks=["b2", "b1"],
                output_blank_id="t",
            )
        ],
    )
    order = topological_sort_blanks(plan)
    # b1 and b2 must appear before t.
    assert order.index("b1") < order.index("t")
    assert order.index("b2") < order.index("t")


def test_topo_sort_argmax_chain():
    # Example C: missile argmax.
    entities = [
        Entity(id=f"e{i}", kind="filled", name=f"missile_{i}") for i in range(1, 5)
    ]
    blanks = [
        Entity(id=f"b{i}", kind="blank", value_type="date") for i in range(1, 5)
    ]
    target = Entity(id="t", kind="blank", value_type="entity", is_target=True)
    vps = [
        VerbPhrase(
            id=f"vp{i}",
            phrase="entered_service_in",
            subject_id=f"e{i}",
            object_id=f"b{i}",
        )
        for i in range(1, 5)
    ]
    plan = GSWPlan(
        entities=entities + blanks + [target],
        verb_phrases=vps,
        constraints=[
            Constraint(
                id="c1",
                kind="argmax",
                candidate_entity_ids=[f"e{i}" for i in range(1, 5)],
                sort_by_blank_ids=[f"b{i}" for i in range(1, 5)],
                output_blank_id="t",
            )
        ],
    )
    order = topological_sort_blanks(plan)
    for b in ("b1", "b2", "b3", "b4"):
        assert order.index(b) < order.index("t")


def test_topo_sort_projection_chain():
    # Example D: b_person (identify) → b_birth_year, b_death_year (project).
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="NYC"),
            Entity(id="e2", kind="filled", name="Catwoman"),
            Entity(id="b_person", kind="blank", value_type="entity"),
            Entity(id="b_birth", kind="blank", value_type="date"),
            Entity(id="b_death", kind="blank", value_type="date"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="born_in", subject_id="b_person", object_id="e1"),
            VerbPhrase(id="vp2", phrase="created", subject_id="b_person", object_id="e2"),
            VerbPhrase(
                id="vp3", phrase="born_in_year", subject_id="b_person", object_id="b_birth"
            ),
            VerbPhrase(
                id="vp4", phrase="died_in_year", subject_id="b_person", object_id="b_death"
            ),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="derived",
                op="diff",
                args_blanks=["b_death", "b_birth"],
                output_blank_id="t",
            )
        ],
    )
    order = topological_sort_blanks(plan)
    # b_person (entity blank) must come before b_birth and b_death
    # (projection VPs). b_birth and b_death must come before t.
    assert order.index("b_person") < order.index("b_birth")
    assert order.index("b_person") < order.index("b_death")
    assert order.index("b_birth") < order.index("t")
    assert order.index("b_death") < order.index("t")


def test_topo_sort_cycle_raises():
    plan = GSWPlan(
        entities=[
            Entity(id="b1", kind="blank", value_type="number"),
            Entity(id="b2", kind="blank", value_type="number", is_target=True),
        ],
        constraints=[
            Constraint(
                id="c1", kind="derived", op="sum", args_blanks=["b2"], output_blank_id="b1"
            ),
            Constraint(
                id="c2", kind="derived", op="sum", args_blanks=["b1"], output_blank_id="b2"
            ),
        ],
    )
    with pytest.raises(ExecutionError) as excinfo:
        topological_sort_blanks(plan)
    assert excinfo.value.kind == "cyclic_plan"


# ---------------------------------------------------------------------------
# constraint computation — derived ops
# ---------------------------------------------------------------------------


def _constraint_state(**values: Any) -> dict[str, BlankResult]:
    return {
        bid: BlankResult(blank_id=bid, value=value, status="resolved")
        for bid, value in values.items()
    }


def test_compute_constraint_mul_div_round_nearest():
    plan = GSWPlan(
        entities=[
            Entity(id="b_a", kind="blank", value_type="number"),
            Entity(id="b_b", kind="blank", value_type="number"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        constraints=[
            Constraint(
                id="c_mul",
                kind="derived",
                op="mul",
                args_blanks=["b_a", "b_b"],
                output_blank_id="t",
            )
        ],
    )
    state = _constraint_state(b_a=6, b_b=7)
    res = _compute_constraint(plan.constraints[0], plan, state)
    assert res.status == "resolved"
    assert res.value == 42

    div = Constraint(
        id="c_div",
        kind="derived",
        op="div",
        args_blanks=["b_a", "b_b"],
        output_blank_id="t",
    )
    assert _compute_constraint(div, plan, _constraint_state(b_a=84, b_b=2)).value == 42

    round_plan = GSWPlan(
        entities=[
            Entity(id="b_value", kind="blank", value_type="number"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        constraints=[
            Constraint(
                id="c_round",
                kind="derived",
                op="round_nearest",
                args_blanks=["b_value"],
                output_blank_id="t",
            )
        ],
    )
    rounded = _compute_constraint(
        round_plan.constraints[0],
        round_plan,
        _constraint_state(b_value=28.026),
    )
    assert rounded.value == 28

    nearest_ten_plan = GSWPlan(
        entities=[
            Entity(id="b_value", kind="blank", value_type="number"),
            Entity(
                id="e_ten",
                kind="filled",
                name="ten",
                role="constraint-value",
                literal_value=10,
            ),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        constraints=[
            Constraint(
                id="c_round",
                kind="derived",
                op="round_nearest",
                args_refs=["b_value", "e_ten"],
                output_blank_id="t",
            )
        ],
    )
    rounded_ten = _compute_constraint(
        nearest_ten_plan.constraints[0],
        nearest_ten_plan,
        _constraint_state(b_value=86),
    )
    assert rounded_ten.value == 90


def _single_list_operand_plan(op: str) -> GSWPlan:
    return GSWPlan(
        entities=[
            Entity(id="b_values", kind="blank", value_type="list"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        constraints=[
            Constraint(
                id="c_agg",
                kind="derived",
                op=op,
                args_blanks=["b_values"],
                output_blank_id="t",
            )
        ],
    )


def test_compute_constraint_avg_over_single_list_operand():
    # FIX 1: avg(b_attendance_list) — a plan whose sole operand resolved to
    # a LIST used to silently no-op (coercer returns None on a bare list ->
    # unknown). The list is the population to average over its elements.
    plan = _single_list_operand_plan("avg")
    res = _compute_constraint(
        plan.constraints[0],
        plan,
        _constraint_state(b_values=[2, 4, 6]),
    )
    assert res.status == "resolved"
    assert res.value == 4.0


def test_compute_constraint_min_over_single_list_operand():
    plan = _single_list_operand_plan("min")
    res = _compute_constraint(
        plan.constraints[0],
        plan,
        _constraint_state(b_values=[5, 1, 9]),
    )
    assert res.status == "resolved"
    assert res.value == 1


def test_compute_constraint_max_over_single_list_operand():
    plan = _single_list_operand_plan("max")
    res = _compute_constraint(
        plan.constraints[0],
        plan,
        _constraint_state(b_values=[5, 1, 9]),
    )
    assert res.status == "resolved"
    assert res.value == 9


def test_compute_constraint_div_with_list_operand_stays_unknown():
    # div/mul are explicitly OUT of scope for the list-population fix: a
    # list operand under div is a planner error and must not be silently
    # summed/averaged away.
    plan = GSWPlan(
        entities=[
            Entity(id="b_values", kind="blank", value_type="list"),
            Entity(id="b_denom", kind="blank", value_type="number"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        constraints=[
            Constraint(
                id="c_div",
                kind="derived",
                op="div",
                args_blanks=["b_values", "b_denom"],
                output_blank_id="t",
            )
        ],
    )
    res = _compute_constraint(
        plan.constraints[0],
        plan,
        _constraint_state(b_values=[2, 4, 6], b_denom=2),
    )
    assert res.status == "unknown"


def test_compute_constraint_avg_over_empty_list_stays_unknown():
    # avg of nothing is not 0 -- must stay unknown, not silently become 0.
    plan = _single_list_operand_plan("avg")
    res = _compute_constraint(
        plan.constraints[0],
        plan,
        _constraint_state(b_values=[]),
    )
    assert res.status == "unknown"


def test_compute_constraint_avg_over_list_with_bad_element_stays_unknown():
    # A single uncoercible element must not be silently dropped -- that
    # would change the answer instead of surfacing the failure.
    plan = _single_list_operand_plan("avg")
    res = _compute_constraint(
        plan.constraints[0],
        plan,
        _constraint_state(b_values=[2, "not-a-number", 6]),
    )
    assert res.status == "unknown"


def test_compute_count_constraint_counts_title_letters_when_plan_says_length():
    plan = GSWPlan(
        entities=[
            Entity(id="b_title", kind="blank", value_type="text"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(
                id="vp_len",
                phrase="title_length",
                subject_id="b_title",
                object_id="t",
            ),
        ],
        constraints=[
            Constraint(
                id="c_count",
                kind="derived",
                op="count",
                args_refs=["b_title"],
                output_blank_id="t",
            ),
        ],
    )
    res = _compute_constraint(
        plan.constraints[0],
        plan,
        _constraint_state(b_title="The Young Americans"),
    )
    assert res.status == "resolved"
    assert res.value == 17


def test_compute_diff_between_birth_dates_preserves_month_day_granularity():
    plan = GSWPlan(
        entities=[
            Entity(id="b_person_a", kind="blank", value_type="entity"),
            Entity(id="b_person_b", kind="blank", value_type="entity"),
            Entity(id="b_birth_a", kind="blank", value_type="date"),
            Entity(id="b_birth_b", kind="blank", value_type="date"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(
                id="vp_a",
                phrase="born_on",
                subject_id="b_person_a",
                object_id="b_birth_a",
            ),
            VerbPhrase(
                id="vp_b",
                phrase="born_on",
                subject_id="b_person_b",
                object_id="b_birth_b",
            ),
        ],
        constraints=[
            Constraint(
                id="c_diff",
                kind="derived",
                op="diff",
                args_refs=["b_birth_a", "b_birth_b"],
                output_blank_id="t",
            ),
        ],
    )
    res = _compute_constraint(
        plan.constraints[0],
        plan,
        _constraint_state(b_birth_a="1916-11-04", b_birth_b="1916-04-09"),
    )
    assert res.status == "resolved"
    assert res.value == "6 months 26 days"


# ---------------------------------------------------------------------------
# q209 fix 3 — SAFE floor-to-years post-processor. NOT the a95281c3 branch
# reorder (reverted in 011f4bb3): the birth-birth branch inside
# _date_diff_constraint_value is left untouched (still returns the "N
# months M days" duration string first), and this fires strictly AFTER
# that string is produced, only when the OUTPUT wants a number of elapsed
# years. A plan shape that never reaches the birth-birth string branch
# (e.g. a decomposed plan with two separately-computed ages) can never
# trigger this post-processor either.
# ---------------------------------------------------------------------------


def test_compute_diff_between_birth_dates_with_numeric_years_target_floors_to_whole_years():
    plan = GSWPlan(
        entities=[
            Entity(id="b_person_a", kind="blank", value_type="entity"),
            Entity(id="b_person_b", kind="blank", value_type="entity"),
            Entity(id="b_birth_a", kind="blank", value_type="date"),
            Entity(id="b_birth_b", kind="blank", value_type="date"),
            Entity(
                id="t",
                kind="blank",
                value_type="number",
                is_target=True,
                verification_claim=(
                    "The age difference between the two people, in years."
                ),
            ),
        ],
        verb_phrases=[
            VerbPhrase(id="vp_a", phrase="born_on", subject_id="b_person_a", object_id="b_birth_a"),
            VerbPhrase(id="vp_b", phrase="born_on", subject_id="b_person_b", object_id="b_birth_b"),
        ],
        constraints=[
            Constraint(
                id="c_diff",
                kind="derived",
                op="diff",
                args_refs=["b_birth_a", "b_birth_b"],
                output_blank_id="t",
            ),
        ],
    )
    # 1940-11-13 -> 1948-01-19 is exactly "86 months 6 days" (verified by
    # _format_month_day_delta); floored whole years = 86 // 12 = 7.
    res = _compute_constraint(
        plan.constraints[0],
        plan,
        _constraint_state(b_birth_a="1940-11-13", b_birth_b="1948-01-19"),
    )
    assert res.status == "resolved"
    assert res.value == 7
    assert not isinstance(res.value, str)


def test_compute_diff_between_birth_dates_with_text_target_keeps_duration_string():
    """Same birth-birth pair, but the output blank is NOT numeric -- the
    duration string must be left exactly as-is, qualifier or not."""
    plan = GSWPlan(
        entities=[
            Entity(id="b_person_a", kind="blank", value_type="entity"),
            Entity(id="b_person_b", kind="blank", value_type="entity"),
            Entity(id="b_birth_a", kind="blank", value_type="date"),
            Entity(id="b_birth_b", kind="blank", value_type="date"),
            Entity(
                id="t",
                kind="blank",
                value_type="text",
                is_target=True,
                verification_claim="The age difference between the two people, in years.",
            ),
        ],
        verb_phrases=[
            VerbPhrase(id="vp_a", phrase="born_on", subject_id="b_person_a", object_id="b_birth_a"),
            VerbPhrase(id="vp_b", phrase="born_on", subject_id="b_person_b", object_id="b_birth_b"),
        ],
        constraints=[
            Constraint(
                id="c_diff",
                kind="derived",
                op="diff",
                args_refs=["b_birth_a", "b_birth_b"],
                output_blank_id="t",
            ),
        ],
    )
    res = _compute_constraint(
        plan.constraints[0],
        plan,
        _constraint_state(b_birth_a="1940-11-13", b_birth_b="1948-01-19"),
    )
    assert res.status == "resolved"
    assert res.value == "86 months 6 days"


def test_compute_diff_between_birth_dates_numeric_target_without_years_cue_stays_string():
    """Conservative gate: value_type=='number' alone is not enough -- an
    explicit years cue must also be present, matching the same
    _wants_elapsed_years_diff / _output_wants_years_diff signals used
    elsewhere in this module."""
    plan = GSWPlan(
        entities=[
            Entity(id="b_person_a", kind="blank", value_type="entity"),
            Entity(id="b_person_b", kind="blank", value_type="entity"),
            Entity(id="b_birth_a", kind="blank", value_type="date"),
            Entity(id="b_birth_b", kind="blank", value_type="date"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp_a", phrase="born_on", subject_id="b_person_a", object_id="b_birth_a"),
            VerbPhrase(id="vp_b", phrase="born_on", subject_id="b_person_b", object_id="b_birth_b"),
        ],
        constraints=[
            Constraint(
                id="c_diff",
                kind="derived",
                op="diff",
                args_refs=["b_birth_a", "b_birth_b"],
                output_blank_id="t",
            ),
        ],
    )
    res = _compute_constraint(
        plan.constraints[0],
        plan,
        _constraint_state(b_birth_a="1940-11-13", b_birth_b="1948-01-19"),
    )
    assert res.status == "resolved"
    assert res.value == "86 months 6 days"


def test_compute_diff_between_full_dates_uses_elapsed_years():
    plan = GSWPlan(
        entities=[
            Entity(id="e_state", kind="filled", name="Commonwealth of Pennsylvania"),
            Entity(id="e_rep", kind="filled", name="Robert D. Heaton"),
            Entity(id="b_start", kind="blank", value_type="date"),
            Entity(id="b_end", kind="blank", value_type="date"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(
                id="vp_admit",
                phrase="admitted_in_year",
                subject_id="e_state",
                object_id="b_start",
            ),
            VerbPhrase(
                id="vp_birth",
                phrase="born_in_year",
                subject_id="e_rep",
                object_id="b_end",
            ),
        ],
        constraints=[
            Constraint(
                id="c_diff",
                kind="derived",
                op="diff",
                args_refs=["b_end", "b_start"],
                output_blank_id="t",
            ),
        ],
    )
    res = _compute_constraint(
        plan.constraints[0],
        plan,
        _constraint_state(b_start="1787-12-12", b_end="1873-07-01"),
    )
    assert res.status == "resolved"
    assert res.value == 85


def test_compute_diff_between_month_precision_dates_uses_elapsed_years():
    plan = GSWPlan(
        entities=[
            Entity(id="e_person", kind="filled", name="Yuri Shefler"),
            Entity(id="e_yacht", kind="filled", name="Serene"),
            Entity(id="b_birth", kind="blank", value_type="date"),
            Entity(id="b_delivery", kind="blank", value_type="date"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(
                id="vp_birth",
                phrase="born_in",
                subject_id="e_person",
                object_id="b_birth",
            ),
            VerbPhrase(
                id="vp_delivery",
                phrase="delivered_in",
                subject_id="e_yacht",
                object_id="b_delivery",
            ),
        ],
        constraints=[
            Constraint(
                id="c_diff",
                kind="derived",
                op="diff",
                args_refs=["b_delivery", "b_birth"],
                output_blank_id="t",
            ),
        ],
    )
    res = _compute_constraint(
        plan.constraints[0],
        plan,
        _constraint_state(b_birth="1967-09-10", b_delivery="August 2011"),
    )
    assert res.status == "resolved"
    assert res.value == 43


# ---------------------------------------------------------------------------
# concat separator — q39 fix 1c. The concat op used to hardcode "; " no
# matter what the constraint's verification_claim said, so a name-building
# concat ("gymnast's middle name followed by a SPACE and then the
# swimmer's first name") produced "<middle>; <first>" instead of
# "<middle> <first>". The separator now derives from the OUTPUT blank's
# verification_claim/retrieval_hint when it names a space join; the
# default ("; ") is unchanged for plans that give no such signal.
# ---------------------------------------------------------------------------


def _concat_plan(*, claim: str | None = None, hint: str | None = None) -> GSWPlan:
    return GSWPlan(
        entities=[
            Entity(id="b_middle", kind="blank", value_type="text"),
            Entity(id="b_first", kind="blank", value_type="text"),
            Entity(
                id="t",
                kind="blank",
                value_type="text",
                is_target=True,
                verification_claim=claim,
                retrieval_hint=hint,
            ),
        ],
        constraints=[
            Constraint(
                id="c_concat",
                kind="derived",
                op="concat",
                args_refs=["b_middle", "b_first"],
                output_blank_id="t",
            ),
        ],
    )


def test_concat_joins_with_space_when_verification_claim_signals_name_construction():
    # Fictional q39-shape claim: a middle name followed by a space and a
    # first name, concatenated into a full name.
    plan = _concat_plan(
        claim=(
            "The target is Ada Voss's middle name followed by a space and "
            "then Lin Karaczynski's full first name as the new middle name."
        )
    )
    res = _compute_constraint(
        plan.constraints[0], plan, _constraint_state(b_middle="Ada", b_first="Sybil"),
    )
    assert res.status == "resolved"
    assert res.value == "Ada Sybil"


def test_concat_joins_with_space_when_retrieval_hint_signals_it():
    plan = _concat_plan(hint="Concatenate the two names with a space between them.")
    res = _compute_constraint(
        plan.constraints[0], plan, _constraint_state(b_middle="Ada", b_first="Sybil"),
    )
    assert res.status == "resolved"
    assert res.value == "Ada Sybil"


def test_concat_keeps_semicolon_separator_by_default():
    """No claim/hint at all -- existing behavior for plans that don't
    specify a separator must stay unchanged."""
    plan = _concat_plan()
    res = _compute_constraint(
        plan.constraints[0], plan, _constraint_state(b_middle="Ada", b_first="Sybil"),
    )
    assert res.status == "resolved"
    assert res.value == "Ada; Sybil"


def test_concat_keeps_semicolon_separator_for_list_style_claims():
    """A claim describing list-style aggregation (no space-join signal)
    must not be misread as a name-construction join."""
    plan = _concat_plan(claim="The target lists both award winners.")
    res = _compute_constraint(
        plan.constraints[0], plan, _constraint_state(b_middle="Ada", b_first="Sybil"),
    )
    assert res.status == "resolved"
    assert res.value == "Ada; Sybil"


# ---------------------------------------------------------------------------
# sort — plan-verifier 2026-08-07: FRAMES q403 asked for three years in
# chronological order; the planner had only `concat`, which joins in
# ARGUMENT order, so an out-of-order plan produced "2002; 1987; 1914"
# against gold "1914; 1987; 2002". `sort` is the missing derived op:
# modeled on `concat` (same separator helper), ordering numerically when
# every value coerces via `_coerce_constraint_number`, else falling back
# to a total case-insensitive string order.
# ---------------------------------------------------------------------------


def _sort_plan(*, descending: bool | None = None) -> GSWPlan:
    kwargs = {} if descending is None else {"descending": descending}
    return GSWPlan(
        entities=[
            Entity(id="b1", kind="blank", value_type="number"),
            Entity(id="b2", kind="blank", value_type="number"),
            Entity(id="b3", kind="blank", value_type="number"),
            Entity(id="t", kind="blank", value_type="text", is_target=True),
        ],
        constraints=[
            Constraint(
                id="c_sort",
                kind="derived",
                op="sort",
                args_refs=["b1", "b2", "b3"],
                output_blank_id="t",
                **kwargs,
            )
        ],
    )


def test_sort_orders_year_values_ascending_by_default():
    """`descending` is left at the shared Constraint default (True); for
    `sort` that default is interpreted as ASCENDING (see the executor
    comment) since chronological/earliest-first is the common case."""
    plan = _sort_plan()
    res = _compute_constraint(
        plan.constraints[0], plan, _constraint_state(b1="2002", b2="1987", b3="1914"),
    )
    assert res.status == "resolved"
    assert res.value == "1914; 1987; 2002"


def test_sort_descending_set_the_other_way_orders_descending():
    """Flipping `descending` away from the shared default flips the
    output order to descending."""
    plan = _sort_plan(descending=False)
    res = _compute_constraint(
        plan.constraints[0], plan, _constraint_state(b1="2002", b2="1987", b3="1914"),
    )
    assert res.status == "resolved"
    assert res.value == "2002; 1987; 1914"


def test_sort_non_numeric_values_use_case_insensitive_string_order():
    plan = _sort_plan()
    res = _compute_constraint(
        plan.constraints[0],
        plan,
        _constraint_state(b1="Cherry", b2="apple", b3="Banana"),
    )
    assert res.status == "resolved"
    assert res.value == "apple; Banana; Cherry"


def test_sort_mixed_coercible_and_non_coercible_falls_back_to_string_order():
    """One value that fails numeric coercion pulls the WHOLE list onto the
    string fallback rather than mixing numeric and string keys, so
    ordering stays total and deterministic."""
    plan = _sort_plan()
    res = _compute_constraint(
        plan.constraints[0],
        plan,
        _constraint_state(b1="Banana", b2="10", b3="2"),
    )
    assert res.status == "resolved"
    # String-sorted (casefold): "10" < "2" < "banana"
    assert res.value == "10; 2; Banana"


def test_sort_plan_validates_under_gswplan_model_validate():
    payload = {
        "entities": [
            {"id": "b1", "kind": "blank", "value_type": "number"},
            {"id": "b2", "kind": "blank", "value_type": "number"},
            {
                "id": "t",
                "kind": "blank",
                "value_type": "text",
                "is_target": True,
            },
        ],
        "verb_phrases": [],
        "constraints": [
            {
                "id": "c_sort",
                "kind": "derived",
                "op": "sort",
                "args_refs": ["b1", "b2"],
                "output_blank_id": "t",
            }
        ],
    }
    plan = GSWPlan.model_validate(payload)
    assert plan.constraints[0].op == "sort"


def test_concat_on_sort_inputs_still_returns_argument_order():
    """No regression: the same out-of-order year values through `concat`
    still join in ARGUMENT order (the original q403 bug), proving `sort`
    is additive and `concat` is untouched."""
    plan = GSWPlan(
        entities=[
            Entity(id="b1", kind="blank", value_type="number"),
            Entity(id="b2", kind="blank", value_type="number"),
            Entity(id="b3", kind="blank", value_type="number"),
            Entity(id="t", kind="blank", value_type="text", is_target=True),
        ],
        constraints=[
            Constraint(
                id="c_concat",
                kind="derived",
                op="concat",
                args_refs=["b1", "b2", "b3"],
                output_blank_id="t",
            )
        ],
    )
    res = _compute_constraint(
        plan.constraints[0], plan, _constraint_state(b1="2002", b2="1987", b3="1914"),
    )
    assert res.status == "resolved"
    assert res.value == "2002; 1987; 1914"


def test_compute_constraint_args_refs_accepts_filled_literal_value():
    plan = GSWPlan(
        entities=[
            Entity(id="b_year", kind="blank", value_type="number"),
            Entity(
                id="e_twelve",
                kind="filled",
                name="twelve",
                role="constraint-value",
                literal_value=12,
            ),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        constraints=[
            Constraint(
                id="c_prior",
                kind="derived",
                op="diff",
                args_refs=["b_year", "e_twelve"],
                output_blank_id="t",
            )
        ],
    )
    deps = build_dependency_graph(plan)
    assert deps["t"] == {"b_year"}

    res = _compute_constraint(plan.constraints[0], plan, _constraint_state(b_year=2011))
    assert res.status == "resolved"
    assert res.value == 1999


def test_args_blanks_still_rejects_filled_literal_refs():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=[
                Entity(id="b_year", kind="blank", value_type="number"),
                Entity(
                    id="e_twelve",
                    kind="filled",
                    name="twelve",
                    role="constraint-value",
                    literal_value=12,
                ),
                Entity(id="t", kind="blank", value_type="number", is_target=True),
            ],
            constraints=[
                Constraint(
                    id="c_prior",
                    kind="derived",
                    op="diff",
                    args_blanks=["b_year", "e_twelve"],
                    output_blank_id="t",
                )
            ],
        )
    assert "args_blanks" in str(excinfo.value)
    assert "e_twelve" in str(excinfo.value)


def test_derived_constraint_model_dump_omits_row_aggregation_fields():
    plan = GSWPlan(
        entities=[
            Entity(id="b_a", kind="blank", value_type="number"),
            Entity(id="b_b", kind="blank", value_type="number"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        constraints=[
            Constraint(
                id="c_diff",
                kind="derived",
                op="diff",
                args_blanks=["b_a", "b_b"],
                output_blank_id="t",
            ),
        ],
    )

    dumped = plan.model_dump()["constraints"][0]

    assert "numeric_attribute" not in dumped
    assert "n" not in dumped
    assert "descending" not in dumped
    assert "filter_attribute" not in dumped
    assert "filter_ref" not in dumped
    assert "filter_value" not in dumped


def test_row_aggregation_model_dump_keeps_operation_fields():
    plan = GSWPlan(
        entities=[
            Entity(id="b_rows", kind="blank", value_type="list"),
            Entity(id="t", kind="blank", value_type="entity", is_target=True),
        ],
        constraints=[
            Constraint(
                id="c_rank",
                kind="rank_n",
                args_blanks=["b_rows"],
                numeric_attribute="sales",
                n=2,
                descending=False,
                output_blank_id="t",
            ),
        ],
    )

    dumped = plan.model_dump()["constraints"][0]

    assert dumped["numeric_attribute"] == "sales"
    assert dumped["n"] == 2
    assert dumped["descending"] is False


def test_compute_relational_constraints_and_in_list_dependencies():
    plan = GSWPlan(
        entities=[
            Entity(id="b_member", kind="blank", value_type="text"),
            Entity(id="b_list", kind="blank", value_type="list"),
            Entity(id="t", kind="blank", value_type="bool", is_target=True),
        ],
        constraints=[
            Constraint(
                id="c_in",
                kind="in_list",
                args_blanks=["b_member", "b_list"],
                output_blank_id="t",
            )
        ],
    )
    deps = build_dependency_graph(plan)
    assert deps["t"] == {"b_member", "b_list"}
    res = _compute_constraint(
        plan.constraints[0],
        plan,
        _constraint_state(b_member="Beta", b_list=["Alpha", "Beta"]),
    )
    assert res.status == "resolved"
    assert res.value is True

    eq = Constraint(
        id="c_eq",
        kind="equals",
        left_ref="b_member",
        right_ref="b_list",
        output_blank_id="t",
    )
    assert _compute_constraint(
        eq,
        plan,
        _constraint_state(b_member="42", b_list=42),
    ).value is True

    gt = Constraint(
        id="c_gt",
        kind="gt",
        left_ref="b_member",
        right_ref="b_list",
        output_blank_id="t",
    )
    assert _compute_constraint(
        gt,
        plan,
        _constraint_state(b_member="1999", b_list="1984"),
    ).value is True


def _row_dict(candidate: str, **attributes: Any) -> dict[str, Any]:
    return {
        "candidate": candidate,
        "attributes": dict(attributes),
        "evidence_chunk_ids": [f"{candidate}:chunk"],
    }


def _aggregation_plan(constraint: Constraint, *, extra_entities: list[Entity] | None = None) -> GSWPlan:
    return GSWPlan(
        entities=[
            Entity(id="b_rows", kind="blank", value_type="list", role="candidate-set"),
            *(extra_entities or []),
            Entity(id="t", kind="blank", value_type="entity", is_target=True),
        ],
        constraints=[constraint],
    )


def test_compute_rank_n_constraint_picks_second_highest_row():
    constraint = Constraint(
        id="c_rank",
        kind="rank_n",
        args_blanks=["b_rows"],
        numeric_attribute="sales",
        n=2,
        descending=True,
        output_blank_id="t",
    )
    plan = _aggregation_plan(constraint)
    deps = build_dependency_graph(plan)
    assert deps["t"] == {"b_rows"}

    res = _compute_constraint(
        constraint,
        plan,
        _constraint_state(
            b_rows=[
                _row_dict("Alpha", sales=10),
                _row_dict("Bravo", sales=30),
                _row_dict("Charlie", sales=20),
            ],
        ),
    )

    assert res.status == "resolved"
    assert res.value == "Charlie"


def test_compute_count_constraint_counts_candidate_rows():
    constraint = Constraint(
        id="c_count",
        kind="count",
        args_blanks=["b_rows"],
        output_blank_id="t",
    )
    plan = _aggregation_plan(constraint)

    res = _compute_constraint(
        constraint,
        plan,
        _constraint_state(
            b_rows=[
                _row_dict("Alpha", sales=10),
                _row_dict("Bravo", sales=30),
            ],
        ),
    )

    assert res.status == "resolved"
    assert res.value == 2


def test_compute_filter_constraint_filters_rows_by_resolved_bridge_value():
    constraint = Constraint(
        id="c_filter",
        kind="filter",
        args_blanks=["b_rows"],
        filter_attribute="year",
        filter_ref="b_year",
        output_blank_id="t",
    )
    plan = _aggregation_plan(
        constraint,
        extra_entities=[Entity(id="b_year", kind="blank", value_type="number")],
    )
    deps = build_dependency_graph(plan)
    assert deps["t"] == {"b_rows", "b_year"}

    res = _compute_constraint(
        constraint,
        plan,
        _constraint_state(
            b_year=1928,
            b_rows=[
                _row_dict("Alpha", year=1928, sales=10),
                _row_dict("Bravo", year=1929, sales=30),
            ],
        ),
    )

    assert res.status == "resolved"
    assert res.value == [_row_dict("Alpha", year=1928, sales=10)]


@pytest.mark.parametrize(
    ("state", "constraint"),
    [
        ({}, Constraint(id="c", kind="rank_n", args_blanks=["b_rows"], numeric_attribute="sales", n=1, output_blank_id="t")),
        ({"b_rows": []}, Constraint(id="c", kind="rank_n", args_blanks=["b_rows"], numeric_attribute="sales", n=1, output_blank_id="t")),
        ({"b_rows": [_row_dict("Alpha", year=1928)]}, Constraint(id="c", kind="rank_n", args_blanks=["b_rows"], numeric_attribute="sales", n=1, output_blank_id="t")),
        ({"b_rows": [_row_dict("Alpha", sales="many")]}, Constraint(id="c", kind="rank_n", args_blanks=["b_rows"], numeric_attribute="sales", n=1, output_blank_id="t")),
        ({"b_rows": [_row_dict("Alpha", sales=1)]}, Constraint(id="c", kind="rank_n", args_blanks=["b_rows"], numeric_attribute="sales", n=2, output_blank_id="t")),
        ({"b_rows": [{"candidate": "Alpha", "attributes": {"sales": 1}}]}, Constraint(id="c", kind="rank_n", args_blanks=["b_rows"], numeric_attribute="sales", n=1, output_blank_id="t")),
    ],
)
def test_compute_rank_n_abstains_without_sufficient_numeric_rows(
    state: dict[str, Any],
    constraint: Constraint,
):
    plan = _aggregation_plan(constraint)

    res = _compute_constraint(constraint, plan, _constraint_state(**state))

    assert res.status == "unknown"
    assert res.value is None


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"candidate": "Alpha", "attributes": {}}],
    ],
)
def test_compute_count_abstains_over_zero_evidenced_rows(rows: list[dict[str, Any]]):
    constraint = Constraint(
        id="c_count",
        kind="count",
        args_blanks=["b_rows"],
        output_blank_id="t",
    )
    plan = _aggregation_plan(constraint)

    res = _compute_constraint(
        constraint,
        plan,
        _constraint_state(b_rows=rows),
    )

    assert res.status == "unknown"
    assert res.value is None


def _resolved(plan: GSWPlan, values: dict[str, Any]) -> dict[str, BlankResult]:
    """State with each blank resolved to its value coerced to the blank's
    value_type, as a researcher commit leaves it."""
    return {
        bid: BlankResult(blank_id=bid, status="resolved",
                         value=_coerce_value(v, plan.entity_by_id(bid).value_type))
        for bid, v in values.items()
    }


def _cascade_answer(plan: GSWPlan, state: dict[str, BlankResult]) -> str:
    _cascade_auto_compute(plan, state)
    res = state.get(plan.target().id)
    return _stringify(res.value) if res is not None else ""


def test_cascade_diff_numeric():
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="Portland Maine"),
            Entity(id="e2", kind="filled", name="Portland Oregon"),
            Entity(id="b1", kind="blank", value_type="number"),
            Entity(id="b2", kind="blank", value_type="number"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="has population", subject_id="e1", object_id="b1"),
            VerbPhrase(id="vp2", phrase="has population", subject_id="e2", object_id="b2"),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="derived",
                op="diff",
                args_blanks=["b2", "b1"],
                output_blank_id="t",
            )
        ],
    )
    state = _resolved(plan, {"b1": 68408, "b2": 152920})
    assert _cascade_answer(plan, state) == "84512"
    assert state["t"].resolved_by == "constraint"


def test_cascade_diff_uses_years_from_date_strings():
    """Year-difference constraints should still fire when upstream date
    blanks store full dates. FRAMES questions often ask "how many years"
    while evidence snippets contain day/month/year dates."""
    plan = GSWPlan(
        entities=[
            Entity(id="e_birth", kind="filled", name="Ben Darwin"),
            Entity(id="e_opened", kind="filled", name="Australian Institute of Sport"),
            Entity(id="b_birth", kind="blank", value_type="date"),
            Entity(id="b_opened", kind="blank", value_type="date"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="born_in_year", subject_id="e_birth", object_id="b_birth"),
            VerbPhrase(id="vp2", phrase="opened_in_year", subject_id="e_opened", object_id="b_opened"),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="derived",
                op="diff",
                args_blanks=["b_opened", "b_birth"],
                output_blank_id="t",
            )
        ],
    )
    state = _resolved(plan, {"b_birth": "1976-10-17", "b_opened": "1981-01-01"})
    assert _cascade_answer(plan, state) == "5"


def test_cascade_sum_digits_from_text_postcode():
    """A sum over a text postcode should add the postcode digits, not
    leave the target unresolved."""
    plan = GSWPlan(
        entities=[
            Entity(id="e_hospital", kind="filled", name="Liverpool Maternity Hospital"),
            Entity(id="b_postcode", kind="blank", value_type="text"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(
                id="vp1",
                phrase="has_postcode",
                subject_id="e_hospital",
                object_id="b_postcode",
            ),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="derived",
                op="sum",
                args_blanks=["b_postcode"],
                output_blank_id="t",
            )
        ],
    )
    state = _resolved(plan, {"b_postcode": "L7 7AB"})
    assert _cascade_answer(plan, state) == "14"


def test_cascade_argmax_entity_selector():
    names = ["Alpha", "Bravo", "Charlie", "Delta"]
    years = {"Alpha": "1996", "Bravo": "1968", "Charlie": "1992", "Delta": "1981"}
    entities = [
        Entity(id=f"e{i}", kind="filled", name=names[i - 1]) for i in range(1, 5)
    ]
    blanks = [Entity(id=f"b{i}", kind="blank", value_type="date") for i in range(1, 5)]
    target = Entity(id="t", kind="blank", value_type="entity", is_target=True)
    plan = GSWPlan(
        entities=entities + blanks + [target],
        verb_phrases=[
            VerbPhrase(
                id=f"vp{i}",
                phrase="entered service in",
                subject_id=f"e{i}",
                object_id=f"b{i}",
            )
            for i in range(1, 5)
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="argmax",
                candidate_entity_ids=[f"e{i}" for i in range(1, 5)],
                sort_by_blank_ids=[f"b{i}" for i in range(1, 5)],
                output_blank_id="t",
            )
        ],
    )
    state = _resolved(plan, {f"b{i}": years[names[i - 1]] for i in range(1, 5)})
    assert _cascade_answer(plan, state) == "Alpha"  # 1996 is max


def test_cascade_leaves_target_unresolved_when_an_input_is_unknown():
    # An unresolved input never fires the constraint: no fabricated target.
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="UnknownEntity"),
            Entity(id="b1", kind="blank", value_type="number"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="has_attribute", subject_id="e1", object_id="b1"),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="derived",
                op="sum",
                args_blanks=["b1"],
                output_blank_id="t",
            )
        ],
    )
    state = {"b1": BlankResult(blank_id="b1", status="unknown")}
    assert _cascade_auto_compute(plan, state) == []
    assert _cascade_answer(plan, state) == ""  # empty string for unresolved target


def test_argmax_with_blank_winner_returns_unknown():
    # q469 regression: a degenerate argmax whose winning candidate is a BLANK
    # (unnamed) entity must resolve to status='unknown', NOT emit the raw
    # blank-id as the answer. (Plan-churn made the target its own argmax
    # candidate; the compute then returned the literal blank-id "t".)
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="Alpha"),
            Entity(id="e2", kind="filled", name="Bravo"),
            Entity(id="b_bad", kind="blank", value_type="entity"),
            Entity(id="b1", kind="blank", value_type="number"),
            Entity(id="b2", kind="blank", value_type="number"),
            Entity(id="t", kind="blank", value_type="entity", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="ranks", subject_id="e1", object_id="b1"),
            VerbPhrase(id="vp2", phrase="ranks", subject_id="e2", object_id="b2"),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="argmax",
                candidate_entity_ids=["e1", "b_bad"],
                sort_by_blank_ids=["b1", "b2"],
                output_blank_id="t",
            )
        ],
    )
    # b_bad (a blank) has the higher sort value -> it would "win" the argmax.
    state = {
        "b1": BlankResult(blank_id="b1", value=10, status="resolved"),
        "b2": BlankResult(blank_id="b2", value=99, status="resolved"),
    }
    res = _compute_constraint(plan.constraints[0], plan, state)
    assert res.status == "unknown"
    assert res.value in (None, "")  # must NOT emit the raw blank-id "b_bad"


def test_plan_without_target_raises_no_target():
    # Wire the blanks with a VP so the no-dangling validator passes,
    # but leave is_target unset so target() raises no_target.
    plan = GSWPlan(
        entities=[
            Entity(id="b1", kind="blank", value_type="number"),
            Entity(id="b2", kind="blank", value_type="number"),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="relates_to", subject_id="b1", object_id="b2"),
        ],
    )
    with pytest.raises(ExecutionError) as excinfo:
        plan.target()
    assert excinfo.value.kind == "no_target"


# ---------------------------------------------------------------------------
# Prompt-v2 additions: role fields + no-dangling validator
# ---------------------------------------------------------------------------


def test_entity_role_and_state_fields_accepted():
    """Plans that include `role` (and optionally `state`) on entities
    should validate cleanly; the executor ignores the fields but the
    schema must round-trip them."""
    plan = GSWPlan(
        entities=[
            Entity(
                id="e1",
                kind="filled",
                name="Pablo Picasso",
                role="subject",
                state="alive_in_question_context",
            ),
            Entity(id="e2", kind="filled", name="Pink Floyd", role="subject"),
            Entity(id="b1", kind="blank", value_type="date", role="bridge-date"),
            Entity(
                id="t",
                kind="blank",
                value_type="entity",
                is_target=True,
                role="target",
            ),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="died_in", subject_id="e1", object_id="b1"),
            VerbPhrase(
                id="vp2", phrase="released_in_year", subject_id="e2", object_id="b1"
            ),
            VerbPhrase(id="vp3", phrase="released", subject_id="e2", object_id="t"),
        ],
    )
    # Round-trip via model_dump / model_validate to ensure fields persist.
    round_tripped = GSWPlan.model_validate(plan.model_dump())
    assert round_tripped.entity_by_id("e1").role == "subject"
    assert round_tripped.entity_by_id("e1").state == "alive_in_question_context"
    assert round_tripped.entity_by_id("t").role == "target"


def test_dangling_entity_rejected():
    """A plan whose entity appears in NO verb-phrase and NO constraint
    reference must raise a ValidationError that names the dangling id."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=[
                Entity(id="e1", kind="filled", name="Used Entity", role="subject"),
                Entity(id="e_dangles", kind="filled", name="Orphan", role="subject"),
                Entity(
                    id="t",
                    kind="blank",
                    value_type="entity",
                    is_target=True,
                    role="target",
                ),
            ],
            verb_phrases=[
                VerbPhrase(id="vp1", phrase="relates_to", subject_id="e1", object_id="t"),
            ],
        )
    # The error message should mention the dangling id.
    assert "e_dangles" in str(excinfo.value)


def test_dangling_validator_accepts_constraint_only_references():
    """Entities referenced only by a constraint (never by a VP) must
    count as non-dangling. This covers argmax target blanks whose only
    incoming edge is the constraint's output_blank_id."""
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="A", role="candidate"),
            Entity(id="e2", kind="filled", name="B", role="candidate"),
            Entity(id="b1", kind="blank", value_type="number", role="bridge-number"),
            Entity(id="b2", kind="blank", value_type="number", role="bridge-number"),
            Entity(
                id="t",
                kind="blank",
                value_type="entity",
                is_target=True,
                role="target",
            ),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="has_x", subject_id="e1", object_id="b1"),
            VerbPhrase(id="vp2", phrase="has_x", subject_id="e2", object_id="b2"),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="argmax",
                candidate_entity_ids=["e1", "e2"],
                sort_by_blank_ids=["b1", "b2"],
                output_blank_id="t",
            )
        ],
    )
    # Simply constructing without raising is the assertion.
    assert plan.target().id == "t"


def test_build_dependency_graph_respects_vp_projection():
    # Ensure that a blank-entity → blank VP creates a dep edge.
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="NYC"),
            Entity(id="b_person", kind="blank", value_type="entity"),
            Entity(id="b_age", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="born_in", subject_id="b_person", object_id="e1"),
            VerbPhrase(id="vp2", phrase="has_age", subject_id="b_person", object_id="b_age"),
        ],
    )
    deps = build_dependency_graph(plan)
    assert "b_person" in deps["b_age"]
    assert deps["b_person"] == set()


def test_build_dependency_graph_target_as_subject_is_sink():
    # vp4: (t, won_in_year, b1) — the TARGET is the VP subject. The target must
    # depend on the bridge (it is a sink), and the bridge must NOT depend on the
    # target. Without this, deps[b1]={t} makes the target a source the
    # orchestrator resolves from priors and submits, skipping the bridge.
    plan = GSWPlan(
        entities=[
            Entity(id="e_serbia", kind="filled", name="Serbia"),
            Entity(id="b1", kind="blank", value_type="date"),
            Entity(id="t", kind="blank", value_type="entity", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="became_independent_in_year",
                       subject_id="e_serbia", object_id="b1"),
            VerbPhrase(id="vp4", phrase="won_in_year",
                       subject_id="t", object_id="b1"),
        ],
    )
    deps = build_dependency_graph(plan)
    assert deps["t"] == {"b1"}      # target depends on the bridge (sink)
    assert deps["b1"] == set()      # bridge depends on nothing, NOT on the target


def _vp_projection_plan() -> GSWPlan:
    # b_age depends on b_person (blank-entity --VP--> blank).
    return GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="NYC"),
            Entity(id="b_person", kind="blank", value_type="entity"),
            Entity(id="b_age", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="born_in", subject_id="b_person", object_id="e1"),
            VerbPhrase(id="vp2", phrase="has_age", subject_id="b_person", object_id="b_age"),
        ],
    )


def test_unresolved_upstream_deps_flags_blank_with_pending_dep():
    plan = _vp_projection_plan()
    # b_person not resolved yet -> b_age is NOT dispatch-ready.
    blocked = unresolved_upstream_deps(plan, resolved_ids=set(), blank_ids=["b_age"])
    assert blocked == {"b_age": ["b_person"]}


def test_unresolved_upstream_deps_clears_when_dep_resolved():
    plan = _vp_projection_plan()
    # b_person resolved -> b_age IS ready (omitted from result).
    blocked = unresolved_upstream_deps(plan, resolved_ids={"b_person"}, blank_ids=["b_age"])
    assert blocked == {}


def test_unresolved_upstream_deps_ignores_depless_blank():
    plan = _vp_projection_plan()
    # b_person has no upstream deps -> always ready.
    blocked = unresolved_upstream_deps(plan, resolved_ids=set(), blank_ids=["b_person"])
    assert blocked == {}


def test_unresolved_upstream_deps_constraint_inputs():
    # derived constraint: t depends on b1 and b2.
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="A"),
            Entity(id="e2", kind="filled", name="B"),
            Entity(id="b1", kind="blank", value_type="number"),
            Entity(id="b2", kind="blank", value_type="number"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="pop", subject_id="e1", object_id="b1"),
            VerbPhrase(id="vp2", phrase="pop", subject_id="e2", object_id="b2"),
        ],
        constraints=[
            Constraint(id="c1", kind="derived", op="diff",
                       args_blanks=["b2", "b1"], output_blank_id="t"),
        ],
    )
    # only b1 resolved -> t still blocked on b2.
    blocked = unresolved_upstream_deps(plan, resolved_ids={"b1"}, blank_ids=["t"])
    assert blocked == {"t": ["b2"]}


# ---------------------------------------------------------------------------
# Phase-3.6.2 — per-kind constraint-shape validation
#
# A constraint missing its required input fields is uncomputable. The
# planner emit-repair loop relies on these ValueErrors to surface a
# concrete problem to the LLM for retry.
# ---------------------------------------------------------------------------


def _base_entities_two_candidates() -> list[Entity]:
    return [
        Entity(id="e1", kind="filled", name="A", role="candidate"),
        Entity(id="e2", kind="filled", name="B", role="candidate"),
        Entity(id="b1", kind="blank", value_type="number", role="bridge-number"),
        Entity(id="b2", kind="blank", value_type="number", role="bridge-number"),
        Entity(
            id="t",
            kind="blank",
            value_type="entity",
            is_target=True,
            role="target",
        ),
    ]


def _base_vps_two_candidates() -> list[VerbPhrase]:
    return [
        VerbPhrase(id="vp1", phrase="has_x", subject_id="e1", object_id="b1"),
        VerbPhrase(id="vp2", phrase="has_x", subject_id="e2", object_id="b2"),
    ]


def test_constraint_argmax_without_candidates_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=_base_entities_two_candidates(),
            verb_phrases=_base_vps_two_candidates(),
            constraints=[
                Constraint(
                    id="c1",
                    kind="argmax",
                    candidate_entity_ids=[],  # empty — invalid
                    sort_by_blank_ids=["b1", "b2"],
                    output_blank_id="t",
                )
            ],
        )
    msg = str(excinfo.value)
    assert "candidate_entity_ids" in msg
    assert "argmax" in msg


def test_constraint_argmin_without_sort_by_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=_base_entities_two_candidates(),
            verb_phrases=_base_vps_two_candidates(),
            constraints=[
                Constraint(
                    id="c1",
                    kind="argmin",
                    candidate_entity_ids=["e1", "e2"],
                    sort_by_blank_ids=[],  # empty — invalid
                    output_blank_id="t",
                )
            ],
        )
    msg = str(excinfo.value)
    assert "sort_by_blank_ids" in msg
    assert "argmin" in msg


def test_constraint_argmax_candidate_sort_length_mismatch_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=_base_entities_two_candidates(),
            verb_phrases=_base_vps_two_candidates(),
            constraints=[
                Constraint(
                    id="c1",
                    kind="argmax",
                    candidate_entity_ids=["e1", "e2"],
                    sort_by_blank_ids=["b1"],
                    output_blank_id="t",
                )
            ],
        )
    msg = str(excinfo.value)
    assert "different lengths" in msg
    assert "argmax" in msg


def test_constraint_derived_with_empty_args_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=[
                Entity(id="e1", kind="filled", name="A", role="subject"),
                Entity(id="b1", kind="blank", value_type="number", role="bridge-number"),
                Entity(
                    id="t",
                    kind="blank",
                    value_type="number",
                    is_target=True,
                    role="target",
                ),
            ],
            verb_phrases=[
                VerbPhrase(id="vp1", phrase="has_x", subject_id="e1", object_id="b1"),
            ],
            constraints=[
                Constraint(
                    id="c1",
                    kind="derived",
                    op="diff",
                    args_blanks=[],  # empty — invalid
                    output_blank_id="t",
                )
            ],
        )
    msg = str(excinfo.value)
    assert "args_blanks" in msg
    assert "derived" in msg


def test_constraint_gt_without_left_ref_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=[
                Entity(id="e1", kind="filled", name="A", role="subject"),
                Entity(id="b1", kind="blank", value_type="number", role="bridge-number"),
                Entity(
                    id="t",
                    kind="blank",
                    value_type="bool",
                    is_target=True,
                    role="target",
                ),
            ],
            verb_phrases=[
                VerbPhrase(id="vp1", phrase="has_x", subject_id="e1", object_id="b1"),
            ],
            constraints=[
                Constraint(
                    id="c1",
                    kind="gt",
                    left_ref=None,  # missing — invalid
                    right_ref="b1",
                    output_blank_id="t",
                )
            ],
        )
    msg = str(excinfo.value)
    assert "left_ref" in msg or "right_ref" in msg
    assert "gt" in msg


def test_constraint_equals_without_right_ref_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=[
                Entity(id="e1", kind="filled", name="A", role="subject"),
                Entity(id="b1", kind="blank", value_type="text", role="bridge-text"),
                Entity(
                    id="t",
                    kind="blank",
                    value_type="bool",
                    is_target=True,
                    role="target",
                ),
            ],
            verb_phrases=[
                VerbPhrase(id="vp1", phrase="has_x", subject_id="e1", object_id="b1"),
            ],
            constraints=[
                Constraint(
                    id="c1",
                    kind="equals",
                    left_ref="b1",
                    right_ref=None,  # missing — invalid
                    output_blank_id="t",
                )
            ],
        )
    msg = str(excinfo.value)
    assert "left_ref" in msg or "right_ref" in msg
    assert "equals" in msg


def test_constraint_in_list_with_empty_args_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=[
                Entity(id="e1", kind="filled", name="A", role="subject"),
                Entity(id="b1", kind="blank", value_type="list", role="bridge-list"),
                Entity(
                    id="t",
                    kind="blank",
                    value_type="bool",
                    is_target=True,
                    role="target",
                ),
            ],
            verb_phrases=[
                VerbPhrase(id="vp1", phrase="has_x", subject_id="e1", object_id="b1"),
            ],
            constraints=[
                Constraint(
                    id="c1",
                    kind="in_list",
                    args_blanks=[],  # empty — invalid
                    output_blank_id="t",
                )
            ],
        )
    msg = str(excinfo.value)
    assert "args_blanks" in msg
    assert "in_list" in msg


def test_duplicate_constraint_outputs_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=[
                Entity(id="e1", kind="filled", name="A", role="subject"),
                Entity(id="b1", kind="blank", value_type="number", role="bridge-number"),
                Entity(id="b2", kind="blank", value_type="number", role="bridge-number"),
                Entity(
                    id="t",
                    kind="blank",
                    value_type="number",
                    is_target=True,
                    role="target",
                ),
            ],
            verb_phrases=[
                VerbPhrase(id="vp1", phrase="x", subject_id="e1", object_id="b1"),
                VerbPhrase(id="vp2", phrase="y", subject_id="e1", object_id="b2"),
            ],
            constraints=[
                Constraint(
                    id="c1",
                    kind="derived",
                    op="sum",
                    args_blanks=["b1"],
                    output_blank_id="t",
                ),
                Constraint(
                    id="c2",
                    kind="derived",
                    op="diff",
                    args_blanks=["b2"],
                    output_blank_id="t",
                ),
            ],
        )
    msg = str(excinfo.value)
    assert "both output" in msg
    assert "t" in msg


def test_constraint_argmax_with_full_shape_accepted():
    """A well-formed argmax constraint must validate without errors."""
    plan = GSWPlan(
        entities=_base_entities_two_candidates(),
        verb_phrases=_base_vps_two_candidates(),
        constraints=[
            Constraint(
                id="c1",
                kind="argmax",
                candidate_entity_ids=["e1", "e2"],
                sort_by_blank_ids=["b1", "b2"],
                output_blank_id="t",
            )
        ],
    )
    assert plan.constraints[0].kind == "argmax"


def test_constraint_unknown_candidate_ref_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        GSWPlan(
            entities=_base_entities_two_candidates(),
            verb_phrases=_base_vps_two_candidates(),
            constraints=[
                Constraint(
                    id="c1",
                    kind="argmax",
                    candidate_entity_ids=["e1", "e_missing"],
                    sort_by_blank_ids=["b1", "b2"],
                    output_blank_id="t",
                )
            ],
        )
    msg = str(excinfo.value)
    assert "candidate_entity_ids" in msg
    assert "e_missing" in msg


def test_constraint_derived_with_args_accepted():
    """A well-formed derived constraint must validate without errors."""
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="A", role="subject"),
            Entity(id="b1", kind="blank", value_type="number", role="bridge-number"),
            Entity(id="b2", kind="blank", value_type="number", role="bridge-number"),
            Entity(
                id="t",
                kind="blank",
                value_type="number",
                is_target=True,
                role="target",
            ),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="x", subject_id="e1", object_id="b1"),
            VerbPhrase(id="vp2", phrase="y", subject_id="e1", object_id="b2"),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="derived",
                op="diff",
                args_blanks=["b1", "b2"],
                output_blank_id="t",
            )
        ],
    )
    assert plan.constraints[0].args_blanks == ["b1", "b2"]


def test_nested_date_diff_over_shared_reference_recovers_precision():
    """q209: t=diff(A,B), A=diff(R,X_birth), B=diff(R,Y_birth). The shared
    reference R cancels; the age difference is elapsed_years(X,Y)=7, not the
    naive |30-22|=8 from the integer-year intermediate ages."""
    plan = GSWPlan.model_validate({
        "entities": [
            {"id": "b_grad", "kind": "blank", "value_type": "date"},
            {"id": "b_bp", "kind": "blank", "value_type": "date"},
            {"id": "b_ba", "kind": "blank", "value_type": "date"},
            {"id": "b_ap", "kind": "blank", "value_type": "number"},
            {"id": "b_aa", "kind": "blank", "value_type": "number"},
            {"id": "t", "kind": "blank", "value_type": "number", "is_target": True},
        ],
        "constraints": [
            {"id": "cA", "kind": "derived", "op": "diff", "args_refs": ["b_grad", "b_bp"], "output_blank_id": "b_ap"},
            {"id": "cB", "kind": "derived", "op": "diff", "args_refs": ["b_grad", "b_ba"], "output_blank_id": "b_aa"},
            {"id": "cT", "kind": "derived", "op": "diff", "args_refs": ["b_ap", "b_aa"], "output_blank_id": "t"},
        ],
    })
    state = {
        "b_grad": BlankResult(blank_id="b_grad", value="1970", status="resolved"),
        "b_bp": BlankResult(blank_id="b_bp", value="1940-11-13", status="resolved"),
        "b_ba": BlankResult(blank_id="b_ba", value="1948-01-19", status="resolved"),
        "b_ap": BlankResult(blank_id="b_ap", value=30.0, status="resolved"),
        "b_aa": BlankResult(blank_id="b_aa", value=22.0, status="resolved"),
    }
    cT = [c for c in plan.constraints if c.id == "cT"][0]
    assert _compute_constraint(cT, plan, state).value == 7


def test_ordinary_diff_of_two_numbers_unchanged():
    """Guard: a plain diff of two resolved numbers (not diff-of-diffs) is
    still the numeric absolute difference — the q209 simplification must not
    fire here."""
    plan = GSWPlan.model_validate({
        "entities": [
            {"id": "a", "kind": "blank", "value_type": "number"},
            {"id": "b", "kind": "blank", "value_type": "number"},
            {"id": "t", "kind": "blank", "value_type": "number", "is_target": True},
        ],
        "constraints": [{"id": "c", "kind": "derived", "op": "diff", "args_refs": ["a", "b"], "output_blank_id": "t"}],
    })
    state = {
        "a": BlankResult(blank_id="a", value=30.0, status="resolved"),
        "b": BlankResult(blank_id="b", value=22.0, status="resolved"),
    }
    assert _compute_constraint(plan.constraints[0], plan, state).value == 8.0


def test_floor_to_years_does_not_zero_out_subyear_duration():
    """q284 regression: two people born 6 months apart. The birth-birth diff
    is "6 months 26 days" and the gold answer IS that duration -- flooring to
    0 years (because the hint says "in years" and the target is numeric)
    destroys it. The floor must only fire when >=1 whole year is present."""
    # Birth-cued blank ids (as the real q284 plan has) route through the
    # birth-birth month/day branch that produces the duration string.
    plan = GSWPlan.model_validate({
        "entities": [
            {"id": "b_birth_barbie", "kind": "blank", "value_type": "date", "retrieval_hint": "full birth date of the inventor of Barbie"},
            {"id": "b_birth_hot_wheels", "kind": "blank", "value_type": "date", "retrieval_hint": "full birth date of the inventor of Hot Wheels"},
            {"id": "t", "kind": "blank", "value_type": "number", "is_target": True, "retrieval_hint": "age difference in years"},
        ],
        "constraints": [{"id": "c", "kind": "derived", "op": "diff", "args_refs": ["b_birth_barbie", "b_birth_hot_wheels"], "output_blank_id": "t"}],
    })
    state = {
        "b_birth_barbie": BlankResult(blank_id="b_birth_barbie", value="1916-11-04", status="resolved"),
        "b_birth_hot_wheels": BlankResult(blank_id="b_birth_hot_wheels", value="1916-04-09", status="resolved"),
    }
    out = _compute_constraint(plan.constraints[0], plan, state).value
    assert out == "6 months 26 days"  # NOT 0
