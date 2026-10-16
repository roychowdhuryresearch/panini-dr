"""Module-level helpers of gsw_planner_react_v1 that the orchestrator uses live.

The ReAct adapter class in that module is dead on the paper path, but the
constraint cascade (_cascade_auto_compute, which resolved roughly one blank in
five in the paper runs) and render_plan_brief are imported by the orchestrator.
Ported from the research repo's test_planner_react.py; the adapter-loop tests
were dropped with the adapter's role.
"""

from __future__ import annotations

from typing import Any

from panini_dr.adapters.ours._planner_exec import (
    BlankResult,
    Constraint,
    Entity,
    GSWPlan,
)
from panini_dr.adapters.ours.gsw_planner_react_v1 import (
    _cascade_auto_compute,
    _collect_constraint_inputs,
)


def _fixture_plan() -> dict[str, Any]:
    return {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pablo Picasso", "role": "subject"},
            {"id": "e2", "kind": "filled", "name": "Pink Floyd", "role": "subject"},
            {"id": "b_year", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {
                "id": "vp1",
                "phrase": "died_in",
                "subject_id": "e1",
                "object_id": "b_year",
            },
            {
                "id": "vp2",
                "phrase": "released_in_year",
                "subject_id": "e2",
                "object_id": "b_year",
            },
            {"id": "vp3", "phrase": "released", "subject_id": "e2", "object_id": "t"},
        ],
        "constraints": [],
    }


def test_render_plan_brief_slice_filters_to_assigned_blanks():
    """With ``slice_blank_ids=['t']`` (only the target), the rendered
    plan brief should mention entity `t` and the VP wiring `t` to
    Pink Floyd, but should drop `b_year` from the topological-order
    section (it belongs to an earlier level)."""
    from panini_dr.adapters.ours._planner_react_prompt import render_plan_brief

    plan_dict = _fixture_plan()
    order = ["b_year", "t"]
    levels = {"b_year": 0, "t": 1}

    full = render_plan_brief(plan_dict, order, levels)
    sliced = render_plan_brief(plan_dict, order, levels, slice_blank_ids=["t"])

    # Sanity: full render contains both blanks in topo order
    assert "b_year" in full
    assert "`t`" in full

    # Sliced render: t is there, b_year is NOT listed as a topo step.
    assert "`t`" in sliced
    # b_year should not appear as its own Step heading.
    assert "Step 1" in sliced
    assert "Step 2" not in sliced  # only one step in the slice
    # Filled anchor "Pink Floyd" must survive (it's wired to t).
    assert "Pink Floyd" in sliced
    # Picasso was only wired to b_year (not in slice) → should be dropped.
    assert "Pablo Picasso" not in sliced


def test_render_plan_brief_slice_keeps_constraint_when_output_in_slice():
    """A constraint whose output blank is in the slice must appear
    in the slice rendering (so the researcher knows not to retrieve it)."""
    from panini_dr.adapters.ours._planner_react_prompt import render_plan_brief

    plan_dict = {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Alice", "role": "subject"},
            {"id": "b_a", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {"id": "b_b", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "text",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_a", "subject_id": "e1", "object_id": "b_a"},
            {"id": "vp2", "phrase": "has_b", "subject_id": "e1", "object_id": "b_b"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "derived",
                "op": "concat",
                "args_blanks": ["b_a", "b_b"],
                "output_blank_id": "t",
            }
        ],
    }
    order = ["b_a", "b_b", "t"]
    levels = {"b_a": 0, "b_b": 0, "t": 1}

    sliced = render_plan_brief(plan_dict, order, levels, slice_blank_ids=["t"])
    # Target appears in topo order as a constraint-computed step.
    assert "`t`" in sliced
    assert "constraint" in sliced.lower() or "auto-computed" in sliced.lower()
    # b_a / b_b are upstream — they shouldn't show as their own Step headings.
    assert "Step 1" in sliced
    assert "Step 2" not in sliced


# ---------------------------------------------------------------------------
# Cascade block_unverified guard (Plan 2, Task 4)
# ---------------------------------------------------------------------------


def _minimal_diff_cascade() -> tuple[GSWPlan, dict[str, BlankResult]]:
    """A minimal ``(diff a b) -> t`` derived constraint plus state.

    ``a`` and ``b`` are resolved with real numeric values; ``t`` is the
    unresolved output blank. Returned fresh each call so tests don't share
    mutable state.
    """
    plan = GSWPlan(
        entities=[
            Entity(id="a", kind="blank", value_type="number"),
            Entity(id="b", kind="blank", value_type="number"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        constraints=[
            Constraint(
                id="c1",
                kind="derived",
                op="diff",
                args_blanks=["a", "b"],
                output_blank_id="t",
            )
        ],
    )
    state: dict[str, BlankResult] = {
        "a": BlankResult(blank_id="a", value=10, status="resolved"),
        "b": BlankResult(blank_id="b", value=3, status="resolved"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    return plan, state


def test_cascade_default_off_byte_identical():
    plan, state = _minimal_diff_cascade()
    state["b"].verifier_status = "unverified_capped"
    # default (block_unverified=False): fires exactly as today
    fired = _cascade_auto_compute(plan, state)  # default arg
    assert "t" in fired
    assert state["t"].status == "resolved"


def test_cascade_sink_none_safe_and_off_records_nothing():
    plan, state = _minimal_diff_cascade()
    state["b"].verifier_status = "unverified_capped"
    sink = []
    _cascade_auto_compute(plan, state, block_unverified=False, blocked_sink=sink)  # off: fires, no block
    assert sink == []
    _cascade_auto_compute(plan, state, block_unverified=True)  # sink None: must not crash


def test_cascade_fires_in_list_wired_via_args_blanks():
    """q101: an in_list bool target whose (member, list) are wired via
    args_blanks (not left_ref/right_ref) must fire in the cascade. Before the
    fix _collect_constraint_inputs returned [] for it, so the membership
    check never ran and the bool target looped to max_turns -> empty."""
    plan = GSWPlan.model_validate({
        "entities": [
            {"id": "b_member", "kind": "blank", "value_type": "entity"},
            {"id": "b_list", "kind": "blank", "value_type": "list"},
            {"id": "t", "kind": "blank", "value_type": "bool", "is_target": True},
        ],
        "constraints": [{
            "id": "c1", "kind": "in_list",
            "args_blanks": ["b_member", "b_list"], "output_blank_id": "t",
        }],
    })
    state = {
        "b_member": BlankResult(blank_id="b_member", value="Ryan Gosling", status="resolved"),
        "b_list": BlankResult(blank_id="b_list", value=["Anthony Hopkins", "Tom Hanks"], status="resolved"),
    }
    assert _collect_constraint_inputs(plan.constraints[0]) == ["b_member", "b_list"]
    fired = _cascade_auto_compute(plan, state)
    assert "t" in fired
    assert state["t"].status == "resolved"
    assert state["t"].value is False  # member not in list


def test_cascade_in_list_left_right_refs_unchanged():
    """in_list wired the classic way (left_ref/right_ref) still collects those."""
    c = _cascade_in_list_leftright()
    assert set(_collect_constraint_inputs(c)) == {"b_member", "b_list"}


def _cascade_in_list_leftright():
    from panini_dr.adapters.ours._planner_exec import Constraint
    return Constraint(id="c1", kind="in_list", left_ref="b_member",
                      right_ref="b_list", output_blank_id="t")
