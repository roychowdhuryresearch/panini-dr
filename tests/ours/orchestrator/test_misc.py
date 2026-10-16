"""Odds and ends: default role clients, constraint-output dispatch, final-answer normalisation.

Split from the research repo's test_planner_orchestrator.py."""

from __future__ import annotations

from tests.ours.orchestrator._helpers import *  # noqa: F401,F403


def test_dispatching_constraint_output_keeps_existing_rejection():
    plan = GSWPlan.model_validate(_concat_plan())
    state = {
        "b_a": BlankResult(blank_id="b_a", status="unknown"),
        "b_b": BlankResult(blank_id="b_b", status="unknown"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    adapter = _mk_llm_adapter(
        _RoutedLLM({}),
        plan_dict=plan.model_dump(),
    )

    result, action = adapter._run_orchestrator_tool(
        name="dispatch_subplan",
        args={"blank_ids": ["t"]},
        plan=plan,
        plan_dict=plan.model_dump(),
        state=state,
        order=["b_a", "b_b", "t"],
        levels={"b_a": 0, "b_b": 0, "t": 1},
        question="Q",
        dispatch_idx=0,
        constraint_outputs={"t"},
    )

    assert result["ok"] is False
    assert "constraint outputs" in result["error"]
    assert action == {}


def test_resolved_target_answer_fills_empty_from_resolved_target():
    # q429: the orchestrator resolved the target in state but never
    # submit_answer'd it (ran to the turn cap thrashing), so the final answer
    # came out empty. The loop-exit safety-net must surface the resolved target
    # value instead of returning empty.
    resolved = {"t": BlankResult(blank_id="t", value="13", status="resolved")}
    assert _resolved_target_answer("", "t", resolved) == "13"
    # A non-empty answer is never overridden.
    assert _resolved_target_answer("already", "t", resolved) == "already"
    # No resolved target -> stays empty (no fabrication).
    assert _resolved_target_answer("", "t", {}) == ""
    assert (
        _resolved_target_answer(
            "", "t", {"t": BlankResult(blank_id="t", status="unknown")}
        )
        == ""
    )
    # An empty resolved value is not surfaced.
    assert (
        _resolved_target_answer(
            "", "t", {"t": BlankResult(blank_id="t", value="", status="resolved")}
        )
        == ""
    )


def test_no_prose_record_means_no_trace_key():
    """Default off must not add the key at all."""
    plan_dict = _one_level_plan()
    adapter = _mk_llm_adapter(_RoutedLLM({}), plan_dict=plan_dict)
    traj = adapter.run_question("Q", question_id="q_noprose")
    assert "gsw_prose" not in traj.extra


def test_final_answer_normalizer_fixes_measured_artifacts():
    """Deterministic answer hygiene at the run_question choke point.

    MEASURED (arm C7, 2026-08-17): two judge losses were formatting, not
    content -- monaco_1512 emitted the correct Moneypenny actresses as
    'A → B → C' (narrow no-break spaces + arrow
    separators; harness judge 1.00 -> 0.00) and monaco_920 emitted
    'J.R.R. Tolkien**' with markdown-bold residue. The answer-format
    contract says comma-separated values, no markup; every arm shares this
    normalizer, so it can only converge answers TOWARD the contract.
    """
    from panini_dr.adapters.ours.gsw_planner_orchestrator_v1 import (
        _normalize_final_answer as norm,
    )
    # the monaco_1512 artifact: narrow no-break spaces + arrow separators
    assert norm("Barbara Bouchet → Caroline Bliss → Samantha Bond") == \
        "Barbara Bouchet, Caroline Bliss, Samantha Bond"
    # the monaco_920 artifact: markdown bold residue
    assert norm("J.R.R. Tolkien**") == "J.R.R. Tolkien"
    # the render's middot joiner leaking into an answer
    assert norm("Kimpembe · Mbappé · Dagba") == "Kimpembe, Mbappé, Dagba"
    # answers already on contract are untouched
    assert norm("Halsey, Khalid, Calvin Harris") == "Halsey, Khalid, Calvin Harris"
    assert norm("True") == "True"
    assert norm("") == ""
    assert norm("36.36363636363637") == "36.36363636363637"
    # a SINGLE arrow inside prose is not a list separator -- untouched
    assert norm("A → B") == "A → B"
    # nbsp variants normalize even without separators
    assert norm("Margaret I of Denmark") == "Margaret I of Denmark"
