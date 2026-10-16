"""Smoke tests for the eval contract — no LLM calls, no vLLM, just shapes."""

from __future__ import annotations

import pytest

from panini_dr.adapters.base import (
    Adapter,
    AdapterContext,
    RegistryError,
    get_adapter,
    register_adapter,
)
from panini_dr.eval.failure_classifier import classify
from panini_dr.eval.frames_dataset import FramesQuestion
from panini_dr.eval.harness import run_cell
from panini_dr.eval.scoring import exact_match, token_f1
from panini_dr.eval.subset import select_stratified_subset
from panini_dr.models.failure_modes import FailureMode
from panini_dr.models.trace import QuestionResult, ToolCall, Trajectory


@pytest.fixture
def fake_question() -> FramesQuestion:
    return FramesQuestion(
        id="999",
        question="What year was the Boeing 777 first flown?",
        answer="1994",
        articles=[{"title": "Boeing 777", "url": "https://en.wikipedia.org/wiki/Boeing_777"}],
        reasoning_types=["Temporal reasoning"],
        num_hops=2,
    )


@pytest.fixture
def fake_adapter_class() -> type[Adapter]:
    @register_adapter
    class FakeAdapter(Adapter):
        system_id = "test_fake_adapter"
        display_name = "Fake adapter for tests"

        def run_question(self, question, *, question_id, articles=None):
            return Trajectory(
                system_id=self.system_id,
                model_id=self.ctx.model_id,
                question_id=question_id,
                turns=2,
                tool_calls=[
                    ToolCall(turn=1, name="search", args={"q": "Boeing 777 first flight"}, result_preview="Boeing 777 first flew in 1994."),
                    ToolCall(turn=2, name="read", args={"id": "doc_1"}, result_preview="Maiden flight 12 June 1994."),
                ],
                final_answer="1994",
                prompt_tokens=500,
                completion_tokens=100,
            )

    return FakeAdapter


def test_scoring_exact_match_and_f1():
    assert exact_match("1994", "1994") is True
    assert exact_match("The year 1994.", "1994") is False
    assert token_f1("first flown in 1994", "1994") > 0.3
    assert token_f1("completely wrong", "1994") == 0.0


def test_subset_selection_is_reproducible(tmp_path):
    questions = [
        FramesQuestion(id=str(i), question=f"q{i}", answer="a", num_hops=(i % 6) + 1)
        for i in range(50)
    ]
    s1 = select_stratified_subset(questions, seed=42)
    s2 = select_stratified_subset(questions, seed=42)
    assert s1.question_ids == s2.question_ids
    assert sum(s1.stratification.values()) == len(s1.question_ids)

    out = tmp_path / "subset.json"
    s1.save(out)
    loaded = type(s1).load(out)
    assert loaded.question_ids == s1.question_ids


def test_registry_lookup_rejects_unknown():
    with pytest.raises(RegistryError):
        get_adapter("definitely_not_registered")


def test_run_cell_end_to_end(fake_adapter_class, fake_question):
    ctx = AdapterContext(
        system_id=fake_adapter_class.system_id,
        model_id="fake-model",
        max_turns=4,
    )
    adapter = fake_adapter_class(ctx)
    cell = run_cell(
        adapter,
        [fake_question],
        subset_id="fake_subset",
        benchmark="frames",
    )
    assert cell.n_total == 1
    assert cell.n_correct == 1
    assert cell.accuracy == 1.0
    assert cell.questions[0].failure_mode is FailureMode.CORRECT
    assert cell.mean_turns == 2.0


def test_start_log_line_omits_a_missing_hop_count(fake_adapter_class, fake_question, caplog,
                                                  monkeypatch):
    """FRAMES-style hop counts are logged; datasets without one (your own data,
    num_hops=0) get no `hops=0` field. Only the log line changes."""
    import logging

    no_hops = fake_question.model_copy(update={"id": "1000", "num_hops": 0})
    adapter = fake_adapter_class(AdapterContext(
        system_id=fake_adapter_class.system_id, model_id="fake-model", max_turns=4))
    # an earlier run in this process may have set up the run logger (propagate=False)
    monkeypatch.setattr(logging.getLogger("panini_dr"), "propagate", True)
    caplog.set_level(logging.INFO, logger="panini_dr")
    cell = run_cell(adapter, [fake_question, no_hops], subset_id="s", benchmark="frames")
    starts = {r.getMessage().split(" ", 1)[0]: r.getMessage()
              for r in caplog.records if " start (" in r.getMessage()}
    assert " hops=2 |" in starts["q999"]
    assert "hops=" not in starts["q1000"] and starts["q1000"].startswith("q1000 start (2/2) |")
    assert cell.n_total == 2


def test_classifier_flags_hallucination(fake_question):
    traj = Trajectory(
        system_id="x",
        model_id="y",
        question_id=fake_question.id,
        turns=1,
        tool_calls=[],
        final_answer="1492",  # Wrong + no retrieval
    )
    qr = QuestionResult(
        question_id=fake_question.id,
        question=fake_question.question,
        gold_answer=fake_question.answer,
        trajectory=traj,
        predicted_answer="1492",
        exact_match=False,
        f1=0.0,
    )
    assert classify(qr, num_hops=fake_question.num_hops) is FailureMode.HALLUCINATION
