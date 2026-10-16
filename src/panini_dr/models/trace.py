"""Pydantic schemas for trajectories, per-question results, and aggregated cell results.

The eval harness persists these to JSON so any adapter run can be
replayed / audited / diffed against another cell's run.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from panini_dr.models.failure_modes import FailureMode


class ToolCall(BaseModel):
    """A single tool invocation inside a trajectory."""

    turn: int
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result_preview: str = Field(
        "",
        description="First ~500 chars of the tool result — for quick terminal logs.",
    )
    result_full: str = Field(
        "",
        description="Full untruncated tool result JSON. Persists the whole payload "
        "into the trajectory trace so we can audit later without re-running.",
    )
    duration_s: float = 0.0
    error: str = ""
    level: int | None = None  # set by orchestrator adapters to identify the researcher slice


class Trajectory(BaseModel):
    """The full trace of one agent's attempt at one question."""

    system_id: str
    model_id: str
    question_id: str

    turns: int = 0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_s: float = 0.0

    final_answer: str = ""
    reasoning: str = Field(
        "",
        description="Adapter-synthesized reasoning summary (tag contents, aggregated thoughts, etc.).",
    )
    hidden_reasoning: str = Field(
        "",
        description="Raw hidden-reasoning content from the model (Bedrock gpt-oss "
        "`reasoning_content`, OpenAI reasoning summaries). Concatenated across all "
        "LLM calls in one question.",
    )

    # Full message history is optional + large — adapters may skip to save disk.
    messages: list[dict[str, Any]] = Field(default_factory=list)

    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Adapter-specific state: GSW snapshot, retrieved doc IDs, etc.",
    )


class QuestionResult(BaseModel):
    """Per-question scored result for one cell."""

    question_id: str
    question: str
    gold_answer: str

    trajectory: Trajectory
    predicted_answer: str = ""

    # Scoring
    exact_match: bool = False
    f1: float = 0.0
    judge_correct: bool | None = Field(
        None,
        description="LLM-as-judge verdict (None = not yet judged or skipped).",
    )
    failure_mode: FailureMode = FailureMode.UNKNOWN

    notes: str = ""


class CellResult(BaseModel):
    """Aggregated result for one (system, model, benchmark, subset) run."""

    system_id: str
    model_id: str
    benchmark: str
    subset_id: str

    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    questions: list[QuestionResult] = Field(default_factory=list)

    # Headline
    n_total: int = 0
    n_correct: int = 0
    accuracy: float = 0.0
    mean_f1: float = 0.0

    # LLM-judge aggregates (populated when a judge is configured).
    n_judge_correct: int = 0
    n_judge_seen: int = 0
    judge_accuracy: float = 0.0

    # Agent-behavior stats (means over questions)
    mean_turns: float = 0.0
    mean_tool_calls: float = 0.0
    mean_prompt_tokens: float = 0.0
    mean_completion_tokens: float = 0.0
    mean_wall_time_s: float = 0.0

    # Failure-mode histogram — keys are FailureMode values, values are counts.
    failure_histogram: dict[str, int] = Field(default_factory=dict)

    # Free-form config snapshot so the run is reproducible.
    config: dict[str, Any] = Field(default_factory=dict)

    def summary_row(self) -> dict[str, Any]:
        """One row for the substitution grid table."""
        return {
            "system": self.system_id,
            "model": self.model_id,
            "benchmark": self.benchmark,
            "subset": self.subset_id,
            "n": self.n_total,
            "accuracy": round(self.accuracy, 4),
            "judge_accuracy": round(self.judge_accuracy, 4),
            "n_judge_correct": self.n_judge_correct,
            "n_judge_seen": self.n_judge_seen,
            "mean_f1": round(self.mean_f1, 4),
            "mean_turns": round(self.mean_turns, 2),
            "mean_tools": round(self.mean_tool_calls, 2),
            "mean_tokens_out": round(self.mean_completion_tokens, 0),
            "mean_wall_s": round(self.mean_wall_time_s, 2),
            **{f"fm_{k}": v for k, v in self.failure_histogram.items()},
        }
