"""Failure-mode taxonomy for per-question auto-classification.

Every question in every cell gets exactly one of these tags so we can
histogram failure modes across the substitution grid.
"""

from __future__ import annotations

from enum import Enum


class FailureMode(str, Enum):
    """Canonical failure modes for the substitution pilot.

    ``correct`` is the only non-failure tag. Everything else represents
    a specific way the trajectory went wrong — we want to see *which*
    failure dominates at each (system, model) cell.
    """

    CORRECT = "correct"

    # Answer is wrong AND the model confidently produced it without
    # sufficient grounding (tool-calls or retrieved docs).
    HALLUCINATION = "hallucination"

    # Retrieval was invoked but the relevant documents never made it
    # into the working context (top-k misses, bad query formulation).
    WRONG_RETRIEVAL = "wrong_retrieval"

    # The question has N sub-questions; the agent answered M<N before
    # finalizing (missed a constraint / skipped a hop).
    INCOMPLETE_DECOMPOSITION = "incomplete_decomposition"

    # Agent emitted a final answer (or gave up) with fewer tool calls
    # than the question needs (<min_turns heuristic).
    EARLY_STOP = "early_stop"

    # Agent looped: repeated near-identical queries or tool calls
    # without progress for >= N turns.
    LOOP = "loop"

    # A tool returned an error (retriever timeout, 4xx, parse fail) and
    # the agent did not recover.
    TOOL_ERROR = "tool_error"

    # Wrong answer that doesn't fit any of the above — default bucket
    # for audit. Kept distinct from hallucination because there IS
    # grounding; the model just mis-synthesized.
    WRONG_SYNTHESIS = "wrong_synthesis"

    # Ran over the turn / token budget before reaching an answer.
    BUDGET_EXCEEDED = "budget_exceeded"

    # Couldn't classify (e.g. adapter crashed, no trajectory). Should
    # be rare; surface in audit.
    UNKNOWN = "unknown"

    @classmethod
    def failures(cls) -> list["FailureMode"]:
        return [m for m in cls if m is not cls.CORRECT]
