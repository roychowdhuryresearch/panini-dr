"""Deterministic worksheet operators over evidence-backed CandidateRows."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class CandidateRow:
    """Evidence-backed candidate for one blank. Formerly lived in the
    (removed) log-only evidence workspace; the deterministic worksheet
    operators and the constraint solver are its remaining consumers."""

    candidate: str
    attributes: dict[str, Any]
    evidence_by_field: dict[str, list[str]]
    passes_filters: dict[str, bool]
    rank: int | None
    status: Literal["candidate", "rejected", "winner", "needs_evidence"]


def filter_rows(
    rows: Sequence[CandidateRow],
    predicate: Callable[[CandidateRow], bool] | Mapping[str, Any],
) -> list[CandidateRow]:
    if callable(predicate):
        return [row for row in rows if bool(predicate(row))]
    return [row for row in rows if _matches(row, predicate)]


def argmax(rows: Sequence[CandidateRow], numeric_attribute: str) -> CandidateRow:
    return max(_numeric_rows(rows, numeric_attribute), key=lambda item: item[0])[1]


def argmin(rows: Sequence[CandidateRow], numeric_attribute: str) -> CandidateRow:
    return min(_numeric_rows(rows, numeric_attribute), key=lambda item: item[0])[1]


def rank_n(
    rows: Sequence[CandidateRow],
    numeric_attribute: str,
    n: int,
    *,
    descending: bool = True,
) -> CandidateRow:
    if n < 1:
        raise ValueError("n is one-based and must be >= 1")
    ranked = sorted(
        _numeric_rows(rows, numeric_attribute),
        key=lambda item: item[0],
        reverse=descending,
    )
    if n > len(ranked):
        raise ValueError(f"rank {n} requested from only {len(ranked)} rows")
    return ranked[n - 1][1]


def count(rows: Sequence[CandidateRow]) -> int:
    return len(rows)


def _matches(row: CandidateRow, conditions: Mapping[str, Any]) -> bool:
    for field, expected in conditions.items():
        if field == "candidate":
            actual = row.candidate
        else:
            actual = row.attributes.get(field)
        if actual != expected:
            return False
    return True


def _numeric_rows(
    rows: Sequence[CandidateRow], numeric_attribute: str
) -> list[tuple[float, CandidateRow]]:
    out: list[tuple[float, CandidateRow]] = []
    for row in rows:
        value = row.attributes.get(numeric_attribute)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"row {row.candidate!r} has no numeric attribute "
                f"{numeric_attribute!r}"
            )
        out.append((float(value), row))
    if not out:
        raise ValueError("no candidate rows supplied")
    return out
