from __future__ import annotations

import pytest

from panini_dr.adapters.ours._gsw_worksheet_ops import (
    CandidateRow,
    argmax,
    argmin,
    count,
    filter_rows,
    rank_n,
)


def _row(candidate: str, **attributes):
    return CandidateRow(
        candidate=candidate,
        attributes=dict(attributes),
        evidence_by_field={k: [f"{candidate}:{k}:chunk"] for k in attributes},
        passes_filters={},
        rank=None,
        status="candidate",
    )


def test_filter_rows_with_callable_and_mapping_returns_existing_rows():
    rows = [
        _row("Blue Yodel No. 1", year=1928, sales=200),
        _row("Ramona", year=1928, sales=100),
        _row("Other", year=1929, sales=300),
    ]

    same_year = filter_rows(rows, {"year": 1928})
    high_sales = filter_rows(rows, lambda row: row.attributes["sales"] > 150)

    assert same_year == rows[:2]
    assert high_sales == [rows[0], rows[2]]
    assert same_year[0] is rows[0]


def test_argmax_and_argmin_choose_existing_candidate_rows():
    rows = [
        _row("Franklin D. Roosevelt", years_served=12),
        _row("Dwight D. Eisenhower", years_served=8),
    ]

    assert argmax(rows, "years_served") is rows[0]
    assert argmin(rows, "years_served") is rows[1]


def test_rank_n_is_one_based_and_returns_existing_row_without_mutating():
    rows = [
        _row("Blue Yodel No. 1 (T for Texas)", sales=500),
        _row("Ramona", sales=300),
        _row("Highest Song", sales=800),
    ]

    selected = rank_n(rows, "sales", n=2, descending=True)

    assert selected is rows[0]
    assert [row.rank for row in rows] == [None, None, None]


def test_count_does_not_invent_values():
    rows = [_row("A", year=1928), _row("B", year=1929)]

    assert count(rows) == 2


def test_operators_reject_missing_numeric_attributes():
    rows = [_row("A", year="unknown")]

    with pytest.raises(ValueError):
        argmax(rows, "year")
    with pytest.raises(ValueError):
        rank_n(rows, "year", n=1)
