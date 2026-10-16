"""Seed skill functions for the codegen compute library.

These are the hand-written, trusted pure functions that seed the library so
that common compute ops are a deterministic *library hit* from turn one (no
LLM call). They are intentionally **self-contained** (only ``datetime`` +
stdlib) so that the ``codegen_library`` package has no dependency on
``_planner_exec`` (avoids a circular import once ``_planner_exec`` routes into
this package).

The two seeded date functions deterministically fix the date-question failure
class:

* ``date_diff_days`` — day-granular absolute date difference. Fixes the
  ``diff`` path that used to collapse both operands to their 4-digit year
  (``abs(2015-2015)==0`` for every candidate -> ties -> wrong pick), e.g.
  q413.
* ``age_years_on`` — birthday-aware completed years between a birth date and
  an as-of date. Fixes the off-by-one from raw year subtraction, e.g.
  q209 / q25.
"""

from __future__ import annotations

import datetime
import re

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _to_date(value: str) -> datetime.date:
    """Parse an ISO or long-form date string into a ``datetime.date``.

    Accepts ``YYYY-MM-DD`` / ``YYYY/MM/DD``, ``Month DD, YYYY`` and
    ``DD Month YYYY``. Missing day defaults to 1. Raises ``ValueError`` on an
    unparseable string.
    """
    s = str(value).strip()
    if not s:
        raise ValueError("empty date")

    m = re.search(r"\b(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?\b", s)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3)) if m.group(3) else 1
        return datetime.date(year, month, day)

    m = re.search(r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b", s)
    if m and m.group(1).casefold() in _MONTHS:
        return datetime.date(int(m.group(3)), _MONTHS[m.group(1).casefold()], int(m.group(2)))

    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b", s)
    if m and m.group(2).casefold() in _MONTHS:
        return datetime.date(int(m.group(3)), _MONTHS[m.group(2).casefold()], int(m.group(1)))

    m = re.search(r"\b([A-Za-z]+)\s+(\d{4})\b", s)
    if m and m.group(1).casefold() in _MONTHS:
        return datetime.date(int(m.group(2)), _MONTHS[m.group(1).casefold()], 1)

    raise ValueError(f"unparseable date: {value!r}")


def date_diff_days(a: str, b: str) -> int:
    """Absolute number of whole days between two dates (day-granular).

    Accepts ISO (``2015-03-20``) or long-form (``March 15, 2015``) dates and
    returns ``abs(days)`` so argument order does not matter.
    """
    return abs((_to_date(a) - _to_date(b)).days)


def age_years_on(born: str, on_date: str) -> int:
    """Completed years elapsed from ``born`` as of ``on_date`` (birthday-aware).

    Returns the number of full anniversary years reached by ``on_date``; the
    count ticks over on the birthday itself, not on Jan 1. Argument order does
    not matter: the earlier of the two dates is treated as the birth date, so
    the constraint may pass the dates in either order.
    """
    b = _to_date(born)
    o = _to_date(on_date)
    if o < b:
        b, o = o, b
    years = o.year - b.year
    if (o.month, o.day) < (b.month, b.day):
        years -= 1
    return max(years, 0)


# Seed table consumed by ``registry.Library.load()``. Each entry maps a stable
# operation key -> the callable + its manifest metadata (name/signature/
# docstring surfaced to the LLM on a library miss).
SEEDS: list[dict] = [
    {
        "key": ("diff", ("date", "date"), "days"),
        "name": "date_diff_days",
        "fn": date_diff_days,
        "signature": "date_diff_days(a: str, b: str) -> int",
        "docstring": (
            "Absolute number of whole days between two dates (day-granular). "
            "Accepts ISO or long-form dates; argument order does not matter."
        ),
    },
    {
        "key": ("diff", ("date", "date"), "age_years"),
        "name": "age_years_on",
        "fn": age_years_on,
        "signature": "age_years_on(born: str, on_date: str) -> int",
        "docstring": (
            "Completed years elapsed from a birth date as of a given date "
            "(birthday-aware; the count ticks over on the birthday)."
        ),
    },
]
