"""Final-answer finishing: ``finish_value``.

What is left of the GSW-Champion L1 temporal layer. The layer (anchor
extraction, valid-time intervals, anchored overwrite selection, NOTE lines)
was never enabled and was removed; ``finish_value`` runs on every final
answer (see the orchestrator's readouts). Pure code, no LLM calls.

``finish_value`` was copied (copy-not-couple) from the research repo's
``investigator.derive.finish_value``, with the divergence documented below.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

_BARE_YEAR_RE = re.compile(r"^\s*(?:19|20)\d{2}\s*$")


# ---------------------------------------------------------------------------
# Finishing (copied from investigator/derive.py — copy-not-couple)
# ---------------------------------------------------------------------------

# Broad years-cue: drives question-format half-up rounding of a *non-integral*
# value (e.g. "in years only" -> 12.15 -> 12).
# copied from investigator/derive.py — copy-not-couple
_YEARS_CUE_RE = re.compile(r"\b(age|years?|how old|older|younger)\b", re.I)
# Narrow AGE-cue: drives ONLY the days->years defense-in-depth below.
# DIVERGENCE from the copied investigator original (review finding 1): the
# original reused the broad _YEARS_CUE_RE for the days-defense, so ANY numeric
# answer whose magnitude exceeds 150 was treated as a day-count whenever the
# question merely contained the word "year(s)". That converts correct 4-digit
# year answers ("In what year was it founded?" -> 1994 -> 5) and legitimate
# large year-counts ("how many years between 1800 and 2020" -> 220 -> 1) into
# nonsense — regressing the exact temporal/year class L1 exists to improve.
# Restricting the days-defense to genuine age semantics (and never firing on a
# value that is itself a plain 4-digit calendar year or a date-typed target)
# keeps the q630 day-count recovery while leaving year answers untouched.
_AGE_CUE_RE = re.compile(r"\b(age|how old|older|younger)\b", re.I)
_ROUND_CUE_RE = re.compile(r"\bround(ed|ing)?\b|\bnearest\b", re.I)
_PERCENT_CUE_RE = re.compile(r"\bpercent(age|ages)?\b|%", re.I)
_DATE_LIKE_TYPES = frozenset({"date", "datetime", "year"})
_DAYS_DEFENSE_THRESHOLD = 150
_DAYS_PER_YEAR = 365.2425


def _round_half_up(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(-value + 0.5))


def finish_value(value: Any, *, value_type: Optional[str], question: str) -> str:
    """Format an already-resolved target value into the final answer string.

    # copied from investigator/derive.py — copy-not-couple

    Floats that are integral render without a trailing ``.0``. Non-integral
    numbers round half-up to an int when the question carries a years-cue
    or an explicit "round"/"nearest" instruction. As defense-in-depth
    against a mis-routed day-scale value reaching here under an age question
    (the q630 class — 8140 "days" instead of an age in years), an age-cue
    value whose magnitude exceeds 150 is treated as days and converted to
    years first. That day-defense DIVERGES from the copied investigator
    original (review finding 1): it fires only on genuine age semantics
    (``_AGE_CUE_RE``, not the broad years-cue) and never on a value that is
    itself a plain 4-digit calendar year or a date-typed target, so correct
    year answers survive. Non-numeric values (entities, text) pass through
    unchanged.
    """
    raw = str(value).strip()
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return raw

    if (
        _AGE_CUE_RE.search(question)
        and abs(v) > _DAYS_DEFENSE_THRESHOLD
        and not _BARE_YEAR_RE.match(raw)
        and not (v == int(v) and 1900 <= abs(v) <= 2099)
        and (value_type or "").strip().lower() not in _DATE_LIKE_TYPES
    ):
        return str(int(_round_half_up(v / _DAYS_PER_YEAR)))
    # A "what percentage" question answered with a bare fraction (monaco_448:
    # 0.2857142857 against gold 38.46) scores 0 while carrying the right
    # quantity. Scale only a value strictly inside (0, 1): 0 has no ambiguity
    # to resolve, and 1.0 cannot be told apart from a legitimate answer of 1%.
    # The round() strips the float noise 0.2857142857 * 100 would otherwise
    # leave (28.571428570000002).
    if _PERCENT_CUE_RE.search(question) and 0 < v < 1:
        v = round(v * 100, 10)
    if v == int(v):
        return str(int(v))
    if _YEARS_CUE_RE.search(question) or _ROUND_CUE_RE.search(question):
        return str(_round_half_up(v))
    return str(v)
