"""Cheap typed candidate filtering for blank resolution.

Value-local heuristics only (brief §8: "do not overbuild"). The hard type
filter is an admission lever: a ``date`` blank should not accept a person's
name, a ``number`` blank should not accept prose. It is permissive for
entity/attribute/text/list (those have no cheap shape check). Needs only the
value + its declared type — no other blank — so it is safe in Stage A.
"""

from __future__ import annotations

import re
from typing import Any

_DATE_RE = re.compile(
    r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})\b"
)
_YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")
_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _looks_like_year(v: str) -> bool:
    return bool(_YEAR_RE.search(v))


def _looks_like_date(v: str) -> bool:
    return bool(_DATE_RE.search(v)) or _looks_like_year(v)


def _looks_like_number(v: str) -> bool:
    s = v.strip().replace(",", "")
    return bool(_NUM_RE.fullmatch(s)) or bool(_NUM_RE.search(v))


def object_matches_value_type(value: Any, value_type: str) -> bool:
    """True iff ``value`` plausibly matches ``value_type``. Permissive for
    entity/attribute/text/list/unknown (no cheap shape check)."""
    if value is None:
        return False
    v = str(value).strip()
    if not v:
        return False
    vt = (value_type or "").lower()
    if vt == "date":
        return _looks_like_date(v)
    if vt == "number":
        return _looks_like_number(v)
    if vt == "bool":
        return v.lower() in {"true", "false", "yes", "no"}
    # entity / attribute / text / list / unknown → permissive
    return True
