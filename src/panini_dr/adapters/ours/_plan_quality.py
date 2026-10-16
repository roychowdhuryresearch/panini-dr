"""Deterministic plan-quality checks (W5.1). Pure functions over a GSWPlan.

CHEAP and STABLE structural/heuristic checks emitting PlanIssue records. These
are log_only signals (spec §6 W5) — nothing here changes run behavior. The
date/source checks (C8/C9) are HEURISTIC: they flag *likely* dropped scope, not
guaranteed semantic correctness (the W5.2 LLM auditor handles subtler cases).
"""
from __future__ import annotations

import re

from ._gsw_workspace import PlanIssue, PlanQualityReport
from ._planner_exec import ExecutionError, GSWPlan, topological_sort_blanks
from ._blank_resolver import get_retrieval_hint
from ._blank_verifier import get_verification_claim

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")
_MONTH = (r"(?:January|February|March|April|May|June|July|August|September|"
          r"October|November|December)")
# An explicit as-of date scope: "as of August 3, 2024", "on 8/3/2024", "current as of 2024".
_AS_OF_RE = re.compile(
    rf"\b(?:as of|current as of|on)\s+"
    rf"({_MONTH}\s+\d{{1,2}},?\s*\d{{4}}|\d{{1,2}}/\d{{1,2}}/\d{{4}}|{_MONTH}\s+\d{{4}}|(?:19|20)\d{{2}})",
    re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_SOURCE_RE = re.compile(
    r"\b(?:according to|per)\s+(?:the\s+)?([A-Z][\w&'-]*(?:\s+[A-Z][\w&'-]*)*)"
    r"|\b([A-Z][\w&'-]*(?:\s+[A-Z][\w&'-]*)*\s+(?:Index|Ranking|Report|Survey))\b")


def _hint_and_claim_corpus(plan: GSWPlan) -> str:
    parts: list[str] = []
    for e in plan.blank_entities():
        for text in (get_retrieval_hint(plan, e.id), get_verification_claim(plan, e.id)):
            if text:
                parts.append(text)
    return "  ".join(parts)


def assess_plan_quality_deterministic(
    plan: GSWPlan, *, question: str, use_hints: bool,
) -> PlanQualityReport:
    """Run all deterministic checks; return a PlanQualityReport (log_only)."""
    issues: list[PlanIssue] = []
    blanks = list(plan.blank_entities())
    blank_ids = {e.id for e in blanks}

    # C1: exactly one target.
    targets = [e for e in blanks if getattr(e, "is_target", False)]
    if len(targets) != 1:
        issues.append(PlanIssue(
            severity="high", issue_type="target_unreachable", blank_id=None,
            evidence=f"plan has {len(targets)} target blanks (expected exactly 1)",
            suggested_fix="mark exactly one blank is_target=True"))

    # C2: every blank has a value_type.
    for e in blanks:
        if not (getattr(e, "value_type", None) or "").strip():
            issues.append(PlanIssue(
                severity="medium", issue_type="underspecified_blank", blank_id=e.id,
                evidence=f"blank {e.id!r} has no value_type",
                suggested_fix="set a value_type (entity/number/date/...) on the blank"))

    # C3: no dependency cycle (defensive — usually pre-filtered upstream).
    try:
        topological_sort_blanks(plan)
    except ExecutionError as exc:
        if exc.kind == "cyclic_plan":
            issues.append(PlanIssue(
                severity="high", issue_type="cycle", blank_id=None,
                evidence=f"dependency cycle: {exc.detail}",
                suggested_fix="break the cycle in the blank dependencies"))

    # C4/C5: hint/claim presence when hints are on.
    if use_hints:
        for e in blanks:
            if not get_retrieval_hint(plan, e.id):
                issues.append(PlanIssue(
                    severity="medium", issue_type="bad_hint", blank_id=e.id,
                    evidence=f"blank {e.id!r} has no retrieval_hint (hints are on)",
                    suggested_fix="emit a retrieval_hint for this blank"))
        for e in targets:
            if not get_verification_claim(plan, e.id):
                issues.append(PlanIssue(
                    severity="medium", issue_type="bad_hint", blank_id=e.id,
                    evidence=f"target {e.id!r} has no verification_claim (hints are on)",
                    suggested_fix="emit a verification_claim for the target"))

    # C6/C7: placeholder integrity.
    for e in blanks:
        for field_name, text in (("retrieval_hint", get_retrieval_hint(plan, e.id)),
                                 ("verification_claim", get_verification_claim(plan, e.id))):
            if not text:
                continue
            for ref in set(_PLACEHOLDER_RE.findall(text)):
                if ref not in blank_ids:
                    issues.append(PlanIssue(
                        severity="high", issue_type="placeholder_error", blank_id=e.id,
                        evidence=f"{field_name} of {e.id!r} references unknown blank {{{ref}}}",
                        suggested_fix=f"reference a real upstream blank id, not {{{ref}}}"))
            stripped = _PLACEHOLDER_RE.sub("", text)
            for bid in blank_ids:
                if re.search(rf"\b{re.escape(bid)}\b", stripped):
                    issues.append(PlanIssue(
                        severity="medium", issue_type="placeholder_error", blank_id=e.id,
                        evidence=f"{field_name} of {e.id!r} has raw blank id {bid!r} outside a placeholder",
                        suggested_fix=f"wrap upstream references as {{{bid}}} or remove the raw id"))

    # C8/C9: heuristic scope/source survival (hints on). Corpus matched
    # case-insensitively to avoid trivial false positives ("Democracy Index"
    # vs "democracy index").
    if use_hints:
        corpus = _hint_and_claim_corpus(plan)
        corpus_l = corpus.lower()
        # C8a: explicit as-of date spans -> HIGH. Collect the YEARS inside those
        # spans so C8b does not double-flag them as bare years.
        as_of_spans = {m.group(1) for m in _AS_OF_RE.finditer(question)}
        as_of_years: set[str] = set()
        for span in as_of_spans:
            as_of_years.update(_YEAR_RE.findall(span))
        for d in as_of_spans:
            if d and d.lower() not in corpus_l:
                issues.append(PlanIssue(
                    severity="high", issue_type="missing_scope", blank_id=None,
                    evidence=f"question scopes 'as of {d}' but no hint/claim mentions {d!r}",
                    suggested_fix=f"add the temporal scope 'as of {d}' to the relevant hint/claim"))
        # C8b: bare years NOT already inside an explicit as-of span -> LOW.
        for y in set(_YEAR_RE.findall(question)):
            if y not in corpus and y not in as_of_years:
                issues.append(PlanIssue(
                    severity="low", issue_type="missing_scope", blank_id=None,
                    evidence=f"question mentions year {y!r} not present in any hint/claim (may be fine)",
                    suggested_fix=f"check whether {y!r} should scope a hint/claim"))
        # C9: named source/ranking -> MEDIUM (case-insensitive).
        for m in _SOURCE_RE.finditer(question):
            src = (m.group(1) or m.group(2) or "").strip()
            if src and src.lower() not in corpus_l:
                issues.append(PlanIssue(
                    severity="medium", issue_type="missing_scope", blank_id=None,
                    evidence=f"question names source {src!r} but no hint/claim mentions it",
                    suggested_fix=f"add the source constraint {src!r} to the relevant hint/claim"))

    return PlanQualityReport(issues=issues)
