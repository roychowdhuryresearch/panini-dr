"""Shared plan-emission helper for the GSW-fragment planner family.

Lifted from ``gsw_planner_v1.py`` so multiple adapters (the deterministic
Python executor in ``gsw_planner_v1`` and the ReAct-reasoner in
``gsw_planner_react_v1``) can share the same "one LLM call → Pydantic
validate → one repair retry" discipline without reimplementing it.

The helper does NOT decide fallback semantics. It raises a structured
``PlanEmitError`` and the caller chooses whether to fall back to a flat
adapter or surface the failure some other way.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from panini_dr.adapters.ours._planner_exec import (
    GSWPlan,
    _extract_json_object,
)
from panini_dr.adapters.ours._planner_prompts import (
    build_planner_messages,
    build_repair_messages,
)

# ---------------------------------------------------------------------------
# Error + metadata shapes
# ---------------------------------------------------------------------------


class PlanEmitError(Exception):
    """Raised when the planner LLM call or plan validation fails terminally.

    ``kind`` is one of:
      - ``"llm_error"`` — network / API / client raised before a response.
      - ``"parse_failure"`` — a single parse attempt failed Pydantic
        validation / JSON extraction. Used by callers (e.g. the plan
        updater) that don't run the multi-retry repair loop.
      - ``"exhausted"`` — the multi-retry repair loop in :func:`emit_plan`
        ran ``MAX_REPAIR_ATTEMPTS`` repair calls and every one of them
        failed validation. Callers should NOT silently fall back to a
        flat reasoner — the planner is broken for this question and a
        sentinel-from-priors answer is worse than an empty submission.
    ``detail`` contains a short human-readable error summary.
    """

    def __init__(
        self,
        kind: Literal["llm_error", "parse_failure", "exhausted"],
        detail: str = "",
    ):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail


@dataclass
class PlanEmitMeta:
    """Bookkeeping the caller needs to populate its Trajectory.extra."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: str = ""
    raw_reasoning: str = ""
    repair_used: bool = False
    # Most-recent repair attempt — kept for legacy callers / debug UI.
    repair_response: str = ""
    repair_reasoning: str = ""
    # Per-attempt log; index 0 is the first repair, index n-1 is the last.
    # Each entry: {"response": str, "reasoning": str, "error": str}.
    repair_attempts: list[dict[str, str]] = field(default_factory=list)
    parse_error: str = ""  # first-pass validation error, even when repair succeeded
    extras: dict[str, Any] = field(default_factory=dict)


# Number of repair calls to attempt before raising ``exhausted``.
# The first LLM call + this many repairs = 1 + MAX_REPAIR_ATTEMPTS total parses.
MAX_REPAIR_ATTEMPTS = 3
_DORMANT_ROW_AGGREGATION_CONSTRAINT_KINDS = {"rank_n", "filter", "count"}


# ---------------------------------------------------------------------------
# Main helper
# ---------------------------------------------------------------------------


def emit_plan(
    question: str,
    llm_client: Any,
    *,
    max_tokens: int = 4096,
    enable_repair: bool = True,
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
    llm_seed: int | None = None,
    blank_qualifier: bool = False,
    blank_hints: bool = False,
) -> tuple[GSWPlan, PlanEmitMeta]:
    """Emit + validate a ``GSWPlan`` from a question with multi-retry repair.

    Workflow:
      1. One initial planner LLM call.
      2. If it parses, return.
      3. Otherwise, run up to ``max_repair_attempts`` repair calls. Each
         repair message is sharpened with the attempt index so the LLM
         knows it has already failed earlier rounds.
      4. If every repair fails, raise ``PlanEmitError(kind="exhausted")``
         carrying the last validation error.

    Returns ``(plan, meta)`` on success. Raises ``PlanEmitError`` on
    terminal failure (LLM exception OR every repair retry failing).

    The caller is responsible for any fallback plumbing — this function
    is intentionally unaware of alternate execution paths.
    """
    meta = PlanEmitMeta()

    # ---- First call ----------------------------------------------------
    try:
        resp = llm_client.chat(
            messages=build_planner_messages(
                question,
                blank_qualifier=blank_qualifier,
                blank_hints=blank_hints,
            ),
            max_tokens=max_tokens,
            seed=llm_seed,
        )
    except Exception as exc:  # noqa: BLE001
        raise PlanEmitError(kind="llm_error", detail=str(exc)) from exc

    text = getattr(resp, "text", "") or ""
    meta.raw_response = text
    meta.raw_reasoning = getattr(resp, "reasoning_content", "") or ""
    meta.prompt_tokens += int(getattr(resp, "prompt_tokens", 0) or 0)
    meta.completion_tokens += int(getattr(resp, "completion_tokens", 0) or 0)

    # ---- Parse attempt 1 ----------------------------------------------
    try:
        plan = _parse_plan(
            text,
            question=question,
        )
        _attach_plan_audit(meta, question, plan)
        return plan, meta
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        meta.parse_error = str(exc)[:400]
        if not enable_repair:
            raise PlanEmitError(kind="parse_failure", detail=str(exc)) from exc

    # ---- Repair loop (up to max_repair_attempts) ------------------------
    last_bad_response = text
    last_error = meta.parse_error
    last_exc: Exception | None = None
    for attempt_idx in range(max_repair_attempts):
        try:
            repair_resp = llm_client.chat(
                messages=build_repair_messages(
                    question,
                    last_bad_response,
                    last_error,
                    attempt=attempt_idx + 1,
                    max_attempts=max_repair_attempts,
                    blank_hints=blank_hints,
                ),
                max_tokens=max_tokens,
                seed=llm_seed,
            )
        except Exception as exc:  # noqa: BLE001
            raise PlanEmitError(
                kind="llm_error",
                detail=f"repair call {attempt_idx + 1} failed: {exc}",
            ) from exc

        repair_text = getattr(repair_resp, "text", "") or ""
        meta.repair_used = True
        meta.repair_response = repair_text
        meta.repair_reasoning = getattr(repair_resp, "reasoning_content", "") or ""
        meta.prompt_tokens += int(getattr(repair_resp, "prompt_tokens", 0) or 0)
        meta.completion_tokens += int(getattr(repair_resp, "completion_tokens", 0) or 0)

        try:
            plan = _parse_plan(
                repair_text,
                question=question,
            )
            _attach_plan_audit(meta, question, plan)
            meta.repair_attempts.append({
                "response": repair_text,
                "reasoning": meta.repair_reasoning,
                "error": "",
            })
            return plan, meta
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_bad_response = repair_text
            last_error = str(exc)[:400]
            last_exc = exc
            meta.repair_attempts.append({
                "response": repair_text,
                "reasoning": meta.repair_reasoning,
                "error": last_error,
            })

    # All repair attempts exhausted.
    detail = (
        f"after {max_repair_attempts} repair attempts: {last_error}"
        if last_exc is not None
        else f"after {max_repair_attempts} repair attempts"
    )
    raise PlanEmitError(kind="exhausted", detail=detail) from last_exc


def _parse_plan(
    text: str,
    *,
    question: str = "",
) -> GSWPlan:
    """Balanced-brace JSON extract + Pydantic validate."""
    obj = _extract_json_object(text)
    plan = GSWPlan.model_validate(obj)
    _reject_dormant_row_aggregation_constraints(plan)
    _validate_plan_quality(question, plan)
    return plan


def _reject_dormant_row_aggregation_constraints(plan: GSWPlan) -> None:
    blocked = [
        c.kind for c in plan.constraints
        if c.kind in _DORMANT_ROW_AGGREGATION_CONSTRAINT_KINDS
    ]
    if blocked:
        raise ValueError(
            "row_aggregation_constraints_unavailable: rank_n/filter/count "
            "constraints are dormant executor primitives and are not emitted "
            "by the planner"
        )


_NUMERIC_DERIVED_OPS = {
    "diff",
    "sum",
    "avg",
    "max",
    "min",
    "mul",
    "div",
    "round_nearest",
}

# Ops where two-or-more numeric operands are actually COMBINED across
# sources (round_nearest is unary-ish: one value + an optional rounding
# interval, never two independently-retrieved measurements, so it is
# deliberately excluded from the unit-consistency check below).
_UNIT_BEARING_OPS = _NUMERIC_DERIVED_OPS - {"round_nearest"}

# Closed unit vocabulary for the cross-unit-arithmetic guard (see
# _reject_unit_mismatch_constraints). Each canonical unit maps to the
# surface variants matched with \b...\b word boundaries in a
# planner-authored retrieval_hint. Deliberately closed/small -- this is
# NOT a general unit parser, just enough to catch "feet vs metres" class
# mismatches in the hints this project's planner actually emits.
_UNIT_VOCAB: dict[str, tuple[str, ...]] = {
    "feet": ("feet", "foot", "ft"),
    "metres": ("metres", "meters", "metre", "meter"),
    "centimetres": ("centimetres", "centimeters", "centimetre", "centimeter", "cm"),
    "km": ("km", "kilometres", "kilometers", "kilometre", "kilometer"),
    "miles": ("miles", "mile"),
    "kg": ("kg", "kilograms", "kilogram"),
    "pounds": ("pounds", "pound", "lbs", "lb"),
    "seconds": ("seconds", "second"),
    "years": ("years", "year"),
    "people": ("people", "persons", "person"),
    "dollars": ("dollars", "dollar", "usd"),
    "percent": ("percent", "percentage"),
}
# Duration units (minutes/hours/days) are deliberately NOT included: unlike
# "cm" (unambiguous) they collide with common English words that show up
# incidentally in freeform planner-authored hints ("minute" as an adjective,
# "day" in "as of the day of..."), and this vocabulary already gates a
# hard-reject ValueError in _reject_unit_mismatch_constraints below -- a
# false-positive extraction here would newly reject previously-valid plans.
# "seconds" is the one duration-ish exception, kept as pre-existing behavior
# from before this audit; not proof the same tradeoff is safe for the rest.
# Bare "m" (metres) is handled separately from the vocab table above: it
# must be a standalone token (not part of another word) to avoid matching
# random text.
_BARE_METRES_RE = re.compile(r"(?<![A-Za-z0-9])[mM](?![A-Za-z0-9])")
_UNIT_VARIANT_RES: dict[str, re.Pattern] = {
    unit: re.compile(r"\b(?:" + "|".join(re.escape(v) for v in variants) + r")\b")
    for unit, variants in _UNIT_VOCAB.items()
}

# Narrow "this comparison is over a physical measurement" cue. Deliberately
# NOT a scan of the raw question for generic vocabulary words (that would
# false-positive constantly: "twelve YEARS earlier", "how many PEOPLE" are
# idiomatic and carry no cross-source unit-mismatch risk). This instead
# matches the specific magnitude-comparison phrasing that actually produces
# a cross-unit bug -- "how many times deeper/taller/...", "how much
# shorter/longer/...", "how many Xs makes up the height/length/... of Y" --
# which is exactly the shape of the three verified failures (q16, q692,
# q771) this check exists to catch.
_MAGNITUDE_COMPARISON_CUES = re.compile(
    r"\b("
    r"how many times|"
    r"times (?:as )?(?:deep|deeper|tall|taller|long|longer|short|shorter|"
    r"large|larger|small|smaller|heavy|heavier|wide|wider|far|farther|"
    r"high|higher)|"
    r"how much (?:deeper|taller|shorter|longer|heavier|farther|further|"
    r"larger|smaller|higher|wider)|"
    r"makes? up the (?:height|length|depth|width|size|weight) of"
    r")\b",
    flags=re.IGNORECASE,
)

_LITERAL_VALUE_PHRASES = {
    "value",
    "value_is",
    "equals_number",
    "number_value",
    "year_value",
}

_RISKY_HUB_CUES = re.compile(
    r"\b("
    r"difference|sum|total|average|ratio|percent|percentage|rounded|nearest|"
    r"older|younger|longer|shorter|earlier|later|before|after|"
    r"how many|how much|more|less|count|"
    r"top|leaders?|leaderboard|ranked|ranking|list|lists|both|career|"
    r"same award|same [a-z0-9 .,'-]{0,40} award|"
    r"population|score|head-to-head"
    r")\b",
    flags=re.IGNORECASE,
)

_COMPOUND_ANSWER_CUE = re.compile(
    r"\b(?:what|which|who|name|give|provide)\b.{0,80}\band\b.{0,80}"
    r"\b(?:percentage|percent|population|character|role|movie|film|team|score|number|date|year)\b",
    flags=re.IGNORECASE | re.DOTALL,
)

_COMPUTED_TARGET_HUB_CUES = re.compile(
    r"\b("
    r"difference|sum|total|average|ratio|percent|percentage|rounded|nearest|"
    r"how many|how much|how old|how long|how far|"
    r"how many more|age difference|"
    r"older|younger|longer|shorter"
    r")\b",
    flags=re.IGNORECASE,
)

_FINAL_NUMBER_TARGET_CUES = re.compile(
    r"\b("
    r"how many letters|number of letters|"
    r"how many more|how much more|"
    r"how old|age difference|"
    r"difference|sum|total|average|ratio|percent|percentage|"
    r"rounded|nearest"
    r")\b",
    flags=re.IGNORECASE,
)

_COMPARISON_THRESHOLD_CUES = re.compile(
    r"\b("
    r"more than|less than|at least|or more|or less|"
    r"larger than|smaller than|older than|younger than|"
    r"greater than|fewer than"
    r")\b",
    flags=re.IGNORECASE,
)

_FORMAT_CUES = re.compile(
    r"\b(rounded|round|nearest|whole number|decimal places?|two decimals?)\b",
    flags=re.IGNORECASE,
)

_DATE_ANCHOR_CUES = re.compile(
    r"\b("
    r"as of|in the year|the year that|"
    r"\d{4}|"
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december"
    r")\b",
    flags=re.IGNORECASE,
)

_AGE_DATE_GRANULARITY_CUES = re.compile(
    r"\b("
    r"how old|age difference|age in years|age (?:at|when|on)|aged|"
    r"turned|turns|birthday|"
    r"anniversary|on the day|by the time|"
    r"stood in (?:the )?(?:\d{4} )?[a-z ]*election"
    r")\b",
    flags=re.IGNORECASE,
)

_COMPARISON_CONSTRAINT_KINDS = {"gt", "lt", "equals", "argmax", "argmin"}


def _validate_plan_quality(
    question: str,
    plan: GSWPlan,
) -> None:
    """Reject only objective structural planner defects.

    These checks intentionally raise ``ValueError`` so ``emit_plan`` uses
    the existing repair loop. Semantic quality concerns are logged by
    :func:`audit_plan_quality` instead of hard-rejecting otherwise
    executable plans.
    """
    # BCP q7 fix (2026-08-18): first-failure-only turned the repair loop
    # into one-defect-per-round discovery -- a question tripping k of these
    # needs k rounds against a budget of 3. Collect EVERY defect and raise
    # them together so one repair round sees the whole bill.
    _errors: list[str] = []
    for _check in (
        lambda: _reject_literal_constant_blanks(plan),
        lambda: _reject_numeric_literal_refs_without_values(plan),
        lambda: _reject_arg_selection_over_blank_candidates(plan),
        lambda: _reject_date_granularity_missing(question, plan),
        lambda: _reject_unit_mismatch_constraints(question, plan),
        lambda: _reject_dropped_qualifier_hints(question, plan),
    ):
        try:
            _check()
        except ValueError as _exc:
            _errors.append(str(_exc))
    if _errors:
        raise ValueError(
            "; ".join(_errors) if len(_errors) == 1
            else "MULTIPLE DEFECTS, fix ALL in one revision: "
                 + " | ".join(_errors)
        )


def _reject_literal_constant_blanks(plan: GSWPlan) -> None:
    literal_ids = {
        e.id
        for e in plan.filled_entities()
        if e.role == "constraint-value" or e.literal_value is not None
    }
    if not literal_ids:
        return
    blanks = {e.id for e in plan.blank_entities()}
    for vp in plan.verb_phrases:
        literal_to_blank = (
            vp.subject_id in literal_ids and vp.object_id in blanks
        ) or (
            vp.object_id in literal_ids and vp.subject_id in blanks
        )
        if not literal_to_blank:
            continue
        phrase = vp.phrase.strip().lower()
        if phrase in _LITERAL_VALUE_PHRASES or "value" in phrase:
            raise ValueError(
                "literal_constant_blank: do not create a blank whose only "
                f"purpose is to hold literal {vp.subject_id!r}/{vp.object_id!r}; "
                "use a filled constraint-value entity with literal_value and "
                "reference it directly in args_refs"
            )


def _reject_numeric_literal_refs_without_values(plan: GSWPlan) -> None:
    ent_by_id = {e.id: e for e in plan.entities}
    for c in plan.constraints:
        if c.kind != "derived" or c.op not in _NUMERIC_DERIVED_OPS:
            continue
        for ref in c.args_refs:
            ent = ent_by_id.get(ref)
            if (
                ent is not None
                and ent.kind == "filled"
                and ent.role == "constraint-value"
                and ent.literal_value is None
            ):
                raise ValueError(
                    "literal_value_missing: numeric derived constraint "
                    f"{c.id!r} references filled constraint-value {ref!r} "
                    "without literal_value; keep name grounded in the "
                    "question and add the typed literal_value"
                )


def audit_plan_quality(question: str, plan: GSWPlan) -> list[str]:
    """Return non-blocking plan-shape tags for run inspection.

    These tags intentionally do not participate in emit validation. They
    are heuristics for plan-quality analysis, so a false positive should
    be cheap: it appears in logs, but it does not force a repair loop or
    kill a runnable plan.
    """
    tags: set[str] = set()

    if _has_literal_constant_blank(plan) or _has_numeric_literal_ref_without_value(plan):
        tags.add("bad_literal")
    if _arg_selection_blank_candidates(plan):
        tags.add("bad_arg_selector")
    if _has_duplicate_constraint_outputs(plan):
        tags.add("bad_constraint_outputs")

    question = question or ""
    try:
        target = plan.target()
    except Exception:
        return ["bad_target"]

    target_type = (target.value_type or "").lower()
    if _FINAL_NUMBER_TARGET_CUES.search(question) and target_type != "number":
        tags.add("ambiguous_target")

    if _COMPARISON_THRESHOLD_CUES.search(question) and not _has_comparison_constraint(plan):
        tags.add("missing_operation")
    if _FORMAT_CUES.search(question) and not _has_rounding_or_format_constraint(plan):
        tags.add("format_missing")
    if _DATE_ANCHOR_CUES.search(question) and not _has_temporal_anchor(plan):
        tags.add("missing_anchor")
    if _has_date_granularity_gap(question, plan):
        tags.add("date_granularity_missing")
    if _looks_over_decomposed(question, plan):
        tags.add("over_decomposed")
    if _looks_under_decomposed(question, plan):
        tags.add("under_decomposed")

    return sorted(tags) if tags else ["ok_minimal"]


def _attach_plan_audit(meta: PlanEmitMeta, question: str, plan: GSWPlan) -> None:
    meta.extras["plan_audit_tags"] = audit_plan_quality(question, plan)


def _reject_arg_selection_over_blank_candidates(plan: GSWPlan) -> None:
    ent_by_id = {e.id: e for e in plan.entities}
    for c in plan.constraints:
        if c.kind not in {"argmax", "argmin"}:
            continue
        blank_candidates = [
            eid
            for eid in c.candidate_entity_ids
            if (ent := ent_by_id.get(eid)) is not None and ent.kind == "blank"
        ]
        if blank_candidates:
            raise ValueError(
                "arg_selection_blank_candidates: argmax/argmin candidate_entity_ids "
                f"must be filled candidate entities, not blank ids {blank_candidates!r}; "
                "otherwise the executor returns the blank id instead of the resolved value"
            )


def _reject_date_granularity_missing(question: str, plan: GSWPlan) -> None:
    if not _has_date_granularity_gap(question, plan):
        return
    raise ValueError(
        "date_granularity_missing: age/anniversary questions must retrieve "
        "full birth/event dates and compute elapsed age from those dates; "
        "do not compute the target from *_year number blanks because the "
        "answer can be off by one before the birthday/anniversary"
    )


def _extract_unit(text: str | None) -> str | None:
    """Scan ``text`` for a closed-vocabulary unit token, or None.

    Only used on (a) a blank's own ``retrieval_hint`` -- deliberate,
    planner-authored text -- and (b) the narrow magnitude-comparison gate
    below. Deliberately NOT used as a general-purpose scan of arbitrary
    question prose (see module docs on _MAGNITUDE_COMPARISON_CUES for why).
    """
    if not text:
        return None
    for unit, pattern in _UNIT_VARIANT_RES.items():
        if pattern.search(text):
            return unit
    if _BARE_METRES_RE.search(text):
        return "metres"
    if "$" in text:
        return "dollars"
    if "%" in text:
        return "percent"
    return None


def _constraint_numeric_blank_operands(plan: GSWPlan, c) -> list:
    """Blank entities with value_type=='number' referenced by ``c``'s inputs.

    Ignores filled entities (constraint-value literals like "twelve" are
    handled by the literal-value checks above, not here) and de-duplicates
    while preserving first-seen order.
    """
    ent_by_id = {e.id: e for e in plan.entities}
    seen: set[str] = set()
    out = []
    for ref in list(c.args_refs) + list(c.args_blanks):
        if ref in seen:
            continue
        ent = ent_by_id.get(ref)
        if ent is not None and ent.kind == "blank" and ent.value_type == "number":
            out.append(ent)
            seen.add(ref)
    return out


def _reject_unit_mismatch_constraints(question: str, plan: GSWPlan) -> None:
    """Reject a derived constraint whose numeric-blank operands disagree on
    (or omit) a declared measurement unit, when a unit is plausibly needed.

    Gating (false-positive guard): most numeric blanks are dimensionless
    counts, years, or rankings ("how many Emmy nominations", "what year"),
    where forcing a unit declaration would trigger a pointless repair on a
    plan that already answers the question correctly. This check only
    fires when there is already a signal that a physical/measurement unit
    is in play:
      (a) at least one operand's retrieval_hint already names a unit, OR
      (b) the question matches a narrow magnitude-comparison phrasing
          ("how many times deeper/taller/...", "how much shorter/longer/
          ...", "how many Xs make up the height/length/... of Y") -- the
          exact shape of the three verified cross-unit failures (q16,
          q692, q771) this check exists to catch.
    Signal (b) intentionally does NOT scan the question for generic unit
    words (years/people/feet/...): "twelve YEARS earlier" and "how many
    PEOPLE" are common idioms with no cross-source unit-mismatch risk, and
    scanning for them produced false positives on unrelated diff/sum
    constraints in this project's own test fixtures.
    """
    question = question or ""
    question_has_magnitude_cue = bool(_MAGNITUDE_COMPARISON_CUES.search(question))

    for c in plan.constraints:
        if c.kind != "derived" or c.op not in _UNIT_BEARING_OPS:
            continue
        operands = _constraint_numeric_blank_operands(plan, c)
        if len(operands) < 2:
            continue

        declared = [(_extract_unit(e.retrieval_hint), e) for e in operands]
        any_operand_unit = any(u is not None for u, _ in declared)
        if not any_operand_unit and not question_has_magnitude_cue:
            continue  # no signal a unit is needed -- likely counts/years/ranks

        non_null_units = {u for u, _ in declared if u is not None}
        operand_ids = [e.id for _, e in declared]

        if not non_null_units:
            raise ValueError(
                f"unit_missing: derived constraint {c.id!r} ({c.op}) combines "
                f"numeric blanks {operand_ids!r} but none of their "
                "retrieval_hint fields declare a unit; this looks like a "
                "physical-magnitude comparison, so every blank feeding this "
                "constraint must declare the SAME unit in its retrieval_hint "
                "(e.g. 'in feet') so the executor never combines mismatched "
                "units"
            )

        if len(non_null_units) > 1:
            raise ValueError(
                f"unit_mismatch: derived constraint {c.id!r} ({c.op}) combines "
                f"numeric blanks {operand_ids!r} whose retrieval_hint fields "
                f"declare DIFFERENT units {sorted(non_null_units)!r}; every "
                f"blank feeding a {c.op} constraint must declare the SAME "
                "unit in its retrieval_hint, matching the unit the question's "
                "answer implies if there is one"
            )

        missing_ids = [e.id for u, e in declared if u is None]
        if missing_ids:
            (only_unit,) = non_null_units
            raise ValueError(
                f"unit_mismatch: derived constraint {c.id!r} ({c.op}) -- "
                f"blank(s) {missing_ids!r} declare no unit in their "
                f"retrieval_hint while sibling operand(s) declare "
                f"{only_unit!r}; give every numeric blank feeding this "
                f"constraint the same declared unit"
            )


# q39 fix 1b (Hard rule 8 backstop): a closed set of qualifier words that,
# when attached to a noun phrase in the question, change what counts as a
# correct value ("full first name" != "first name" -- a nickname satisfies
# the latter but not the former). See _planner_prompts.py Hard rule 8 for
# the prompt-side half of this fix.
_QUALIFIER_WORDS = (
    "full", "exact", "official", "original", "legal", "formal", "complete", "maiden",
)

# Closed noun vocabulary the qualifier must attach to, so this stays narrow
# ("full first name" fires; "played a full match" does not, since "match"
# isn't in the vocabulary). Deliberately small and conservative -- false
# negatives (a qualifier construction we don't catch) are fine, false
# positives (rejecting a plan that already handled the qualifier fine) are
# not, since this raises a hard ValueError that drives the repair loop.
_QUALIFIER_NOUN_HEADS = (
    "name", "date", "title", "spelling", "address", "list",
)

_QUALIFIER_NOUN_RE = re.compile(
    r"\b(" + "|".join(_QUALIFIER_WORDS) + r")\s+"
    r"((?:[a-z]+\s+){0,2}?(?:" + "|".join(_QUALIFIER_NOUN_HEADS) + r"))\b",
    flags=re.IGNORECASE,
)


def _reject_dropped_qualifier_hints(question: str, plan: GSWPlan) -> None:
    """Reject a blank whose retrieval_hint/verification_claim covers a
    qualified noun phrase from the question ("full first name") but drops
    the qualifier word itself ("first name").

    Narrowly gated to avoid false-firing on unrelated uses of "full" /
    "original" / etc. elsewhere in the question: this only fires when (a)
    the question has a qualifier word immediately followed by a noun
    phrase ending in the closed vocabulary above, AND (b) some blank's
    retrieval_hint/verification_claim already contains that exact noun
    phrase -- i.e. the blank clearly targets the qualified noun, it just
    silently dropped the qualifier. A qualifier match with no blank
    covering the noun at all is not flagged (nothing to repair).
    """
    question = question or ""
    for match in _QUALIFIER_NOUN_RE.finditer(question):
        qualifier = match.group(1).lower()
        noun_phrase = " ".join(match.group(2).lower().split())
        for ent in plan.blank_entities():
            hint_text = " ".join(
                filter(None, (ent.retrieval_hint, ent.verification_claim))
            ).lower()
            if not hint_text.strip():
                continue
            if noun_phrase not in hint_text:
                continue
            if re.search(r"\b" + re.escape(qualifier) + r"\b", hint_text):
                continue
            raise ValueError(
                f"dropped_qualifier: the question qualifies {noun_phrase!r} "
                f"with {qualifier!r}, but blank {ent.id!r}'s retrieval_hint/"
                f"verification_claim covers {noun_phrase!r} without the "
                f"qualifier; keep {qualifier!r} verbatim (e.g. 'the "
                f"{qualifier} {noun_phrase}', not just 'the {noun_phrase}') "
                "-- an informal/shortened value can satisfy the unqualified "
                "phrase but not the qualified one"
            )


def _has_literal_constant_blank(plan: GSWPlan) -> bool:
    literal_ids = {
        e.id
        for e in plan.filled_entities()
        if e.role == "constraint-value" or e.literal_value is not None
    }
    if not literal_ids:
        return False
    blanks = {e.id for e in plan.blank_entities()}
    for vp in plan.verb_phrases:
        literal_to_blank = (
            vp.subject_id in literal_ids and vp.object_id in blanks
        ) or (
            vp.object_id in literal_ids and vp.subject_id in blanks
        )
        phrase = vp.phrase.strip().lower()
        if literal_to_blank and (phrase in _LITERAL_VALUE_PHRASES or "value" in phrase):
            return True
    return False


def _has_numeric_literal_ref_without_value(plan: GSWPlan) -> bool:
    ent_by_id = {e.id: e for e in plan.entities}
    for c in plan.constraints:
        if c.kind != "derived" or c.op not in _NUMERIC_DERIVED_OPS:
            continue
        for ref in c.args_refs:
            ent = ent_by_id.get(ref)
            if (
                ent is not None
                and ent.kind == "filled"
                and ent.role == "constraint-value"
                and ent.literal_value is None
            ):
                return True
    return False


def _arg_selection_blank_candidates(plan: GSWPlan) -> list[str]:
    ent_by_id = {e.id: e for e in plan.entities}
    out: list[str] = []
    for c in plan.constraints:
        if c.kind not in {"argmax", "argmin"}:
            continue
        out.extend(
            eid
            for eid in c.candidate_entity_ids
            if (ent := ent_by_id.get(eid)) is not None and ent.kind == "blank"
        )
    return out


def _has_duplicate_constraint_outputs(plan: GSWPlan) -> bool:
    seen: set[str] = set()
    for c in plan.constraints:
        if not c.output_blank_id:
            continue
        if c.output_blank_id in seen:
            return True
        seen.add(c.output_blank_id)
    return False


def _has_comparison_constraint(plan: GSWPlan) -> bool:
    return any(
        c.kind in _COMPARISON_CONSTRAINT_KINDS
        or (c.kind == "derived" and c.op in _COMPARISON_CONSTRAINT_KINDS)
        for c in plan.constraints
    )


def _has_rounding_or_format_constraint(plan: GSWPlan) -> bool:
    return any(c.kind == "derived" and c.op == "round_nearest" for c in plan.constraints)


def _has_temporal_anchor(plan: GSWPlan) -> bool:
    return any(
        e.kind == "filled" and e.role in {"year-anchor", "date-anchor", "as-of-date"}
        for e in plan.entities
    )


def _has_date_granularity_gap(question: str, plan: GSWPlan) -> bool:
    if not _AGE_DATE_GRANULARITY_CUES.search(question or ""):
        return False
    try:
        target_id = plan.target().id
    except Exception:
        return False
    if not _depends_on_year_number_blank(plan, target_id):
        return False
    return any(
        c.kind == "derived"
        and c.op in {"diff", "sum"}
        and (c.output_blank_id in _target_dependency_closure(plan, target_id))
        for c in plan.constraints
    )


def _depends_on_year_number_blank(plan: GSWPlan, target_id: str) -> bool:
    closure = _target_dependency_closure(plan, target_id)
    blank_by_id = {e.id: e for e in plan.blank_entities()}
    for bid in closure:
        if bid == target_id:
            continue
        ent = blank_by_id.get(bid)
        if ent is None or ent.value_type != "number":
            continue
        if ent.role == "target" or ent.is_target:
            continue
        if _blank_has_year_or_birth_signal(plan, bid):
            return True
    return False


def _target_dependency_closure(plan: GSWPlan, target_id: str) -> set[str]:
    by_output = {
        c.output_blank_id: c for c in plan.constraints
        if c.output_blank_id
    }
    blank_ids = {e.id for e in plan.blank_entities()}
    seen: set[str] = set()
    stack = [target_id]
    while stack:
        bid = stack.pop()
        if bid in seen:
            continue
        seen.add(bid)
        c = by_output.get(bid)
        if c is None:
            continue
        refs: list[str] = []
        if c.kind == "derived":
            refs.extend(c.args_refs)
            refs.extend(c.args_blanks)
        elif c.kind in {"equals", "in_list", "gt", "lt"}:
            refs.extend(ref for ref in (c.left_ref, c.right_ref) if ref)
        elif c.kind in {"argmax", "argmin"}:
            refs.extend(c.sort_by_blank_ids or [])
        stack.extend(ref for ref in refs if ref in blank_ids and ref not in seen)
    return seen


def _blank_has_year_or_birth_signal(plan: GSWPlan, blank_id: str) -> bool:
    ent = plan.entity_by_id(blank_id)
    text_bits = [ent.id, ent.role or "", ent.name or ""]
    for vp in plan.verb_phrases:
        if blank_id in {vp.subject_id, vp.object_id}:
            text_bits.append(vp.phrase)
    text = " ".join(text_bits).replace("-", "_").lower()
    return any(
        signal in text
        for signal in (
            "birth_year",
            "born_year",
            "year_born",
            "anniversary_year",
            "event_year",
            "election_year",
            "release_year",
            "death_year",
            "year_of",
            "in_year",
            "born_in_year",
        )
    )


def _has_semantic_operation_cue(question: str) -> bool:
    return bool(
        _RISKY_HUB_CUES.search(question)
        or _COMPOUND_ANSWER_CUE.search(question)
        or _COMPARISON_THRESHOLD_CUES.search(question)
        or _FORMAT_CUES.search(question)
    )


def _looks_over_decomposed(question: str, plan: GSWPlan) -> bool:
    if _has_semantic_operation_cue(question or ""):
        return False
    blank_count = len(plan.blank_entities())
    return (blank_count >= 7 and not plan.constraints) or len(plan.constraints) >= 3


def _looks_under_decomposed(question: str, plan: GSWPlan) -> bool:
    try:
        target = plan.target()
    except Exception:
        return False
    target_id = target.id
    incident = [
        vp
        for vp in plan.verb_phrases
        if target_id in (vp.subject_id, vp.object_id)
    ]
    if len(incident) < 3:
        return False
    if any(c.output_blank_id == target_id for c in plan.constraints):
        return False

    question = question or ""
    target_type = (target.value_type or "").lower()
    computed_target = target_type in {"number", "date", "bool"} and bool(
        _COMPUTED_TARGET_HUB_CUES.search(question)
    )
    compound_text_target = target_type == "text" and bool(
        _COMPOUND_ANSWER_CUE.search(question)
    )
    broad_risky_cue = bool(
        _RISKY_HUB_CUES.search(question) or _COMPOUND_ANSWER_CUE.search(question)
    )
    return computed_target or compound_text_target or broad_risky_cue


# ---------------------------------------------------------------------------
# Shared fallback plumbing for planner-family adapters
# ---------------------------------------------------------------------------


class _PlannerFallbackMixin:
    """Mixin that provides `_get_fallback_adapter` + `_run_fallback`.

    Assumes the including class has:
      - ``self.ctx`` (AdapterContext)
      - ``self.corpus`` (shared article corpus)
      - ``self.retriever`` (shared BM25 retriever)
      - ``self.system_id`` (class-level string)

    and attribute ``_fallback_adapter: Optional[Adapter]`` initialised to
    ``None`` in ``__init__``.
    """

    def _get_fallback_adapter(self):
        """Lazily instantiate the ours_gsw_v1 flat adapter for fallbacks."""
        existing = getattr(self, "_fallback_adapter", None)
        if existing is not None:
            return existing
        # Import lazily to avoid circular-import noise.
        from panini_dr.adapters.base import AdapterContext
        from panini_dr.adapters.ours.ours_gsw_v1 import OursGSWv1Adapter

        fallback_ctx = AdapterContext(
            system_id="ours_gsw_v1",
            model_id=self.ctx.model_id,
            model_name=self.ctx.model_name,
            base_url=self.ctx.base_url,
            api_key=self.ctx.api_key,
            max_turns=self.ctx.max_turns,
            max_completion_tokens=self.ctx.max_completion_tokens,
            extra={
                "corpus": self.corpus,
                "retriever": self.retriever,
            },
        )
        self._fallback_adapter = OursGSWv1Adapter(fallback_ctx)
        return self._fallback_adapter

    def _run_fallback(
        self,
        question: str,
        question_id: str,
        articles,
        reason: str,
    ):
        """Delegate to the flat fallback adapter, stamp trajectory flags."""
        traj = self._get_fallback_adapter().run_question(
            question,
            question_id=question_id,
            articles=articles,
        )
        traj.extra["fallback_flag"] = True
        traj.extra["fallback_reason"] = reason
        # Preserve caller's system_id so harness bookkeeping stays consistent.
        traj.system_id = self.system_id
        return traj
