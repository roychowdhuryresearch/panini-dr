"""The plan contract: what a generated plan must satisfy before dispatch.

Two tiers, declared together so the split is real from the start:

- ``BINDING`` clauses block dispatch (absent a caller-supplied demotion).
- ``ADVISORY`` clauses (``numeric_operands``, ``date_diff_granularity``) are
  reported with ``binding=False`` and never block dispatch: both name plan
  shapes that cannot be repaired without inventing content the plan never
  asked for, so gating on them would spin the repair loop forever.

This module owns its clause definitions outright — it does not restate the
executor's rules. In particular the ``ordering`` clause calls
:func:`_planner_exec.build_dependency_graph` and derives levels from what it
returns, rather than re-deriving the dependency direction by hand. Two prior
hand-written copies of that rule shipped in this codebase and were wrong both
times: they missed that a verb phrase whose subject is the target flips the
edge, and that a projection edge exists only when the subject's
``value_type == "entity"``. Likewise ``date_diff_granularity`` imports
``_ref_has_birth_cue`` / ``_wants_elapsed_years_diff`` /
``_output_wants_years_diff`` from ``_planner_exec`` rather than re-deriving
which branch of ``_date_diff_constraint_value`` a plan takes, and
``numeric_operands`` is deliberately scoped to NOT duplicate
``_planner_emit._reject_unit_mismatch_constraints``, which already
hard-rejects operand-versus-operand unit disagreement.

Pure: ``(plan, question, demoted) -> list[Violation]``. No model calls, no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from panini_dr.adapters.ours._planner_emit import _NUMERIC_DERIVED_OPS
from panini_dr.adapters.ours._planner_exec import (
    Constraint,
    ExecutionError,
    GSWPlan,
    _output_wants_years_diff,
    _ref_has_birth_cue,
    _wants_elapsed_years_diff,
    build_dependency_graph,
)

BINDING: frozenset[str] = frozenset(
    {"ordering", "acyclic", "target_type", "target_producible"}
)
ADVISORY: frozenset[str] = frozenset({"numeric_operands", "date_diff_granularity"})

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z_0-9]*)\}")
_BOOLEAN_KINDS = {"equals", "gt", "lt", "in_list"}
# Operand fields a constraint's own output_blank_id must not appear among.
_LIST_OPERAND_FIELDS = (
    "args_blanks",
    "args_refs",
    "candidate_entity_ids",
    "sort_by_blank_ids",
)
_SCALAR_OPERAND_FIELDS = ("left_ref", "right_ref", "filter_ref")

# Granularities no branch of `_date_diff_constraint_value` (_planner_exec.py
# :1111) can return. Branch 2 gives completed YEARS; branch 1 gives a
# "N months and M days" string. Neither yields a day, week, hour or minute
# count, so a question asking for one cannot be served by a date diff at all.
_UNSERVABLE_GRANULARITY = re.compile(
    r"\bhow many (days|weeks|hours|minutes)\b|\bnumber of (days|weeks|hours)\b",
    re.IGNORECASE,
)

# `literal_value` is what `_coerce_number` (_planner_exec.py:723) actually
# receives; a bare digit-shaped `name` is the fallback for entities that
# never got a `literal_value` populated.
_NUMERIC = re.compile(r"^[-+]?[\d,. ]+%?$")


@dataclass
class Violation:
    clause: str
    binding: bool
    blank_id: str | None
    evidence: str


def _levels(plan: GSWPlan) -> dict[str, int]:
    """Dependency level per blank, from the executor's OWN graph."""
    deps = build_dependency_graph(plan)
    level: dict[str, int] = {}

    def _lvl(b: str, seen: frozenset = frozenset()) -> int:
        if b in level:
            return level[b]
        if b in seen:
            return 0
        v = 1 + max(
            (_lvl(d, seen | {b}) for d in (deps.get(b) or set())), default=-1
        )
        level[b] = v
        return v

    for b in deps:
        _lvl(b)
    return level


def _mk(clause: str, blank_id: str | None, evidence: str, demoted: frozenset[str]) -> Violation:
    return Violation(
        clause=clause,
        binding=clause in BINDING and clause not in demoted,
        blank_id=blank_id,
        evidence=evidence,
    )


def _ordering_violations(plan: GSWPlan, demoted: frozenset[str]) -> list[Violation]:
    """A placeholder ``{b_x}`` in a blank's retrieval_hint is only renderable
    if ``b_x`` resolves at a strictly earlier dependency wave. Same level
    means same dispatch wave -- the placeholder can render unresolved.
    """
    levels = _levels(plan)
    out: list[Violation] = []
    for b in plan.blank_entities():
        hint = b.retrieval_hint or ""
        if not hint:
            continue
        owner_level = levels.get(b.id, 0)
        for ref in _PLACEHOLDER.findall(hint):
            if ref not in levels:
                continue
            ref_level = levels[ref]
            if ref_level >= owner_level:
                out.append(_mk(
                    "ordering",
                    b.id,
                    f"{b.id}.retrieval_hint interpolates {{{ref}}}, but "
                    f"{ref} is at dependency level {ref_level}, not "
                    f"strictly below {b.id}'s own level {owner_level}",
                    demoted,
                ))
    return out


def _constraint_operands(c: Constraint) -> set[str]:
    operands: set[str] = set()
    for field_name in _LIST_OPERAND_FIELDS:
        operands.update(getattr(c, field_name) or [])
    for field_name in _SCALAR_OPERAND_FIELDS:
        value = getattr(c, field_name)
        if value:
            operands.add(value)
    return operands


def _acyclic_violations(plan: GSWPlan, demoted: frozenset[str]) -> list[Violation]:
    """A constraint that reads the very blank it writes can never resolve --
    its own output would have to be known before it runs."""
    out: list[Violation] = []
    for c in plan.constraints:
        out_id = c.output_blank_id
        if not out_id:
            continue
        if out_id in _constraint_operands(c):
            out.append(_mk(
                "acyclic",
                out_id,
                f"constraint {c.id!r} writes into {out_id!r}, which also "
                "appears among its own operands",
                demoted,
            ))
    return out


def _target_type_violations(plan: GSWPlan, demoted: frozenset[str]) -> list[Violation]:
    """equals/gt/lt/in_list constraints produce a boolean; when one of them
    writes into THE TARGET, the target must be typed to receive one. A
    non-target blank of the "wrong" type is not this clause's concern -- the
    repair for this clause (route the comparison into a fresh blank so the
    target keeps its declared type) is meaningless for a non-target blank.
    """
    try:
        target = plan.target()
    except ExecutionError:
        return []
    out: list[Violation] = []
    for c in plan.constraints:
        if c.kind not in _BOOLEAN_KINDS:
            continue
        if c.output_blank_id != target.id:
            continue
        if target.value_type != "bool":
            out.append(_mk(
                "target_type",
                target.id,
                f"constraint {c.id!r} (kind={c.kind!r}) writes a boolean "
                f"result into the target {target.id!r}, whose value_type "
                f"is {target.value_type!r}, not 'bool'",
                demoted,
            ))
    return out


def _target_producible_violations(plan: GSWPlan, demoted: frozenset[str]) -> list[Violation]:
    """The target must either be filled by a constraint or have a
    retrieval_hint the researcher can act on directly."""
    produced = {c.output_blank_id for c in plan.constraints if c.output_blank_id}
    try:
        target = plan.target()
    except ExecutionError:
        return []
    if target.id in produced:
        return []
    if target.retrieval_hint:
        return []
    return [_mk(
        "target_producible",
        target.id,
        f"target {target.id!r} is produced by no constraint and carries "
        "no retrieval_hint",
        demoted,
    )]


def _is_date_diff(plan: GSWPlan, c: Constraint) -> bool:
    """A `diff` whose first two operands are date-typed blanks."""
    if c.kind != "derived" or c.op != "diff":
        return False
    by_id = {e.id: e for e in plan.entities}
    refs = (list(c.args_blanks) + list(c.args_refs))[:2]
    if len(refs) < 2:
        return False
    return all(
        (getattr(by_id.get(r), "value_type", None) or "") == "date" for r in refs
    )


def _date_diff_granularity_violations(
    plan: GSWPlan, question: str, demoted: frozenset[str]
) -> list[Violation]:
    """Two failure modes of a date `diff`, both ADVISORY under the single
    ``date_diff_granularity`` clause name -- neither can be repaired without
    inventing content (a birth cue, a granularity the executor doesn't have),
    so binding on either would spin the repair loop.

    `_date_diff_constraint_value` (_planner_exec.py:1111) picks exactly one
    of three outcomes:

    1. both operands carry a BIRTH cue -> `_format_month_day_delta`, a
       "N months and M days" STRING;
    2. else a year cue (`_YEARS_OUTPUT_CUES`) in the question or the output
       blank's hint/claim, AND the output blank typed `number` -> completed
       years;
    3. else -> FALLS THROUGH to `abs(_coerce_constraint_number(a) -
       _coerce_constraint_number(b))`, and that coercer falls back to the
       first 4-digit year, so the result is raw YEAR subtraction: month and
       day are discarded. This does NOT resolve unknown -- it returns a
       plausible wrong number, exactly 0 when both dates share a year.

    Neither branch 1 nor branch 2 can ever return a day/week/hour/minute
    count, so a question asking for one is unservable no matter which branch
    the diff takes -- this is reported even when branch 1 or 2 routes fine.
    """
    by_id = {e.id: e for e in plan.entities}
    out: list[Violation] = []
    for c in plan.constraints:
        if not _is_date_diff(plan, c):
            continue
        oid = c.output_blank_id
        refs = (list(c.args_blanks) + list(c.args_refs))[:2]

        birth = all(_ref_has_birth_cue(plan, r) for r in refs)
        years_cue = _wants_elapsed_years_diff(plan, refs) or _output_wants_years_diff(
            plan, c, question or ""
        )
        numeric_out = (getattr(by_id.get(oid), "value_type", None) or "") == "number"

        if not birth and not (years_cue and numeric_out):
            why = (
                "the output is not typed `number`"
                if years_cue
                else "neither the question nor the output hint asks for "
                "elapsed years"
            )
            out.append(_mk(
                "date_diff_granularity",
                oid,
                f"constraint {c.id!r} subtracts two dates but no operand "
                f"carries a birth cue, and {why}, so it falls through to raw "
                f"year subtraction: month and day are discarded and {oid!r} "
                "is off by up to a year -- exactly 0 when both dates share "
                "a year",
                demoted,
            ))
            continue

        if question and _UNSERVABLE_GRANULARITY.search(question):
            out.append(_mk(
                "date_diff_granularity",
                oid,
                f"the question asks for a day/week/hour/minute count, but "
                "date diff returns completed years (or a 'months and days' "
                f"string); constraint {c.id!r} cannot produce that "
                "granularity",
                demoted,
            ))
    return out


def _constraint_operand_list(c: Constraint) -> list[str]:
    """Every operand field, in a fixed order -- unlike `_constraint_operands`
    (a set), this preserves order so a multi-operand violation's evidence
    text is deterministic across runs."""
    out: list[str] = []
    for field_name in _LIST_OPERAND_FIELDS:
        out.extend(x for x in (getattr(c, field_name) or []) if x)
    for field_name in _SCALAR_OPERAND_FIELDS:
        value = getattr(c, field_name)
        if value:
            out.append(value)
    return out


def _is_numeric_operand(ent) -> bool:
    """`literal_value` FIRST -- it is what `_coerce_number`
    (_planner_exec.py:723) actually receives. Reading `name` instead flags
    every constant the planner spelled in words and never a digit-named one.
    """
    lit = ent.literal_value
    if isinstance(lit, (int, float)):
        return True
    for cand in (lit, ent.name):
        if cand not in (None, "") and _NUMERIC.match(str(cand).strip()):
            return True
    return False


def _numeric_operands_violations(
    plan: GSWPlan, demoted: frozenset[str]
) -> list[Violation]:
    """An arithmetic constraint with a FILLED operand that has no numeric
    value at all. `_coerce_number` hard-fails on a non-parseable string and
    silently kills the chain -- e.g. `sum(b_year, e_tenth)` where `e_tenth`
    is named "nearest tenth" with no `literal_value`.

    Deliberately narrower than it could be: operand-VERSUS-operand unit
    disagreement (feet vs metres) is already hard-rejected at emit time by
    `_planner_emit._reject_unit_mismatch_constraints`; duplicating that here
    would just re-detect the same plans a second way. This clause only
    covers what that check does not -- a single operand with nothing numeric
    to coerce.
    """
    by_id = {e.id: e for e in plan.entities}
    out: list[Violation] = []
    for c in plan.constraints:
        if c.kind != "derived" or c.op not in _NUMERIC_DERIVED_OPS:
            continue
        bad: list[str] = []
        for r in _constraint_operand_list(c):
            ent = by_id.get(r)
            if ent is None or ent.kind != "filled":
                continue
            if not _is_numeric_operand(ent):
                label = ent.name or r
                bad.append(f"{r} ({label!r})")
        if bad:
            out.append(_mk(
                "numeric_operands",
                c.output_blank_id,
                f"constraint {c.id!r} ({c.op}) has operand(s) with no "
                f"numeric value: {', '.join(bad)}",
                demoted,
            ))
    return out


def contract_violations(
    plan: GSWPlan,
    question: str = "",
    demoted: frozenset[str] = frozenset(),
) -> list[Violation]:
    """Every clause violation the contract can detect in ``plan``.

    ``demoted`` names BINDING clauses whose violations are still reported
    (so a caller can see what was waived) but no longer set ``binding``.
    ADVISORY clauses always report with ``binding=False``, demoted or not --
    ``_mk`` already restricts ``binding`` to clauses in ``BINDING``.

    Pure: no model calls, no I/O.
    """
    out: list[Violation] = []
    out.extend(_ordering_violations(plan, demoted))
    out.extend(_acyclic_violations(plan, demoted))
    out.extend(_target_type_violations(plan, demoted))
    out.extend(_target_producible_violations(plan, demoted))
    out.extend(_date_diff_granularity_violations(plan, question, demoted))
    out.extend(_numeric_operands_violations(plan, demoted))
    return out
