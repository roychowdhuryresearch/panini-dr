"""The gate. One op in, accept or reject out, and the reason why.

The verifier is the only thing that changes a plan. The repairer proposes; this
decides. Nothing here calls a model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from panini_dr.adapters.ours._plan_contract import contract_violations
from panini_dr.adapters.ours._plan_repair_ops import (
    RepairApplyError, RepairOp, apply_op, is_destructive, parse_repair_op,
)
from panini_dr.adapters.ours._planner_exec import GSWPlan


@dataclass
class Verdict:
    accept: bool
    reason: str
    plan: GSWPlan | None = None


def _binding_count(plan: GSWPlan, question: str, demoted: frozenset[str]) -> int:
    return sum(1 for v in contract_violations(plan, question, demoted) if v.binding)


def _clauses(plan: GSWPlan, question: str, demoted: frozenset[str]) -> set[str]:
    return {v.clause for v in contract_violations(plan, question, demoted) if v.binding}


def _entity_texts(plan: GSWPlan) -> dict[str, str]:
    """Each entity's OWN retrieval_hint/verification_claim/name, joined.

    Deliberately per-entity, not pooled across the plan: pooling let a
    destroyed hint hide behind an unrelated node that happened to carry the
    same string (review finding, task 4 round 2) -- two blanks sharing
    identical hint text made emptying one look survivable because the
    other's copy was still in the pool. The deletion test must ask whether
    THIS node still carries its own content, never whether the string
    appears somewhere else in the plan.
    """
    out: dict[str, str] = {}
    for e in plan.entities:
        parts = []
        for f in ("retrieval_hint", "verification_claim", "name"):
            val = getattr(e, f, None)
            if val:
                parts.append(str(val))
        out[e.id] = " || ".join(parts).lower()
    return out


def verify_op(
    plan: GSWPlan, op: RepairOp, question: str = "",
    demoted: frozenset[str] = frozenset(),
) -> Verdict:
    """Accept only a strictly-improving, validating, non-destructive op."""
    before_n = _binding_count(plan, question, demoted)
    before_c = _clauses(plan, question, demoted)
    before_texts = _entity_texts(plan)

    try:
        candidate = apply_op(plan, op)
    except RepairApplyError as exc:
        return Verdict(False, f"op does not apply: {exc}")
    except Exception as exc:  # noqa: BLE001 — pydantic validation
        return Verdict(False, f"result fails plan validation: {exc}")

    if is_destructive(op):
        # For each node that had its own content before: the SAME node
        # (never a different one) must still carry it, or the op must have
        # replaced it in the same operation. A node missing entirely from
        # `candidate` (deleted) counts as carrying nothing.
        after_texts = _entity_texts(candidate)
        for node_id, before_text in before_texts.items():
            if not before_text.strip():
                continue
            after_text = after_texts.get(node_id, "")
            lost = [s for s in before_text.split(" || ")
                    if s.strip() and s not in after_text]
            if lost:
                return Verdict(False,
                               f"destructive: {node_id!r} loses content "
                               f"nothing on it replaces ({lost[0][:60]!r})")

    after_n = _binding_count(candidate, question, demoted)
    if after_n >= before_n:
        return Verdict(False,
                       f"binding violations did not decrease ({before_n} -> {after_n})")
    new_clauses = _clauses(candidate, question, demoted) - before_c
    if new_clauses:
        return Verdict(False, f"introduces a new binding clause: {sorted(new_clauses)}")
    return Verdict(True, "accepted", candidate)


MAX_ATTEMPTS_PER_VIOLATION = 3


@dataclass
class RepairTranscript:
    ops_accepted: list[dict] = field(default_factory=list)
    ops_rejected: list[dict] = field(default_factory=list)
    demoted: list[str] = field(default_factory=list)
    violations_before: int = 0
    violations_after: int = 0


def verify_and_repair(plan, question, propose,
                      max_attempts: int = MAX_ATTEMPTS_PER_VIOLATION):
    """Loop until no binding clause is violated.

    Accepted ops strictly reduce binding violations, so they are bounded by the
    initial count. Only rejections can loop, and `max_attempts` bounds those:
    on the third consecutive rejection the clause demotes to advisory FOR THIS
    PLAN, which is an escape from one unsatisfiable clause, not from the loop.
    A proposer that returns None gives up on the violation outright and
    demotes its clause immediately, without waiting out the remaining
    attempts.

    Never mutates the caller's `plan`: every accepted op is a fresh object
    from `verify_op`/`apply_op`, and reassigning the local `plan` name here
    never touches the object the caller passed in.
    """
    demoted: set[str] = set()
    tr = RepairTranscript()
    tr.violations_before = _binding_count(plan, question, frozenset())

    while True:
        binding = [v for v in contract_violations(plan, question, frozenset(demoted))
                   if v.binding]
        if not binding:
            break
        v = binding[0]
        last_reject = None
        gave_up = False
        for _ in range(max_attempts):
            raw = propose(plan, v, last_reject)
            if raw is None:
                gave_up = True
                break
            try:
                op = parse_repair_op(raw)
            except Exception as exc:  # noqa: BLE001
                last_reject = f"unparseable op: {exc}"
                tr.ops_rejected.append({"op": raw, "reason": last_reject})
                continue
            verdict = verify_op(plan, op, question, frozenset(demoted))
            if verdict.accept:
                # `v.clause` is only what PROMPTED this attempt -- verify_op's
                # own accept rule ("strictly fewer binding violations, no new
                # clause") never requires the op to fix THAT clause, so the op
                # may have cleared a different one instead (or several).
                # Recording `v.clause` as "the clause this op fixed" is a
                # misattribution: capture the real before/after clause
                # difference so a transcript reader can't draw that false
                # conclusion.
                before_clauses = _clauses(plan, question, frozenset(demoted))
                plan = verdict.plan
                after_clauses = _clauses(plan, question, frozenset(demoted))
                tr.ops_accepted.append({
                    "op": raw,
                    "prompted_by": v.clause,
                    "clauses_resolved": sorted(before_clauses - after_clauses),
                })
                break
            last_reject = verdict.reason
            tr.ops_rejected.append({"op": raw, "reason": last_reject})
        else:
            gave_up = True
        if gave_up:
            demoted.add(v.clause)
            tr.demoted.append(v.clause)

    residual = contract_violations(plan, question, frozenset(demoted))
    tr.violations_after = sum(1 for r in residual if r.binding)
    return plan, residual, tr
