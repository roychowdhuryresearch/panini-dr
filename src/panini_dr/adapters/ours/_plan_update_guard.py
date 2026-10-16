"""W6 plan-update identity guard. Pure functions over old + new GSWPlan.

The plan-updater preserves blanks by id only, so a preserved blank whose
meaning changed (or a resolved blank downstream of one) would keep a stale
resolved value across a replan. These helpers find those blanks; the workspace
decides (gated) whether to invalidate them.
"""
from __future__ import annotations

from typing import Iterable

from ._planner_exec import GSWPlan, build_dependency_graph
from ._blank_resolver import get_retrieval_hint
from ._blank_verifier import get_verification_claim


def _identity_sig(plan: GSWPlan, bid: str, deps: dict[str, set[str]]) -> tuple:
    ent = plan.entity_by_id(bid)
    # DIRECTIONAL incident relations: carry the blank's role so an object->subject
    # flip (same phrase + endpoint) still counts as an identity change.
    incident = tuple(sorted(
        (
            "subject" if vp.subject_id == bid else "object",
            vp.phrase,
            vp.object_id if vp.subject_id == bid else vp.subject_id,
        )
        for vp in plan.verb_phrases if bid in (vp.subject_id, vp.object_id)
    ))
    return (
        getattr(ent, "value_type", None),
        get_retrieval_hint(plan, bid),
        get_verification_claim(plan, bid),
        tuple(sorted(deps.get(bid, set()))),
        bool(getattr(ent, "is_target", False)),
        incident,
    )


def changed_identity_blanks(
    old_plan: GSWPlan, new_plan: GSWPlan, *,
    preserved_ids: Iterable[str], resolved_ids: set[str],
) -> list[str]:
    """Preserved + currently-resolved blank ids whose identity changed."""
    old_deps = build_dependency_graph(old_plan)
    new_deps = build_dependency_graph(new_plan)
    old_blanks = {e.id for e in old_plan.blank_entities()}
    new_blanks = {e.id for e in new_plan.blank_entities()}
    changed: list[str] = []
    for bid in preserved_ids:
        if bid not in resolved_ids or bid not in old_blanks or bid not in new_blanks:
            continue
        if _identity_sig(old_plan, bid, old_deps) != _identity_sig(new_plan, bid, new_deps):
            changed.append(bid)
    return sorted(changed)


def downstream_resolved_blanks(
    new_plan: GSWPlan, *, seeds: Iterable[str], resolved_ids: set[str],
) -> set[str]:
    """Resolved blanks (excluding the seeds) that TRANSITIVELY depend on any seed,
    per the new plan's dependency graph. Traversal passes through unresolved nodes
    to reach resolved descendants."""
    deps = build_dependency_graph(new_plan)        # bid -> {upstream ids}
    children: dict[str, set[str]] = {}
    for b, ups in deps.items():
        for u in ups:
            children.setdefault(u, set()).add(b)
    seeds = list(seeds)
    seen: set[str] = set(seeds)
    out: set[str] = set()
    stack = list(seeds)
    while stack:
        cur = stack.pop()
        for child in children.get(cur, ()):
            if child in seen:
                continue
            seen.add(child)
            if child in resolved_ids:
                out.add(child)
            stack.append(child)          # keep traversing even through unresolved
    return out
