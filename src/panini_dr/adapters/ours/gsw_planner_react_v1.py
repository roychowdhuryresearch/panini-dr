"""Constraint-cascade helpers shared by the orchestrator.

``_cascade_auto_compute`` fires every constraint whose inputs are resolved
(``_collect_constraint_inputs`` mirrors ``_planner_exec._compute_constraint``'s
input semantics); ``_compute_levels`` layers the blank dependency graph;
``_stringify`` renders a blank value as text.

The module kept its name from the ReAct planner adapter
(``ours_gsw_planner_react_v1``), which the paper never ran and which was
removed.
"""

from __future__ import annotations

from typing import Any

from panini_dr.adapters.ours._planner_exec import (
    BlankResult,
    Constraint,
    GSWPlan,
    _compute_constraint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stringify(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, (list, tuple)):
        return ", ".join(_stringify(x) for x in v)
    return str(v)


def _collect_constraint_inputs(c: Constraint, blank_ids: set[str] | None = None) -> list[str]:
    """Return the list of blank ids a constraint reads from.

    Mirrors the input semantics in ``_planner_exec._compute_constraint``:
    - derived: args_refs when present, otherwise args_blanks
    - argmax / argmin: sort_by_blank_ids  (candidate_entity_ids are filled
      entities, not retrievable inputs)
    - relational (equals / in_list / gt / lt): left_ref + right_ref
    """
    inputs: list[str] = []
    if c.kind == "derived":
        inputs.extend(c.args_refs or c.args_blanks or [])
    elif c.kind in ("argmax", "argmin"):
        inputs.extend(c.sort_by_blank_ids or [])
    elif c.kind in ("equals", "in_list", "gt", "lt"):
        for ref in (c.left_ref, c.right_ref):
            if ref:
                inputs.append(ref)
        # in_list may wire its (member, list) operands via args_blanks
        # instead of left_ref/right_ref -- _compute_constraint's
        # _relational_ref_pair already falls back to args_blanks[:2], but
        # this collector did not, so the cascade saw "no inputs", never
        # fired the membership check, and a bool target sat unresolved
        # until the orchestrator hit max_turns and returned empty (q101:
        # "did the costar win Best Actor?" -> in_list(b_costar,
        # b_actor_winners) -> t never computed). Mirror _relational_ref_pair.
        if not inputs and c.kind == "in_list" and len(c.args_blanks) >= 2:
            inputs.extend(c.args_blanks[:2])
    if blank_ids is not None:
        inputs = [i for i in inputs if i in blank_ids]
    return [i for i in inputs if i]


# Cascade reliability guards. When ``block_unverified=True`` the cascade
# refuses to consume an input flagged unreliable (verifier) or incomplete
# (completeness). Split sets — verifier_status is checked against the
# verifier set, completeness_status against the completeness set; they are
# never crossed.
_CASCADE_BLOCKED_VERIFIER = {"unverified_capped", "verifier_rejected"}
_CASCADE_BLOCKED_COMPLETENESS = {"partial"}


def _cascade_auto_compute(
    plan: GSWPlan,
    state: dict[str, BlankResult],
    block_unverified: bool = False,
    blocked_sink: list | None = None,
    codegen: Any = None,
) -> list[str]:
    """Auto-fire any constraint whose inputs are all resolved.

    Returns the list of blank ids that were newly resolved by the cascade.
    Loops until no more constraints can fire — so a chain like
    ``(diff A B) → t1`` followed by ``(concat t1 C) → t2`` resolves in
    one pass.

    Skips constraints whose output is already resolved; the model may have
    pre-filled the target with its own computed value and we do NOT
    overwrite it.
    """
    newly_filled: list[str] = []
    changed = True
    blank_ids = {e.id for e in plan.blank_entities()}
    while changed:
        changed = False
        for c in plan.constraints:
            out_id = c.output_blank_id
            if not out_id:
                continue
            out = state.get(out_id)
            if out is not None and out.status == "resolved":
                continue  # already set (by LLM commit or prior cascade)
            refs = _collect_constraint_inputs(c)
            if not refs:
                continue
            inputs = [ref for ref in refs if ref in blank_ids]
            ready = True
            for i in inputs:
                bi = state.get(i)
                if bi is None or bi.status != "resolved" or bi.value is None:
                    ready = False
                    break
                if block_unverified:
                    vs = getattr(bi, "verifier_status", None)
                    cs = getattr(bi, "completeness_status", None)
                    if vs in _CASCADE_BLOCKED_VERIFIER or cs in _CASCADE_BLOCKED_COMPLETENESS:
                        if blocked_sink is not None:
                            blocked_sink.append({
                                "output_blank_id": out_id,
                                "constraint_id": getattr(c, "id", None),
                                "blocked_input": i,
                                "kind": "unverified" if vs in _CASCADE_BLOCKED_VERIFIER else "partial",
                            })
                        ready = False
                        break
            if not ready:
                continue
            result = _compute_constraint(c, plan, state, codegen=codegen)
            if result.status == "resolved":
                result.resolved_by = "constraint"
                state[out_id] = result
                newly_filled.append(out_id)
                changed = True
    return newly_filled


def _compute_levels(deps: dict[str, set[str]]) -> dict[str, int]:
    """Blank-id → dependency level via iterative Kahn's layers.

    Level 0 = no blank deps (can run first). Level n = depends on some
    blank at level n-1. Blanks at the same level have no mutual
    dependencies.
    """
    levels: dict[str, int] = {}
    remaining = {b: set(ds) for b, ds in deps.items()}
    lvl = 0
    while remaining:
        frontier = [b for b, ds in remaining.items() if not ds]
        if not frontier:
            break  # cycle — shouldn't happen post-validator
        for b in frontier:
            levels[b] = lvl
            remaining.pop(b)
        for ds in remaining.values():
            ds.difference_update(frontier)
        lvl += 1
    return levels
