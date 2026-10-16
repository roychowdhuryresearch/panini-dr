"""The contract vocabulary -- five clauses, each a pure (before, after) -> bool.

WHY A RESTRICTED VOCABULARY. The plan updater's acceptance test today is a
LIVENESS test: does the op parse, is it non-trivial, does it apply, did the plan
signature change. It never asks whether the result is EXECUTABLE. Measured on
FRAMES-824: 15 of 191 updates increased the binding-violation count, and 19
questions executed a plan carrying an `ordering` or `acyclic` violation. Two of
those were `acyclic` -- a class emit-time produced ZERO of across 400 plans, so
the updater manufactures a failure mode the planner never does. `acyclic` does
not mean a verb-phrase graph cycle -- `_acyclic_violations` (`_plan_contract.py`)
flags a constraint whose `output_blank_id` appears among its own operands, i.e.
a constraint that reads the very blank it writes and so can never resolve. The
diff vocabulary edits one blank/VP/constraint at a time with no such check
re-run globally after the edit, so this self-reference can be introduced by an
otherwise-valid op and ship undetected.

WHY THESE FIVE AND NOTHING ELSE. An LLM authors the contract once, from the
researcher's hint. If the LLM also RULED on each proposed op, the loop could
accept, reject and re-accept indefinitely with only a hard cap to stop it. By
restricting the author to clauses a machine can decide, semantic judgment happens
exactly once and every accept/reject afterwards is deterministic -- which is what
gives termination, the same property that makes the emit-time repair loop
trustworthy.

Pure: no model calls, no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from panini_dr.adapters.ours._plan_contract import BINDING, contract_violations
from panini_dr.adapters.ours._planner_exec import GSWPlan, build_dependency_graph

# `hint_changed` was REMOVED after the 824 run: authored 21 times, satisfied 0.
# It is unsatisfiable by construction -- `PlanDiffOp`'s entire vocabulary is
# add_blank / remove_blank / change_target / add_vp / remove_vp /
# add|replace|remove_constraint / change_value_type, and none of them edits an
# existing blank's `retrieval_hint`; the only retrieval_hint write in
# `_plan_diff.py` sets it on a NEWLY created blank. The author was being invited
# to spend one of its 2-4 slots on a clause the updater has no operation to
# deliver, which also guaranteed no contract could ever read as fully satisfied.
# Root cause: the spec recorded that this clause vocabulary and PlanDiffOp are
# DIFFERENT vocabularies, then never checked every clause is REACHABLE by some
# op. `depends_on` was checked at the same time and KEPT -- it is reachable
# (4/33 satisfied); its low rate is the model wiring the new VP backwards
# (X -> new instead of new -> X), which is a comprehension gap, not a missing op.
CLAUSE_KINDS: frozenset[str] = frozenset(
    {"blank_added", "depends_on", "preserved", "retyped"}
)


@dataclass(frozen=True)
class Clause:
    """One requirement the repaired plan must meet.

    Frozen and hashable so `satisfied_set` can return a frozenset and the
    acceptance test can compare sets directly.
    """

    kind: str
    blank_id: Optional[str] = None
    value_type: Optional[str] = None


def _ids(plan: GSWPlan) -> set[str]:
    return {e.id for e in plan.entities}


def _blank_ids(plan: GSWPlan) -> set[str]:
    """Ids of BLANK entities only.

    `blank_added` must use this, not `_ids` -- `_ids` includes filled
    entities, so a repair that only adds a filled (grounded) entity and no
    blank would otherwise satisfy `blank_added` and let the acceptance test
    accept an op that added no blank at all.
    """
    return {e.id for e in plan.blank_entities()}


def _entity(plan: GSWPlan, node_id: Optional[str]):
    if not node_id:
        return None
    for e in plan.entities:
        if e.id == node_id:
            return e
    return None


def _deps(plan: GSWPlan, blank_id: str) -> set[str]:
    """Dependencies of `blank_id` per the EXECUTOR's own graph.

    Never restate the edge rule here. `S -phrase-> O` creates a dependency only
    when S is the target (the edge flips) or `S.value_type == "entity"`; date,
    number and list subjects create none. A local copy of that rule drifted
    from the executor once already and pointed half of a findings set the wrong
    way -- caught only by a held-out set.
    """
    graph = build_dependency_graph(plan)
    return set(graph.get(blank_id, ()) or ())


def satisfied(clause: Clause, before: GSWPlan, after: GSWPlan) -> bool:
    """Is `clause` true of the change from `before` to `after`?

    Total: an unknown kind or a clause naming a nonexistent blank returns False
    rather than raising, so a malformed contract can only make the acceptance
    test stricter, never vacuously permissive.
    """
    kind = clause.kind
    if kind == "blank_added":
        return bool(_blank_ids(after) - _blank_ids(before))
    if kind == "preserved":
        b_before, b_after = _entity(before, clause.blank_id), _entity(after, clause.blank_id)
        if b_before is None or b_after is None:
            return False
        return b_before.model_dump() == b_after.model_dump()
    if kind == "retyped":
        b_after = _entity(after, clause.blank_id)
        return b_after is not None and b_after.value_type == clause.value_type
    if kind == "depends_on":
        if clause.blank_id is None or clause.blank_id not in _ids(after):
            return False
        return len(_deps(after, clause.blank_id)) > len(_deps(before, clause.blank_id))
    return False


def satisfied_set(
    clauses: list[Clause], before: GSWPlan, after: GSWPlan
) -> frozenset[Clause]:
    """The subset of `clauses` true of this change."""
    return frozenset(c for c in clauses if satisfied(c, before, after))


def binding_violations(plan: GSWPlan, question: str = "") -> list[str]:
    """Names of BINDING clauses `plan` violates. ADVISORY clauses are excluded.

    `contract_violations` takes (plan, question, demoted) -- `question` is the
    SECOND positional argument and is read by the date-diff detector. Passing a
    frozenset there silently mis-parameterises the call.
    """
    # `question` only reaches date_diff_granularity, which is ADVISORY and is
    # filtered out below -- no BINDING clause reads it today. Forward-looking:
    # keep threading it through rather than dropping the parameter as dead.
    return [v.clause for v in contract_violations(plan, question) if v.clause in BINDING]
