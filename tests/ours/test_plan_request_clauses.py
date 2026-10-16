"""The contract vocabulary: five clauses, each a pure (before, after) -> bool.

The point of restricting the author to these five is termination. An LLM writes
the contract ONCE; every accept/reject afterwards is mechanical, so the loop
cannot accept, reject and re-accept forever. Every function here must therefore
be total, pure, and free of model calls.
"""
from __future__ import annotations

import pytest

from panini_dr.adapters.ours._plan_contract import contract_violations
from panini_dr.adapters.ours._plan_request_clauses import (
    Clause, CLAUSE_KINDS, satisfied, satisfied_set, binding_violations,
)
from panini_dr.adapters.ours._planner_exec import GSWPlan


def _plan(**over) -> GSWPlan:
    """Two blanks, b_x depends on nothing, t is the target."""
    base = {
        "entities": [
            {"id": "b_x", "kind": "blank", "value_type": "entity",
             "retrieval_hint": "Who founded Acme?"},
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
             "retrieval_hint": "Where was the founder born?"},
        ],
        "verb_phrases": [
            {"id": "vp1", "subject_id": "b_x", "phrase": "born_in", "object_id": "t"},
        ],
        "constraints": [],
    }
    base.update(over)
    return GSWPlan.model_validate(base)


def test_the_vocabulary_is_exactly_four_kinds():
    """The author is constrained to these. Widening the set is a spec change.

    `hint_changed` was removed after the 824 run: authored 21 times, satisfied
    0. No `PlanDiffOp` can edit an existing blank's `retrieval_hint` -- the only
    retrieval_hint write in `_plan_diff.py` sets it on a NEWLY created blank --
    so the author was spending one of its 2-4 slots on a clause the updater has
    no operation to deliver, and no contract could ever read as fully satisfied.

    Every kind here must be REACHABLE by some diff op. That is the property the
    original spec never checked, and it is what this test now guards.
    """
    assert CLAUSE_KINDS == {"blank_added", "depends_on", "preserved", "retyped"}


def test_blank_added_is_false_for_the_identity_and_true_when_a_blank_appears():
    p = _plan()
    assert satisfied(Clause("blank_added"), p, p) is False
    # b_new needs a verb-phrase edge of its own -- GSWPlan._check_no_dangling
    # rejects any entity unreferenced by a verb-phrase or constraint -- so vp1
    # is carried over unchanged and a second, minimal edge wires b_new to t.
    after = _plan(
        entities=[
            *[e.model_dump() for e in p.entities],
            {"id": "b_new", "kind": "blank", "value_type": "entity",
             "retrieval_hint": "What is the address?"},
        ],
        verb_phrases=[
            *[vp.model_dump() for vp in p.verb_phrases],
            {"id": "vp2", "subject_id": "b_new", "phrase": "located_at", "object_id": "t"},
        ],
    )
    assert satisfied(Clause("blank_added"), p, after) is True


def test_blank_added_is_false_when_only_a_filled_entity_is_added():
    """`blank_added` must count BLANKS only. A repair that grounds a new
    FILLED (already-resolved) entity into the plan and adds no blank must
    not satisfy this clause -- otherwise the acceptance test would accept
    an op that added no blank at all."""
    p = _plan()
    after = _plan(
        entities=[
            *[e.model_dump() for e in p.entities],
            {"id": "f_extra", "kind": "filled", "name": "Extra Corp"},
        ],
        verb_phrases=[
            *[vp.model_dump() for vp in p.verb_phrases],
            {"id": "vp2", "subject_id": "f_extra", "phrase": "related_to", "object_id": "t"},
        ],
    )
    assert satisfied(Clause("blank_added"), p, after) is False


def test_blank_added_is_false_for_a_removal_only_change():
    """Not a symmetric difference: `blank_added` diffs after-minus-before,
    so dropping a blank and adding none must not satisfy it -- a symmetric-
    difference mutant would wrongly see the dropped id and return True."""
    p = _plan(entities=[
        {"id": "b_x", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "Who founded Acme?"},
        {"id": "b_extra", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "Which subsidiary?"},
        {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
         "retrieval_hint": "Where was the founder born?"},
    ], verb_phrases=[
        {"id": "vp1", "subject_id": "b_x", "phrase": "born_in", "object_id": "t"},
        {"id": "vp2", "subject_id": "b_extra", "phrase": "owns", "object_id": "t"},
    ])
    after = _plan(entities=[
        {"id": "b_x", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "Who founded Acme?"},
        {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
         "retrieval_hint": "Where was the founder born?"},
    ], verb_phrases=[
        {"id": "vp1", "subject_id": "b_x", "phrase": "born_in", "object_id": "t"},
    ])
    assert satisfied(Clause("blank_added"), p, after) is False


def test_hint_changed_is_no_longer_a_clause_and_is_never_satisfied():
    """Removed from the vocabulary, so it falls through `satisfied`'s
    unknown-kind branch and can never be true -- even if an author smuggles it
    past the membership filter. Kept as a test because a future edit that
    re-adds the kind without adding an op to satisfy it would silently
    reintroduce 21-authored/0-satisfied."""
    p = _plan()
    after = _plan(entities=[
        {"id": "b_x", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "Who founded Acme Corporation in 1953?"},
        {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
         "retrieval_hint": "Where was the founder born?"},
    ])
    assert "hint_changed" not in CLAUSE_KINDS
    assert satisfied(Clause("hint_changed", blank_id="b_x"), p, after) is False


def test_preserved_is_true_for_the_identity_and_false_once_the_blank_is_gone():
    """`preserved` is satisfied by the identity (before == after) -- so is
    `retyped`, whenever the declared type already matches. Neither is unique
    in that respect, and that's fine: what makes an already-satisfied clause
    SAFE to name in a contract is that it appears in both sides of Task 4's
    strict-superset comparison. It can only make the acceptance test
    STRICTER -- an op that disturbs the blank drops the clause, shrinking
    the satisfied set below what the contract requires -- never vacuously
    permissive."""
    p = _plan()
    assert satisfied(Clause("preserved", blank_id="b_x"), p, p) is True
    # b_x is gone, so vp1 (which named it) must go too; t still needs SOME
    # edge to stay non-dangling, so a minimal filled entity replaces b_x's
    # wiring without reintroducing b_x itself.
    after = _plan(entities=[
        {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
         "retrieval_hint": "Where was the founder born?"},
        {"id": "f_acme", "kind": "filled", "name": "Acme"},
    ], verb_phrases=[
        {"id": "vp1", "subject_id": "f_acme", "phrase": "born_in", "object_id": "t"},
    ])
    assert satisfied(Clause("preserved", blank_id="b_x"), p, after) is False


def test_preserved_is_false_once_the_blank_is_edited():
    """The clause's entire meaning is the model_dump() EQUALITY, not mere
    presence: b_x survives into `after` under the same id, but its
    retrieval_hint changed, so `preserved` must be False even though the
    blank was never removed. Without this, a presence-only mutant
    (`return b_before is not None and b_after is not None`) would still
    pass every other test in this file."""
    p = _plan()
    after = _plan(entities=[
        {"id": "b_x", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "Who founded Acme Corporation in 1953?"},
        {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
         "retrieval_hint": "Where was the founder born?"},
    ])
    assert satisfied(Clause("preserved", blank_id="b_x"), p, after) is False


def test_retyped_checks_the_declared_value_type():
    p = _plan()
    after = _plan(entities=[
        {"id": "b_x", "kind": "blank", "value_type": "list",
         "retrieval_hint": "Who founded Acme?"},
        {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
         "retrieval_hint": "Where was the founder born?"},
    ])
    assert satisfied(Clause("retyped", "b_x", "list"), p, after) is True
    assert satisfied(Clause("retyped", "b_x", "entity"), p, after) is False


def test_depends_on_uses_the_executors_own_graph_not_a_restatement():
    """The dependency rule is subtle -- an edge S -> O only creates a dependency
    if S is the target or S.value_type == 'entity'. Restating it here would drift
    from the executor. `build_dependency_graph` is the single source of truth."""
    # b_x and b_y each need SOME edge to stay non-dangling in `before`, so
    # both are wired to t (harmless: t already depends on both, and neither
    # wiring touches b_x's or b_y's own dependency sets -- only t's). `after`
    # then adds the one edge under test: b_x -founded_in-> b_y.
    p = _plan(entities=[
        {"id": "b_x", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "Who founded Acme?"},
        {"id": "b_y", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "What year?"},
        {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
         "retrieval_hint": "Where born?"},
    ], verb_phrases=[
        {"id": "vp_x", "subject_id": "b_x", "phrase": "born_in", "object_id": "t"},
        {"id": "vp_y", "subject_id": "b_y", "phrase": "born_in", "object_id": "t"},
    ])
    after = _plan(entities=[e.model_dump() for e in p.entities], verb_phrases=[
        {"id": "vp_x", "subject_id": "b_x", "phrase": "born_in", "object_id": "t"},
        {"id": "vp_y", "subject_id": "b_y", "phrase": "born_in", "object_id": "t"},
        {"id": "vp1", "subject_id": "b_x", "phrase": "founded_in", "object_id": "b_y"},
    ])
    assert satisfied(Clause("depends_on", blank_id="b_y"), p, after) is True
    assert satisfied(Clause("depends_on", blank_id="b_x"), p, after) is False


def test_an_unknown_clause_kind_is_never_satisfied():
    """Defensive: an author that smuggles a kind past validation must not be
    able to make a clause vacuously true."""
    p = _plan()
    assert satisfied(Clause("wishful_thinking"), p, p) is False


def test_a_clause_naming_a_nonexistent_blank_is_not_satisfied():
    p = _plan()
    assert satisfied(Clause("hint_changed", blank_id="b_nope"), p, p) is False
    assert satisfied(Clause("preserved", blank_id="b_nope"), p, p) is False
    assert satisfied(Clause("depends_on", blank_id="b_nope"), p, p) is False


def test_satisfied_set_returns_the_satisfied_clauses():
    p = _plan()
    cl = [Clause("preserved", blank_id="b_x"), Clause("blank_added")]
    assert satisfied_set(cl, p, p) == frozenset({Clause("preserved", blank_id="b_x")})


def test_binding_violations_names_only_binding_clauses():
    """A clean plan has none; the helper must not leak ADVISORY clause names,
    which would make the acceptance test reject on non-binding grounds."""
    assert binding_violations(_plan()) == []


def test_binding_violations_excludes_a_real_advisory_violation():
    """The clean-plan case above only proves "no violations -> []"; it never
    exercises the `if v.clause in BINDING` filter, since there is nothing to
    filter. This plan provokes a genuine `numeric_operands` violation --
    ADVISORY, per `_plan_contract.BINDING` -- on a derived `sum` constraint
    whose only operand is a filled entity with no numeric value at all.
    `contract_violations` must report it; `binding_violations` must not."""
    p = GSWPlan.model_validate({
        "entities": [
            {"id": "t", "kind": "blank", "value_type": "number", "is_target": True,
             "retrieval_hint": "What is the total?"},
            {"id": "e_bad", "kind": "filled", "name": "a lot"},
        ],
        "verb_phrases": [],
        "constraints": [
            {"id": "c1", "kind": "derived", "op": "sum", "args_refs": ["e_bad"],
             "output_blank_id": "t"},
        ],
    })
    reported = {v.clause for v in contract_violations(p)}
    assert "numeric_operands" in reported
    assert "numeric_operands" not in binding_violations(p)


def test_binding_violations_catches_a_forward_reference():
    """The flip in `build_dependency_graph` only fires when the verb-phrase's
    SUBJECT is the target (see test_the_target_as_subject_flips_the_edge).
    Here vp1's subject is b_x, not t, so the default rule applies: the
    object (t) depends on the subject (b_x). b_x therefore resolves at
    dependency level 0 and t at level 1 -- t is downstream of b_x, never the
    reverse. b_x's hint referencing {t} asks for a value that does not exist
    yet when b_x is dispatched: a genuine forward reference, and the exact
    class that 34 of the 36 measured post-update violations belong to.
    (t's hint may safely reference {b_x} -- that's a backward reference to
    an already-resolved blank, and is not tested here.)
    """
    p = _plan(entities=[
        {"id": "b_x", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "Who founded {t}?"},
        {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
         "retrieval_hint": "Where was the founder born?"},
    ])
    assert "ordering" in binding_violations(p)


def test_the_target_as_subject_flips_the_edge():
    """The flip keys on the SUBJECT being the target -- not on either
    endpoint merely being the target. When t (the target) sits in the
    subject slot of a verb-phrase, the edge still routes t -> object: t
    depends on its object, never the reverse, so the target can never
    become a dependency SOURCE. This rule has been misread twice in this
    project (most recently in this file's own forward-reference fixture,
    fixed above); pin it directly.

    `before` wires t and b_movie to a shared third blank (b_studio) so
    neither is dangling, without creating any t<->b_movie edge yet. `after`
    adds vp1, t -directed-> b_movie -- t as subject. If the flip works, t
    gains a new dependency (on b_movie) and b_movie gains none.
    """
    p = _plan(entities=[
        {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
         "retrieval_hint": "Who directed it?"},
        {"id": "b_movie", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "Which movie?"},
        {"id": "b_studio", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "Which studio?"},
    ], verb_phrases=[
        {"id": "vp_t", "subject_id": "t", "phrase": "worked_at", "object_id": "b_studio"},
        {"id": "vp_m", "subject_id": "b_movie", "phrase": "made_by", "object_id": "b_studio"},
    ])
    after = _plan(entities=[e.model_dump() for e in p.entities], verb_phrases=[
        {"id": "vp_t", "subject_id": "t", "phrase": "worked_at", "object_id": "b_studio"},
        {"id": "vp_m", "subject_id": "b_movie", "phrase": "made_by", "object_id": "b_studio"},
        {"id": "vp1", "subject_id": "t", "phrase": "directed", "object_id": "b_movie"},
    ])
    assert satisfied(Clause("depends_on", blank_id="t"), p, after) is True
    assert satisfied(Clause("depends_on", blank_id="b_movie"), p, after) is False
