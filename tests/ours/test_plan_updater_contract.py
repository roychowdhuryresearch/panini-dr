"""The plan updater's acceptance test, upgraded from liveness to correctness.

Today `update_plan` accepts any op that parses, is non-trivial, applies, and
changes the plan signature. It never asks whether the RESULT is executable.
Measured on FRAMES-824: 15 of 191 updates increased the binding-violation count
and 19 questions went on to execute a plan carrying an `ordering` or `acyclic`
violation. Two were `acyclic`, which emit-time never produces.

The regression this file exists to prevent is the one in
`test_an_op_that_satisfies_a_clause_but_breaks_the_binding_contract_is_rejected`.

DEVIATION FROM THE BRIEF (see task-4-report.md for the full account): every
diff-op payload dict quoted in the brief fails to parse against the real
`PlanDiffOp` schema (`_plan_diff.py`) -- `add_blank` requires a `wire_via_vp`
key (one VP object, not a `verb_phrases` list), and `change_value_type` names
its target field `to`, not `value_type`. Verified empirically with
`parse_diff_op` before writing this file: all three of the brief's op literals
(`ADD_BLANK`, the cyclic op, `noop_ish`) raise `pydantic.ValidationError`. Had
they shipped as written, every test below would have failed by exhausting
`MAX_REVISION_ATTEMPTS` and raising `PlanEmitError` -- not the assertion each
test claims to check. The op literals here are corrected to the real schema;
each test's assertions and stated intent are unchanged. The flagship
regression test additionally could not be reproduced with the brief's
VP-cycle shape: `apply_diff` already runs `topological_sort_blanks`, which
rejects any *dependency*-graph cycle with `DiffApplyError` before the new
binding-violations check ever sees the plan -- but a literal 2-node VP cycle
touching the TARGET is not one: the target-is-always-a-sink rule
(`_planner_exec.build_dependency_graph`) flips the edge on either VP
direction, so both `b_x -> t` and `t -> b_x` collapse to the same single
dependency ("t depends on b_x"), no dependency cycle forms, and `apply_diff`
ACCEPTS it (verified directly: `add_vp` wiring `t -> b_x` on top of the
existing `b_x -> t` applies with no `DiffApplyError`). Separately, a single
`add_blank` op (one new node, one new edge) cannot introduce a 2-node cycle
by construction regardless of target involvement. And separately again,
`_plan_contract.py`'s `acyclic` clause is not a VP-graph cycle at all -- it
flags a constraint whose `output_blank_id` also appears among its own
operands (see `test_an_op_that_satisfies_depends_on_but_introduces_an_acyclic_violation_is_rejected`
below, added in fix round 1). The flagship test below instead triggers the
`ordering` BINDING clause (a blank's `retrieval_hint` self-interpolating
`{b_new}` before it can ever resolve), which is reachable through a single
`add_blank` op and was verified to reproduce
`binding_violations(new_plan, question) == ["ordering"]` while still
satisfying `blank_added`.

One more addition beyond the brief: `_Scripted` now counts `author_calls`
(invocations with `tools` set), and both `test_flag_off_keeps_todays_behaviour`
and `test_no_plan_request_means_no_contract_even_when_enabled` assert it is
zero. The "flag off -> byte-identical behaviour" constraint explicitly
includes "no extra LLM call", but the brief's own version of these tests only
checked `diff.contract == []`, which stays empty regardless of whether the
author ran -- `contract_active` gates its USE, not its invocation, and
`author_contract` fails open on a `None` request, so a bug that called the
author on either the flag-off path or the no-request path produced no
observable difference in `diff`. Verified by mutation, twice: without these
assertions, (a) a mutant that runs `author_contract` unconditionally on
`contract_enabled`, and (b) a mutant that drops the `plan_request is not
None` half of the guard before calling the author, both pass all 6 tests
unchanged.

A third round of mutation revealed the assertions above still were not
enough: `author_contract(None, ...)` raises `AttributeError` inside its own
message-building step -- BEFORE ever calling `llm_client.chat()` -- so a
`None` request is indistinguishable from "author never invoked" through
`author_calls` too, no matter which guard is removed. The only way to make
the `plan_request is None` guard observably load-bearing is to give it an
op that a liveness-only test accepts but a binding check would reject, and
assert the op is ACCEPTED with no plan_request -- proving the binding check
never ran. `test_no_plan_request_means_no_contract_even_when_enabled` does
that with the same ordering-violating op as the flagship test.

FIX ROUND 1 (review findings). The acceptance rule changed from "the new
plan must be BINDING-clean" to "the new plan must not have MORE binding
violations than the old plan" -- measured cost of the absolute rule: 31 of
824 real plans ship with an `ordering` violation the emit-time repair loop
already demoted, not fixed, and every candidate op on those 31 was being
rejected for a violation that predates the op, burning all
`MAX_REVISION_ATTEMPTS` and losing the replan outright. Two tests were
added: `test_an_op_that_leaves_a_preexisting_binding_violation_unchanged_is_accepted`
(the regression 7 of those 31 questions represent -- an unrelated, useful op
on an already-broken plan must not be punished for a violation it didn't
introduce) and
`test_an_op_that_satisfies_depends_on_but_introduces_an_acyclic_violation_is_rejected`
(the OTHER BINDING clause this feature was measured to need, previously
uncovered in this file -- `acyclic` is not a VP-graph cycle, it's a
constraint whose own output appears among its inputs, and it is reachable
through `add_constraint` in one op, unlike a VP-graph cycle). Both fixtures
were verified against the real `_plan_contract.py`/`_planner_exec.py`
functions with a standalone script before being written into a test, same
discipline as everything else in this file.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from panini_dr.adapters.ours._plan_request import PlanRequest
from panini_dr.adapters.ours._plan_request_clauses import Clause
from panini_dr.adapters.ours._planner_emit import PlanEmitError
from panini_dr.adapters.ours._plan_updater import update_plan
from panini_dr.adapters.ours._planner_exec import GSWPlan

OLD = GSWPlan.model_validate({
    "entities": [
        {"id": "b_x", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "Who founded Acme?"},
        {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
         "retrieval_hint": "Where born?"},
    ],
    "verb_phrases": [
        {"id": "vp1", "subject_id": "b_x", "phrase": "born_in", "object_id": "t"},
    ],
    "constraints": [],
})

REQ = PlanRequest(request_id="q1:b_x:0", blank_id="b_x", assigned_blank_ids=["b_x"],
                  reason="no chunk names a founder", hint="add an intermediate blank")


class _Scripted:
    """Returns diff-op responses in order; also serves the contract author.

    ``contract_clauses`` overrides the clause list the scripted author
    "emits" (default ``[{"kind": "blank_added"}]``, matching every test
    written before fix round 1). Needed by the `acyclic` test, whose op
    never adds a blank -- `blank_added` could never be satisfied by it, so
    that test scripts a `depends_on` clause instead.

    ``message_log`` (fix round 2) records the `messages` argument of every
    non-author call, one entry per attempt, so a test can inspect exactly
    what the SECOND attempt's prompt contained -- needed to pin the
    evidence-clobbering regression, which is invisible from `update_plan`'s
    return value (the accepted plan and diff look identical either way; only
    the discarded retry prompt was wrong).
    """

    def __init__(self, *responses, contract_clauses=None):
        self.responses = list(responses)
        self.calls = 0
        self.author_calls = 0
        self.message_log: list[list[dict]] = []
        self.contract_clauses = (
            contract_clauses if contract_clauses is not None
            else [{"kind": "blank_added"}]
        )

    def chat(self, messages=None, tools=None, **kw):
        if tools:  # the contract author
            self.author_calls += 1
            return SimpleNamespace(
                text="", reasoning_content="", prompt_tokens=0, completion_tokens=0,
                tool_calls=[SimpleNamespace(function=SimpleNamespace(
                    name="emit_contract",
                    arguments=json.dumps({"clauses": self.contract_clauses})))])
        self.message_log.append(messages)
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r


def _op(payload) -> SimpleNamespace:
    return SimpleNamespace(text=json.dumps(payload), reasoning_content="",
                           prompt_tokens=0, completion_tokens=0, tool_calls=[])


# Corrected to the real `AddBlankOp` schema: `wire_via_vp` is required (a
# single VP wiring the new blank to an existing entity); there is no
# `verb_phrases` field.
ADD_BLANK = {"op": "add_blank", "blank": {
    "id": "b_new", "kind": "blank", "value_type": "entity",
    "retrieval_hint": "What is the address?"},
    "wire_via_vp": {"id": "vp_new", "phrase": "located_at",
                    "subject_id": "b_x", "object_id": "b_new"}}


def test_flag_off_keeps_todays_behaviour():
    """The default path must not move. `contract_enabled=False` means the
    contract is never authored and the old liveness test decides."""
    llm = _Scripted(_op(ADD_BLANK))
    plan, diff, _ = update_plan(
        old_plan=OLD, state={}, question="q", reason="r", evidence="e",
        llm_client=llm, plan_request=REQ, contract_enabled=False)
    assert "b_new" in {e.id for e in plan.entities}
    assert diff.contract == []
    # "no extra LLM call" is part of the flag-off contract, not just "no
    # observable effect" -- assert the author was never invoked at all.
    assert llm.author_calls == 0


def test_an_op_satisfying_the_contract_is_accepted():
    llm = _Scripted(_op(ADD_BLANK))
    plan, diff, _ = update_plan(
        old_plan=OLD, state={}, question="q", reason="r", evidence="e",
        llm_client=llm, plan_request=REQ, contract_enabled=True)
    assert "b_new" in {e.id for e in plan.entities}
    assert {c["kind"] for c in diff.contract} == {"blank_added"}
    assert {c["kind"] for c in diff.contract_satisfied} == {"blank_added"}


def test_an_op_that_satisfies_a_clause_but_breaks_the_binding_contract_is_rejected():
    """THE regression this feature exists to prevent.

    The first op adds a blank (satisfying `blank_added`) but gives it a
    retrieval_hint that interpolates `{b_new}` -- itself -- which
    `_plan_contract._ordering_violations` flags BINDING: a blank can never
    resolve a placeholder that names its own not-yet-resolved value. It must
    be rejected on the binding check even though the authored clause is
    satisfied. The second, clean op is then accepted.

    Still holds under fix round 1's non-increase rule: `OLD` starts with
    zero binding violations, so 0 -> 1 is still an increase. The rule this
    test needs is `test_an_op_that_leaves_a_preexisting_binding_violation_unchanged_is_accepted`
    below, which starts from a plan that already has one."""
    ordering_violation = {"op": "add_blank", "blank": {
        "id": "b_new", "kind": "blank", "value_type": "entity",
        "retrieval_hint": "What is near {b_new}?"},
        "wire_via_vp": {"id": "vp_ordering", "phrase": "near",
                        "subject_id": "b_new", "object_id": "b_x"}}
    llm = _Scripted(_op(ordering_violation), _op(ADD_BLANK))
    plan, diff, _ = update_plan(
        old_plan=OLD, state={}, question="q", reason="r", evidence="e",
        llm_client=llm, plan_request=REQ, contract_enabled=True)
    assert llm.calls == 2, "the ordering-violating op should have been rejected and retried"
    assert not any(vp.id == "vp_ordering" for vp in plan.verb_phrases)


def test_an_op_satisfying_nothing_in_the_contract_is_rejected():
    """Progress is required: the satisfied set must STRICTLY grow."""
    # Corrected field name: `ChangeValueTypeOp.to`, not `value_type`.
    noop_ish = {"op": "change_value_type", "blank_id": "b_x", "to": "text"}
    llm = _Scripted(_op(noop_ish), _op(ADD_BLANK))
    plan, _diff, _ = update_plan(
        old_plan=OLD, state={}, question="q", reason="r", evidence="e",
        llm_client=llm, plan_request=REQ, contract_enabled=True)
    assert llm.calls == 2
    assert "b_new" in {e.id for e in plan.entities}


def test_no_plan_request_means_no_authored_contract_even_when_enabled():
    """The orchestrator can call request_plan_update with no escalation behind
    it -- 71 of 134 questions did exactly that. The AUTHORED-clause machinery
    still requires a `PlanRequest` (there is nothing to author a contract
    from), so no author call happens and `diff.contract` stays empty.

    Final-review fix wave, Finding 3, superseded this test's old assertion
    that the op was ACCEPTED on the first try. It measured that at least 71
    of the 134 replanning questions never escalate, so gating the BINDING
    check on `contract_active` (which also requires a PlanRequest) meant the
    check never ran on that majority -- even though it costs no LLM call and
    is a strict improvement. See
    `test_binding_check_runs_with_no_plan_request_when_contract_enabled`
    below for the corrected (now-unconditional) binding behaviour; this test
    keeps only the half that is still true: no PlanRequest means no author
    call and no authored contract.
    """
    llm = _Scripted(_op(ADD_BLANK))
    plan, diff, _ = update_plan(
        old_plan=OLD, state={}, question="q", reason="r", evidence="e",
        llm_client=llm, plan_request=None, contract_enabled=True)
    assert llm.calls == 1
    assert "b_new" in {e.id for e in plan.entities}
    assert diff.contract == []
    assert diff.contract_active is False
    # Same reasoning as the flag-off test: the author must not be invoked at
    # all when there is no request to author from, not merely unobserved.
    assert llm.author_calls == 0


def test_binding_check_runs_with_no_plan_request_when_contract_enabled():
    """Finding 3 (final-review fix wave, half a): the BINDING non-increase
    check must fire whenever `contract_enabled` is True, regardless of
    whether a `PlanRequest` is attached. It costs no LLM call and is
    strictly an improvement -- it is the half of the feature that addresses
    the measured "15 of 191 updates increased the binding-violation count",
    and gating it on `contract_active` (PlanRequest-dependent) meant it never
    ran on the (at least) 71 of 134 replanning questions that never
    escalated. Only the AUTHORED-clause machinery (the strict-superset
    progress check) still requires a PlanRequest.

    Same ordering-violating op as the flagship regression test
    (`test_an_op_that_satisfies_a_clause_but_breaks_the_binding_contract_is_rejected`),
    but with NO `PlanRequest` attached: it must still be rejected and
    retried, not accepted on the first try the way it used to be before this
    fix (see the superseded assertion in
    `test_no_plan_request_means_no_authored_contract_even_when_enabled`'s
    docstring above).
    """
    ordering_violation = {"op": "add_blank", "blank": {
        "id": "b_new", "kind": "blank", "value_type": "entity",
        "retrieval_hint": "What is near {b_new}?"},
        "wire_via_vp": {"id": "vp_ordering", "phrase": "near",
                        "subject_id": "b_new", "object_id": "b_x"}}
    llm = _Scripted(_op(ordering_violation), _op(ADD_BLANK))
    plan, diff, _ = update_plan(
        old_plan=OLD, state={}, question="q", reason="r", evidence="e",
        llm_client=llm, plan_request=None, contract_enabled=True)
    assert llm.calls == 2, (
        "the binding check must fire and reject the ordering-violating op "
        "even with no plan_request attached"
    )
    assert not any(vp.id == "vp_ordering" for vp in plan.verb_phrases)
    assert "b_new" in {e.id for e in plan.entities}
    # No PlanRequest -> no authored contract, even though the binding check
    # ran and gated acceptance.
    assert diff.contract == []
    assert diff.contract_active is False
    assert llm.author_calls == 0


def test_an_author_failure_falls_through_to_todays_behaviour():
    """Fail-open. An empty contract leaves only the BINDING check, which is
    still stronger than the liveness test it replaces."""
    class AuthorBoom(_Scripted):
        def chat(self, messages=None, tools=None, **kw):
            if tools:
                raise RuntimeError("author down")
            return super().chat(messages=messages, tools=None, **kw)

    llm = AuthorBoom(_op(ADD_BLANK))
    plan, diff, _ = update_plan(
        old_plan=OLD, state={}, question="q", reason="r", evidence="e",
        llm_client=llm, plan_request=REQ, contract_enabled=True)
    assert "b_new" in {e.id for e in plan.entities}
    assert diff.contract == []


# ---------------------------------------------------------------------------
# Fix round 1 (review findings 1 and 2).
# ---------------------------------------------------------------------------

# A plan that ALREADY carries one `ordering` BINDING violation before any op
# runs: `b_bad`'s retrieval_hint interpolates `{b_bad}`, itself. Verified
# empirically (`binding_violations(ALREADY_VIOLATING_OLD, "q") ==
# ["ordering"]`) before being written into a test.
ALREADY_VIOLATING_OLD = GSWPlan.model_validate({
    "entities": [
        {"id": "b_x", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "Who founded Acme?"},
        {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
         "retrieval_hint": "Where born?"},
        {"id": "b_bad", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "What is near {b_bad}?"},
    ],
    "verb_phrases": [
        {"id": "vp1", "subject_id": "b_x", "phrase": "born_in", "object_id": "t"},
        {"id": "vp_bad", "subject_id": "b_bad", "phrase": "located_near",
         "object_id": "b_x"},
    ],
    "constraints": [],
})

ALREADY_VIOLATING_REQ = PlanRequest(
    request_id="q3:b_x:0", blank_id="b_x", assigned_blank_ids=["b_x"],
    reason="need an unrelated bridge fact", hint="add an unrelated blank")


def test_an_op_that_leaves_a_preexisting_binding_violation_unchanged_is_accepted():
    """The regression 7 of the 31 already-violating FRAMES-824 questions
    represent: the orchestrator asks for a genuinely useful, UNRELATED fix
    on a plan that already carries a binding violation from an earlier
    step. Rejecting every op on such a plan because the pre-existing
    violation is still present after the op -- an absolute "must be clean"
    reading of the binding check -- burns all attempts and drops the replan
    entirely, on exactly the plans that most need one.

    The op here adds `b_extra`, wired only to `b_x`; it never touches
    `b_bad`, so the SAME single `ordering` violation survives, unchanged
    (1 -> 1, not an increase). Verified empirically before writing this
    test: `binding_violations` returns `["ordering"]` both before and after.
    """
    unrelated_add = {"op": "add_blank", "blank": {
        "id": "b_extra", "kind": "blank", "value_type": "entity",
        "retrieval_hint": "What is the address?"},
        "wire_via_vp": {"id": "vp_extra", "phrase": "located_at",
                        "subject_id": "b_x", "object_id": "b_extra"}}
    llm = _Scripted(_op(unrelated_add))
    plan, diff, _ = update_plan(
        old_plan=ALREADY_VIOLATING_OLD, state={}, question="q", reason="r",
        evidence="e", llm_client=llm, plan_request=ALREADY_VIOLATING_REQ,
        contract_enabled=True)
    assert llm.calls == 1, "an unchanged violation count must not be rejected"
    assert "b_extra" in {e.id for e in plan.entities}
    assert {c["kind"] for c in diff.contract_satisfied} == {"blank_added"}


# A plan where `add_constraint` can introduce a genuine `acyclic` violation
# (a `derived` constraint whose own `output_blank_id` also appears among its
# `args_blanks`) while ALSO satisfying `depends_on(t)`: "b_x" reaches "t"
# ONLY through the new constraint (there is no verb-phrase between them), so
# `t`'s dependency set gains a member it didn't have before. Verified
# empirically before writing this test -- `apply_diff` applies the op
# cleanly (the executor's own dependency-graph builder skips an operand
# equal to its own output when building edges, so this is not a topology
# cycle `topological_sort_blanks` would catch), `binding_violations` goes
# from `[]` to `["acyclic"]`, and `satisfied_set` for `depends_on(t)` goes
# from empty to `{depends_on(t)}` (a strict superset).
ACYCLIC_OLD = GSWPlan.model_validate({
    "entities": [
        {"id": "b_x", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "Who is nearby?"},
        {"id": "b_y", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "Where is it?"},
        {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
         "retrieval_hint": "Where born?"},
    ],
    "verb_phrases": [
        {"id": "vp_by", "subject_id": "b_y", "phrase": "located_at", "object_id": "t"},
        {"id": "vp_bx", "subject_id": "b_x", "phrase": "near", "object_id": "b_y"},
    ],
    "constraints": [],
})

ACYCLIC_REQ = PlanRequest(
    request_id="q4:t:0", blank_id="t", assigned_blank_ids=["t"],
    reason="target needs one more bridge fact", hint="add a derived constraint")


def test_an_op_that_satisfies_depends_on_but_introduces_an_acyclic_violation_is_rejected():
    """The OTHER BINDING clause this feature was measured to need
    (`ordering` is covered by the flagship test above); previously
    uncovered in this file. `acyclic` is not a VP-graph cycle -- it is a
    `derived` constraint whose `output_blank_id` also appears among its own
    operands, a shape only reachable through `add_constraint`, not
    `add_blank`. The scripted author here returns `depends_on(t)` instead
    of `blank_added` (via `_Scripted`'s `contract_clauses` override), since
    this op never adds a blank.

    The first op is exactly this shape and satisfies `depends_on(t)` -- it
    must still be rejected, because under the non-increase rule the
    violation count goes from 0 to 1. The second op (`add_blank`, wiring a
    fresh blank directly to `t`) is clean AND also satisfies `depends_on(t)`
    a different way (the new blank itself becomes a dependency of `t`), so
    it is accepted.
    """
    acyclic_op = {"op": "add_constraint", "constraint": {
        "id": "c_self", "kind": "derived", "op": "sum",
        "args_blanks": ["b_x", "t"], "output_blank_id": "t"}}
    clean_op = {"op": "add_blank", "blank": {
        "id": "b_new2", "kind": "blank", "value_type": "entity",
        "retrieval_hint": "What connects to it?"},
        "wire_via_vp": {"id": "vp_new2", "phrase": "connects_to",
                        "subject_id": "b_new2", "object_id": "t"}}
    llm = _Scripted(_op(acyclic_op), _op(clean_op),
                     contract_clauses=[{"kind": "depends_on", "blank_id": "t"}])
    plan, diff, _ = update_plan(
        old_plan=ACYCLIC_OLD, state={}, question="q", reason="r", evidence="e",
        llm_client=llm, plan_request=ACYCLIC_REQ, contract_enabled=True)
    assert llm.calls == 2, "the acyclic op should have been rejected and retried"
    assert not any(c.id == "c_self" for c in plan.constraints)
    assert "b_new2" in {e.id for e in plan.entities}
    assert {c["kind"] for c in diff.contract_satisfied} == {"depends_on"}


# ---------------------------------------------------------------------------
# Fix round 2 (review finding: the `evidence` clobbering bug).
# ---------------------------------------------------------------------------


def test_a_binding_rejection_does_not_clobber_the_researchers_evidence():
    """Regression pin: `evidence` is a required `update_plan` PARAMETER (the
    researcher's retrieved evidence) threaded into `build_updater_messages`
    on attempt 1 and into every `build_updater_retry_messages` call on later
    attempts. Fix round 1 introduced a local named `evidence` inside the
    binding-violation rejection branch (`"; ".join(...)` over the newly
    increased clauses' evidence strings) that REASSIGNED the same local --
    Python has no block scoping -- permanently replacing the researcher's
    evidence with violation text for the remainder of the loop, the moment
    the first binding rejection fired. Silent: `update_plan`'s return value
    (the accepted plan and diff) looks identical either way; only the
    discarded retry PROMPT was wrong, which is why this needs its own test
    rather than being implied by the others.

    Demonstrated with a sentinel evidence string that must NOT appear
    anywhere in the attempt-1 diagnostic text (so a false pass can't happen
    via coincidental substring overlap), and must survive into attempt 2's
    prompt unchanged after the ordering-violating op is rejected on attempt
    1. Hand-reverting the rename (`violation_evidence` -> `evidence`)
    reproduces this test failing -- see task-4-report.md fix round 2.
    """
    sentinel = "SENTINEL-RETRIEVED-EVIDENCE-XYZ"
    ordering_violation = {"op": "add_blank", "blank": {
        "id": "b_new", "kind": "blank", "value_type": "entity",
        "retrieval_hint": "What is near {b_new}?"},
        "wire_via_vp": {"id": "vp_ordering", "phrase": "near",
                        "subject_id": "b_new", "object_id": "b_x"}}
    llm = _Scripted(_op(ordering_violation), _op(ADD_BLANK))
    plan, diff, _ = update_plan(
        old_plan=OLD, state={}, question="q", reason="r", evidence=sentinel,
        llm_client=llm, plan_request=REQ, contract_enabled=True)
    assert llm.calls == 2, "the ordering-violating op should have been rejected and retried"
    assert "b_new" in {e.id for e in plan.entities}
    assert len(llm.message_log) == 2
    second_attempt_text = json.dumps(llm.message_log[1])
    assert sentinel in second_attempt_text, (
        "the researcher's evidence must survive a binding-violation retry, "
        "not be replaced by the violation diagnostic text"
    )


# ---------------------------------------------------------------------------
# Final-review fix wave (2026-08-09): Finding 1 -- an identity-satisfied
# contract used to destroy the replan.
# ---------------------------------------------------------------------------


def test_a_preserved_only_contract_still_accepts_a_reasonable_op():
    """Finding 1 (CRITICAL): `preserved(x)` is true of the IDENTITY transform
    (before == after == old_plan) by construction -- b_x's own fields are
    trivially unchanged when before and after are the SAME plan object. The
    author is explicitly invited to emit exactly this clause for "blanks
    already carrying a resolved value" (`_plan_request_author.py:53`). If the
    author emits ONLY such clauses, `base_satisfied` already equals the full
    clause set before any op runs, so the strict-superset progress check can
    never pass -- no matter what the op does. Before this fix, every op was
    rejected, all `MAX_REVISION_ATTEMPTS` burned, and `update_plan` raised
    `PlanEmitError`, losing the replan entirely. Reproduced by the reviewer:
        preserved-only   RAISED   updater_calls=3   PlanEmitError

    The fix drops to BINDING-only checking whenever the authored contract is
    already fully satisfied by the identity. `ADD_BLANK` adds `b_new` and
    does not touch `b_x`, so this must be ACCEPTED on the first try, not
    raise.
    """
    llm = _Scripted(_op(ADD_BLANK),
                     contract_clauses=[{"kind": "preserved", "blank_id": "b_x"}])
    plan, diff, _ = update_plan(
        old_plan=OLD, state={}, question="q", reason="r", evidence="e",
        llm_client=llm, plan_request=REQ, contract_enabled=True)
    assert llm.calls == 1
    assert "b_new" in {e.id for e in plan.entities}
    assert diff.contract_identity_satisfied is True
    # The clauses must not be silently discarded even though the strict-
    # superset check was skipped for them.
    assert diff.contract == [
        {"kind": "preserved", "blank_id": "b_x", "value_type": None}]


def test_a_retyped_identity_only_contract_still_accepts_a_reasonable_op():
    """Finding 1, second reproduction: `retyped(x, T)` is also true of the
    identity whenever `x`'s CURRENT type already equals `T`. `b_x` starts as
    `value_type=entity`; an author that emits `retyped(b_x, entity)` produces
    the same unsatisfiable-progress trap as the preserved-only case above.
    Reproduced by the reviewer:
        retyped-identity-only   RAISED   updater_calls=3   PlanEmitError

    `ADD_BLANK` does not touch `b_x`'s `value_type`, so `retyped(b_x,
    entity)` stays satisfied after the op too -- this must be ACCEPTED on the
    first try under BINDING-only checking, not raise.
    """
    llm = _Scripted(
        _op(ADD_BLANK),
        contract_clauses=[{"kind": "retyped", "blank_id": "b_x", "value_type": "entity"}],
    )
    plan, diff, _ = update_plan(
        old_plan=OLD, state={}, question="q", reason="r", evidence="e",
        llm_client=llm, plan_request=REQ, contract_enabled=True)
    assert llm.calls == 1
    assert "b_new" in {e.id for e in plan.entities}
    assert diff.contract_identity_satisfied is True
    assert diff.contract == [
        {"kind": "retyped", "blank_id": "b_x", "value_type": "entity"}]


def test_identity_satisfied_contract_still_enforces_the_binding_check():
    """Finding 1's fix drops the strict-superset check but must KEEP the
    non-increase binding check -- "BINDING-only", not "no check at all". A
    preserved-only (identity-satisfied) contract must still reject an op
    that raises the binding-violation count, exactly like a normal
    (non-identity-satisfied) contract does in
    `test_an_op_that_satisfies_a_clause_but_breaks_the_binding_contract_is_rejected`
    above.
    """
    ordering_violation = {"op": "add_blank", "blank": {
        "id": "b_new", "kind": "blank", "value_type": "entity",
        "retrieval_hint": "What is near {b_new}?"},
        "wire_via_vp": {"id": "vp_ordering", "phrase": "near",
                        "subject_id": "b_new", "object_id": "b_x"}}
    llm = _Scripted(_op(ordering_violation), _op(ADD_BLANK),
                     contract_clauses=[{"kind": "preserved", "blank_id": "b_x"}])
    plan, diff, _ = update_plan(
        old_plan=OLD, state={}, question="q", reason="r", evidence="e",
        llm_client=llm, plan_request=REQ, contract_enabled=True)
    assert llm.calls == 2, (
        "the binding check must still fire and reject even in "
        "identity-satisfied (BINDING-only) mode"
    )
    assert not any(vp.id == "vp_ordering" for vp in plan.verb_phrases)
    assert diff.contract_identity_satisfied is True


def test_an_unreachable_contract_keeps_the_update_instead_of_losing_the_replan():
    """FAIL-OPEN on contract exhaustion. Observed live, FRAMES q351.

    The author wrote `depends_on(b_other_astronaut)`; no op the updater
    produced reached it in `MAX_REVISION_ATTEMPTS`; every attempt was rejected
    by the progress check; `update_plan` raised and the orchestrator lost the
    replan entirely -- on a question whose researcher had escalated precisely
    because the plan was wrong. The logged error was:

        "plan-updater could not produce a structural change after 3 attempts.
         last failure: the revised plan does not advance the contract;
         still unmet: ['depends_on(b_other_astronaut)']"

    An authored clause is a PREFERENCE about how the plan should change. It
    must never be a licence to discard a structurally-valid update: the op
    below parses, is non-trivial, applies, changes the signature and does NOT
    increase the binding-violation count. It is kept, flagged `contract_unmet`
    so the trace shows the contract went unhonoured rather than the replan
    silently vanishing.

    This is the third door onto the same failure: absolute-zero binding (fix
    round 1) and the identity-satisfied contract (final review) were the first
    two. All three burn every attempt and lose the update.
    """
    # `depends_on(b_new)` can never be satisfied by ADD_BLANK: the new blank is
    # wired as the OBJECT of vp_new with an entity subject, so b_new GAINS a
    # dependency... which would satisfy it. Use a clause naming a blank the op
    # does not touch at all, so the contract is genuinely unreachable.
    llm = _Scripted(_op(ADD_BLANK),
                    contract_clauses=[{"kind": "depends_on", "blank_id": "t"}])
    plan, diff, _ = update_plan(
        old_plan=OLD, state={}, question="q", reason="r", evidence="e",
        llm_client=llm, plan_request=REQ, contract_enabled=True)

    # The replan SURVIVED rather than raising PlanEmitError.
    assert "b_new" in {e.id for e in plan.entities}
    assert diff.contract_unmet is True
    assert diff.contract_active is True
    # It exhausted the budget trying to honour the contract before giving up.
    assert llm.calls == 3
    # The unreachable clause is reported as not satisfied, so a trace shows
    # exactly what the contract asked for and did not get.
    assert {c["kind"] for c in diff.contract} == {"depends_on"}
    assert diff.contract_satisfied == []


def test_exhaustion_with_no_structurally_valid_op_still_raises():
    """Fail-open is not fail-blind. If NO attempt ever produced a usable op --
    every one failed to parse, was trivial, or raised binding violations --
    there is nothing to keep, and `update_plan` must still raise so the
    orchestrator records the failure rather than silently proceeding on an
    unchanged plan."""
    garbage = SimpleNamespace(text="not a diff op at all", reasoning_content="",
                              prompt_tokens=0, completion_tokens=0, tool_calls=[])
    llm = _Scripted(garbage, contract_clauses=[{"kind": "blank_added"}])
    with pytest.raises(PlanEmitError):
        update_plan(old_plan=OLD, state={}, question="q", reason="r", evidence="e",
                    llm_client=llm, plan_request=REQ, contract_enabled=True)


def test_a_carried_contract_rejects_a_later_update_that_undoes_the_first():
    """CARRY-FORWARD. Live regression, FRAMES q456.

    "If the person who played the Joker in The Dark Knight (2008) had children,
    how old will they be on the film's 50th anniversary?"  gold: 52.

    A researcher escalated: the corpus has no 50th-anniversary date, so the plan
    needs a release-date blank plus a derived anniversary step. Update 1 was
    gated by the contract and produced exactly that --
    `sum(b_release_date, e_fifty) -> b_anniversary_date`, so
    age = anniversary - birth = 52. Correct.

    Update 2 cited no `plan_request_id`, so it ran with NO contract, and rewired
    the age constraint to `diff(b_release_date, b_child_birth_date)` -- the
    child's age at RELEASE, not at the anniversary. Answer: 2. The flag-off arm,
    which never replanned at all, answered 52.

    The contract was right and was overruled by absence. Carried forward, it
    refuses that edit: dropping `b_anniversary_date` from the dependency graph
    un-satisfies `depends_on(b_anniversary_date)` -- the very clause the
    escalation existed to establish -- so the satisfied set SHRINKS and the
    strict-superset rule rejects it.
    """
    # b_x depends on nothing; the op below would REMOVE the dependency that the
    # carried clause requires, exactly as q456's second update did.
    plan = GSWPlan.model_validate({
        "entities": [
            {"id": "b_src", "kind": "blank", "value_type": "entity",
             "retrieval_hint": "When was it released?"},
            {"id": "b_mid", "kind": "blank", "value_type": "entity",
             "retrieval_hint": "What is the anniversary date?"},
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
             "retrieval_hint": "How old then?"},
        ],
        "verb_phrases": [
            {"id": "vp1", "subject_id": "b_src", "phrase": "anniversary_of",
             "object_id": "b_mid"},
            {"id": "vp2", "subject_id": "b_mid", "phrase": "age_at", "object_id": "t"},
        ],
        "constraints": [],
    })
    carried = [{"kind": "depends_on", "blank_id": "t", "value_type": None}]
    # This op removes the vp that gives `t` its dependency on b_mid -- the
    # shape of q456's regression.
    undo = {"op": "remove_vp", "vp_id": "vp2"}
    good = {"op": "add_blank", "blank": {
        "id": "b_new", "kind": "blank", "value_type": "entity",
        "retrieval_hint": "extra"},
        "wire_via_vp": {"id": "vp_new", "phrase": "rel",
                        "subject_id": "b_new", "object_id": "t"}}
    llm = _Scripted(_op(undo), _op(good))
    new_plan, diff, _ = update_plan(
        old_plan=plan, state={}, question="q", reason="r", evidence="e",
        llm_client=llm, contract_enabled=True, carried_contract=carried)

    assert llm.calls == 2, "the undoing op should have been rejected and retried"
    assert any(vp.id == "vp2" for vp in new_plan.verb_phrases), \
        "vp2 must survive -- removing it is what the carried contract forbids"
    assert diff.contract_active is True
    # The carried contract was used; no author call was needed.
    assert llm.author_calls == 0


def test_a_destructive_op_is_never_kept_by_the_fail_open():
    """Live regression, FRAMES q806.

    "What is greater: the combined 2011 populations of four villages, or the
    2022 population of London?"  gold: the 2022 population of London.

    The contract rejected `change_target(t -> b_sum_2011)` on all three
    attempts. The fail-open kept it anyway, so the system stopped answering
    "which is greater" and answered "what is the sum" -- 34765. The flag-off arm
    answered "London is greater".

    `change_target` redefines WHICH QUESTION is being answered. Keeping one over
    the contract's explicit objection is strictly worse than losing the replan,
    which merely leaves the previous plan -- and the previous plan was answering
    the right question. Additive ops still fail open; these do not.
    """
    retarget = {"op": "change_target", "from_id": "t", "to_id": "b_x"}
    llm = _Scripted(_op(retarget),
                    contract_clauses=[{"kind": "depends_on", "blank_id": "t"}])
    with pytest.raises(PlanEmitError):
        update_plan(old_plan=OLD, state={}, question="q", reason="r", evidence="e",
                    llm_client=llm, plan_request=REQ, contract_enabled=True)


def test_a_clause_rejection_tells_the_updater_HOW_to_satisfy_it():
    """The rejection message must prescribe the op shape, not just name the clause.

    `depends_on` was satisfied 4/33 in the 824 run and 0/5 after an
    author-side prompt fix, with the same error every time: a verb phrase wired
    `b_X -> b_new` when the clause needs `b_new -> b_X`. The retry message said
    only "still unmet: ['depends_on(b_writer1)']" -- which names WHICH
    requirement failed but not HOW to meet it.

    Three prompt attempts aimed at the AUTHOR (which clause to pick) moved
    nothing. None had told the UPDATER -- the component that must actually
    emit the op -- what shape complies. It follows concrete structural
    instructions faithfully (it implements researcher hints verbatim), so the
    rejection now carries one, including the direction, which is the exact
    thing it gets backwards.
    """
    # `depends_on(t)` cannot be met by ADD_BLANK here, so attempt 1 is rejected
    # and attempt 2's prompt must carry the how-to.
    llm = _Scripted(_op(ADD_BLANK), _op(ADD_BLANK),
                    contract_clauses=[{"kind": "depends_on", "blank_id": "t"}])
    update_plan(old_plan=OLD, state={}, question="q", reason="r", evidence="e",
                llm_client=llm, plan_request=REQ, contract_enabled=True)

    retry = json.dumps(llm.message_log[1])
    assert "depends_on" in retry
    # names the op shapes that would work
    assert "add_vp" in retry and "add_constraint" in retry
    # and states the direction, which is the failure mode
    assert "object_id" in retry and "SUBJECT" in retry


def test_the_how_to_covers_every_clause_kind():
    """Every kind in the vocabulary must render actionable advice -- a clause
    that can be authored but not explained leaves the updater guessing, which
    is the state `depends_on` was in."""
    from panini_dr.adapters.ours._plan_request_clauses import CLAUSE_KINDS
    from panini_dr.adapters.ours._plan_updater import _how_to_satisfy
    generic = _how_to_satisfy(Clause("something_unknown"))
    for kind in CLAUSE_KINDS:
        advice = _how_to_satisfy(Clause(kind, blank_id="b_x", value_type="list"))
        assert advice and advice != generic, f"{kind} has no specific advice"
