"""Flag off ⇒ nothing moves, and the real violations are really caught.

`update_plan` is on the DEFAULT path: every shipped run replans. A feature that
changes behaviour when its flag is unset is a silent regression on the shipped
config, so identity is tested first and explicitly.

The second half uses the structure of plans logged in a FRAMES-824 research
run -- the actual `ordering` and `acyclic` violations the updater produced --
with the entity names replaced by fictional ones (see
tests/fixtures/plan_update_violations/README.md).
"""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from panini_dr.adapters.ours._plan_request import PlanRequest
from panini_dr.adapters.ours._plan_request_clauses import binding_violations
from panini_dr.adapters.ours._plan_updater import update_plan
from panini_dr.adapters.ours._planner_exec import GSWPlan

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "plan_update_violations"

OLD = GSWPlan.model_validate({
    "entities": [
        {"id": "b_x", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "Who founded Acme?"},
        {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
         "retrieval_hint": "Where born?"},
    ],
    # `GSWPlan._check_no_dangling` rejects any entity not referenced by a
    # verb-phrase or constraint edge -- both blanks need one to validate.
    "verb_phrases": [
        {"id": "vp0", "phrase": "relates_to", "subject_id": "b_x", "object_id": "t"},
    ],
    "constraints": [],
})

# `AddBlankOp.wire_via_vp` is required (`_plan_diff.py:107`) -- a blank with no
# wiring VP is dangling by construction and `GSWPlan` would reject it, so the
# op must carry one atomically with the new blank.
ADD_BLANK = {"op": "add_blank", "blank": {
    "id": "b_new", "kind": "blank", "value_type": "entity",
    "retrieval_hint": "What address?"},
    "wire_via_vp": {"id": "vp_new", "phrase": "located_at",
                    "subject_id": "t", "object_id": "b_new"}}

# A real escalation, not `None`. `update_plan`'s author-invocation guard is
# `contract_enabled and plan_request is not None` -- with no `PlanRequest`
# supplied, the SECOND half of that guard already blocks the author on its
# own, and a flag-off test passing no request cannot tell "the flag stopped
# it" from "there was never anything to author from". `REQ` makes the flag
# the only thing standing between this call and an author invocation.
REQ = PlanRequest(request_id="q:t:0", blank_id="t", assigned_blank_ids=("t",),
                  reason="no chunk names an address", hint="add a bridge blank")


class _One:
    def __init__(self):
        self.tool_calls_seen = 0

    def chat(self, messages=None, tools=None, **kw):
        if tools:
            self.tool_calls_seen += 1
        return SimpleNamespace(text=json.dumps(ADD_BLANK), reasoning_content="",
                               prompt_tokens=0, completion_tokens=0, tool_calls=[])


def test_flag_off_never_calls_the_contract_author():
    """Not just 'same result' -- the author must not be invoked at all, or an
    unset run pays for a model call it does not use.

    `plan_request=REQ` is load-bearing here, not decoration: without a real
    request, `plan_request is not None` already fails on its own and the
    author is unreachable regardless of the flag, so the assertion below
    would pass even if `contract_enabled` were ignored entirely. With `REQ`
    supplied, the flag is the only thing left standing between this call and
    an author invocation."""
    llm = _One()
    update_plan(old_plan=OLD, state={}, question="q", reason="r", evidence="e",
                llm_client=llm, plan_request=REQ, contract_enabled=False)
    assert llm.tool_calls_seen == 0


def test_flag_off_produces_the_same_plan_and_an_empty_contract_record():
    llm = _One()
    plan, diff, _ = update_plan(
        old_plan=OLD, state={}, question="q", reason="r", evidence="e",
        llm_client=llm, plan_request=REQ, contract_enabled=False)
    assert {e.id for e in plan.entities} == {"b_x", "t", "b_new"}
    assert diff.contract == [] and diff.contract_satisfied == []


@pytest.mark.parametrize("clause", ["ordering", "acyclic"])
def test_a_real_logged_post_update_violation_is_detected(clause):
    """Regression fixtures with the structure of plans logged in a FRAMES-824
    research run. 36 BINDING violations survived a plan update there -- 34
    ordering, 2 acyclic -- and nothing in the pipeline noticed. These two are the
    proof the checker sees them."""
    path = FIX / f"{clause}.json"
    rec = json.loads(path.read_text())
    plan = GSWPlan.model_validate(rec["plan"])
    assert clause in binding_violations(plan, rec["question"])
