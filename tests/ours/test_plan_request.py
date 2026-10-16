"""The researcher's structural proposal, carried intact.

`suggest_plan_revision(reason, hint)` asks the researcher for a CONCRETE
structural fix -- "add b_address (entity, role=bridge-attribute) wired by VP
(b_hospital, located_at, b_address)". Today that reaches the orchestrator and is
then paraphrased into the orchestrator's own prose before `request_plan_update`,
so the best-informed proposal in the pipeline never reaches the thing that edits
the plan. `PlanRequest` is the vehicle that carries it the last hop.
"""
from __future__ import annotations

import pytest

from panini_dr.adapters.ours._plan_request import (
    PlanRequest, plan_request_from_revision,
)

RR = {
    "blank_id": "b_postcode",
    "assigned_blank_ids": ["b_postcode", "b_extra"],
    "reason": "no chunk in the corpus lists a postcode for Wren Clinic",
    "hint": "add b_address (entity, role=bridge-attribute) wired by VP "
            "(b_hospital, located_at, b_address)",
}


def test_it_carries_the_hint_verbatim():
    """The hint is the payload. Any paraphrase here defeats the whole feature."""
    pr = plan_request_from_revision(RR, question_id="q42", seq=0)
    assert pr.hint == RR["hint"]
    assert pr.reason == RR["reason"]
    assert pr.blank_id == "b_postcode"
    assert pr.assigned_blank_ids == ("b_postcode", "b_extra")


def test_the_id_is_stable_and_identifies_the_blank():
    """The orchestrator passes this id back as a tool argument, so it must be
    reproducible from the same inputs and readable in a trace."""
    a = plan_request_from_revision(RR, question_id="q42", seq=0)
    b = plan_request_from_revision(RR, question_id="q42", seq=0)
    assert a.request_id == b.request_id == "q42:b_postcode:0"


def test_seq_disambiguates_two_escalations_on_the_same_blank():
    a = plan_request_from_revision(RR, question_id="q42", seq=0)
    b = plan_request_from_revision(RR, question_id="q42", seq=1)
    assert a.request_id != b.request_id


@pytest.mark.parametrize("bad", [
    {},                                        # empty
    {"reason": "x"},                           # no hint
    {"hint": "x"},                             # no reason
    {"reason": "", "hint": "x"},               # blank reason
    {"reason": "x", "hint": "   "},            # whitespace hint
    {"reason": "   ", "hint": "x"},            # whitespace reason
])
def test_an_unusable_revision_dict_yields_none_rather_than_a_junk_request(bad):
    """A malformed escalation must not produce a contract. Returning None lets
    the caller fall through to today's behaviour."""
    assert plan_request_from_revision(bad, "q1", 0) is None


def test_it_is_frozen():
    """Nothing downstream may edit the researcher's words."""
    pr = plan_request_from_revision(RR, "q1", 0)
    with pytest.raises(Exception):
        pr.hint = "something else"  # type: ignore[misc]


def test_a_missing_blank_id_is_tolerated():
    """`allowed_order` can be empty; the request is still usable."""
    # Test with blank_id=None
    pr = plan_request_from_revision({**RR, "blank_id": None}, "q1", 0)
    assert pr is not None
    assert pr.blank_id is None
    assert pr.request_id == "q1:_:0"

    # Test with empty assigned_blank_ids (what "allowed_order can be empty" really means)
    pr_empty = plan_request_from_revision(
        {**RR, "assigned_blank_ids": []},
        "q1",
        0,
    )
    assert pr_empty is not None
    assert pr_empty.assigned_blank_ids == ()

    # Test with missing assigned_blank_ids key entirely
    pr_missing = plan_request_from_revision(
        {k: v for k, v in RR.items() if k != "assigned_blank_ids"},
        "q1",
        0,
    )
    assert pr_missing is not None
    assert pr_missing.assigned_blank_ids == ()


def test_assigned_blank_ids_is_immutable_tuple():
    """assigned_blank_ids is stored as a tuple, not a list, so the entire
    object is immutable. Direct construction with a list literal must also
    result in a tuple."""
    # Direct construction with a list literal should convert to tuple
    pr = PlanRequest(
        request_id="q1:b_x:0",
        blank_id="b_x",
        assigned_blank_ids=["b_x", "b_y"],  # type: ignore[arg-type]
        reason="test reason",
        hint="test hint",
    )
    assert isinstance(pr.assigned_blank_ids, tuple)
    assert pr.assigned_blank_ids == ("b_x", "b_y")

    # Verify mutation is impossible
    with pytest.raises(Exception):
        pr.assigned_blank_ids.append("b_z")  # type: ignore[attr-defined]
