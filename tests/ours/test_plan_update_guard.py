"""Unit tests for the W6 plan-update identity guard. Pure Python."""
from __future__ import annotations

from panini_dr.adapters.ours._planner_exec import (
    Constraint, Entity, GSWPlan, VerbPhrase,
)
from panini_dr.adapters.ours._plan_update_guard import (
    changed_identity_blanks, downstream_resolved_blanks,
)


def _plan(*, t_hint="h", t_claim=None, t_vtype="entity", t_target=True,
          phrase="rel", t_is_object=True):
    """Single target blank t wired to filled anchor e1 (direction controllable)."""
    if t_is_object:
        vp = VerbPhrase(id="vp1", phrase=phrase, subject_id="e1", object_id="t")
    else:
        vp = VerbPhrase(id="vp1", phrase=phrase, subject_id="t", object_id="e1")
    return GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="Anchor"),
            Entity(id="t", kind="blank", value_type=t_vtype, is_target=t_target,
                   retrieval_hint=t_hint, verification_claim=t_claim),
        ],
        verb_phrases=[vp], constraints=[])


def _c(old, new):
    return changed_identity_blanks(old, new, preserved_ids=["t"], resolved_ids={"t"})


def test_identical_plan_no_changes():
    assert _c(_plan(), _plan()) == []

def test_changed_hint_flags():
    assert _c(_plan(t_hint="a"), _plan(t_hint="b")) == ["t"]

def test_changed_claim_flags():
    assert _c(_plan(t_claim="a"), _plan(t_claim="b")) == ["t"]

def test_changed_value_type_flags():
    assert _c(_plan(t_vtype="entity"), _plan(t_vtype="number")) == ["t"]

def test_changed_target_role_flags():
    assert _c(_plan(t_target=True), _plan(t_target=False)) == ["t"]

def test_changed_relation_phrase_flags():
    assert _c(_plan(phrase="released"), _plan(phrase="directed")) == ["t"]

def test_flipped_relation_direction_flags():
    # same phrase + same other endpoint, but t goes object -> subject (meaning change)
    assert _c(_plan(t_is_object=True), _plan(t_is_object=False)) == ["t"]

def test_unresolved_blank_not_flagged():
    assert changed_identity_blanks(_plan(t_hint="a"), _plan(t_hint="b"),
                                   preserved_ids=["t"], resolved_ids=set()) == []

def test_not_preserved_blank_ignored():
    assert changed_identity_blanks(_plan(t_hint="a"), _plan(t_hint="b"),
                                   preserved_ids=[], resolved_ids={"t"}) == []


def _bridge_plan():
    # b_country --biggest_religion--> t ; t depends on b_country
    return GSWPlan(
        entities=[
            Entity(id="b_country", kind="blank", value_type="entity",
                   retrieval_hint="highest in 2023 Democracy Index"),
            Entity(id="t", kind="blank", value_type="entity", is_target=True,
                   retrieval_hint="largest religion in {b_country}"),
        ],
        verb_phrases=[VerbPhrase(id="vp1", phrase="biggest_religion",
                                 subject_id="b_country", object_id="t")],
        constraints=[])


def test_downstream_resolved_dependent_collected():
    plan = _bridge_plan()
    # b_country changed -> t (resolved) depends on it -> t is downstream-stale
    out = downstream_resolved_blanks(plan, seeds=["b_country"],
                                     resolved_ids={"b_country", "t"})
    assert out == {"t"}

def test_downstream_skips_unresolved():
    plan = _bridge_plan()
    out = downstream_resolved_blanks(plan, seeds=["b_country"],
                                     resolved_ids={"b_country"})  # t unresolved
    assert out == set()
