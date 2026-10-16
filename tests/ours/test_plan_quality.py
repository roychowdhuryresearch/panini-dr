"""Unit tests for deterministic plan-quality checks (W5.1). Pure Python."""
from __future__ import annotations

from panini_dr.adapters.ours._planner_exec import (
    Constraint, Entity, GSWPlan, VerbPhrase,
)
from panini_dr.adapters.ours._gsw_workspace import PlanQualityReport
from panini_dr.adapters.ours._plan_quality import assess_plan_quality_deterministic


def _types(report: PlanQualityReport) -> set[str]:
    return {i.issue_type for i in report.issues}


def _sev_of(report, issue_type):
    return {i.severity for i in report.issues if i.issue_type == issue_type}


def test_clean_plan_has_no_issues():
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="Pink_Floyd"),
            Entity(id="t", kind="blank", value_type="entity", is_target=True),
        ],
        verb_phrases=[VerbPhrase(id="vp1", phrase="released", subject_id="e1", object_id="t")],
        constraints=[],
    )
    rep = assess_plan_quality_deterministic(plan, question="What did Pink Floyd release?", use_hints=False)
    assert rep.issues == []


def test_two_targets_flagged_high():
    # Two blanks connected via a VP (satisfies no-dangling validator).
    plan = GSWPlan(
        entities=[
            Entity(id="a", kind="blank", value_type="entity", is_target=True),
            Entity(id="b", kind="blank", value_type="entity", is_target=True),
        ],
        verb_phrases=[VerbPhrase(id="vp1", phrase="related_to", subject_id="a", object_id="b")],
        constraints=[])
    rep = assess_plan_quality_deterministic(plan, question="Q", use_hints=False)
    assert "target_unreachable" in _types(rep) and rep.has_high_severity_issue


def test_missing_value_type_flagged():
    # value_type=None (omitted) is valid pydantic; the check flags the absent type.
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="X"),
            Entity(id="t", kind="blank", value_type=None, is_target=True),
        ],
        verb_phrases=[VerbPhrase(id="vp1", phrase="r", subject_id="e1", object_id="t")],
        constraints=[])
    rep = assess_plan_quality_deterministic(plan, question="Q", use_hints=False)
    assert "underspecified_blank" in _types(rep)


def test_hints_on_missing_retrieval_hint_flagged():
    plan = GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="X"),
            Entity(id="t", kind="blank", value_type="entity", is_target=True),
        ],
        verb_phrases=[VerbPhrase(id="vp1", phrase="r", subject_id="e1", object_id="t")],
        constraints=[])
    rep = assess_plan_quality_deterministic(plan, question="Q", use_hints=True)
    assert "bad_hint" in _types(rep)


def test_placeholder_references_unknown_blank_flagged_high():
    # Need a filled entity + VP to satisfy no-dangling validator.
    plan = GSWPlan(
        entities=[
            Entity(id="anchor", kind="filled", name="World"),
            Entity(id="t", kind="blank", value_type="entity", is_target=True,
                   retrieval_hint="largest religion in {b_nope}",
                   verification_claim="must be the largest religion in {b_nope}"),
        ],
        verb_phrases=[VerbPhrase(id="vp1", phrase="has_religion", subject_id="anchor", object_id="t")],
        constraints=[])
    rep = assess_plan_quality_deterministic(plan, question="Q", use_hints=True)
    assert "placeholder_error" in _types(rep)
    assert "high" in _sev_of(rep, "placeholder_error")


def test_dropped_as_of_date_is_high_bare_year_not_flagged():
    # Democracy-Index / largest-religion: "2023" IS present (in b_country hint) so
    # not flagged; "as of August 3, 2024" is dropped -> HIGH missing_scope.
    plan = GSWPlan(
        entities=[
            Entity(id="b_country", kind="blank", value_type="entity",
                   retrieval_hint="Which country ranked highest in the 2023 Democracy Index?",
                   verification_claim="the country ranked highest in the 2023 Democracy Index"),
            Entity(id="t", kind="blank", value_type="entity", is_target=True,
                   retrieval_hint="What is the largest religion in {b_country}?",
                   verification_claim="the largest religion in {b_country}"),
        ],
        verb_phrases=[VerbPhrase(id="vp1", phrase="biggest_religion", subject_id="b_country", object_id="t")],
        constraints=[])
    q = "What is the largest religion in the country highest in the 2023 Democracy Index as of August 3, 2024?"
    rep = assess_plan_quality_deterministic(plan, question=q, use_hints=True)
    assert "missing_scope" in _types(rep)
    assert "high" in _sev_of(rep, "missing_scope")     # the as-of date
    # 2023 is present in the corpus, so it is NOT separately flagged as a missing bare year
    miss = [i for i in rep.issues if i.issue_type == "missing_scope"]
    assert all("2023" not in i.evidence for i in miss)


def test_dropped_source_constraint_flagged():
    # Need a filled entity + VP to satisfy no-dangling validator.
    plan = GSWPlan(
        entities=[
            Entity(id="anchor", kind="filled", name="Democracy"),
            Entity(id="t", kind="blank", value_type="entity", is_target=True,
                   retrieval_hint="Which country is most democratic?",
                   verification_claim="the most democratic country"),
        ],
        verb_phrases=[VerbPhrase(id="vp1", phrase="ranked_in", subject_id="t", object_id="anchor")],
        constraints=[])
    q = "Which country ranked highest according to the Democracy Index?"
    rep = assess_plan_quality_deterministic(plan, question=q, use_hints=True)
    assert "missing_scope" in _types(rep)
