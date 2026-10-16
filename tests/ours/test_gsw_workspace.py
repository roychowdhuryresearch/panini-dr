"""Unit tests for GSWWorkspace (W1: read-only wrapper + shared types).

Pure Python, no LLM/retriever — mirrors tests/ours/test_planner_exec.py.
"""
from __future__ import annotations

from panini_dr.adapters.ours._planner_exec import (
    BlankResult,
    Constraint,
    Entity,
    GSWPlan,
    VerbPhrase,
    build_dependency_graph,
    topological_sort_blanks,
)
from panini_dr.adapters.ours.gsw_planner_react_v1 import _compute_levels
from panini_dr.adapters.ours._gsw_workspace import (
    BlankAttempt,
    BlankContext,
    BlankResearchResult,
    GSWWorkspace,
    PlanIssue,
    PlanQualityReport,
)


def test_plan_issue_and_report_high_severity():
    low = PlanIssue(severity="low", issue_type="bad_hint", blank_id="t",
                    evidence="e", suggested_fix="f")
    high = PlanIssue(severity="high", issue_type="missing_scope", blank_id="t",
                     evidence="e2", suggested_fix="f2")
    assert PlanQualityReport(issues=[low]).has_high_severity_issue is False
    assert PlanQualityReport(issues=[low, high]).has_high_severity_issue is True


def test_blank_attempt_and_result_defaults():
    att = BlankAttempt(blank_id="b1", attempt_id=0)
    assert att.status == "unknown"
    assert att.evidence_chunk_ids == []
    res = BlankResearchResult(blank_id="b1", status="unresolved")
    assert res.value is None
    assert res.confidence == 0.0
    assert res.plan_issue is None


def test_blank_context_holds_hints_and_upstream():
    ctx = BlankContext(
        blank_id="t", value_type="entity",
        retrieval_hint="What is the largest religion in {b_country}?",
        verification_claim="The candidate must be the largest religion in {b_country}.",
        upstream_values={"b_country": "Norway"},
    )
    assert ctx.upstream_values["b_country"] == "Norway"
    assert ctx.previous_attempts == []


def _diff_plan() -> GSWPlan:
    """Portland ME/OR population-diff plan: b1,b2 -> t (derived diff)."""
    return GSWPlan(
        entities=[
            Entity(id="e1", kind="filled", name="Portland_Maine"),
            Entity(id="e2", kind="filled", name="Portland_Oregon"),
            Entity(id="b1", kind="blank", value_type="number"),
            Entity(id="b2", kind="blank", value_type="number"),
            Entity(id="t", kind="blank", value_type="number", is_target=True),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="has_population", subject_id="e1", object_id="b1"),
            VerbPhrase(id="vp2", phrase="has_population", subject_id="e2", object_id="b2"),
        ],
        constraints=[
            Constraint(id="c1", kind="derived", op="diff",
                       args_blanks=["b2", "b1"], output_blank_id="t"),
        ],
    )


def _make_workspace(plan: GSWPlan, resolved: dict[str, object]) -> GSWWorkspace:
    deps = build_dependency_graph(plan)
    levels = _compute_levels(deps)
    order = topological_sort_blanks(plan)
    state = {
        e.id: BlankResult(blank_id=e.id, status="unknown")
        for e in plan.blank_entities()
    }
    for bid, val in resolved.items():
        state[bid].status = "resolved"
        state[bid].value = val
    return GSWWorkspace(
        question="Q", plan=plan, state=state,
        deps=deps, levels=levels, order=order, qgsw_state=None,
    )


def test_ready_blanks_nothing_resolved():
    ws = _make_workspace(_diff_plan(), resolved={})
    assert set(ws.get_ready_blanks()) == {"b1", "b2"}


def test_ready_blanks_after_upstream_resolved():
    ws = _make_workspace(_diff_plan(), resolved={"b1": 68000, "b2": 650000})
    assert ws.get_ready_blanks() == ["t"]
    assert ws.resolved_ids() == {"b1", "b2"}


def test_ready_blanks_all_resolved_is_empty():
    ws = _make_workspace(_diff_plan(), resolved={"b1": 1, "b2": 2, "t": 1})
    assert ws.get_ready_blanks() == []


def _hint_plan() -> GSWPlan:
    """b_country -> t projection, with hints on both blanks."""
    return GSWPlan(
        entities=[
            Entity(id="b_country", kind="blank", value_type="entity",
                   retrieval_hint="Which country ranked highest in the 2023 Democracy Index?",
                   verification_claim="The candidate must be the country ranked highest in the 2023 Democracy Index."),
            Entity(id="t", kind="blank", value_type="entity", is_target=True,
                   retrieval_hint="What is the largest religion in {b_country}?",
                   verification_claim="The candidate must be the largest religion in {b_country}."),
        ],
        verb_phrases=[
            VerbPhrase(id="vp1", phrase="biggest_religion", subject_id="b_country", object_id="t"),
        ],
        constraints=[],
    )


def test_blank_context_includes_hint_and_upstream_value():
    ws = _make_workspace(_hint_plan(), resolved={"b_country": "Norway"})
    ctx = ws.get_blank_context("t")
    assert ctx.blank_id == "t"
    assert ctx.value_type == "entity"
    assert ctx.retrieval_hint == "What is the largest religion in {b_country}?"
    assert ctx.upstream_values == {"b_country": "Norway"}
    assert "b_country -> t" in ctx.related_edges
    assert ctx.previous_attempts == []


def test_ready_blanks_hint_plan_downstream_blocked():
    # Single projection chain b_country -> t: with nothing resolved, only the
    # upstream is ready; the downstream target stays blocked until it resolves.
    ws = _make_workspace(_hint_plan(), resolved={})
    assert ws.get_ready_blanks() == ["b_country"]


def test_summary_shape():
    ws = _make_workspace(_hint_plan(), resolved={"b_country": "Norway"})
    s = ws.summary()
    assert s["question"] == "Q"
    assert s["ready_blanks"] == ["t"]
    assert s["resolved"] == ["b_country"]
    assert s["blanks"]["b_country"]["value"] == "Norway"
    assert s["plan_issues"] == []


def test_summary_plan_issues_do_not_alias_live_dataclass():
    # summary() must COPY each PlanIssue, not expose the live __dict__, so a
    # caller mutating the snapshot can't corrupt workspace state (matters once
    # W5 populates plan_issues).
    ws = _make_workspace(_hint_plan(), resolved={})
    ws.plan_issues.append(
        PlanIssue(severity="high", issue_type="missing_scope", blank_id="t",
                  evidence="e", suggested_fix="f")
    )
    snap = ws.summary()
    snap["plan_issues"][0]["severity"] = "low"
    assert ws.plan_issues[0].severity == "high"


def test_state_view_matches_get_state_shape():
    # Reproduces the orchestrator get_state view exactly (key order included).
    ws = _make_workspace(_diff_plan(), resolved={"b1": 68000})
    constraint_outputs = {"t"}  # t is produced by the derived diff constraint
    view = ws.state_view(constraint_outputs)
    assert list(view.keys()) == ["blanks"]
    # blank order mirrors state.items() insertion order (what the LLM reads)
    assert [b["blank_id"] for b in view["blanks"]] == ["b1", "b2", "t"]
    by_id = {b["blank_id"]: b for b in view["blanks"]}
    # every blank present
    assert set(by_id) == {"b1", "b2", "t"}
    # exact per-blank key order
    assert list(by_id["b1"].keys()) == [
        "blank_id", "status", "value", "is_target", "is_constraint_output",
    ]
    # resolved upstream carries its value+status
    assert by_id["b1"]["status"] == "resolved"
    assert by_id["b1"]["value"] == 68000
    assert by_id["b1"]["is_target"] is False
    assert by_id["b1"]["is_constraint_output"] is False
    # target + constraint-output flags
    assert by_id["t"]["is_target"] is True
    assert by_id["t"]["is_constraint_output"] is True
    # unresolved blank
    assert by_id["b2"]["status"] == "unknown"
    assert by_id["b2"]["value"] is None


def test_record_research_result_matches_inline_resolver_commit():
    from panini_dr.adapters.ours._planner_exec import BlankResult as _BR

    ws = _make_workspace(_diff_plan(), resolved={})
    # what the inline resolver-commit at gsw_planner_orchestrator_v1.py:2319-2326
    # would have written for a resolved, non-partial blank:
    inline = _BR(
        blank_id="b1", value=68000, status="resolved",
        evidence_chunk_ids=["c1", "c2"],
        resolved_by="resolver", llm_calls=0, wall_time_s=0.0,
        verifier_status=None, completeness_status=None,
    )
    ws.record_research_result(
        BlankResearchResult(
            blank_id="b1", status="resolved", value=68000,
            resolved_by="resolver", evidence_chunk_ids=["c1", "c2"],
        )
    )
    got = ws.state["b1"]
    assert got == inline
    # the workspace wrote into its OWN state dict (same object the orchestrator owns)
    assert ws.state["b1"].status == "resolved"
    assert ws.state["b1"].resolved_by == "resolver"
    assert ws.state["b1"].evidence_chunk_ids == ["c1", "c2"]


def test_record_research_result_partial_flag():
    ws = _make_workspace(_diff_plan(), resolved={})
    ws.record_research_result(
        BlankResearchResult(
            blank_id="b2", status="resolved", value=650000,
            resolved_by="resolver", evidence_chunk_ids=["c9"],
            completeness_status="partial",
        )
    )
    assert ws.state["b2"].completeness_status == "partial"
    assert ws.state["b2"].status == "resolved"


def test_record_research_result_copies_evidence_list():
    # the written BlankResult must not alias the result's evidence list
    ws = _make_workspace(_diff_plan(), resolved={})
    ev = ["c1"]
    res = BlankResearchResult(blank_id="b1", status="resolved", value=1,
                              resolved_by="resolver", evidence_chunk_ids=ev)
    ws.record_research_result(res)
    ev.append("c2")
    assert ws.state["b1"].evidence_chunk_ids == ["c1"]


def test_record_research_result_writes_shared_state_object():
    # THE core W3 claim: the workspace writes into the SAME state dict the
    # orchestrator owns (not a private copy) — that is what makes the write-path
    # migration behavior-neutral. Build the state dict externally and assert the
    # mutation is visible on that external reference.
    plan = _diff_plan()
    state = {
        e.id: BlankResult(blank_id=e.id, status="unknown")
        for e in plan.blank_entities()
    }
    deps = build_dependency_graph(plan)
    ws = GSWWorkspace(
        question="Q", plan=plan, state=state, deps=deps,
        levels=_compute_levels(deps), order=topological_sort_blanks(plan),
        qgsw_state=None,
    )
    ws.record_research_result(
        BlankResearchResult(
            blank_id="b1", status="resolved", value=68000,
            resolved_by="resolver", evidence_chunk_ids=["c1"],
        )
    )
    # assert on the EXTERNAL dict reference — proves ws.state IS that object
    assert ws.state is state
    assert state["b1"].status == "resolved"
    assert state["b1"].value == 68000


def test_resolved_values_matches_inline_dep_values():
    plan = _diff_plan()
    state = {e.id: BlankResult(blank_id=e.id, status="unknown")
             for e in plan.blank_entities()}
    state["b1"].status = "resolved"; state["b1"].value = 68000
    state["b2"].status = "resolved"; state["b2"].value = None   # resolved but None -> excluded
    deps = build_dependency_graph(plan)
    ws = GSWWorkspace(question="Q", plan=plan, state=state, deps=deps,
                      levels=_compute_levels(deps),
                      order=topological_sort_blanks(plan), qgsw_state=None)
    # exactly the inline filter: resolved AND value is not None
    inline = {bid: br.value for bid, br in state.items()
              if br is not None and br.status == "resolved" and br.value is not None}
    assert ws.resolved_values() == inline
    assert ws.resolved_values() == {"b1": 68000}   # b2 excluded (value None)


def test_resolved_values_is_a_snapshot_not_live_view():
    # The dispatch captures dep_values = ws.resolved_values() ONCE before the
    # resolver loop. resolved_values() must return a FRESH dict, so a later
    # record_research_result (a same-batch sibling commit) does NOT retroactively
    # appear in the already-captured snapshot. This is what keeps same-batch
    # siblings invisible to later same-batch resolver attempts.
    ws = _make_workspace(_diff_plan(), resolved={"b1": 68000})
    snap = ws.resolved_values()                       # captured "before the loop"
    ws.record_research_result(BlankResearchResult(
        blank_id="b2", status="resolved", value=650000, resolved_by="resolver"))
    assert snap == {"b1": 68000}                      # frozen — b2 not added retroactively
    assert ws.resolved_values() == {"b1": 68000, "b2": 650000}  # live view DID move


def test_assess_plan_quality_deterministic_and_summary():
    # Two target blanks → _check_no_dangling requires ≥1 verb-phrase referencing
    # each entity, so connect them with a VerbPhrase to satisfy the validator.
    plan = GSWPlan(
        entities=[
            Entity(id="a", kind="blank", value_type="entity", is_target=True),
            Entity(id="b", kind="blank", value_type="entity", is_target=True),
        ],
        verb_phrases=[VerbPhrase(id="vp1", phrase="rel", subject_id="a", object_id="b")],
        constraints=[],
    )
    deps = build_dependency_graph(plan)
    ws = GSWWorkspace(question="Q", plan=plan, state={}, deps=deps,
                      levels={}, order=[], qgsw_state=None)
    rep = ws.assess_plan_quality_deterministic(use_hints=False)
    assert rep.has_high_severity_issue
    s = ws.plan_quality_summary(rep.issues)
    assert s["has_high_severity_issue"] is True and s["high_count"] >= 1
    assert s["high_count"] + s["medium_count"] + s["low_count"] == len(rep.issues)


def _bridge_ws(resolved):
    plan = GSWPlan(
        entities=[
            Entity(id="b_country", kind="blank", value_type="entity",
                   retrieval_hint="highest in 2023 Democracy Index"),
            Entity(id="t", kind="blank", value_type="entity", is_target=True,
                   retrieval_hint="largest religion in {b_country}"),
        ],
        verb_phrases=[VerbPhrase(id="vp1", phrase="biggest_religion",
                                 subject_id="b_country", object_id="t")],
        constraints=[])
    state = {bid: BlankResult(blank_id=bid, status="resolved", value=val, resolved_by="resolver")
             for bid, val in resolved.items()}
    deps = build_dependency_graph(plan)
    return plan, GSWWorkspace(question="Q", plan=plan, state=state, deps=deps,
                              levels={}, order=[], qgsw_state=None)


def _bridge_changed(hint):
    return GSWPlan(
        entities=[
            Entity(id="b_country", kind="blank", value_type="entity", retrieval_hint=hint),
            Entity(id="t", kind="blank", value_type="entity", is_target=True,
                   retrieval_hint="largest religion in {b_country}"),
        ],
        verb_phrases=[VerbPhrase(id="vp1", phrase="biggest_religion",
                                 subject_id="b_country", object_id="t")],
        constraints=[])


def test_invalidate_enforce_resets_changed_and_downstream():
    old, ws = _bridge_ws({"b_country": "Norway", "t": "Church of Norway"})
    new = _bridge_changed("highest in 2024 Democracy Index")   # b_country meaning changed
    diag = ws.invalidate_changed_blanks(old_plan=old, new_plan=new,
                                        preserved_ids=["b_country", "t"], mode="enforce")
    assert diag["changed_identity"] == ["b_country"]
    assert diag["downstream_stale"] == ["t"]
    assert diag["invalidated"] == ["b_country", "t"]
    assert ws.state["b_country"].status == "unknown" and ws.state["t"].status == "unknown"


def test_invalidate_log_only_records_but_resets_nothing():
    old, ws = _bridge_ws({"b_country": "Norway", "t": "Church of Norway"})
    new = _bridge_changed("highest in 2024 Democracy Index")
    diag = ws.invalidate_changed_blanks(old_plan=old, new_plan=new,
                                        preserved_ids=["b_country", "t"], mode="log_only")
    assert diag["changed_identity"] == ["b_country"] and diag["downstream_stale"] == ["t"]
    assert diag["invalidated"] == []
    assert ws.state["b_country"].status == "resolved" and ws.state["t"].status == "resolved"


def test_invalidate_unknown_mode_is_safe_noop():
    old, ws = _bridge_ws({"b_country": "Norway", "t": "Church of Norway"})
    new = _bridge_changed("changed")
    diag = ws.invalidate_changed_blanks(old_plan=old, new_plan=new,
                                        preserved_ids=["b_country", "t"], mode="bogus")
    assert diag["invalidated"] == [] and ws.state["b_country"].status == "resolved"


def test_plan_issue_summary_det_first_auditor_high_only():
    ws = _make_workspace(_diff_plan(), resolved={})
    det = [
        PlanIssue(severity="high", issue_type="missing_scope", blank_id="t",
                  evidence="x"*400, suggested_fix="y"*400),
        PlanIssue(severity="low", issue_type="bad_hint", blank_id="b1",
                  evidence="e", suggested_fix="f"),
    ]
    aud = [
        PlanIssue(severity="high", issue_type="bad_dependency", blank_id="b2",
                  evidence="e", suggested_fix="f"),
        PlanIssue(severity="medium", issue_type="missing_qualifier", blank_id=None,
                  evidence="e", suggested_fix="f"),   # auditor-medium -> SUPPRESSED
    ]
    out = ws.plan_issue_summary(deterministic=det, auditor=aud, max_issues=3)
    ids = [o["issue_id"] for o in out]
    # rank: det-high(0) < aud-high(2) < det-low(3); auditor-medium dropped
    assert ids == ["pq_det_0", "pq_aud_0", "pq_det_1"]
    assert out[0]["severity"] == "high" and out[1]["severity"] == "high"
    assert out[2]["severity"] == "low"
    assert all(o["issue_type"] != "missing_qualifier" for o in out)   # aud-medium gone
    # truncation to 240
    assert len(out[0]["evidence"]) <= 240 and len(out[0]["suggested_fix"]) <= 240
    assert set(out[0]) == {"issue_id","severity","issue_type","blank_id","evidence","suggested_fix"}


def test_plan_issue_summary_auditor_medlow_suppressed_even_with_empty_slots():
    ws = _make_workspace(_diff_plan(), resolved={})
    aud = [
        PlanIssue(severity="high",   issue_type="bad_dependency", blank_id="b2", evidence="e", suggested_fix="f"),
        PlanIssue(severity="medium", issue_type="missing_qualifier", blank_id=None, evidence="e", suggested_fix="f"),
        PlanIssue(severity="low",    issue_type="other", blank_id=None, evidence="e", suggested_fix="f"),
    ]
    out = ws.plan_issue_summary(deterministic=[], auditor=aud, max_issues=3)
    # only the auditor HIGH surfaces; med/low suppressed even though slots are free
    assert [o["issue_id"] for o in out] == ["pq_aud_0"]


def test_plan_issue_summary_empty():
    ws = _make_workspace(_diff_plan(), resolved={})
    assert ws.plan_issue_summary(deterministic=[], auditor=[]) == []
