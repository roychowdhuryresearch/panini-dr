"""GSWWorkspace paths, plan-quality checks, plan-update guard (W6) and the advisory plan-issue surface.

Split from the research repo's test_planner_orchestrator.py."""

from __future__ import annotations

from tests.ours.orchestrator._helpers import *  # noqa: F401,F403


# ---------------------------------------------------------------------------
# Regression: verifier_reject_counts reset per question (M3 leak fix)
# ---------------------------------------------------------------------------


def test_verifier_reject_counts_reset_between_questions():
    """_checker_sink_tls.verifier_reject_counts must be wiped at the start of
    each run_question call so that stale reject tallies from a prior question
    cannot spuriously trigger M3's over-decomposition branch on the next one.

    Strategy: seed stale counts on _checker_sink_tls directly, run a minimal
    two-turn question that resolves cleanly, then assert the post-run counts
    contain only entries from the current question (i.e. none of the stale
    entries remain).
    """
    plan_dict = _one_level_plan()  # target blank "t"
    script = [
        _StubResponse(
            tool_calls=[_tc("c1", "update_blank", {
                "blank_id": "t",
                "value": "answer",
                "evidence_chunk_ids": ["chunk_pf_0"],
            })]
        ),
    ]
    llm = _orchestrated_llm(script)
    adapter = _mk_adapter(llm, plan_dict=plan_dict)

    # Seed stale reject counts as if a prior question had 3 verifier rejects.
    adapter._checker_sink_tls.verifier_reject_counts = {"stale_blank": 3}

    traj = adapter.run_question("What album?", question_id="q_reset_test")

    # Stale entry must be gone — only the current question's counts survive.
    counts = getattr(adapter._checker_sink_tls, "verifier_reject_counts", {})
    assert "stale_blank" not in counts, (
        "verifier_reject_counts was NOT reset per question: "
        f"stale key still present with value {counts.get('stale_blank')}"
    )
    # Sanity: the question itself resolved correctly.
    assert traj.final_answer == "answer"


def test_orchestrator_reads_blank_hints_flag():
    from panini_dr.adapters.ours.gsw_planner_orchestrator_v1 import (
        OursGSWPlannerOrchestratorV1Adapter,
    )
    from panini_dr.adapters.base import AdapterContext

    ctx = AdapterContext(
        system_id="ours_gsw_planner_orchestrator_v1",
        model_id="stub",
        max_turns=10,
        max_completion_tokens=256,
        # A corpus is mandatory now (no silent fallback to a default one).
        extra={"gsw_blank_hints": True, "corpus": _StubCorpus({}),
               "retriever": _StubRetriever({})},
    )
    assert OursGSWPlannerOrchestratorV1Adapter(ctx).gsw_blank_hints is True


def test_get_state_workspace_path_matches_inline_path():
    # W2: routing get_state through GSWWorkspace.state_view must return the
    # exact same Python dict as the inline build (output-identical). Mirror the
    # adapter+plan construction used by
    # test_llm_dispatch_allows_dependent_batch_when_guard_disabled.
    from panini_dr.adapters.ours._planner_exec import (
        BlankResult, build_dependency_graph, topological_sort_blanks,
    )
    from panini_dr.adapters.ours.gsw_planner_react_v1 import _compute_levels
    from panini_dr.adapters.ours._gsw_workspace import GSWWorkspace

    plan = GSWPlan.model_validate(_two_level_plan())
    state = {
        e.id: BlankResult(blank_id=e.id, status="unknown")
        for e in plan.blank_entities()
    }
    adapter = _mk_llm_adapter(_RoutedLLM({}), plan_dict=plan.model_dump())

    # resolve one upstream blank so value/status are exercised
    first_blank = plan.blank_entities()[0].id
    state[first_blank].status = "resolved"
    state[first_blank].value = "X"

    deps = build_dependency_graph(plan)
    levels = _compute_levels(deps)
    order = topological_sort_blanks(plan)
    constraint_outputs = {c.output_blank_id for c in plan.constraints if c.output_blank_id}

    kwargs = dict(
        name="get_state", args={}, plan=plan, plan_dict=plan.model_dump(),
        state=state, order=order, levels=levels, question="Q",
        dispatch_idx=0, constraint_outputs=constraint_outputs,
    )
    inline_view, _ = adapter._run_orchestrator_tool(**kwargs, workspace=None)

    ws = GSWWorkspace(question="Q", plan=plan, state=state, deps=deps,
                      levels=levels, order=order, qgsw_state=None)
    ws_view, _ = adapter._run_orchestrator_tool(**kwargs, workspace=ws)

    assert ws_view == inline_view
    assert inline_view["blanks"]  # non-empty sanity
    # The LLM reads the blanks as an ordered list, so order must match exactly
    # (state.items() insertion order), not merely agree by blank_id.
    assert [b["blank_id"] for b in ws_view["blanks"]] == [
        b["blank_id"] for b in inline_view["blanks"]
    ]


def test_research_blank_all_four_branches():
    from panini_dr.adapters.ours._gsw_workspace import BlankResearchResult
    from panini_dr.adapters.ours._gsw_workspace import GSWWorkspace
    from panini_dr.adapters.ours._planner_exec import (
        BlankResult, build_dependency_graph, topological_sort_blanks,
    )
    from panini_dr.adapters.ours.gsw_planner_react_v1 import _compute_levels

    plan = GSWPlan.model_validate(_two_level_plan())
    adapter = _mk_llm_adapter(_RoutedLLM({}), plan_dict=plan.model_dump())
    target = plan.target().id

    state = {e.id: BlankResult(blank_id=e.id, status="unknown")
             for e in plan.blank_entities()}
    deps = build_dependency_graph(plan)
    ws = GSWWorkspace(question="Q", plan=plan, state=state, deps=deps,
                      levels=_compute_levels(deps),
                      order=topological_sort_blanks(plan), qgsw_state=None)
    ctx = ws.get_blank_context(target)

    class _Res:  # stub TypedBlankResolver result
        def __init__(self, status, value=None, chunks=None, message=""):
            self.status = status
            self.value = value
            self.evidence_chunk_ids = chunks or []
            self.retrieved_chunks = []
            self.message = message

    class _Resolver:
        def __init__(self, res): self._res = res
        def resolve_blank(self, plan, bid, dep_values, question, feedback=None):
            return self._res

    class _Verdict:
        def __init__(self, accepted, partial=False, reason=""):
            self.accepted = accepted; self.partial = partial; self.reason = reason
            self.satisfies_all = accepted; self.unique = True; self.complete = True
            self.ambiguous_candidates = []; self.deferred_constraints = []

    class _Verifier:
        def __init__(self, verdict): self._v = verdict
        def verify(self, plan, bid, value, chunks, dep_values): return self._v

    def _call(resolver, verifier):
        return adapter._research_blank(
            workspace=ws, context=ctx, dep_values={},
            question="Q", resolver=resolver, verifier=verifier)

    # 1) resolved, no verifier -> resolved result
    r = _call(_Resolver(_Res("resolved", value="X", chunks=["c1"])), None)
    assert r.status == "resolved" and r.value == "X"
    assert r.resolved_by == "resolver" and r.evidence_chunk_ids == ["c1"]

    # 2) resolved + verifier accepted + partial -> resolved, completeness=partial
    r = _call(_Resolver(_Res("resolved", value="X", chunks=["c1"])),
              _Verifier(_Verdict(accepted=True, partial=True)))
    assert r.status == "resolved" and r.completeness_status == "partial"

    # 3) resolved + verifier rejected -> unresolved (no commit), handoff
    n0 = len(adapter._resolver_attempts)
    r = _call(_Resolver(_Res("resolved", value="BAD", chunks=["c1"])),
              _Verifier(_Verdict(accepted=False, reason="wrong relation")))
    assert r.status == "unresolved" and r.candidate == "BAD"
    assert r.local_feedback["candidate"] == "BAD"
    assert r.local_feedback["reason"] == "wrong relation"
    assert r.local_feedback["chunk_ids"] == ["c1"]
    last = adapter._resolver_attempts[-1]
    assert len(adapter._resolver_attempts) == n0 + 1
    assert last["blank_id"] == target and last["candidate"] == "BAD"
    assert "verifier" in last and "observed_only" not in last

    # 4) unresolved -> unresolved + handoff payload in local_feedback
    r = _call(_Resolver(_Res("unresolved", message="not found")), None)
    assert r.status == "unresolved" and r.candidate is None
    assert r.local_feedback["reason"] == "not found"
    assert "query" in r.local_feedback and "chunk_ids" in r.local_feedback


# ---------------------------------------------------------------------------
# W5 Task 3: behavior-neutrality test (flag OFF vs ON)
# ---------------------------------------------------------------------------


def test_plan_quality_checks_flag_behavior_neutral():
    """gsw_plan_quality_checks default-off is byte-identical to flag-on
    for final_answer and committed blank values.  Flag OFF -> no
    'gsw_plan_quality' key; Flag ON -> key present with expected shape."""
    plan_dict = _one_level_plan()

    def _make_script():
        return [
            _StubResponse(
                tool_calls=[_tc("c1", "update_blank", {
                    "blank_id": "t",
                    "value": "Dark Side of the Moon",
                    "evidence_chunk_ids": ["chunk_pf_0"],
                })]
            ),
        ]

    # Run with flag OFF (default).
    llm_off = _orchestrated_llm(_make_script())
    adapter_off = _mk_adapter(llm_off, plan_dict=plan_dict)
    traj_off = adapter_off.run_question(
        "What album did Pink Floyd release?", question_id="q_pq_off"
    )

    # Run with flag ON.
    llm_on = _orchestrated_llm(_make_script())
    adapter_on = _mk_adapter(llm_on, plan_dict=plan_dict,
                              extra={"gsw_plan_quality_checks": True})
    traj_on = adapter_on.run_question(
        "What album did Pink Floyd release?", question_id="q_pq_on"
    )

    # Flag OFF: no gsw_plan_quality key written.
    assert "gsw_plan_quality" not in traj_off.extra

    # Flag ON: key present with required shape.
    pq = traj_on.extra["gsw_plan_quality"]
    assert "deterministic" in pq
    assert "summary" in pq
    assert "stage" in pq
    assert pq["stage"] == "initial_plan"
    assert pq["llm_auditor"] is None

    # Behavior neutral: final answer and committed blank values are identical.
    assert traj_off.final_answer == traj_on.final_answer == "Dark Side of the Moon"
    assert traj_off.extra["stopped_reason"] == traj_on.extra["stopped_reason"]


def test_plan_quality_mode_matrix_off_deterministic_auditor():
    """W5 behavior contract — same plan cell, two modes.

    Mode 1 (checks OFF): no 'gsw_plan_quality' key written.
    Mode 2 (checks=True): key present; deterministic filled; llm_auditor is
        always None (the LLM auditor was removed; a verifier will populate
        this next); summary has all 4 count keys.
    Both modes: final_answer and stopped_reason are IDENTICAL (log_only).
    """
    import json as _json

    plan_dict = _one_level_plan()

    _CANNED_AUDITOR_ANSWER = _json.dumps({
        "issues": [{
            "severity": "low",
            "issue_type": "other",
            "blank_id": None,
            "evidence": "plan looks good enough",
            "suggested_fix": "none needed",
        }]
    })

    class _AuditorRoutedLLM:
        """Stub LLM that routes auditor calls by system-prompt content,
        then falls back to a sequential script for orchestrator/researcher calls.

        Routing rule: if the system message contains 'plan-quality auditor',
        return a canned auditor JSON response.  All other calls are served from
        the sequential script in arrival order.
        """

        def __init__(self, script: list[_StubResponse], auditor_json: str) -> None:
            self._script = list(script)
            self._auditor_json = auditor_json
            self._orchestrator = _dispatch_then_submit(["t"])
            self.calls: list[dict] = []

        def chat(self, messages, *, tools=None, max_tokens=None, **kwargs):
            opening = _planning_phase_opening(messages, tools)
            if opening is not None:
                return opening
            self.calls.append({"messages": list(messages), "tools": tools, **kwargs})
            # Detect auditor call: system message contains the auditor marker.
            sys_content = ""
            for m in messages:
                if m.get("role") == "system":
                    sys_content = m.get("content", "") or ""
                    break
            if "plan-quality auditor" in sys_content:
                return _StubResponse(text=self._auditor_json, tool_calls=[])
            if "You are the **Orchestrator**" in sys_content:
                return self._orchestrator.pop(0)
            # Regular orchestrator/researcher call — serve from script.
            if not self._script:
                return _StubResponse(text="(script exhausted)", tool_calls=[])
            return self._script.pop(0)

    def _researcher_script():
        return [
            _StubResponse(
                tool_calls=[_tc("c1", "update_blank", {
                    "blank_id": "t",
                    "value": "Dark Side of the Moon",
                    "evidence_chunk_ids": ["chunk_pf_0"],
                })]
            ),
        ]

    QUESTION = "What album did Pink Floyd release?"

    # --- Mode 1: both flags OFF -----------------------------------------------
    llm_off = _AuditorRoutedLLM(_researcher_script(), _CANNED_AUDITOR_ANSWER)
    adapter_off = _mk_adapter(llm_off, plan_dict=plan_dict)
    traj_off = adapter_off.run_question(QUESTION, question_id="q_wm_off")

    # --- Mode 2: checks=True ---------------------------------------------------
    llm_det = _AuditorRoutedLLM(_researcher_script(), _CANNED_AUDITOR_ANSWER)
    adapter_det = _mk_adapter(llm_det, plan_dict=plan_dict,
                               extra={"gsw_plan_quality_checks": True})
    traj_det = adapter_det.run_question(QUESTION, question_id="q_wm_det")

    # ---- Mode 1 assertions ---------------------------------------------------
    assert "gsw_plan_quality" not in traj_off.extra, (
        "flag-off run must not write gsw_plan_quality"
    )

    # ---- Mode 2 assertions ---------------------------------------------------
    pq_det = traj_det.extra["gsw_plan_quality"]
    assert "deterministic" in pq_det
    assert pq_det["stage"] == "initial_plan"
    assert pq_det["llm_auditor"] is None, "llm_auditor is always None (auditor removed)"
    summary_det = pq_det["summary"]
    for key in ("has_high_severity_issue", "high_count", "medium_count", "low_count"):
        assert key in summary_det, f"summary missing key {key!r} in mode 2"

    # ---- Behavior-neutral: answer identical in both modes ---------------
    assert traj_off.final_answer == traj_det.final_answer, (
        "final_answer must be identical across off/deterministic modes (log_only)"
    )
    assert (traj_off.extra["stopped_reason"]
            == traj_det.extra["stopped_reason"]), (
        "stopped_reason must be identical across both modes"
    )


def test_w6_guard_flag_defaults_to_off():
    """Default adapter (no extra override) has gsw_plan_update_guard='off'."""
    adapter = _mk_llm_adapter(_RoutedLLM({}), plan_dict=_one_level_plan())
    assert adapter.gsw_plan_update_guard == "off"


def test_w6_guard_flag_reads_mode(monkeypatch):
    """Adapter reads gsw_plan_update_guard from extra."""
    for mode in ("off", "log_only", "enforce"):
        adapter = _mk_llm_adapter(
            _RoutedLLM({}), plan_dict=_one_level_plan(),
            extra={"gsw_plan_update_guard": mode},
        )
        assert adapter.gsw_plan_update_guard == mode


def test_w6_guard_invalid_mode_coerces_to_off():
    """An unknown mode string is coerced to 'off'."""
    adapter = _mk_llm_adapter(
        _RoutedLLM({}), plan_dict=_one_level_plan(),
        extra={"gsw_plan_update_guard": "bogus"},
    )
    assert adapter.gsw_plan_update_guard == "off"


def test_w6_result_dict_identical_off_vs_log_only(monkeypatch):
    """TRACE-ONLY guarantee: the result dict returned to the orchestrator LLM
    is byte-identical between 'off' and 'log_only'. In both, the resolved
    blank keeps its value unchanged."""
    from panini_dr.adapters.ours._plan_updater import PlanReconcileDiff
    from panini_dr.adapters.ours._planner_emit import PlanEmitMeta

    old_dict, new_dict = _make_w6_test_plans()
    new_plan = GSWPlan.model_validate(new_dict)

    def _fake_update_plan(**kwargs):
        diff = PlanReconcileDiff(
            preserved_ids=["b_country", "t"],
            added_ids=[], dropped_ids=[], invalidated_ids=[],
            summary="hint changed", op_summary="update_hint", attempts=1,
        )
        meta = PlanEmitMeta(prompt_tokens=1, completion_tokens=1, raw_response="{}")
        return new_plan, diff, meta

    # --- mode=off ---
    adapter_off, ws_off, old_plan, _, _, state_off = _make_w6_adapter_and_workspace("off")
    monkeypatch.setattr(
        "panini_dr.adapters.ours.gsw_planner_orchestrator_v1.update_plan",
        _fake_update_plan,
    )
    result_off, action_off = adapter_off._run_orchestrator_tool(
        name="request_plan_update",
        args={"reason": "hint changed", "evidence": "2024 data"},
        plan=old_plan,
        plan_dict=old_plan.model_dump(),
        state=state_off,
        order=["b_country", "t"],
        levels={"b_country": 0, "t": 1},
        question="Q",
        dispatch_idx=0,
        constraint_outputs=set(),
        workspace=ws_off,
    )

    # --- mode=log_only ---
    adapter_log, ws_log, old_plan2, _, _, state_log = _make_w6_adapter_and_workspace("log_only")
    result_log, action_log = adapter_log._run_orchestrator_tool(
        name="request_plan_update",
        args={"reason": "hint changed", "evidence": "2024 data"},
        plan=old_plan2,
        plan_dict=old_plan2.model_dump(),
        state=state_log,
        order=["b_country", "t"],
        levels={"b_country": 0, "t": 1},
        question="Q",
        dispatch_idx=0,
        constraint_outputs=set(),
        workspace=ws_log,
    )

    # Core guarantee: result dicts are identical
    assert result_off == result_log, (
        "result dict (what the orchestrator LLM sees) must be identical for "
        "off vs log_only — the trace-only guarantee"
    )
    # Both resolved blanks keep their values unchanged
    assert state_off["b_country"].status == "resolved"
    assert state_off["b_country"].value == "Norway"
    assert state_log["b_country"].status == "resolved"
    assert state_log["b_country"].value == "Norway"


def test_w6_log_only_records_diag_in_action_not_result(monkeypatch):
    """In log_only: action['plan_update_guard'] records the changed blank;
    invalidated==[] (state unchanged); diag NOT in result dict."""
    from panini_dr.adapters.ours._plan_updater import PlanReconcileDiff
    from panini_dr.adapters.ours._planner_emit import PlanEmitMeta

    old_dict, new_dict = _make_w6_test_plans()
    new_plan = GSWPlan.model_validate(new_dict)

    def _fake_update_plan(**kwargs):
        diff = PlanReconcileDiff(
            preserved_ids=["b_country", "t"],
            added_ids=[], dropped_ids=[], invalidated_ids=[],
            summary="hint changed", op_summary="update_hint", attempts=1,
        )
        meta = PlanEmitMeta(prompt_tokens=1, completion_tokens=1, raw_response="{}")
        return new_plan, diff, meta

    adapter, ws, old_plan, _, _, state = _make_w6_adapter_and_workspace("log_only")
    monkeypatch.setattr(
        "panini_dr.adapters.ours.gsw_planner_orchestrator_v1.update_plan",
        _fake_update_plan,
    )
    result, action = adapter._run_orchestrator_tool(
        name="request_plan_update",
        args={"reason": "hint changed", "evidence": "2024 data"},
        plan=old_plan,
        plan_dict=old_plan.model_dump(),
        state=state,
        order=["b_country", "t"],
        levels={"b_country": 0, "t": 1},
        question="Q",
        dispatch_idx=0,
        constraint_outputs=set(),
        workspace=ws,
    )

    # Diag in action
    assert "plan_update_guard" in action
    guard = action["plan_update_guard"]
    assert "b_country" in guard["changed_identity"]
    assert guard["invalidated"] == []
    # State unchanged
    assert state["b_country"].status == "resolved"
    assert state["t"].status == "resolved"
    # Diag NOT in result dict (trace-only)
    assert "plan_update_guard" not in result
    assert "w6" not in result


def test_w6_enforce_resets_changed_and_downstream(monkeypatch):
    """In enforce: the changed blank (b_country) and its resolved dependent
    (t) are reset to 'unknown'; action['plan_update_guard']['invalidated']
    lists both."""
    from panini_dr.adapters.ours._plan_updater import PlanReconcileDiff
    from panini_dr.adapters.ours._planner_emit import PlanEmitMeta

    old_dict, new_dict = _make_w6_test_plans()
    new_plan = GSWPlan.model_validate(new_dict)

    def _fake_update_plan(**kwargs):
        diff = PlanReconcileDiff(
            preserved_ids=["b_country", "t"],
            added_ids=[], dropped_ids=[], invalidated_ids=[],
            summary="hint changed", op_summary="update_hint", attempts=1,
        )
        meta = PlanEmitMeta(prompt_tokens=1, completion_tokens=1, raw_response="{}")
        return new_plan, diff, meta

    adapter, ws, old_plan, _, _, state = _make_w6_adapter_and_workspace("enforce")
    monkeypatch.setattr(
        "panini_dr.adapters.ours.gsw_planner_orchestrator_v1.update_plan",
        _fake_update_plan,
    )
    result, action = adapter._run_orchestrator_tool(
        name="request_plan_update",
        args={"reason": "hint changed", "evidence": "2024 data"},
        plan=old_plan,
        plan_dict=old_plan.model_dump(),
        state=state,
        order=["b_country", "t"],
        levels={"b_country": 0, "t": 1},
        question="Q",
        dispatch_idx=0,
        constraint_outputs=set(),
        workspace=ws,
    )

    # Diag in action
    assert "plan_update_guard" in action
    guard = action["plan_update_guard"]
    assert "b_country" in guard["changed_identity"]
    assert "b_country" in guard["invalidated"]
    assert "t" in guard["invalidated"]
    # State mutated: both reset to unknown
    assert state["b_country"].status == "unknown"
    assert state["t"].status == "unknown"
    # Diag NOT in result dict (trace-only)
    assert "plan_update_guard" not in result


def test_plan_issue_surface_flag_off_is_byte_identical():
    """gsw_plan_issue_surface default 'off': no plan_issue_surface key in
    traj.extra, and final_answer matches a run with the flag omitted entirely."""
    plan_dict = _plan_with_high_det_issue()

    def _make_script():
        return [
            _StubResponse(tool_calls=[_tc("c1", "update_blank", {
                "blank_id": "t",
                "value": "The Wall",
                "evidence_chunk_ids": ["chunk1"],
            })]),
        ]

    # Run with flag completely absent (default-off).
    llm_base = _orchestrated_llm(_make_script())
    adapter_base = _mk_adapter(llm_base, plan_dict=plan_dict)
    traj_base = adapter_base.run_question(
        "What album did Pink Floyd release?", question_id="q_surface_base"
    )

    # Run with flag explicitly set to "off".
    llm_off = _orchestrated_llm(_make_script())
    adapter_off = _mk_adapter(llm_off, plan_dict=plan_dict,
                               extra={"gsw_plan_issue_surface": "off"})
    traj_off = adapter_off.run_question(
        "What album did Pink Floyd release?", question_id="q_surface_off"
    )

    # Flag-off: no plan_issue_surface key.
    assert "plan_issue_surface" not in traj_base.extra
    assert "plan_issue_surface" not in traj_off.extra

    # Behavior neutral: same final answer and stopped reason.
    assert traj_base.final_answer == traj_off.final_answer
    assert traj_base.extra["stopped_reason"] == traj_off.extra["stopped_reason"]


def test_plan_issue_surface_advisory_wires_warnings_into_prompt():
    """gsw_plan_issue_surface='advisory' with a plan that has a HIGH deterministic
    issue: the orchestrator system prompt contains the advisory section, and
    traj.extra['plan_issue_surface'] is written with count >= 1."""
    plan_dict = _plan_with_high_det_issue()

    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                         {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "submit_answer",
                                         {"answer": "The Wall"})]),
        ],
        "resolve exactly these blanks": [
            _StubResponse(tool_calls=[_tc("r1", "update_blank", {
                "blank_id": "t",
                "value": "The Wall",
                "evidence_chunk_ids": ["chunk1"],
            })]),
        ],
    }
    llm = _CapturingRoutedLLM(routes)
    adapter = _mk_llm_adapter(
        llm, plan_dict=plan_dict,
        extra={"gsw_plan_issue_surface": "advisory"},
    )
    traj = adapter.run_question(
        "What album did Pink Floyd release?", question_id="q_surface_adv"
    )

    # traj.extra must have plan_issue_surface with count >= 1.
    assert "plan_issue_surface" in traj.extra, (
        "advisory mode must write plan_issue_surface into traj.extra"
    )
    surf = traj.extra["plan_issue_surface"]
    assert surf["count"] >= 1, (
        f"expected at least one surfaced issue, got count={surf['count']}"
    )
    assert "count" in surf and "high_count" in surf and "sources" in surf

    # At least one orchestrator system prompt must contain the advisory section.
    orch_prompts = [p for p in llm.system_prompts if "You are the **Orchestrator**" in p]
    assert orch_prompts, "no orchestrator system prompts captured"
    assert any("Plan-quality warnings" in p for p in orch_prompts), (
        "advisory mode: 'Plan-quality warnings' section must appear in orchestrator system prompt"
    )
    assert any("hints, not instructions" in p for p in orch_prompts), (
        "advisory wording 'hints, not instructions' must appear in orchestrator system prompt"
    )
