"""GSWWorkspace — the orchestrator-facing memory object.

W1 scope: bundle the already-built per-question objects behind one object and
expose READ-ONLY queries the orchestrator can ask instead of reconstructing
context by hand. No field is mutated here; the old `state` dict remains the
owner (spec §5, one-write-path invariant). Result/issue recording (W3) and plan
quality (W5) and plan-update reconciliation (W6) land in later phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from ._planner_exec import BlankResult, GSWPlan, unresolved_upstream_deps

# Closed set of plan-issue categories (spec §4). Keep in sync with the spec.
IssueType = Literal[
    "missing_scope",
    "missing_qualifier",
    "bad_hint",
    "bad_dependency",
    "underspecified_blank",
    "wrong_resolution_unit",
    "repeated_structural_failure",
    "placeholder_error",
    "target_unreachable",
    "cycle",
    "other",
]


@dataclass
class PlanIssue:
    """A plan-level problem surfaced to the orchestrator (advisory)."""

    severity: Literal["low", "medium", "high"]
    issue_type: IssueType
    blank_id: Optional[str]
    evidence: str
    suggested_fix: str


@dataclass
class PlanQualityReport:
    """Result of GSWWorkspace.assess_plan_quality() (populated from W5)."""

    issues: list[PlanIssue] = field(default_factory=list)

    @property
    def has_high_severity_issue(self) -> bool:
        return any(i.severity == "high" for i in self.issues)


@dataclass
class BlankAttempt:
    """One resolution attempt record; the unit of attempt_history."""

    blank_id: str
    attempt_id: int
    resolved_by: Optional[str] = None
    query: Optional[str] = None
    candidate: Optional[Any] = None
    status: str = "unknown"
    verifier_verdict: Optional[str] = None
    evidence_chunk_ids: list[str] = field(default_factory=list)
    local_feedback: Optional[dict[str, Any]] = None
    plan_issue: Optional[PlanIssue] = None


@dataclass
class BlankContext:
    """Structured input to research_blank (W4a). Built by get_blank_context."""

    blank_id: str
    value_type: str
    retrieval_hint: Optional[str] = None
    verification_claim: Optional[str] = None
    upstream_values: dict[str, Any] = field(default_factory=dict)
    related_edges: list[str] = field(default_factory=list)
    previous_attempts: list[BlankAttempt] = field(default_factory=list)


@dataclass
class BlankResearchResult:
    """Structured output of research_blank (W4a). Recorded by W3."""

    blank_id: str
    status: Literal["resolved", "unresolved"]
    value: Any = None
    candidate: Any = None
    resolved_by: Optional[str] = None
    evidence_chunk_ids: list[str] = field(default_factory=list)
    connection_evidence: dict[str, list[str]] = field(default_factory=dict)
    verifier_verdict: Optional[str] = None
    confidence: float = 0.0
    completeness_status: Optional[str] = None  # "partial" when known-incomplete (list/set blank)
    local_feedback: Optional[dict[str, Any]] = None
    plan_issue: Optional[PlanIssue] = None


class GSWWorkspace:
    """Read-only (W1) bundle of the per-question run state.

    Constructed from objects the orchestrator already builds in run_question
    (plan, deps, levels, topo order, the BlankResult state dict, optional
    QueryGSWState). At W1 the workspace MIRRORS the state dict and mutates
    nothing; later phases add the write paths (record_research_result in W3,
    assess_plan_quality in W5, apply_plan_update in W6).
    """

    def __init__(
        self,
        *,
        question: str,
        plan: GSWPlan,
        state: dict[str, Any],          # dict[str, BlankResult]; owned elsewhere at W1
        deps: dict[str, set[str]],
        levels: dict[str, int],
        order: list[str],
        qgsw_state: Any = None,         # Optional[QueryGSWState]
    ) -> None:
        self.question = question
        self.plan = plan
        self.state = state
        self.deps = deps
        self.levels = levels
        self.order = order
        self.qgsw_state = qgsw_state
        self.plan_issues: list[PlanIssue] = []
        self.attempt_history: dict[str, list[BlankAttempt]] = {}

    def resolved_ids(self) -> set[str]:
        return {bid for bid, br in self.state.items() if br.status == "resolved"}

    def resolved_values(self) -> dict[str, Any]:
        """The resolver pre-pass dep_values: every resolved blank's value.

        Reproduces the orchestrator's inline dep_values snapshot exactly
        (resolved AND value is not None). The orchestrator calls this ONCE
        before the resolver loop, preserving the fixed-snapshot semantics.
        """
        return {
            bid: br.value
            for bid, br in self.state.items()
            if br is not None and br.status == "resolved" and br.value is not None
        }

    def get_ready_blanks(self) -> list[str]:
        """Blank ids that are unresolved AND have every upstream dep resolved.

        Uses the same dependency relation the deterministic level-walk uses
        (unresolved_upstream_deps over build_dependency_graph), so the ready
        set matches existing topology.
        """
        resolved = self.resolved_ids()
        all_blanks = [e.id for e in self.plan.blank_entities()]
        pending = [b for b in all_blanks if b not in resolved]
        blocked = unresolved_upstream_deps(self.plan, resolved, pending)
        return [b for b in pending if not blocked.get(b)]

    def get_blank_context(self, blank_id: str) -> BlankContext:
        """Assemble the read-only context the researcher needs for one blank.

        Pulls hints straight off the plan Entity and resolved upstream values
        out of the state dict — the same information run_question assembles by
        hand today before calling the resolver/researcher.
        """
        ent = self.plan.entity_by_id(blank_id)
        resolved = self.resolved_ids()
        upstream = {
            dep: self.state[dep].value
            for dep in self.deps.get(blank_id, set())
            if dep in resolved
        }
        related = [f"{dep} -> {blank_id}" for dep in sorted(self.deps.get(blank_id, set()))]
        return BlankContext(
            blank_id=blank_id,
            value_type=getattr(ent, "value_type", "") or "",
            retrieval_hint=getattr(ent, "retrieval_hint", None),
            verification_claim=getattr(ent, "verification_claim", None),
            upstream_values=upstream,
            related_edges=related,
            previous_attempts=list(self.attempt_history.get(blank_id, [])),
        )

    def state_view(self, constraint_outputs: set[str]) -> dict[str, Any]:
        """The orchestrator's get_state tool view, sourced from the workspace.

        Reproduces gsw_planner_orchestrator_v1's inline get_state dict exactly
        (same keys, same order, same blank order) so routing the tool through
        the workspace returns an identical dict. Read-only.
        """
        target_id = self.plan.target().id
        view: dict[str, Any] = {"blanks": []}
        for bid, br in self.state.items():
            view["blanks"].append(
                {
                    "blank_id": bid,
                    "status": br.status,
                    "value": br.value,
                    "is_target": bid == target_id,
                    "is_constraint_output": bid in constraint_outputs,
                }
            )
        return view

    def record_research_result(self, result: "BlankResearchResult") -> None:
        """Commit a RESOLVED blank result into the workspace state.

        This is the typed write path (spec §6 W3): it reproduces the inline
        resolver-commit's BlankResult exactly (the unset fields — llm_calls,
        wall_time_s, verifier_status, error — fall through to the BlankResult
        defaults, which match the inline construction). `self.state` is the
        same dict the orchestrator owns, so this is the same memory write.

        Only resolved results are recorded here (the orchestrator routes
        unresolved results to its handoff/ReAct path, which does not write
        state). Calling with a non-resolved result is a programmer error.
        """
        if result.status != "resolved":
            raise ValueError(
                f"record_research_result expects a resolved result, "
                f"got status={result.status!r} for {result.blank_id!r}"
            )
        self.state[result.blank_id] = BlankResult(
            blank_id=result.blank_id,
            value=result.value,
            status="resolved",
            evidence_chunk_ids=list(result.evidence_chunk_ids),
            connection_evidence=dict(getattr(result, "connection_evidence", {}) or {}),
            resolved_by=result.resolved_by,
            completeness_status=result.completeness_status,
        )

    def summary(self) -> dict[str, Any]:
        """A compact, JSON-able snapshot for get_state / prompt rendering."""
        return {
            "question": self.question,
            "ready_blanks": self.get_ready_blanks(),
            "resolved": sorted(self.resolved_ids()),
            "blanks": {
                bid: {
                    "status": br.status,
                    "value": br.value,
                    "resolved_by": br.resolved_by,
                }
                for bid, br in self.state.items()
            },
            # dict(vars(i)) COPIES each PlanIssue.__dict__ — a bare vars(i)
            # would alias the live dataclass, letting a caller mutate the
            # issue through the summary once W5 populates plan_issues.
            "plan_issues": [dict(vars(i)) for i in self.plan_issues],
        }

    def invalidate_changed_blanks(
        self, *, old_plan: "GSWPlan", new_plan: "GSWPlan",
        preserved_ids: list[str], mode: str,
    ) -> dict[str, Any]:
        """W6 safety net. Among PRESERVED + currently-resolved blanks, find those
        whose identity changed across a plan-update, plus the resolved blanks that
        transitively depend on them. In mode='enforce' reset all of them to
        'unknown' (self.state is the shared dict, so this is the same write the
        orchestrator would do); mode='log_only' records but mutates nothing; any
        other mode is a safe no-op. Returns a trace-only diag.
        """
        from ._plan_update_guard import changed_identity_blanks, downstream_resolved_blanks
        resolved = self.resolved_ids()
        changed = changed_identity_blanks(
            old_plan, new_plan, preserved_ids=preserved_ids, resolved_ids=resolved)
        downstream = sorted(downstream_resolved_blanks(
            new_plan, seeds=changed, resolved_ids=resolved))
        invalidated: list[str] = []
        if mode == "enforce":
            for bid in [*changed, *downstream]:
                self.state[bid] = BlankResult(blank_id=bid, status="unknown")
                invalidated.append(bid)
        return {
            "mode": mode,
            "changed_identity": changed,
            "downstream_stale": downstream,
            "invalidated": sorted(set(invalidated)),
        }

    def assess_plan_quality_deterministic(self, *, use_hints: bool) -> PlanQualityReport:
        """Deterministic plan-quality report (log_only; see _plan_quality)."""
        from ._plan_quality import assess_plan_quality_deterministic as _assess
        return _assess(self.plan, question=self.question, use_hints=use_hints)

    @staticmethod
    def plan_quality_summary(issues: list[PlanIssue]) -> dict[str, Any]:
        """Aggregate severity counts over a combined issue list (both rungs)."""
        sev = [i.severity for i in issues]
        return {
            "has_high_severity_issue": "high" in sev,
            "high_count": sev.count("high"),
            "medium_count": sev.count("medium"),
            "low_count": sev.count("low"),
        }

    # W6a v1 surfacing rank: deterministic-first, auditor-conservative. Lower
    # number = surfaced first. Auditor medium/low are NOT in the table => they
    # are SUPPRESSED (the auditor is calibration-noisy).
    _SURFACE_RANK = {
        ("det", "high"): 0, ("det", "medium"): 1,
        ("aud", "high"): 2, ("det", "low"): 3,
    }

    @staticmethod
    def plan_issue_summary(
        *, deterministic: list[PlanIssue], auditor: list[PlanIssue],
        max_issues: int = 3, max_chars: int = 240,
    ) -> list[dict[str, Any]]:
        """Compact, prompt-safe surface of plan-quality issues (W6a v1).

        Deterministic-first; auditor only surfaces its HIGH issues (medium/low
        suppressed). Caps to max_issues, truncates text. Each item carries a
        stable issue_id (pq_det_<i>/pq_aud_<i>) the orchestrator can name in a
        request_plan_update reason.
        """
        ranked: list[tuple[int, str, int, "PlanIssue"]] = []
        for src, items in (("det", deterministic), ("aud", auditor)):
            for i, x in enumerate(items):
                key = GSWWorkspace._SURFACE_RANK.get((src, x.severity))
                if key is None:           # auditor medium/low (or unknown) => suppressed
                    continue
                ranked.append((key, src, i, x))
        ranked.sort(key=lambda t: (t[0], t[1], t[2]))   # rank, src, original index (stable)
        out: list[dict[str, Any]] = []
        for _key, src, i, x in ranked[:max_issues]:
            out.append({
                "issue_id": f"pq_{src}_{i}",
                "severity": x.severity,
                "issue_type": x.issue_type,
                "blank_id": x.blank_id,
                "evidence": (x.evidence or "")[:max_chars],
                "suggested_fix": (x.suggested_fix or "")[:max_chars],
            })
        return out
