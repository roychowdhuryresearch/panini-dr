"""Query-level GSW evidence state (Phase 1: state + logging only).

A persistent, per-question evidence workspace layered on top of the
existing planner/orchestrator/executor pipeline. In Phase 1 it is a pure
*observation* layer: it mirrors the blanks the orchestrator already
resolves into a structured ``QueryGSWState`` and serialises it for
inspection. It does NOT change how blanks are resolved or what answer is
returned — with the ``gsw_evidence_state`` flag off the orchestrator is
byte-identical to v8.

Later phases populate ``EvidenceEdge`` objects and gate commits through
this state.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from panini_dr.adapters.ours._planner_exec import (
    BlankResult,
    ExecutionError,
    GSWPlan,
    build_dependency_graph,
)

EvidenceStatus = Literal[
    "candidate", "accepted", "rejected", "ambiguous", "low_confidence",
    "missing_dependency",
]

BlankStatus = Literal[
    "pending", "ready", "resolving", "resolved", "unresolved", "ambiguous", "failed"
]


@dataclass
class EvidenceEdge:
    """A typed, source-backed candidate for one blank's value.

    Schema only in Phase 1 (no edges are constructed yet). Phase 2 builds
    these from the span the researcher actually read.
    """

    edge_id: str
    subject: str
    relation: str
    object: str
    object_type: str
    doc_id: str
    blank_id: str
    chunk_id: Optional[str] = None
    source_title: Optional[str] = None
    source_span: str = ""
    retrieval_score: float = 0.0
    type_score: float = 0.0
    verifier_score: float = 0.0
    confidence: float = 0.0
    status: EvidenceStatus = "candidate"
    rejection_reason: Optional[str] = None
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "object_type": self.object_type,
            "blank_id": self.blank_id,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source_title": self.source_title,
            "source_span": self.source_span,
            "retrieval_score": self.retrieval_score,
            "type_score": self.type_score,
            "verifier_score": self.verifier_score,
            "confidence": self.confidence,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "raw_payload": dict(self.raw_payload),
        }


@dataclass
class BlankState:
    """Resolution state for one plan blank, projected onto subject/relation."""

    blank_id: str
    subject: str
    relation: str
    value_type: str
    deps: list[str] = field(default_factory=list)
    is_target: bool = False
    direction: Literal["forward", "reverse", "none"] = "none"
    # ALL incident verb-phrase edges, not just the primary (subject, relation)
    # summary — so Phase 3 retrieval can build queries from exact edges when a
    # blank participates in several VPs. Each item:
    #   {vp_id, blank_role: "subject"|"object", other_entity_id, other_label,
    #    phrase, direction: "forward"|"reverse"}
    projection_edges: list[dict[str, Any]] = field(default_factory=list)
    status: BlankStatus = "pending"
    value: Optional[Any] = None
    confidence: float = 0.0
    # Provenance: which mechanism resolved this blank — set by the orchestrator
    # at resolution time ("resolver" | "react" | "constraint"); None until then.
    resolved_by: Optional[str] = None
    evidence_chunk_ids: list[str] = field(default_factory=list)
    # Per-connection evidence map: vp_id -> [chunk_id, ...]. Empty unless
    # --gsw-connection-evidence is on. Parallel to evidence_chunk_ids (which
    # stays as the flat union); empty => off = byte-identical.
    connection_evidence: dict[str, list[str]] = field(default_factory=dict)
    accepted_edge_ids: list[str] = field(default_factory=list)
    candidate_edge_ids: list[str] = field(default_factory=list)
    rejected_edge_ids: list[str] = field(default_factory=list)
    attempts: int = 0
    last_error: Optional[str] = None
    # Phase 3-full: lightweight per-attempt trace (query/candidate/verifier verdict)
    # so the inspector + Phase 4 taxonomy can see WHY a blank was rejected even
    # after ReAct overwrites it. Empty by default => no dump change when unused.
    resolver_attempts: list[dict[str, Any]] = field(default_factory=list)
    # Phase 3.6: set to "unverified_capped" when the verifier rejected this
    # blank's value up to the retry cap and it was committed anyway (accept-with-
    # flag policy). None otherwise. Phase 4 coverage treats it as unsupported.
    verifier_status: Optional[str] = None
    # Cascade reliability: "partial" when an accepted value is known-incomplete
    # (list/set blank missing required items). None otherwise. Pure observation.
    completeness_status: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "blank_id": self.blank_id,
            "subject": self.subject,
            "relation": self.relation,
            "value_type": self.value_type,
            "deps": list(self.deps),
            "is_target": self.is_target,
            "direction": self.direction,
            "projection_edges": [dict(e) for e in self.projection_edges],
            "status": self.status,
            "value": self.value,
            "confidence": self.confidence,
            "resolved_by": self.resolved_by,
            "evidence_chunk_ids": list(self.evidence_chunk_ids),
            "connection_evidence": _copy_connection_evidence(self.connection_evidence),
            "accepted_edge_ids": list(self.accepted_edge_ids),
            "candidate_edge_ids": list(self.candidate_edge_ids),
            "rejected_edge_ids": list(self.rejected_edge_ids),
            "attempts": self.attempts,
            "last_error": self.last_error,
            "resolver_attempts": [dict(a) for a in self.resolver_attempts],
            "verifier_status": self.verifier_status,
            "completeness_status": self.completeness_status,
        }


@dataclass
class QueryGSWState:
    """The single per-question evidence workspace."""

    question: str
    blanks: dict[str, BlankState] = field(default_factory=dict)
    evidence_edges: dict[str, EvidenceEdge] = field(default_factory=dict)
    final_answer: Optional[str] = None

    def resolved_values(self) -> dict[str, Any]:
        return {
            bid: b.value
            for bid, b in self.blanks.items()
            if b.status == "resolved" and b.value is not None
        }

    def add_evidence_edge(self, edge: "EvidenceEdge") -> None:
        """Store an edge; link it onto its blank only if it is accepted."""
        self.evidence_edges[edge.edge_id] = edge
        bs = self.blanks.get(edge.blank_id)
        if bs is not None and edge.status == "accepted":
            if edge.edge_id not in bs.accepted_edge_ids:
                bs.accepted_edge_ids.append(edge.edge_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "blanks": {bid: b.to_dict() for bid, b in self.blanks.items()},
            "evidence_edges": {
                eid: e.to_dict() for eid, e in self.evidence_edges.items()
            },
            "final_answer": self.final_answer,
        }


def _copy_connection_evidence(ce: Any) -> dict[str, Any]:
    """Shape-tolerant copy of a ``connection_evidence`` map. Values are
    ``[chunk_id, ...]`` normally; under gsw_candidate_evidence_matrix the map is
    candidate-keyed (values are per-candidate ``{vp_id: [chunk_id]}`` rows), and
    a blanket ``list(v)`` over a dict row would silently copy its KEYS.
    Byte-identical for the flat map."""
    out: dict[str, Any] = {}
    for k, v in (ce or {}).items():
        if isinstance(v, dict):
            out[k] = {vp: list(ids) for vp, ids in v.items()}
        else:
            out[k] = list(v)
    return out


def _entity_label(plan: GSWPlan, entity_id: str) -> str:
    """Human-readable label for an entity: its ``name`` if set, else its id."""
    try:
        ent = plan.entity_by_id(entity_id)
    except ExecutionError:
        return entity_id
    return ent.name or ent.id


def blank_projection_edges(plan: GSWPlan, blank_id: str) -> list[dict[str, Any]]:
    """ALL verb-phrase edges incident to a blank, in plan order.

    Each edge dict preserves the full provenance Phase 3 needs to build a
    precise retrieval query when a blank participates in several VPs:
    ``{vp_id, blank_role, other_entity_id, other_label, phrase, direction}``.
    - blank is the VP **object** → ``direction="forward"`` (value is object).
    - blank is the VP **subject** → ``direction="reverse"`` (value is subject).
    """
    edges: list[dict[str, Any]] = []
    for vp in plan.verb_phrases:
        if vp.object_id == blank_id:
            edges.append({
                "vp_id": vp.id, "blank_role": "object",
                "other_entity_id": vp.subject_id,
                "other_label": _entity_label(plan, vp.subject_id),
                "phrase": vp.phrase, "direction": "forward",
            })
        elif vp.subject_id == blank_id:
            edges.append({
                "vp_id": vp.id, "blank_role": "subject",
                "other_entity_id": vp.object_id,
                "other_label": _entity_label(plan, vp.object_id),
                "phrase": vp.phrase, "direction": "reverse",
            })
    return edges


def blank_projection(plan: GSWPlan, blank_id: str) -> tuple[str, str, str]:
    """Primary ``(subject, relation, direction)`` summary for a blank.

    Picks the FIRST incident verb-phrase edge (deterministic). Returns
    ``("", "", "none")`` when the blank is referenced only by constraints
    (e.g. a derived output). Use :func:`blank_projection_edges` for the full
    multi-VP view.
    """
    edges = blank_projection_edges(plan, blank_id)
    if not edges:
        return "", "", "none"
    e = edges[0]
    return e["other_label"], e["phrase"], e["direction"]


def initialize_query_gsw_state(
    question: str,
    plan: GSWPlan,
    deps: dict[str, set[str]],
) -> QueryGSWState:
    """Build a :class:`QueryGSWState` with one :class:`BlankState` per blank.

    ``subject``/``relation``/``direction`` are projected from the plan's
    verb-phrase edges; ``deps`` come from ``build_dependency_graph``.
    """
    state = QueryGSWState(question=question)
    for ent in plan.blank_entities():
        subject, relation, direction = blank_projection(plan, ent.id)
        state.blanks[ent.id] = BlankState(
            blank_id=ent.id,
            subject=subject,
            relation=relation,
            value_type=ent.value_type or "entity",
            deps=sorted(deps.get(ent.id, set())),
            is_target=ent.is_target,
            direction=direction,
            projection_edges=blank_projection_edges(plan, ent.id),
            status="pending",
        )
    return state


def mirror_results_into_state(
    state: QueryGSWState,
    results: dict[str, BlankResult],
    final_answer: Optional[str],
    plan: Optional[GSWPlan] = None,
) -> None:
    """Phase-1 observation mirror: copy resolved values into ``BlankState``s.

    Pure logging — does NOT create ``EvidenceEdge``s or gate anything. Each
    ``BlankResult``'s committed value / status / ``evidence_chunk_ids`` are
    mirrored onto the matching ``BlankState`` so the serialised
    ``QueryGSWState`` reflects what the orchestrator actually did.

    ``plan`` is the FINAL plan. When given, blanks that a plan revision added
    after :func:`initialize_query_gsw_state` ran are created here exactly as
    that function would have created them, and ``is_target`` is re-synced from
    the plan (a ``change_target`` op moves the target to a new blank). Without
    it, results for blank ids not present in the state are skipped.

    2026-09-04: on every run of record every revision-added blank (50 on
    BCP, 113 on FRAMES, 502 on MoNaCo) was missing from the serialised state;
    its verifier flag was then dropped at the merge and Stage-B grounding saw
    its relation as ``missing_dependency``.
    """
    if plan is not None:
        plan_blank_ids = {ent.id for ent in plan.blank_entities()}
        deps: Optional[dict[str, set[str]]] = None
        for blank_id in results:
            if blank_id in state.blanks or blank_id not in plan_blank_ids:
                continue
            if deps is None:
                deps = build_dependency_graph(plan)
            ent = plan.entity_by_id(blank_id)
            subject, relation, direction = blank_projection(plan, blank_id)
            state.blanks[blank_id] = BlankState(
                blank_id=blank_id,
                subject=subject,
                relation=relation,
                value_type=ent.value_type or "entity",
                deps=sorted(deps.get(blank_id, set())),
                is_target=ent.is_target,
                direction=direction,
                projection_edges=blank_projection_edges(plan, blank_id),
                status="pending",
            )
        for ent in plan.blank_entities():
            bs = state.blanks.get(ent.id)
            if bs is not None:
                bs.is_target = bool(ent.is_target)
    for blank_id, res in results.items():
        bs = state.blanks.get(blank_id)
        if bs is None:
            continue
        bs.value = res.value
        bs.evidence_chunk_ids = list(res.evidence_chunk_ids or [])
        bs.connection_evidence = _copy_connection_evidence(
            getattr(res, "connection_evidence", {}))
        bs.resolved_by = res.resolved_by
        if getattr(res, "completeness_status", None) is not None:
            bs.completeness_status = res.completeness_status
        if res.status == "resolved":
            bs.status = "resolved"
        elif res.status == "error":
            bs.status = "failed"
            bs.last_error = res.error or None
        else:  # "unknown" — orchestrator never committed a value
            bs.status = "unresolved"
    state.final_answer = final_answer


# ---------------------------------------------------------------------------
# Phase 2 — evidence metrics
# ---------------------------------------------------------------------------


def evidence_metrics(state: QueryGSWState) -> dict[str, Any]:
    """Grounding summary. `missing_dependency` edges are NOT checkable yet and
    are excluded from `pct_grounded` (reported separately) — never counted as
    failures."""
    edges = list(state.evidence_edges.values())
    n_committed = len(edges)
    n_grounded = sum(1 for e in edges if e.status == "accepted")
    n_ungrounded = sum(1 for e in edges if e.status == "low_confidence")
    n_missing = sum(1 for e in edges if e.status == "missing_dependency")
    checkable = n_grounded + n_ungrounded
    return {
        "n_committed": n_committed,
        "n_grounded": n_grounded,
        "n_ungrounded": n_ungrounded,
        "n_missing_dependency": n_missing,
        "pct_grounded": round(n_grounded / checkable, 3) if checkable else None,
    }


# ---------------------------------------------------------------------------
# Stage B — graph-level verification (Phase 2.5)
# ---------------------------------------------------------------------------


def read_cited_chunks(corpus: Any, evidence_chunk_ids: list[str]) -> list[dict[str, str]]:
    """Read chunk records for a list of chunk ids via the corpus. Skips empty,
    'insufficient_evidence', and missing ids; returns
    [{"chunk_id":.., "title":.., "text":..}, ...]."""
    out: list[dict[str, str]] = []
    for cid in evidence_chunk_ids:
        if not cid or cid == "insufficient_evidence":
            continue
        try:
            ch = corpus.get_chunk(cid)
        except Exception:  # noqa: BLE001
            ch = None
        if ch is None:
            continue
        out.append({"chunk_id": cid, "title": getattr(ch, "title", "") or "",
                    "text": getattr(ch, "text", "") or ""})
    return out


def has_real_citation(corpus: Any, evidence_chunk_ids: list[str]) -> bool:
    """Stage-A structural admission: at least one cited id is non-placeholder
    AND resolves to a real chunk in the corpus. Cheap, single-endpoint,
    well-formedness only — NOT semantic support."""
    for c in (evidence_chunk_ids or []):
        if c and c != "insufficient_evidence" and corpus.get_chunk(c) is not None:
            return True
    return False


def unknown_chunk_ids(corpus: Any, evidence_chunk_ids: list[str]) -> list[str]:
    """Ids that do NOT resolve to a real corpus chunk (empty strings and the
    'insufficient_evidence' placeholder included). CE-v2: used for hard tool
    errors at the bind/commit boundary so a fake id can never silently read
    as 'no evidence' (the q510 gap-door trigger)."""
    bad: list[str] = []
    for cid in (evidence_chunk_ids or []):
        if not cid or cid == "insufficient_evidence":
            bad.append(cid)
            continue
        try:
            ch = corpus.get_chunk(cid)
        except Exception:  # noqa: BLE001
            ch = None
        if ch is None:
            bad.append(cid)
    return bad


def _endpoint_value(plan: GSWPlan, state: QueryGSWState, entity_id: str) -> Optional[str]:
    """Display value for a verb-phrase endpoint: a filled entity's name, or a
    blank's resolved value, else None (unresolved)."""
    try:
        ent = plan.entity_by_id(entity_id)
    except Exception:
        return None
    if ent.kind == "filled":
        return ent.name or ent.id
    bs = state.blanks.get(entity_id)
    if bs is not None and bs.status == "resolved" and bs.value is not None:
        return str(bs.value)
    return None


def _relation_citations(state: QueryGSWState, vp: Any) -> list[str]:
    """Chunks cited by the blank endpoint(s) of this verb-phrase (deduped,
    placeholder dropped)."""
    out: list[str] = []
    for eid in (vp.subject_id, vp.object_id):
        bs = state.blanks.get(eid)
        if bs is None:
            continue
        for c in bs.evidence_chunk_ids or []:
            if c and c != "insufficient_evidence" and c not in out:
                out.append(c)
    return out


def _ground_relation_substring(obj_value: str, cited: list[str], corpus: Any) -> tuple[bool, str]:
    """Default Stage-B grounder: numeric-tolerant substring of the (blank)
    object value in a cited chunk's OWN text. Returns (grounded, span)."""
    vs = (obj_value or "").lower()
    needles = [vs] + ([vs[:-2]] if vs.endswith(".0") else [])
    for cid in cited:
        chunk = corpus.get_chunk(cid)
        if chunk is None:
            continue
        text = chunk.text or ""
        low = text.lower()
        for n in needles:
            if n and n in low:
                i = low.find(n)
                return True, text[max(0, i - 80): i + len(n) + 80]
    return False, ""


def _blank_endpoint_id(plan: GSWPlan, vp: Any) -> str:
    """The blank endpoint this relation helps fill (prefer object, else subject)."""
    for eid in (vp.object_id, vp.subject_id):
        try:
            if plan.entity_by_id(eid).kind == "blank":
                return eid
        except Exception:
            continue
    return vp.object_id


# "rejected" can arrive from Stage A's enforce gate; Stage B never overwrites it.
_TERMINAL_EDGE_STATUS = {"accepted", "low_confidence", "rejected"}


def verify_graph(
    plan: GSWPlan,
    state: QueryGSWState,
    corpus: Any,
    *,
    judge: Optional[Any] = None,
) -> None:
    """Stage B graph verification over the MERGED state. For each verb-phrase:
    resolve both endpoints from merged values; if either is unresolved ->
    `missing_dependency` (never `low_confidence`); else ground the
    instantiated triple. ``judge`` is an optional callable
    ``(subject, relation, object, cited) -> (grounded, span)``; default uses the
    substring grounder.

    Idempotent: terminal verdicts are left alone; `missing_dependency` edges
    are re-evaluated on each call (the fixpoint)."""
    for vp in plan.verb_phrases:
        edge_id = f"{vp.id}:rel"
        existing = state.evidence_edges.get(edge_id)
        if existing is not None and existing.status in _TERMINAL_EDGE_STATUS:
            continue
        subj = _endpoint_value(plan, state, vp.subject_id)
        obj = _endpoint_value(plan, state, vp.object_id)
        if subj is None or obj is None:
            needs = [
                eid for eid, val in
                ((vp.subject_id, subj), (vp.object_id, obj)) if val is None
            ]
            state.add_evidence_edge(EvidenceEdge(
                edge_id=edge_id,
                subject=(subj or vp.subject_id),
                relation=vp.phrase,
                object=(obj or vp.object_id),
                object_type="entity", doc_id="",
                blank_id=_blank_endpoint_id(plan, vp),
                status="missing_dependency",
                rejection_reason="unresolved_slot",
                confidence=0.0,
                raw_payload={"needs_slots": needs},
            ))
            continue
        cited = _relation_citations(state, vp)
        if judge is not None:
            grounded, span = judge(subj, vp.phrase, obj, cited)
        else:
            grounded, span = _ground_relation_substring(obj, cited, corpus)
        state.add_evidence_edge(EvidenceEdge(
            edge_id=edge_id,
            subject=subj, relation=vp.phrase, object=obj,
            object_type="entity",
            doc_id=(cited[0] if cited else ""),
            blank_id=_blank_endpoint_id(plan, vp),
            chunk_id=(cited[0] if cited else None),
            source_span=span,
            status="accepted" if grounded else "low_confidence",
            rejection_reason=None if grounded
            else "relation_not_supported_by_cited_evidence",
            confidence=1.0 if grounded else 0.0,
        ))


# ---------------------------------------------------------------------------
# Phase 2 — LLM grounding judge for Stage B
# ---------------------------------------------------------------------------


def _value_windows(article: str, value_str: str, radius: int = 400, max_windows: int = 4) -> list[str]:
    """±radius-char windows around occurrences of value_str (and the integer
    form for floats like '10.0') — bounds the Tier-2 escalation."""
    if not article or not value_str:
        return []
    hay = article.lower()
    needles = [value_str.lower()] + ([value_str[:-2].lower()] if value_str.endswith(".0") else [])
    windows: list[str] = []
    used: set[int] = set()
    for needle in needles:
        if not needle:
            continue
        start = 0
        while len(windows) < max_windows:
            i = hay.find(needle, start)
            if i < 0:
                break
            left = max(0, i - radius)
            bucket = left // 100
            if bucket not in used:
                windows.append(article[left: i + len(needle) + radius])
                used.add(bucket)
            start = i + len(needle)
    return windows


def _judge_claim_llm(
    llm: Any, claim: str, passages: list[str], seed: Any
) -> tuple[bool, str, str, str, str]:
    """Returns (grounded, span, prompt_text, raw_output, reasoning)."""
    user_content = (
        f"Claim: {claim}\n\nEvidence:\n" + "\n\n".join(passages)
        + "\n\nDoes the evidence support the claim?"
    )
    messages = [
        {"role": "system", "content": (
            "You verify whether retrieved evidence supports a factual claim. "
            "Judge MEANING, not exact wording: '3554' supports '3554.0'; a "
            "'2004-05 season' mention supports '2004-05'. Reply ONLY with a JSON "
            'object {"grounded": true|false, "span": "<short quote or empty>"}.'
        )},
        {"role": "user", "content": user_content},
    ]
    try:
        resp = llm.chat(
            messages, max_tokens=200, temperature=0.0,
            response_format={"type": "json_object"}, stream=False, seed=seed,
        )
        raw_output = getattr(resp, "text", "") or ""
        reasoning = getattr(resp, "reasoning_content", "") or ""
        data = _json.loads(raw_output.strip())
        return bool(data.get("grounded")), str(data.get("span") or ""), user_content, raw_output, reasoning
    except Exception:
        return True, "", user_content, "", ""


def judge_relation_llm(
    llm: Any, corpus: Any, seed: Any = None, sink: Optional[list] = None
):
    """Build a Stage-B judge closure: (subject, relation, object, cited) ->
    (grounded, span). Tier 1 judges the cited chunks' OWN text; on low
    confidence, Tier 2 re-judges bounded find-windows of the object value
    across the full article. Conservative: no evidence -> ungrounded; LLM/parse
    error -> grounded (never block on infra).

    ``sink`` (optional): a list to append a per-call inspection record to —
    the claim, the passages shown to the judge, and the verdict — so the UI
    can show what each grounding check saw and decided. None records nothing
    and is behaviour-identical."""
    def judge(subject: str, relation: str, obj: str, cited: list[str]) -> tuple[bool, str]:
        chunks = [c for c in (cited or []) if c and c != "insufficient_evidence"]
        claim = f"{subject} {str(relation).replace('_', ' ')} {obj}"
        # Initialised to "" so pre-LLM _emit calls (no_evidence, no_chunk_text)
        # always read valid values from the closure.
        prompt_text: str = ""
        raw_output: str = ""
        reasoning: str = ""

        def _emit(grounded: bool, span: str, tier: str, passages: list) -> tuple[bool, str]:
            if sink is not None:
                sink.append({
                    "kind": "grounding", "subject": subject, "relation": relation,
                    "object": obj, "claim": claim, "cited": list(chunks),
                    "tier": tier, "grounded": grounded, "span": span,
                    "passages_shown": [p[:600] for p in (passages or [])],
                    "prompt": prompt_text, "raw_output": raw_output,
                    "reasoning": reasoning,
                })
            return grounded, span

        if not chunks:
            return _emit(False, "", "no_evidence", [])
        tier1 = []
        for cid in chunks[:4]:
            ch = corpus.get_chunk(cid)
            if ch is not None and ch.text:
                tier1.append(f"[{cid}] {ch.text[:8000]}")
        if not tier1:
            return _emit(False, "", "no_chunk_text", [])
        grounded, span, prompt_text, raw_output, reasoning = _judge_claim_llm(llm, claim, tier1, seed)
        if grounded:
            return _emit(True, span, "tier1", tier1)
        esc = []
        for cid in chunks[:4]:
            ch = corpus.get_chunk(cid)
            if ch is None:
                continue
            try:
                article = corpus.article_text(ch.title) or ""
            except Exception:
                article = ""
            for win in _value_windows(article, str(obj)):
                esc.append(f"[{cid}] …{win}…")
        if esc:
            g2, s2, prompt_text, raw_output, reasoning = _judge_claim_llm(llm, claim, esc, seed)
            if g2:
                return _emit(True, s2, "tier2", esc)
            return _emit(False, span, "tier2_failed", esc)
        return _emit(False, span, "tier1_failed", tier1)
    return judge


def _transitive_deps(state: "QueryGSWState", target_id: str) -> set[str]:
    """All blank ids reachable from target via BlankState.deps (excludes target itself)."""
    seen: set[str] = set()
    stack = list(getattr(state.blanks.get(target_id), "deps", []) or [])
    while stack:
        bid = stack.pop()
        if bid in seen:
            continue
        seen.add(bid)
        bs = state.blanks.get(bid)
        if bs is not None:
            stack.extend(getattr(bs, "deps", []) or [])
    return seen


def collect_supporting_edges(state: QueryGSWState, target_id: str) -> list[str]:
    """Accepted edge ids backing ``target_id`` and its transitive deps.

    Walks ``BlankState.deps`` from the target; unions each visited blank's
    ``accepted_edge_ids`` in first-seen order.
    """
    dep_closure = _transitive_deps(state, target_id)
    order: list[str] = []
    for bid in [target_id] + sorted(dep_closure):
        bs = state.blanks.get(bid)
        if bs is None:
            continue
        for eid in bs.accepted_edge_ids:
            if eid not in order:
                order.append(eid)
    return order


def cascade_diagnostics(block_records: list[dict], state: "QueryGSWState", target_id: str) -> dict:
    """Deterministic per-question Plan-2 mechanism-slice fields from cascade block records."""
    ids = {r["blocked_input"] for r in block_records}
    out_ids = {r["output_blank_id"] for r in block_records}
    cids = {r["constraint_id"] for r in block_records if r.get("constraint_id")}
    n_unver = sum(1 for r in block_records if r.get("kind") == "unverified")
    n_part = sum(1 for r in block_records if r.get("kind") == "partial")
    tgt = state.blanks.get(target_id)
    target_resolved = bool(tgt and tgt.status == "resolved")
    dep_closure = _transitive_deps(state, target_id)
    target_depends_on_blocked_output = (target_id in out_ids) or bool(out_ids & dep_closure)
    target_depends_on_blocked_input = bool(ids & dep_closure)
    return {
        "cascade_blocked_count": len(block_records),
        "blocked_input_blank_ids": sorted(ids),
        "blocked_output_blank_ids": sorted(out_ids),
        "blocked_constraint_ids": sorted(cids),
        "unverified_cascade_blocked": n_unver,
        "partial_cascade_blocked": n_part,
        "target_depends_on_blocked_output": target_depends_on_blocked_output,
        "target_depends_on_blocked_input": target_depends_on_blocked_input,
        "target_resolved": target_resolved,
        "blocked_and_unresolved": bool(block_records) and target_depends_on_blocked_output and not target_resolved,
        "blocked_but_recovered": bool(block_records) and target_depends_on_blocked_output and target_resolved,
    }


@dataclass
class CoverageReport:
    target_resolved: bool
    final_answer_supported: bool
    unsupported_final_value: bool
    resolved_blanks: int
    resolved_without_edge: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_resolved": self.target_resolved,
            "final_answer_supported": self.final_answer_supported,
            "unsupported_final_value": self.unsupported_final_value,
            "resolved_blanks": self.resolved_blanks,
            "resolved_without_edge": self.resolved_without_edge,
        }


def check_evidence_coverage(
    state: QueryGSWState, plan: GSWPlan, final_answer: Optional[str]
) -> CoverageReport:
    """Deterministic faithfulness check over the resolved state.

    ``final_answer_supported`` = the target is resolved AND it either has an
    accepted edge or is a constraint output (its support is its inputs'
    edges). ``unsupported_final_value`` = a non-empty answer that is not
    supported.
    """
    constraint_outputs = {
        c.output_blank_id for c in plan.constraints if c.output_blank_id
    }
    try:
        target_id = plan.target().id
    except Exception:
        target_id = None

    tbs = state.blanks.get(target_id) if target_id else None
    target_resolved = bool(tbs and tbs.status == "resolved")

    resolved = [b for b in state.blanks.values() if b.status == "resolved"]
    resolved_without_edge = sum(
        1
        for b in resolved
        if not b.accepted_edge_ids and b.blank_id not in constraint_outputs
    )

    fa = (final_answer or "").strip()
    final_answer_supported = bool(
        target_resolved
        and tbs is not None
        and (tbs.accepted_edge_ids or tbs.blank_id in constraint_outputs)
    )
    unsupported_final_value = bool(fa) and not final_answer_supported

    return CoverageReport(
        target_resolved=target_resolved,
        final_answer_supported=final_answer_supported,
        unsupported_final_value=unsupported_final_value,
        resolved_blanks=len(resolved),
        resolved_without_edge=resolved_without_edge,
    )


def classify_failure(
    state: QueryGSWState,
    plan: GSWPlan,
    final_answer: Optional[str],
    gold: Optional[str] = None,
) -> str:
    """Deterministic primary failure bucket from state signals.

    Buckets: blank_ambiguous, dependency_unresolved, retrieval_miss,
    unsupported_final_value, blank_without_evidence, empty_answer,
    blank_wrong_value (only with ``gold``), ok.
    """
    if any(b.status == "ambiguous" for b in state.blanks.values()):
        return "blank_ambiguous"

    cov = check_evidence_coverage(state, plan, final_answer)

    if not cov.target_resolved:
        try:
            tbs = state.blanks.get(plan.target().id)
        except Exception:
            tbs = None
        if tbs and any(
            (state.blanks.get(d) is not None and state.blanks[d].status != "resolved")
            for d in tbs.deps
        ):
            return "dependency_unresolved"
        return "retrieval_miss"

    if cov.unsupported_final_value:
        return "unsupported_final_value"
    if cov.resolved_without_edge > 0:
        return "blank_without_evidence"

    fa = (final_answer or "").strip()
    if not fa:
        return "empty_answer"
    if gold is not None and fa.lower() != str(gold).strip().lower():
        return "blank_wrong_value"
    return "ok"
