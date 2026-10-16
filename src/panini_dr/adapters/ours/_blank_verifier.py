"""Per-blank verifier (Phase 3-full §9): judge whether a resolver value
satisfies its blank's full constraint set, uniquely. One structured LLM call.
NO edge building — Stage B (verify_graph) still owns relation grounding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from panini_dr.adapters.ours._blank_resolver import (
    _qualifier_label, substitute_placeholders, has_unresolved_placeholders,
)
from panini_dr.adapters.ours._evidence_prompts import (
    VERIFIER_REASONING_SYSTEM_ADDENDUM,
    VERIFIER_SYSTEM,
    VERIFIER_SYSTEM_SCOPED,
    VERIFIER_USER,
    parse_verifier_output,
)
from panini_dr.adapters.ours._gsw_state import blank_projection_edges

def _chunk_block(chunks: list[dict[str, str]]) -> str:
    """Render a group of chunks for the verifier prompt."""
    return "\n\n".join(
        f"[{c.get('chunk_id','')}] {c.get('title','')}\n{c.get('text','')}"
        for c in chunks
    ) or "(none)"


@dataclass
class VerifierResult:
    satisfies_all: bool
    unique: bool
    complete: bool = True                 # list/set blanks: are all required items present? scalar default True
    missing_items: list[str] = field(default_factory=list)
    missing_constraints: list[str] = field(default_factory=list)
    deferred_constraints: list[str] = field(default_factory=list)
    ambiguous_candidates: list[str] = field(default_factory=list)
    reason: str = ""
    connection_verdicts: dict[str, Any] = field(default_factory=dict)  # vp_id -> {ok, reason}
    unsupported_vp_ids: list[str] = field(default_factory=list)
    composed_unique: bool = True
    verifier_status_hint: Optional[str] = None  # "unverified_capped" when a connection gap was flagged

    @property
    def accepted(self) -> bool:
        return self.satisfies_all and self.unique

    @property
    def partial(self) -> bool:
        return not self.complete


def _render_edge_line(plan: Any, edge: dict[str, Any], dep_values: dict[str, Any]) -> str:
    """Render one incident relation for the prompt, applying the
    unresolved-partner rule: resolved/filled -> value; unresolved+qualified ->
    'some [category]'; unresolved+bare -> '?<id> (deferred)'. Never the raw id
    inside a checkable claim. Shared by _constraint_lines (holistic) and
    BlankVerifier.verify_compositional (per-edge)."""
    pid = edge["other_entity_id"]
    ent = plan.entity_by_id(pid)
    rel = edge["phrase"]
    if ent.kind == "filled":
        partner = ent.name or ent.id
    else:
        val = dep_values.get(pid)
        if val is not None and str(val).strip():
            partner = str(val)
        else:
            qual = _qualifier_label(plan, pid)
            partner = f"some {qual} [category]" if qual else f"?{pid} (deferred)"
    side = "(value as subject)" if edge["direction"] == "reverse" else "(value as object)"
    return f"- {rel} {partner} {side}"


def _constraint_lines(plan: Any, blank_id: str, dep_values: dict[str, Any]) -> str:
    """Render each incident relation for the prompt, applying the
    unresolved-partner rule: resolved/filled -> value; unresolved+qualified ->
    'some [category]'; unresolved+bare -> '?<id> (deferred)'. Never the raw id
    inside a checkable claim."""
    lines = [_render_edge_line(plan, e, dep_values) for e in blank_projection_edges(plan, blank_id)]
    return "\n".join(lines) if lines else "- (no relations)"


def get_verification_claim(plan: Any, blank_id: str) -> Optional[str]:
    """The planner's verification_claim for a blank, or None if absent/blank."""
    try:
        ent = plan.entity_by_id(blank_id)
    except Exception:  # noqa: BLE001
        return None
    claim = getattr(ent, "verification_claim", None)
    return claim.strip() if claim and claim.strip() else None


def render_blank_claim(plan: Any, blank_id: str, dep_values: dict[str, Any]) -> Optional[str]:
    """Substitute dep values into a blank's verification_claim. Returns None when
    there is no claim or it still has unresolved placeholders (caller then falls
    back to _constraint_lines)."""
    claim = get_verification_claim(plan, blank_id)
    if not claim:
        return None
    rendered = substitute_placeholders(claim, dep_values)
    if has_unresolved_placeholders(rendered):
        return None
    return rendered


class BlankVerifier:
    def __init__(self, llm: Any, sink: Optional[list] = None,
                 max_completion_tokens: int = 400, use_hints: bool = False,
                 scoped: bool = False) -> None:
        self.llm = llm
        self.sink = sink
        self.max_completion_tokens = max_completion_tokens
        self.use_hints = use_hints
        self.scoped = scoped

    def _system_prompt(self, reasoning: Optional[str] = None) -> str:
        system_prompt = VERIFIER_SYSTEM_SCOPED if self.scoped else VERIFIER_SYSTEM
        if reasoning:
            system_prompt = system_prompt + "\n\n" + VERIFIER_REASONING_SYSTEM_ADDENDUM
        return system_prompt

    def verify(self, plan: Any, blank_id: str, value: Any,
               chunks: list[dict[str, str]], dep_values: dict[str, Any],
               reasoning: Optional[str] = None) -> VerifierResult:
        researcher_reasoning = reasoning
        try:
            value_type = plan.entity_by_id(blank_id).value_type or "entity"
        except Exception:  # noqa: BLE001
            value_type = "entity"
        claim_source = "projection_edges"
        raw_claim = get_verification_claim(plan, blank_id)
        rendered_claim = render_blank_claim(plan, blank_id, dep_values) if self.use_hints else None
        if rendered_claim is not None:
            constraints = rendered_claim
            claim_source = "hint"
        else:
            constraints = _constraint_lines(plan, blank_id, dep_values)
        context = "; ".join(f"{k}={v}" for k, v in dep_values.items()) or "(none)"
        chunk_text = _chunk_block(chunks)
        user_prompt = VERIFIER_USER.format(
            blank_id=blank_id, value_type=value_type, value=value,
            constraints=constraints, context=context, chunks=chunk_text,
        )
        if researcher_reasoning:
            user_prompt = (
                user_prompt
                + "\n\nResearcher's reasoning (GUIDANCE ONLY):\n"
                + researcher_reasoning
            )
        raw_output, reasoning = "", ""
        try:
            resp = self.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": self._system_prompt(researcher_reasoning),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_completion_tokens,
            )
            raw_output = getattr(resp, "text", "") or ""
            reasoning = getattr(resp, "reasoning_content", "") or ""
            data = parse_verifier_output(raw_output)
        except Exception as exc:  # noqa: BLE001
            data = {"satisfies_all": False, "unique": False,
                    "missing_constraints": [], "deferred_constraints": [],
                    "ambiguous_candidates": [], "reason": f"verifier_error: {exc}"}
        result = VerifierResult(**data)
        if self.sink is not None:
            self.sink.append({
                "kind": "verifier", "blank_id": blank_id, "value": value,
                "satisfies_all": result.satisfies_all, "unique": result.unique,
                "missing_constraints": result.missing_constraints,
                "deferred_constraints": result.deferred_constraints,
                "ambiguous_candidates": result.ambiguous_candidates,
                "complete": result.complete,
                "partial": result.partial,
                "missing_items": result.missing_items,
                "reason": result.reason, "constraints": constraints,
                "claim_source": claim_source,
                "verification_claim_raw": raw_claim,
                "rendered_verification_claim": rendered_claim,
                "prompt": user_prompt, "raw_output": raw_output,
                "reasoning": reasoning,
                "verdict": {"satisfies_all": result.satisfies_all,
                            "unique": result.unique},
            })
        return result


def verify_blank(verifier, plan, blank_id, value, dep_values, *,
                 holistic_chunks,
                 reasoning: Optional[str] = None):
    """Per-blank verification: the holistic single-call verify over
    holistic_chunks. (The compositional per-connection path and its
    RefutationLog lived behind gsw_connection_evidence — removed 2026-08-29,
    archived on campaign-full.)"""
    verify_kwargs: dict[str, Any] = {}
    if reasoning is not None:
        verify_kwargs["reasoning"] = reasoning
    return verifier.verify(
        plan, blank_id, value, holistic_chunks, dep_values, **verify_kwargs
    )
