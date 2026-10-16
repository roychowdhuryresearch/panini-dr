"""Typed blank resolver: structured retrieve -> extract -> type-filter ->
produce (value, citations). NO edge building -- grounding is Stage B
(verify_graph) on the merged state. One structured LLM call per blank.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from panini_dr.adapters.ours._evidence_prompts import (
    RESOLVER_SYSTEM,
    RESOLVER_USER,
    parse_resolver_output,
    render_resolver_unit_instruction,
)
from panini_dr.adapters.ours._evidence_ranker import object_matches_value_type
from panini_dr.adapters.ours._gsw_state import (
    blank_projection,
    blank_projection_edges,
)
from panini_dr.adapters.ours._planner_emit import _extract_unit
from panini_dr.adapters.ours._planner_exec import (
    GSWPlan,
    _format_chunks_for_prompt,
    _retrieve_chunks,
)

# Small override map for relations whose snake_case reads poorly as a query.
# Keep this tiny — NOT a relation ontology. Default is underscore -> space.
_RELATION_OVERRIDES = {
    "cofounder_of": "co-founder of",
    "co_founded": "co-founded",
    "instance_of": "",          # the qualifier label already reads naturally
    "has_occupation": "occupation",
}


def normalize_relation_phrase(phrase: str) -> str:
    """Render a snake_case predicate as a natural retrieval fragment."""
    if phrase in _RELATION_OVERRIDES:
        return _RELATION_OVERRIDES[phrase]
    return (phrase or "").replace("_", " ").strip()


def _qualifier_label(plan: Any, blank_id: str) -> Optional[str]:
    """The category label on a blank's `instance_of` qualifier edge, if any."""
    for e in blank_projection_edges(plan, blank_id):
        if e["phrase"] == "instance_of":
            return e["other_label"] or None
    return None


def _partner_token(plan: Any, partner_id: str, dep_values: dict[str, Any]) -> Optional[str]:
    """Substitution token for an edge's partner; None => omit the clause."""
    ent = plan.entity_by_id(partner_id)
    if ent.kind == "filled":
        return (ent.name or ent.id)
    # blank partner
    val = dep_values.get(partner_id)
    if val is not None and str(val).strip():
        return str(val)                      # resolved -> its value
    return _qualifier_label(plan, partner_id)  # unresolved -> qualifier, or None (omit)


def build_resolver_query(plan: Any, blank_id: str, dep_values: dict[str, Any]) -> str:
    """Direction-aware retrieval query for one blank.

    For each incident edge: normalize the relation; substitute the partner's
    value (resolved/filled) or its category qualifier (unresolved+qualified);
    omit the clause when the partner is an unresolved blank with no qualifier.
    """
    parts: list[str] = []
    for e in blank_projection_edges(plan, blank_id):
        partner = _partner_token(plan, e["other_entity_id"], dep_values)
        if partner is None:
            continue                          # unresolved, unqualified -> omit
        rel = normalize_relation_phrase(e["phrase"])
        if e["direction"] == "forward":       # blank is OBJECT -> "subject relation"
            clause = f"{partner} {rel}".strip()
        else:                                 # reverse: blank is SUBJECT -> "relation object"
            clause = f"{rel} {partner}".strip()
        if clause:
            parts.append(clause)
    return " ".join(parts).strip()


_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")


def substitute_placeholders(text: str, dep_values: dict[str, Any]) -> str:
    """Replace each ``{blank_id}`` with its resolved value. Unknown / empty
    placeholders are left intact so the caller can detect an unusable hint."""
    def _repl(m: "re.Match[str]") -> str:
        key = m.group(1)
        val = dep_values.get(key)
        if val is not None and str(val).strip():
            return str(val)
        return m.group(0)
    return _PLACEHOLDER_RE.sub(_repl, text or "")


def has_unresolved_placeholders(text: str) -> bool:
    """True if any ``{blank_id}`` token remains."""
    return bool(_PLACEHOLDER_RE.search(text or ""))


def get_retrieval_hint(plan: Any, blank_id: str) -> Optional[str]:
    """The planner's retrieval_hint for a blank, or None if absent/blank."""
    try:
        ent = plan.entity_by_id(blank_id)
    except Exception:  # noqa: BLE001
        return None
    hint = getattr(ent, "retrieval_hint", None)
    return hint.strip() if hint and hint.strip() else None


@dataclass
class EffectiveQuery:
    """Resolved retrieval query for one blank, plus where it came from.

    source semantics:
      - "hint":             planner retrieval_hint rendered cleanly -> use query.
      - "unresolved_hint":  a hint exists but still has {placeholders};
                            query is None -> the resolver must NOT search; it
                            returns status="unresolved" (needs dependency).
      - "edge":             no hint -> edge-based query builder produced a string.
      - "question_fallback":no hint and edge builder empty -> raw question.
    """
    query: Optional[str]
    source: str
    raw_hint: Optional[str] = None
    rendered_hint: Optional[str] = None


def effective_query(
    plan: Any, blank_id: str, dep_values: dict[str, Any], question: str,
    *, use_hints: bool,
) -> EffectiveQuery:
    """Decide the retrieval query for a blank. Hint-first when use_hints is on;
    a hint with unresolved placeholders blocks (does NOT fall back to edges)."""
    if use_hints:
        # Derive raw + rendered inline so both can be logged, including on the
        # unresolved-hint path.
        raw = get_retrieval_hint(plan, blank_id)
        if raw:
            rendered = substitute_placeholders(raw, dep_values)
            if has_unresolved_placeholders(rendered):
                return EffectiveQuery(query=None, source="unresolved_hint",
                                      raw_hint=raw, rendered_hint=rendered)
            return EffectiveQuery(query=rendered, source="hint",
                                  raw_hint=raw, rendered_hint=rendered)
    edge = build_resolver_query(plan, blank_id, dep_values)
    if edge:
        return EffectiveQuery(query=edge, source="edge")
    return EffectiveQuery(query=question, source="question_fallback")


@dataclass
class BlankResolutionResult:
    blank_id: str
    status: str  # "resolved" | "unresolved"
    value: Optional[Any]
    confidence: float
    evidence_chunk_ids: list[str]
    connection_evidence: dict[str, list[str]] = field(default_factory=dict)
    source_span: str = ""
    message: str = ""
    retrieved_chunks: list[dict[str, str]] = field(default_factory=list)


def _retrieved_chunk_records(chunks: list[Any]) -> list[dict[str, str]]:
    return [
        {
            "chunk_id": str(getattr(c, "chunk_id", "") or ""),
            "title": str(getattr(c, "title", "") or ""),
            "text": str(getattr(c, "text", "") or ""),
        }
        for c in chunks
    ]


# ---------------------------------------------------------------------------
# Hard rule 7 / q16 class -- deterministic dual-unit reconciliation.
#
# Wikipedia commonly states one measurement twice in the same sentence, once
# per unit: "Tallest building height: 1,070 ft (326.1 m)". A prior audit
# measured this "NUM unit1 (NUM unit2)" shape in 176/1013 (17.4%) of corpus
# articles, 2118 occurrences. When the plan DECLARES a required unit for a
# blank (Hard rule 7's retrieval_hint, e.g. "...in meters?") and the
# resolver's cited source_span has this shape, prefer whichever bracketed
# number matches the declared unit over whatever the LLM picked -- the model
# is known to sometimes guess the wrong one (q16: declared metres, span read
# "1,070 ft (326.1 m)", model answered 1070/feet).
#
# Modeled on the span_autorepair mechanism below (CE-v2 spec S3.4): same
# "detect a class of grounding slip, deterministically correct it, log to
# sink" shape, but repairing UNIT choice instead of chunk-id citation.
# ---------------------------------------------------------------------------

# Two numbers in the same span, each immediately followed by a unit-shaped
# token, the second pair parenthesized: "1,070 ft (326.1 m)". Deliberately
# requires the unit token to sit directly between the number and the "(" (no
# intervening prose) -- this is what keeps it from firing on unrelated
# parentheticals like "population 1,070 (as of 2020 census: 326,000)" or
# "$1,070 million (up from $326 million in 2020)" (currency's "$" prefixes
# the number instead of following it, so it never matches this shape).
_DUAL_UNIT_SPAN_RE = re.compile(
    r"([\d][\d,]*\.?\d*)\s*([A-Za-z%$]+)\.?\s*\(\s*([\d][\d,]*\.?\d*)\s*([A-Za-z%$]+)\.?\s*\)"
)


def _parse_number_token(raw: str) -> Optional[Any]:
    """"1,070" -> 1070 (int); "326.1" -> 326.1 (float); unparsable -> None."""
    try:
        f = float(raw.replace(",", ""))
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else f


def reconcile_declared_unit_value(
    source_span: str, declared_unit: Optional[str],
) -> Optional[Any]:
    """Return the span's number for ``declared_unit``, or None to change
    nothing.

    Conservative by construction -- returns None (do not guess) unless ALL
    of:
      1. a unit is declared,
      2. the span matches the "NUM unit (NUM unit)" shape exactly once,
      3. BOTH bracketed tokens resolve to a *recognized* unit via the SAME
         closed vocabulary ``_planner_emit._extract_unit`` uses for the
         Hard rule 7 planner-side check (no second unit vocabulary),
      4. those two recognized units are DIFFERENT, and
      5. exactly one of them equals ``declared_unit``.
    Any other shape -- no match, an unrecognized token, both sides the same
    unit, or the declared unit absent from both -- returns None so the
    caller leaves the resolver's original value untouched.
    """
    if not source_span or not declared_unit:
        return None
    m = _DUAL_UNIT_SPAN_RE.search(source_span)
    if not m:
        return None
    num1_raw, tok1, num2_raw, tok2 = m.groups()
    unit1 = _extract_unit(tok1)
    unit2 = _extract_unit(tok2)
    if not unit1 or not unit2 or unit1 == unit2:
        return None  # unrecognized token(s), or ambiguous same-unit pair
    if unit1 == declared_unit and unit2 != declared_unit:
        return _parse_number_token(num1_raw)
    if unit2 == declared_unit and unit1 != declared_unit:
        return _parse_number_token(num2_raw)
    return None  # declared unit not (uniquely) present in either bracket


class TypedBlankResolver:
    def __init__(
        self, *, retriever: Any, corpus: Any, llm: Any,
        max_docs: int = 5, hard_type_filter: bool = True,
        max_completion_tokens: int = 600, sink: Optional[list] = None,
        use_hints: bool = False,
    ) -> None:
        self.retriever = retriever
        self.corpus = corpus
        self.llm = llm
        self.max_docs = max_docs
        self.hard_type_filter = hard_type_filter
        self.max_completion_tokens = max_completion_tokens
        # gsw_blank_hints: prefer the planner's retrieval_hint over the edge
        # query builder. Default off -> behavior-identical.
        self.use_hints = use_hints
        # Optional list to append a per-blank inspection record to (query +
        # outcome) so the UI can show what the resolver searched and decided.
        self.sink = sink

    def resolve_blank(
        self, plan: GSWPlan, blank_id: str, dep_values: dict[str, Any],
        question: str,
    ) -> BlankResolutionResult:
        subject, relation, _direction = blank_projection(plan, blank_id)
        try:
            value_type = plan.entity_by_id(blank_id).value_type or "entity"
        except Exception:
            value_type = "entity"

        eq = effective_query(plan, blank_id, dep_values, question,
                             use_hints=self.use_hints)
        query = eq.query

        # Hard rule 7 / q16 class: a numeric blank's retrieval_hint may
        # DECLARE a required unit ("...in meters?"). Only meaningful when the
        # hint was actually used (eq.source == "hint") and the blank is
        # numeric; otherwise stays None so the prompt/reconciliation below
        # are no-ops -> byte-identical to before this feature existed.
        declared_unit: Optional[str] = (
            _extract_unit(eq.rendered_hint)
            if eq.source == "hint" and value_type == "number"
            else None
        )

        chunks: list[Any] = []
        prompt_text, raw_output, reasoning = "", "", ""

        def _emit(result: BlankResolutionResult) -> BlankResolutionResult:
            records = _retrieved_chunk_records(chunks)
            result.retrieved_chunks = records
            if self.sink is not None:
                self.sink.append({
                    "kind": "resolver", "blank_id": blank_id, "query": query,
                    "query_source": eq.source,
                    "retrieval_hint_raw": eq.raw_hint,
                    "rendered_retrieval_hint": eq.rendered_hint,
                    "subject": subject, "relation": relation,
                    "value_type": value_type, "status": result.status,
                    "value": result.value, "confidence": result.confidence,
                    "message": result.message,
                    "evidence_chunk_ids": result.evidence_chunk_ids,
                    "source_span": result.source_span,
                    "retrieved_chunks": records,
                    "prompt": prompt_text, "raw_output": raw_output,
                    "reasoning": reasoning,
                    "verdict": {"status": result.status, "value": result.value,
                                "confidence": result.confidence},
                })
            return result

        if eq.source == "unresolved_hint":
            return _emit(BlankResolutionResult(
                blank_id=blank_id, status="unresolved", value=None,
                confidence=0.0, evidence_chunk_ids=[],
                message="retrieval_hint has unresolved placeholders",
            ))

        chunks = _retrieve_chunks(self.retriever, query, top_k=self.max_docs)
        if not chunks:
            return _emit(BlankResolutionResult(
                blank_id, "unresolved", None, 0.0, [], message="no_chunks"
            ))

        context = "; ".join(f"{k}={v}" for k, v in dep_values.items()) or "(none)"
        try:
            prompt_text = RESOLVER_USER.format(
                subject=subject or "(unknown)",
                relation=relation or "(unknown)",
                value_type=value_type,
                question=question,
                context=context,
                chunks=_format_chunks_for_prompt(chunks),
                unit_instruction=render_resolver_unit_instruction(
                    declared_unit, eq.rendered_hint),
            )
            resp = self.llm.chat(
                messages=[
                    {"role": "system", "content": RESOLVER_SYSTEM},
                    {"role": "user", "content": prompt_text},
                ],
                max_tokens=self.max_completion_tokens,
            )
            raw_output = getattr(resp, "text", "") or ""
            reasoning = getattr(resp, "reasoning_content", "") or ""
            data = parse_resolver_output(raw_output)

            value = data["value"]
            if not data["supported"] or value is None or str(value).strip() == "":
                return _emit(BlankResolutionResult(
                    blank_id, "unresolved", None, data["confidence"], [],
                    message="unsupported",
                ))
            if self.hard_type_filter and not object_matches_value_type(value, value_type):
                return _emit(BlankResolutionResult(
                    blank_id, "unresolved", None, data["confidence"], [],
                    message="type_mismatch",
                ))

            # Hard rule 7 / q16 class, step 2: deterministic dual-unit
            # reconciliation. See reconcile_declared_unit_value's docstring
            # for the exact conservative firing conditions -- a no-op
            # (returns None) for every blank without a declared unit, so
            # this cannot change behavior on unaffected questions.
            reconciled = reconcile_declared_unit_value(
                data["source_span"], declared_unit)
            if reconciled is not None and reconciled != _parse_number_token(str(value)):
                if self.sink is not None:
                    self.sink.append({
                        "kind": "unit_reconcile", "blank_id": blank_id,
                        "declared_unit": declared_unit,
                        "from": value, "to": reconciled,
                        "span": data["source_span"],
                    })
                value = reconciled

            # NO edge building -- Stage B (verify_graph) grounds post-merge.
            ev_ids = data["evidence_chunk_ids"] or [_chunk_id(c) for c in chunks]
            connection_evidence: dict[str, list[str]] = {}
            return _emit(BlankResolutionResult(
                blank_id, "resolved", value, data["confidence"], ev_ids,
                connection_evidence=connection_evidence,
                source_span=data["source_span"], message="resolved",
            ))
        except Exception as exc:  # noqa: BLE001 — contain per-blank failures
            return _emit(BlankResolutionResult(
                blank_id, "unresolved", None, 0.0, [], message=f"llm_error: {exc}"
            ))


def _chunk_id(c: Any) -> str:
    # _retrieve_chunks already wraps hits into _RetrievalChunk objects that
    # expose .chunk_id directly. The getattr(c, "chunk", c) guard is kept for
    # defensive correctness if ever called with a raw hit object instead.
    inner = getattr(c, "chunk", c)
    return getattr(inner, "chunk_id", "")
