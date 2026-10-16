"""Prompt + parser for the single structured resolver call (Phase 3-lite)."""

from __future__ import annotations

from typing import Any

from panini_dr.adapters.ours._planner_exec import _extract_json_object

RESOLVER_SYSTEM = """You resolve exactly ONE typed slot ("blank") in a \
question-answering plan from retrieved encyclopedia chunks.

You are given a blank described by (subject, relation, expected value type) \
and a set of chunks. Find the single value that the chunks DIRECTLY support \
for `subject --relation--> value`.

Return a JSON object only:
{
  "value": <the value, or null>,
  "object_type": "<date|number|entity|attribute|list|text|bool>",
  "source_span": "<verbatim sentence/phrase from a chunk that supports it>",
  "supported": true | false,
  "confidence": <0.0-1.0>,
  "evidence_chunk_ids": ["<chunk id>", ...]
}

Rules:
- Set supported=false (and value=null) if the chunks do not directly support \
the relation, or if you are unsure. NEVER fabricate.
- The value's type must match the expected value type.
- Do NOT answer the overall question — resolve ONLY this blank.
- `evidence_chunk_ids` must be chunk ids you were shown."""

RESOLVER_USER = """Blank to resolve:
- subject: {subject}
- relation: {relation}
- expected value type: {value_type}
{unit_instruction}
Overall question (for disambiguation only): {question}

Resolved context (upstream values): {context}

Retrieved chunks:
{chunks}

Return the JSON object now."""


def render_resolver_unit_instruction(unit: "str | None", hint: "str | None") -> str:
    """Build the RESOLVER_USER `{unit_instruction}` block for a blank whose
    planner-declared retrieval_hint names a required unit (Hard rule 7).

    ``unit`` is the closed-vocabulary canonical unit (e.g. "metres", "feet")
    already extracted by the caller via ``_planner_emit._extract_unit`` --
    this function does no extraction itself, only rendering.

    Returns "" when ``unit`` is falsy. Callers MUST pass that "" straight
    into RESOLVER_USER's `unit_instruction` slot for any blank with no
    declared unit -- the template line collapses back to the single blank
    line that separated "expected value type" from "Overall question"
    before this field existed, so the prompt renders BYTE-IDENTICAL to the
    pre-fix template. This is the behavior-preservation guarantee for every
    question that doesn't hit the q16/q692/q771 cross-unit class.
    """
    if not unit:
        return ""
    hint_text = hint or ""
    return (
        f"- REQUIRED UNIT: {unit}. The plan's retrieval hint ({hint_text!r}) "
        f"requires this value in {unit}. If a retrieved chunk states the "
        f"measurement in a different unit, CONVERT it to {unit} before "
        "reporting `value` -- do NOT silently report the number in "
        "whatever unit the source happens to use."
    )


def parse_resolver_output(text: str) -> dict[str, Any]:
    """Parse the resolver JSON; on any failure return a safe unsupported."""
    default = {
        "value": None, "object_type": None, "source_span": "",
        "supported": False, "confidence": 0.0, "evidence_chunk_ids": [],
    }
    try:
        data = _extract_json_object(text)
    except Exception:  # noqa: BLE001 — any parse failure is unsupported
        return default
    ev = data.get("evidence_chunk_ids") or []
    if not isinstance(ev, list):
        ev = []
    return {
        "value": data.get("value", None),
        "object_type": data.get("object_type", None),
        "source_span": str(data.get("source_span", "") or ""),
        "supported": bool(data.get("supported", False)),
        "confidence": float(data.get("confidence", 0.0) or 0.0),
        "evidence_chunk_ids": [str(x) for x in ev],
    }


VERIFIER_SYSTEM = """You verify whether a proposed VALUE correctly and \
UNIQUELY fills one slot ("blank") in a question-answering plan.

You are given the blank's full constraint set (every relation it must \
satisfy, including a category/type qualifier when present), the proposed \
value, and the retrieved chunks. Judge ONLY this value against ONLY these \
constraints — do not answer the overall question.

Return a JSON object only:
{
  "satisfies_all": true | false,   // value meets every CHECKABLE constraint
  "unique": true | false,          // no other entity also fits the evidence
  "missing_constraints": ["<relation the value fails>", ...],
  "deferred_constraints": ["<relation not checkable until a partner resolves>", ...],
  "ambiguous_candidates": ["<other entity that also fits>", ...],
  "complete": true | false,        // ALL items the question requires are present (LIST/SET blanks)
  "missing_items": ["<name of item still missing>", ...],  // empty if complete
  "reason": "<one sentence>"
}

Rules:
- A constraint whose partner is an unresolved blank WITH a category qualifier \
is checked existentially ("did the value <relation> SOME <category>?").
- A constraint whose partner is an unresolved blank with NO qualifier is NOT \
checkable: put it in deferred_constraints; do NOT count it as missing and do \
NOT invent the partner.
- satisfies_all is true when every CHECKABLE constraint holds (deferred ones \
do not block it).
- Set unique=false and list ambiguous_candidates when the evidence fits more \
than one entity (e.g. two co-founders).
- NEVER fabricate. When unsure a constraint holds, treat it as missing."""

VERIFIER_SYSTEM_SCOPED = VERIFIER_SYSTEM + """

Additional scoped-verifier rules:
- Snapshot/as-of date qualifiers (e.g. "as of August 3, 2024") are GIVEN \
context, not facts to verify. Never list them in `missing_constraints` and \
never set `satisfies_all=false` because a snapshot date is not separately \
evidenced.
- If a clause depends on a fact that is NOT addressable by the cited evidence \
(it belongs to a different/not-yet-resolved step — e.g. a property of another \
entity), put it in `deferred_constraints`, NOT `missing_constraints`. Deferred \
clauses must not set `satisfies_all=false`."""

VERIFIER_REASONING_SYSTEM_ADDENDUM = """The researcher's reasoning explains \
its derivation — use it to follow the logic and scope, but your verdict MUST \
be grounded in the cited EVIDENCE. If the reasoning asserts something the \
cited evidence does not show, do NOT accept it on the reasoning alone."""

VERIFIER_USER = """Blank: {blank_id} (expected type: {value_type})
Proposed value: {value}

Constraints to verify (relation -> partner; "?<id>" = unresolved blank, \
"[category]" = its qualifier):
{constraints}

Resolved upstream context: {context}

Retrieved chunks:
{chunks}

Return the JSON object now."""


def parse_verifier_output(text: str) -> dict[str, Any]:
    """Parse the verifier JSON; on any failure return a safe reject."""
    default = {
        "satisfies_all": False, "unique": False, "missing_constraints": [],
        "deferred_constraints": [], "ambiguous_candidates": [], "reason": "",
        "complete": True, "missing_items": [],
    }
    try:
        data = _extract_json_object(text)
    except Exception:  # noqa: BLE001
        return default

    def _strlist(key: str) -> list[str]:
        v = data.get(key) or []
        return [str(x) for x in v] if isinstance(v, list) else []

    return {
        "satisfies_all": bool(data.get("satisfies_all", False)),
        "unique": bool(data.get("unique", False)),
        "missing_constraints": _strlist("missing_constraints"),
        "deferred_constraints": _strlist("deferred_constraints"),
        "ambiguous_candidates": _strlist("ambiguous_candidates"),
        "reason": str(data.get("reason", "") or ""),
        "complete": bool(data.get("complete", True)),
        "missing_items": _strlist("missing_items"),
    }
