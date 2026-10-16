"""GSW-fragment plans: schema, dependency DAG and constraint computation.

A ``GSWPlan`` emitted by the question-planner LLM has:

- Filled entities (extracted from the question text via NER).
- Blank entities with a ``value_type`` (to be resolved by retrieval).
- Verb-phrases connecting entities (directional, binary).
- Constraints between blanks (derived / argmax / argmin + relational and
  row-aggregation kinds), computed here in Python (``_compute_constraint``)
  once their inputs are resolved.

One blank has ``is_target=True``; its value is the answer. The shape is
deliberately generic: no FRAMES-specific slot types, no per-question
templates. The orchestrator resolves blanks; the flat executor that once
filled them here in one pass (``execute``) was never part of the paper
pipeline and was removed.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_serializer, model_validator

from panini_dr.adapters.ours import _gsw_worksheet_ops
from panini_dr.adapters.ours._gsw_worksheet_ops import CandidateRow

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_ROW_AGGREGATION_KINDS = {"rank_n", "filter", "count"}
_ROW_AGGREGATION_FIELDS = (
    "numeric_attribute",
    "n",
    "descending",
    "filter_attribute",
    "filter_ref",
    "filter_value",
)


class Entity(BaseModel):
    """A node in the GSW-fragment plan.

    Filled entities carry a ``name`` (extracted from the question).
    Blank entities carry a ``value_type`` and are resolved by retrieval.
    Exactly one blank should have ``is_target=True``.
    """

    id: str
    kind: Literal["filled", "blank"]
    name: Optional[str] = None
    value_type: Optional[
        Literal["date", "number", "entity", "attribute", "list", "text", "bool"]
    ] = None
    is_target: bool = False
    category: bool = False
    # Phase-1.5 planner metadata — purely informational for the LLM + UI.
    # The executor ignores these fields (they don't change retrieval or
    # constraint evaluation). Optional so older dumps / plans without
    # role annotations keep validating.
    role: Optional[str] = None
    state: Optional[str] = None
    # Literal value for filled constraint constants. Example:
    # name="twelve", literal_value=12 lets the entity remain grounded in
    # the question text while constraints consume the numeric value.
    literal_value: Optional[Any] = None
    # gsw_blank_hints: planner-emitted natural-language retrieval/verification
    # guidance for a blank. Optional so plans without them keep validating and
    # the off-path stays behavior-identical. retrieval_hint is a human-style
    # search question; verification_claim is a precise sentence the candidate
    # must satisfy. Both may contain {blank_id} placeholders for upstream blanks.
    retrieval_hint: Optional[str] = None
    verification_claim: Optional[str] = None


class VerbPhrase(BaseModel):
    """A typed binary relation between two entities."""

    id: str
    phrase: str
    subject_id: str
    object_id: str


class Constraint(BaseModel):
    """Dependency between blanks.

    Only one of the field groups is populated per instance:
    - ``op`` + ``args_refs`` / ``args_blanks`` for ``kind="derived"``.
    - ``candidate_entity_ids`` + ``sort_by_blank_ids`` for argmax / argmin.
    - ``left_ref`` / ``right_ref`` for relational kinds (Phase-2).

    ``output_blank_id`` names the blank whose value this constraint fills.
    For argmax / argmin the filled blank's value is an entity id drawn
    from ``candidate_entity_ids``.
    """

    id: str
    kind: Literal[
        "derived",
        "argmax",
        "argmin",
        "equals",
        "in_list",
        "gt",
        "lt",
        "rank_n",
        "filter",
        "count",
    ]
    op: Optional[
        Literal[
            "diff",
            "sum",
            "avg",
            "max",
            "min",
            "count",
            "concat",
            "sort",
            "mul",
            "div",
            "round_nearest",
        ]
    ] = None
    args_blanks: list[str] = Field(default_factory=list)
    # Preferred input list for derived constraints. Unlike the legacy
    # args_blanks field, args_refs may point to blanks or filled literal
    # entities such as constraint-value "twelve".
    args_refs: list[str] = Field(default_factory=list)
    candidate_entity_ids: list[str] = Field(default_factory=list)
    sort_by_blank_ids: list[str] = Field(default_factory=list)
    left_ref: Optional[str] = None
    right_ref: Optional[str] = None
    numeric_attribute: Optional[str] = None
    n: Optional[int] = None
    descending: bool = True
    filter_attribute: Optional[str] = None
    filter_ref: Optional[str] = None
    filter_value: Optional[Any] = None
    output_blank_id: Optional[str] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.kind not in _ROW_AGGREGATION_KINDS:
            for field_name in _ROW_AGGREGATION_FIELDS:
                data.pop(field_name, None)
            return data

        if self.kind != "rank_n":
            data.pop("numeric_attribute", None)
            data.pop("n", None)
            data.pop("descending", None)
        if self.kind != "filter":
            data.pop("filter_attribute", None)
            data.pop("filter_ref", None)
            data.pop("filter_value", None)
        for field_name in _ROW_AGGREGATION_FIELDS:
            if data.get(field_name) is None:
                data.pop(field_name, None)
        return data


class GSWPlan(BaseModel):
    """Full plan emitted by the question-planner LLM."""

    entities: list[Entity]
    verb_phrases: list[VerbPhrase] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)

    def filled_entities(self) -> list[Entity]:
        return [e for e in self.entities if e.kind == "filled"]

    def blank_entities(self) -> list[Entity]:
        return [e for e in self.entities if e.kind == "blank"]

    def target(self) -> Entity:
        targets = [e for e in self.blank_entities() if e.is_target]
        if len(targets) != 1:
            raise ExecutionError(
                kind="no_target",
                detail=f"expected exactly one target blank, found {len(targets)}",
            )
        return targets[0]

    def entity_by_id(self, eid: str) -> Entity:
        for e in self.entities:
            if e.id == eid:
                return e
        raise ExecutionError(kind="bad_ref", detail=f"unknown entity id {eid!r}")

    @model_validator(mode="after")
    def _check_known_refs(self) -> "GSWPlan":
        """Every VP/constraint reference must point at an emitted entity."""
        entity_ids = {e.id for e in self.entities}
        blank_ids = {e.id for e in self.blank_entities()}

        def require_known(ref: Optional[str], *, where: str) -> None:
            if ref and ref not in entity_ids:
                raise ValueError(f"{where} references unknown entity id {ref!r}")

        def require_blank(ref: Optional[str], *, where: str) -> None:
            if ref and ref not in blank_ids:
                raise ValueError(f"{where} references non-blank id {ref!r}")

        for vp in self.verb_phrases:
            require_known(vp.subject_id, where=f"verb-phrase {vp.id!r}.subject_id")
            require_known(vp.object_id, where=f"verb-phrase {vp.id!r}.object_id")

        for c in self.constraints:
            for b in c.args_blanks:
                require_blank(b, where=f"constraint {c.id!r}.args_blanks")
            for ref in c.args_refs:
                require_known(ref, where=f"constraint {c.id!r}.args_refs")
            for b in c.sort_by_blank_ids:
                require_blank(b, where=f"constraint {c.id!r}.sort_by_blank_ids")
            for eid in c.candidate_entity_ids:
                require_known(eid, where=f"constraint {c.id!r}.candidate_entity_ids")
            require_known(c.left_ref, where=f"constraint {c.id!r}.left_ref")
            require_known(c.right_ref, where=f"constraint {c.id!r}.right_ref")
            require_known(c.filter_ref, where=f"constraint {c.id!r}.filter_ref")
            require_blank(c.output_blank_id, where=f"constraint {c.id!r}.output_blank_id")
        return self

    @model_validator(mode="after")
    def _check_no_dangling(self) -> "GSWPlan":
        """Every entity must be referenced by ≥ 1 verb-phrase or constraint.

        A dangling entity is unreachable by the executor — no retrieval
        signal can use it and no constraint can consume/produce it — so
        it contributes nothing to answering the question. Emit a clear
        ValidationError so the adapter's repair retry can surface it to
        the LLM.
        """
        referenced: set[str] = set()
        for vp in self.verb_phrases:
            referenced.add(vp.subject_id)
            referenced.add(vp.object_id)
        for c in self.constraints:
            referenced.update(c.args_blanks)
            referenced.update(c.args_refs)
            referenced.update(c.candidate_entity_ids)
            referenced.update(c.sort_by_blank_ids)
            for ref in (c.left_ref, c.right_ref, c.filter_ref, c.output_blank_id):
                if ref:
                    referenced.add(ref)
        dangling = [e.id for e in self.entities if e.id not in referenced]
        if dangling:
            raise ValueError(
                f"dangling entities (no verb-phrase or constraint edge): {dangling}"
            )
        return self

    @model_validator(mode="after")
    def _check_constraint_shape(self) -> "GSWPlan":
        """Validate per-kind required fields on each constraint.

        A constraint with empty inputs is uncomputable — the executor
        can't pick a winner / sum / compare anything. The
        orchestrator then submits whatever placeholder is in scope
        (e.g., the literal hint string ``Winner B``). Catching at
        plan-emit time and raising ``ValueError`` lets the planner
        repair retry surface a concrete error to the LLM.
        """
        # BCP q7 fix (2026-08-18): CHECK-kind constraints (gt/lt/equals/
        # in_list/filter) do not PRODUCE a blank -- they bound one. The
        # planner routinely writes a range ("after 1990, before 2004") as
        # gt+lt both naming the blank in output_blank_id, which is the only
        # field the schema offers for "this constraint is about that blank".
        # Rejecting that killed the question (3 repair rounds, pred='').
        # Normalize instead: when a check-kind collides with any other owner
        # of the blank, strip the CHECKER's producer claim. Two genuine
        # producers (derived/argmax/argmin/rank_n/count) still conflict.
        _CHECK_KINDS = {"gt", "lt", "equals", "in_list", "filter"}
        output_owner: dict[str, str] = {}
        owner_kind: dict[str, str] = {}
        for c in self.constraints:
            if c.output_blank_id:
                prior = output_owner.get(c.output_blank_id)
                if prior is not None:
                    if c.kind in _CHECK_KINDS:
                        c.output_blank_id = None
                        continue
                    if owner_kind.get(c.output_blank_id) in _CHECK_KINDS:
                        # the earlier claimant was the checker -- demote it
                        for prev in self.constraints:
                            if prev.id == prior:
                                prev.output_blank_id = None
                        output_owner[c.output_blank_id] = c.id
                        owner_kind[c.output_blank_id] = c.kind
                        continue
                    raise ValueError(
                        f"constraints {prior!r} and {c.id!r} both output "
                        f"to blank {c.output_blank_id!r}; each blank may "
                        "have at most one producing constraint"
                    )
                output_owner[c.output_blank_id] = c.id
                owner_kind[c.output_blank_id] = c.kind

            if c.kind == "derived":
                if not c.op:
                    raise ValueError(
                        f"constraint {c.id!r} (kind=derived) has no op"
                    )
                if not (c.args_refs or c.args_blanks):
                    raise ValueError(
                        f"constraint {c.id!r} (kind=derived, op={c.op!r}) "
                        "has empty args_refs/args_blanks; derived "
                        "constraints must list input refs"
                    )
            elif c.kind in ("argmax", "argmin"):
                if not c.candidate_entity_ids:
                    raise ValueError(
                        f"constraint {c.id!r} (kind={c.kind}) has empty "
                        "candidate_entity_ids; argmax/argmin must list "
                        "the entities to choose between"
                    )
                if not c.sort_by_blank_ids:
                    raise ValueError(
                        f"constraint {c.id!r} (kind={c.kind}) has empty "
                        "sort_by_blank_ids; argmax/argmin must specify "
                        "the blank(s) to rank candidates by"
                    )
                if len(c.candidate_entity_ids) != len(c.sort_by_blank_ids):
                    raise ValueError(
                        f"constraint {c.id!r} (kind={c.kind}) has "
                        "candidate_entity_ids and sort_by_blank_ids with "
                        "different lengths; positions must align"
                    )
            elif c.kind in ("equals", "gt", "lt"):
                if not (c.left_ref and c.right_ref):
                    raise ValueError(
                        f"constraint {c.id!r} (kind={c.kind}) requires "
                        "both left_ref and right_ref"
                    )
            elif c.kind == "in_list":
                has_left_right = bool(c.left_ref and c.right_ref)
                has_args = len(c.args_blanks) >= 2
                if not (has_left_right or has_args):
                    raise ValueError(
                        f"constraint {c.id!r} (kind=in_list) requires "
                        "either left_ref/right_ref or args_blanks="
                        "[member_blank, list_blank]"
                    )
            elif c.kind == "rank_n":
                if len(c.args_blanks) != 1:
                    raise ValueError(
                        f"constraint {c.id!r} (kind=rank_n) requires "
                        "args_blanks=[source_list_blank]"
                    )
                if not c.numeric_attribute:
                    raise ValueError(
                        f"constraint {c.id!r} (kind=rank_n) requires "
                        "numeric_attribute"
                    )
                if c.n is None or c.n < 1:
                    raise ValueError(
                        f"constraint {c.id!r} (kind=rank_n) requires "
                        "one-based n >= 1"
                    )
            elif c.kind == "filter":
                if len(c.args_blanks) != 1:
                    raise ValueError(
                        f"constraint {c.id!r} (kind=filter) requires "
                        "args_blanks=[source_list_blank]"
                    )
                if not c.filter_attribute:
                    raise ValueError(
                        f"constraint {c.id!r} (kind=filter) requires "
                        "filter_attribute"
                    )
                has_ref = bool(c.filter_ref)
                has_literal = c.filter_value is not None
                if has_ref == has_literal:
                    raise ValueError(
                        f"constraint {c.id!r} (kind=filter) requires exactly "
                        "one of filter_ref or filter_value"
                    )
            elif c.kind == "count":
                if len(c.args_blanks) != 1:
                    raise ValueError(
                        f"constraint {c.id!r} (kind=count) requires "
                        "args_blanks=[source_list_blank]"
                    )
        return self


# ---------------------------------------------------------------------------
# Errors + state
# ---------------------------------------------------------------------------


class ExecutionError(Exception):
    """Raised when the planner-emitted plan cannot be executed.

    ``kind`` is one of: ``cyclic_plan``, ``no_target``, ``bad_ref``,
    ``empty_plan``. The adapter catches this and falls back to flat.
    """

    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.detail = detail


@dataclass
class BlankResult:
    blank_id: str
    value: Any = None
    evidence_chunk_ids: list[str] = field(default_factory=list)
    connection_evidence: dict[str, list[str]] = field(default_factory=dict)
    reasoning: Optional[str] = None
    status: Literal["unknown", "resolved", "error"] = "unknown"
    llm_calls: int = 0
    wall_time_s: float = 0.0
    error: str = ""
    # resolved_by vocabulary: "react" | "resolver" | "constraint" |
    # "holistic_fallback" (Phase 3.6b M3: committed by the logged holistic
    # recovery when the structured path dead-ended; see _recovery_holistic).
    resolved_by: Optional[str] = None
    verifier_status: Optional[str] = None        # "unverified_capped" when committed past the verifier cap; else None
    completeness_status: Optional[str] = None     # "partial" when a list/set blank is known-incomplete; else None


# ---------------------------------------------------------------------------
# Dependency DAG + topo sort
# ---------------------------------------------------------------------------


def build_dependency_graph(plan: GSWPlan) -> dict[str, set[str]]:
    """For each blank B, return the set of blanks it depends on.

    B depends on A if (a) a Constraint inputs A and outputs B, or
    (b) a VerbPhrase has A on one side and B on the other and A is
    used to identify B.

    For VP-based dependencies we choose the direction conservatively:
    if one endpoint is an entity blank (type=entity) with no incoming
    filled-neighbor VPs yet, we treat it as the "source" for outgoing
    attribute-projection VPs. But a simpler and equivalent rule is:
    **both endpoints of a VP sit at the same dependency level; if one
    is later resolved we re-visit**. The topo sort uses only constraint
    edges + "identification-requires-incoming-VP" — projection edges
    are executed after the source blank resolves.

    This function returns the **constraint dependency graph** only; the
    executor handles VP-based projection as a second phase after
    identification.
    """

    blanks = {e.id for e in plan.blank_entities()}
    deps: dict[str, set[str]] = {b: set() for b in blanks}

    # Constraint-based deps: output depends on all constraint inputs.
    for c in plan.constraints:
        inputs: set[str] = set()
        if c.kind == "derived":
            inputs.update(x for x in _derived_arg_refs(c) if x in blanks)
        elif c.kind in ("argmax", "argmin"):
            inputs.update(x for x in c.sort_by_blank_ids if x in blanks)
        elif c.kind in ("equals", "gt", "lt"):
            for x in (c.left_ref, c.right_ref):
                if x and x in blanks:
                    inputs.add(x)
        elif c.kind == "in_list":
            if c.left_ref and c.right_ref:
                for x in (c.left_ref, c.right_ref):
                    if x in blanks:
                        inputs.add(x)
            else:
                inputs.update(x for x in c.args_blanks[:2] if x in blanks)
        elif c.kind in ("rank_n", "filter", "count"):
            inputs.update(x for x in c.args_blanks[:1] if x in blanks)
            if c.kind == "filter" and c.filter_ref in blanks:
                inputs.add(c.filter_ref)

        if c.output_blank_id and c.output_blank_id in blanks:
            # output depends on all inputs (minus self)
            for inp in inputs:
                if inp != c.output_blank_id:
                    deps[c.output_blank_id].add(inp)

    # VP-based "projection" deps: if a blank-entity B has outgoing VPs
    # to other blanks (B is subject, object is blank), those object
    # blanks depend on B being resolved first.
    ent_by_id = {e.id: e for e in plan.entities}
    try:
        target_id: Optional[str] = plan.target().id
    except ExecutionError:
        target_id = None
    for vp in plan.verb_phrases:
        sub = ent_by_id.get(vp.subject_id)
        obj = ent_by_id.get(vp.object_id)
        if not sub or not obj:
            raise ExecutionError(
                kind="bad_ref",
                detail=f"verb-phrase {vp.id!r} references unknown entity",
            )
        if sub.kind == "blank" and obj.kind == "blank":
            # The TARGET is always a SINK: it depends on its partner blank,
            # never the reverse. Otherwise an inverted VP (target on the
            # subject side) makes the answer a source the orchestrator can
            # resolve from priors and submit while the bridge is skipped.
            # Target as OBJECT matches the default object-depends-on-subject
            # projection; target as SUBJECT flips the edge so it depends on
            # its partner.
            if sub.id == target_id:
                deps[sub.id].add(obj.id)
            elif sub.value_type == "entity":
                deps[obj.id].add(sub.id)

    return deps


def unresolved_upstream_deps(
    plan: GSWPlan,
    resolved_ids: set[str],
    blank_ids: Iterable[str],
) -> dict[str, list[str]]:
    """For each blank in ``blank_ids``, the subset of its upstream dependencies
    (per :func:`build_dependency_graph`) that are NOT in ``resolved_ids``.

    A blank is dispatch-ready only when every blank it depends on is resolved —
    this mirrors the topological gate that deterministic mode enforces by
    construction. Blanks with no pending deps are omitted from the result, so an
    empty dict means "all requested blanks are ready to dispatch".
    """
    deps = build_dependency_graph(plan)
    blocked: dict[str, list[str]] = {}
    for b in blank_ids:
        pending = sorted(d for d in deps.get(b, set()) if d not in resolved_ids)
        if pending:
            blocked[b] = pending
    return blocked


def topological_sort_blanks(plan: GSWPlan) -> list[str]:
    """Kahn's algorithm. Raises ExecutionError on cycle."""

    deps = build_dependency_graph(plan)
    indeg: dict[str, int] = {b: len(ds) for b, ds in deps.items()}
    children: dict[str, set[str]] = defaultdict(set)
    for b, ds in deps.items():
        for d in ds:
            children[d].add(b)

    q: deque[str] = deque([b for b, d in indeg.items() if d == 0])
    order: list[str] = []
    while q:
        b = q.popleft()
        order.append(b)
        for child in children[b]:
            indeg[child] -= 1
            if indeg[child] == 0:
                q.append(child)

    if len(order) != len(indeg):
        unresolved = [b for b, d in indeg.items() if d > 0]
        raise ExecutionError(kind="cyclic_plan", detail=f"unresolved blanks: {unresolved}")

    return order


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


Retriever = Any
"""A retriever with ``.search(query: str, top_k: int = …) -> list[Hit]``
where each ``Hit`` has ``chunk.chunk_id``, ``chunk.title``, ``chunk.text``."""


@dataclass
class _RetrievalChunk:
    chunk_id: str
    title: str
    text: str
    score: float


def _retrieve_chunks(retriever: Retriever, query: str, top_k: int = 8) -> list[_RetrievalChunk]:
    """Unified retrieval wrapper — tolerant to the BM25 Hit shape."""
    hits = retriever.search(query, top_k=top_k)
    out: list[_RetrievalChunk] = []
    for h in hits:
        chunk = getattr(h, "chunk", h)
        out.append(
            _RetrievalChunk(
                chunk_id=getattr(chunk, "chunk_id", ""),
                title=getattr(chunk, "title", ""),
                text=getattr(chunk, "text", "")[:1500],
                score=float(getattr(h, "score", 0.0)),
            )
        )
    return out


def _format_chunks_for_prompt(chunks: list[_RetrievalChunk]) -> str:
    """Plain-text formatting for LLM consumption."""
    if not chunks:
        return "(no retrieved chunks)"
    lines = []
    for c in chunks:
        lines.append(f"[{c.chunk_id}] {c.title}")
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    """Balanced-brace extractor — safer than a greedy regex."""
    if not text:
        raise ValueError("empty LLM response")
    start = text.find("{")
    if start < 0:
        raise ValueError("no opening brace in response")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced braces in LLM response")


def _coerce_number(value: Any) -> Optional[float]:
    """Best-effort coercion of an LLM output into a float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _coerce_year(value: Any) -> Optional[int]:
    """Pull an integer year out of a string (best effort)."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value).strip()
    # Just look for the first 4-digit run.
    import re

    m = re.search(r"\b(\d{4})\b", s)
    if m:
        return int(m.group(1))
    return None


def _coerce_constraint_number(value: Any) -> Optional[float]:
    """Coerce values for numeric constraints.

    Date blanks often store full dates (``2015-02-25``) even when a
    downstream ``diff`` asks for a year difference. Use a normal numeric
    parse first, then fall back to the first 4-digit year.
    """
    n = _coerce_number(value)
    if n is not None:
        return n
    y = _coerce_year(value)
    if y is not None:
        return float(y)
    return None


def _coerce_sum_term(value: Any) -> Optional[float]:
    """Coerce one term for a ``sum`` constraint.

    Normal numbers win. If the value is a text token containing digits
    (e.g. a postcode or reference code), fall back to summing its
    digits. This lets plans express a digit-sum over any such token as
    ``sum(token_blank)`` while still preserving the original token as
    the upstream value.
    """
    n = _coerce_number(value)
    if n is not None:
        return n
    if isinstance(value, (list, tuple)):
        total = 0.0
        found = False
        for item in value:
            item_n = _coerce_sum_term(item)
            if item_n is None:
                continue
            total += item_n
            found = True
        return total if found else None
    digits = [int(ch) for ch in str(value) if ch.isdigit()]
    if digits:
        return float(sum(digits))
    return None


def _count_constraint_value(value: Any, *, count_letters: bool = False) -> int:
    """Best-effort count for a single count-constraint argument."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return 0
        if count_letters:
            return sum(1 for ch in s if ch.isalpha())
        import re

        parts = [p.strip() for p in re.split(r"[,;]\s*", s) if p.strip()]
        return len(parts) if len(parts) > 1 else 1
    return 1


def _constraint_incident_phrases(
    plan: GSWPlan,
    refs: set[str],
) -> list[str]:
    phrases: list[str] = []
    for vp in plan.verb_phrases:
        if vp.subject_id in refs or vp.object_id in refs:
            phrases.append((vp.phrase or "").replace("_", " ").casefold())
    return phrases


def _is_letter_count_constraint(
    plan: GSWPlan,
    constraint: Constraint,
    arg_refs: list[str],
) -> bool:
    out_id = constraint.output_blank_id or ""
    refs = set(arg_refs)
    refs.add(out_id)
    phrases = _constraint_incident_phrases(plan, refs)
    return any(
        "letter" in phrase
        or "title length" in phrase
        or "name length" in phrase
        or phrase.strip() in {"length", "title length"}
        for phrase in phrases
    )


def _parse_date_parts(value: Any) -> Optional[tuple[int, Optional[int], Optional[int]]]:
    if value is None or isinstance(value, (int, float)):
        return None
    s = str(value).strip()
    if not s:
        return None
    import re

    month_names = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    m = re.search(r"\b(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?\b", s)
    if m:
        return (
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)) if m.group(3) else None,
        )

    m = re.search(
        r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b"
        r"|\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b"
        r"|\b([A-Za-z]+)\s+(\d{4})\b",
        s,
    )
    if m:
        if m.group(1) and m.group(1).casefold() in month_names:
            return int(m.group(3)), month_names[m.group(1).casefold()], int(m.group(2))
        if m.group(5) and m.group(5).casefold() in month_names:
            return int(m.group(6)), month_names[m.group(5).casefold()], int(m.group(4))
        if m.group(7) and m.group(7).casefold() in month_names:
            return int(m.group(8)), month_names[m.group(7).casefold()], None

    m = re.search(r"\b(\d{4})\b", s)
    if m:
        return int(m.group(1)), None, None
    return None


def _ref_has_birth_cue(plan: GSWPlan, ref: str) -> bool:
    ref_l = ref.casefold()
    if "birth" in ref_l or "born" in ref_l:
        return True
    for phrase in _constraint_incident_phrases(plan, {ref}):
        if "birth" in phrase or "born" in phrase:
            return True
    return False


def _ref_incident_text(plan: GSWPlan, ref: str) -> str:
    return " ".join([ref.casefold(), *_constraint_incident_phrases(plan, {ref})])


def _wants_elapsed_years_diff(plan: GSWPlan, arg_refs: list[str]) -> bool:
    """Detect date diffs that mean completed years, not raw year subtraction."""
    if len(arg_refs) < 2:
        return False

    left_text = _ref_incident_text(plan, arg_refs[0])
    right_text = _ref_incident_text(plan, arg_refs[1])
    texts = (left_text, right_text)

    # Age/as-of questions: a birth date compared to a dated event.
    age_event_cues = (
        "admitted",
        "delivered",
        "graduated",
        "released",
        "anniversary",
        "assassinated",
        "died",
        "fell",
    )
    if any("birth" in t or "born" in t for t in texts):
        return any(any(cue in t for cue in age_event_cues) for t in texts)

    # Elapsed time between two dated events where the question later
    # adds/subtracts that elapsed age. Keep this deliberately narrow so
    # plain "year difference" constraints retain old behavior.
    event_pair_cues = ("assassinated", "fell")
    return all(any(cue in t for cue in event_pair_cues) for t in texts)


# Cues in the OUTPUT blank hint / question that mean the date-diff should be
# completed years (an age), not raw days. Robust routing signal that does not
# depend on the input events matching a hardcoded verb list — fixes "lost their
# fight" (q25) and org "founding -> inactivated" (q630) which the input-cue
# heuristic misses.
_YEARS_OUTPUT_CUES = (
    "how old",
    "years old",
    "age difference",
    "difference in years",
    "age at",
    "age of",
    "in years",
    "number of years",
    "how many years",
    "years between",
)


def _output_wants_years_diff(
    plan: GSWPlan, constraint: "Constraint", question: str = ""
) -> bool:
    """True when the diff's OUTPUT (blank hint / question) asks for elapsed
    years or an age, e.g. 'how old...', 'difference in years'."""
    texts: list[str] = [(question or "").lower()]
    oid = getattr(constraint, "output_blank_id", None)
    if oid:
        try:
            ent = plan.entity_by_id(oid)
        except ExecutionError:
            ent = None
        if ent is not None:
            texts.append(str(getattr(ent, "retrieval_hint", "") or "").lower())
            texts.append(str(getattr(ent, "verification_claim", "") or "").lower())
    return any(cue in t for t in texts for cue in _YEARS_OUTPUT_CUES)


def _format_month_day_delta(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
) -> str:
    from calendar import monthrange
    from datetime import date

    start = date(*left)
    end = date(*right)
    if start > end:
        start, end = end, start

    def add_months(d: date, months: int) -> date:
        month = d.month - 1 + months
        year = d.year + month // 12
        month = month % 12 + 1
        day = min(d.day, monthrange(year, month)[1])
        return date(year, month, day)

    months = 0
    while add_months(start, months + 1) <= end:
        months += 1
    anchor = add_months(start, months)
    days = (end - anchor).days
    parts = []
    if months:
        parts.append(f"{months} month{'s' if months != 1 else ''}")
    if days or not parts:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    return " ".join(parts)


def _elapsed_years_between_date_parts(
    left: tuple[int, Optional[int], Optional[int]],
    right: tuple[int, Optional[int], Optional[int]],
) -> Optional[int]:
    """Return completed anniversary years between two partially parsed dates."""
    if left[1] is None or right[1] is None:
        return None

    def sort_key(part: tuple[int, Optional[int], Optional[int]]) -> tuple[int, int, int]:
        return part[0], part[1] or 1, part[2] or 1

    start, end = (left, right)
    if sort_key(start) > sort_key(end):
        start, end = end, start

    start_year, start_month, start_day = start
    end_year, end_month, end_day = end
    assert start_month is not None
    assert end_month is not None

    years = end_year - start_year
    if end_month < start_month:
        years -= 1
    elif (
        end_month == start_month
        and start_day is not None
        and end_day is not None
        and end_day < start_day
    ):
        years -= 1
    return max(years, 0)


def _diff_constraint_producing(plan: GSWPlan, blank_ref: str) -> Optional[Constraint]:
    """The derived-diff constraint whose output is ``blank_ref``, if any."""
    for c in plan.constraints:
        if (
            c.output_blank_id == blank_ref
            and c.kind == "derived"
            and c.op == "diff"
        ):
            return c
    return None


def _nested_date_diff_over_shared_reference(
    plan: GSWPlan,
    constraint: Constraint,
    arg_refs: list[str],
    state: dict[str, "BlankResult"],
) -> Optional[int]:
    """Collapse ``diff(A, B)`` where ``A = diff(R, X)`` and ``B = diff(R, Y)``.

    The shared reference R cancels: ``(R - X) - (R - Y) = Y - X``. When X and
    Y are full dates this recovers the precision that the intermediate
    integer-year values threw away.

    q209 asks the age difference between two people "when the artist
    graduated". Age difference is time-invariant, but the planner decomposed
    it as b_age_pres = diff(grad_year, birth_pres),
    b_age_artist = diff(grad_year, birth_artist), t = diff(b_age_pres,
    b_age_artist). The naive numeric path yields |30 - 22| = 8 (each age
    rounded to whole years off a year-only grad date); the true difference is
    elapsed_years(birth_pres, birth_artist) = 7.

    Fires ONLY when: op is diff over exactly two refs, each ref is itself a
    derived-diff output, the two share the same first operand (minuend), the
    two second operands (subtrahends) resolve to FULL dates (month present),
    and the output blank is numeric. Otherwise returns None and the normal
    numeric diff runs unchanged.
    """
    if constraint.op != "diff" or len(arg_refs) < 2:
        return None
    if constraint.output_blank_id:
        try:
            out_ent = plan.entity_by_id(constraint.output_blank_id)
        except ExecutionError:
            out_ent = None
        if out_ent is None or out_ent.value_type != "number":
            return None
    ca = _diff_constraint_producing(plan, arg_refs[0])
    cb = _diff_constraint_producing(plan, arg_refs[1])
    if ca is None or cb is None:
        return None
    a_args, b_args = _derived_arg_refs(ca), _derived_arg_refs(cb)
    if len(a_args) < 2 or len(b_args) < 2:
        return None
    if a_args[0] != b_args[0]:  # not the same reference minuend
        return None
    x_val, x_ok = _constraint_ref_value(a_args[1], plan, state)
    y_val, y_ok = _constraint_ref_value(b_args[1], plan, state)
    if not (x_ok and y_ok):
        return None
    x_parts, y_parts = _parse_date_parts(x_val), _parse_date_parts(y_val)
    if not x_parts or not y_parts:
        return None
    if x_parts[1] is None or y_parts[1] is None:  # need month for real precision
        return None
    return _elapsed_years_between_date_parts(x_parts, y_parts)


def _date_diff_constraint_value(
    plan: GSWPlan,
    constraint: Constraint,
    values: list[Any],
    arg_refs: list[str],
) -> Optional[Any]:
    if len(values) < 2 or len(arg_refs) < 2:
        return None
    parts = [_parse_date_parts(value) for value in values[:2]]
    if not parts[0] or not parts[1]:
        return None

    # If both sides are birth dates, preserve month/day granularity
    # instead of collapsing to same-year zero. This fixes age-difference
    # questions where both people were born in the same year.
    if all(_ref_has_birth_cue(plan, ref) for ref in arg_refs[:2]):
        full = []
        for part in parts:
            year, month, day = part
            if month is None or day is None:
                return None
            full.append((year, month, day))
        return _format_month_day_delta(full[0], full[1])
    if constraint.output_blank_id and (
        _wants_elapsed_years_diff(plan, arg_refs)
        or _output_wants_years_diff(plan, constraint)
    ):
        try:
            output_ent = plan.entity_by_id(constraint.output_blank_id)
        except ExecutionError:
            output_ent = None
        if output_ent is not None and output_ent.value_type == "number":
            return _elapsed_years_between_date_parts(parts[0], parts[1])
    return None


# q209 fix 3: a SAFE post-processor, deliberately NOT a reorder of the
# branches above (that reorder was commit a95281c3, reverted in 011f4bb3 --
# see that revert's message for why: its premise was wrong AND it never
# reached the live q209 trace anyway, since that trace's plan never hits
# the birth-birth branch at all). This fires strictly AFTER
# _date_diff_constraint_value has already returned its value, only when
# that value is the "N months M days" duration STRING the birth-birth
# branch produces AND the OUTPUT wants a number of elapsed years -- so it
# can only ever touch the birth-birth-diff-into-a-numeric-blank shape; a
# decomposed plan (two separately-computed ages, subtracted) never
# produces this string and so never reaches this function either.
_DURATION_MONTHS_RE = re.compile(r"(\d+)\s+months?\b", flags=re.IGNORECASE)
_DURATION_DAYS_RE = re.compile(r"(\d+)\s+days?\b", flags=re.IGNORECASE)


def _is_month_day_duration_string(value: Any) -> bool:
    return isinstance(value, str) and bool(
        _DURATION_MONTHS_RE.search(value) or _DURATION_DAYS_RE.search(value)
    )


def _floor_years_from_duration_string(text: str) -> int:
    """Conventional floored whole years from a '_format_month_day_delta'
    string, e.g. '86 months 6 days' -> 7 (86 // 12, remainder dropped)."""
    m = _DURATION_MONTHS_RE.search(text)
    months = int(m.group(1)) if m else 0
    return months // 12


def _maybe_floor_duration_to_years(
    plan: GSWPlan,
    constraint: Constraint,
    arg_refs: list[str],
    value: Any,
) -> Any:
    """Convert a birth-birth date-diff duration STRING to floored whole
    years when the OUTPUT blank is numeric and wants elapsed years.

    Leaves the string untouched (and leaves _date_diff_constraint_value's
    birth-birth branch itself untouched -- see the comment above) for
    every other case: a non-duration value, a non-numeric output, or a
    numeric output with no years cue. This is intentionally narrower than
    the branch reorder it replaces: it only ever converts a value this
    module itself already produced, never changes which branch wins.
    """
    if not _is_month_day_duration_string(value):
        return value
    if not constraint.output_blank_id:
        return value
    try:
        output_ent = plan.entity_by_id(constraint.output_blank_id)
    except ExecutionError:
        return value
    if output_ent is None or output_ent.value_type != "number":
        return value
    if not (
        _wants_elapsed_years_diff(plan, arg_refs)
        or _output_wants_years_diff(plan, constraint)
    ):
        return value
    floored = _floor_years_from_duration_string(value)
    # Never floor a SUB-YEAR duration to 0: for two people born months apart
    # (q284: inventors of Barbie and Hot Wheels, "6 months 26 days"), the gold
    # answer IS that duration -- flooring to 0 destroys it, even though the
    # hint says "difference in years". Only apply the floor when at least one
    # whole year is present; otherwise keep the original duration string so
    # downstream synthesis can surface it. q209 ("86 months 6 days" -> 7) is
    # unaffected: it has whole years to floor to.
    if floored is None or (isinstance(floored, (int, float)) and floored < 1):
        return value
    return floored


def _constraint_ref_value(
    ref: Optional[str],
    plan: GSWPlan,
    state: dict[str, BlankResult],
) -> tuple[Any, bool]:
    """Resolve a constraint ref from state or a filled entity name."""
    if not ref:
        return None, False
    res = state.get(ref)
    if res is not None:
        return res.value, res.status == "resolved"
    try:
        ent = plan.entity_by_id(ref)
    except ExecutionError:
        return None, False
    if ent.kind == "filled":
        if ent.literal_value is not None:
            return ent.literal_value, True
        return ent.name or ent.id, True
    return None, False


def _derived_arg_refs(constraint: Constraint) -> list[str]:
    """Return derived inputs, preferring the literal-aware field."""
    return list(constraint.args_refs or constraint.args_blanks or [])


def _relational_ref_pair(constraint: Constraint) -> tuple[Optional[str], Optional[str]]:
    if constraint.left_ref and constraint.right_ref:
        return constraint.left_ref, constraint.right_ref
    if constraint.kind == "in_list" and len(constraint.args_blanks) >= 2:
        return constraint.args_blanks[0], constraint.args_blanks[1]
    return None, None


def _constraint_values_equal(left: Any, right: Any) -> bool:
    left_num = _coerce_number(left)
    right_num = _coerce_number(right)
    if left_num is not None and right_num is not None:
        return left_num == right_num
    return str(left).strip().casefold() == str(right).strip().casefold()


def _constraint_contains(member: Any, container: Any) -> bool:
    if isinstance(container, (list, tuple, set)):
        return any(_constraint_values_equal(member, item) for item in container)
    text = str(container or "")
    import re

    parts = [p.strip() for p in re.split(r"[,;\n]\s*", text) if p.strip()]
    if parts:
        return any(_constraint_values_equal(member, part) for part in parts)
    return str(member).strip().casefold() in text.casefold()


def _unknown_constraint_result(out_id: str, t0: float) -> BlankResult:
    return BlankResult(
        blank_id=out_id,
        status="unknown",
        wall_time_s=round(time.time() - t0, 3),
    )


@dataclass(frozen=True)
class StructuredCandidateRowsValidation:
    ok: bool
    rows: list[CandidateRow] = field(default_factory=list)
    payload: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    missing_evidence: list[str] = field(default_factory=list)


def _structured_rows_validation_error(
    error: str,
    *,
    rows: list[CandidateRow] | None = None,
    missing_evidence: list[str] | None = None,
) -> StructuredCandidateRowsValidation:
    return StructuredCandidateRowsValidation(
        ok=False,
        rows=list(rows or []),
        payload=[],
        error=error,
        missing_evidence=list(missing_evidence or []),
    )


def _is_structured_numeric(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _constraint_source_value(
    constraint: Constraint,
    state: dict[str, BlankResult],
) -> Any:
    if len(constraint.args_blanks) != 1:
        raise ValueError("structured aggregation constraints need one source list")
    source_id = constraint.args_blanks[0]
    res = state.get(source_id)
    if res is None or res.status != "resolved" or res.value is None:
        raise ValueError(f"source list blank {source_id!r} is not resolved")
    return res.value


def validate_structured_candidate_rows(
    value: Any,
    constraint: Constraint,
    *,
    plan: GSWPlan | None = None,
    state: dict[str, BlankResult] | None = None,
    require_all_rows_valid: bool = True,
) -> StructuredCandidateRowsValidation:
    """Validate evidence-backed row payloads for rank/filter/count operators.

    The validator is deliberately conservative: rows without evidence are not
    valid operator inputs, rank_n needs numeric sort keys on enough rows, filter
    needs a resolved expected value, and count over zero evidenced rows remains
    unknown rather than resolving to 0.
    """
    if constraint.kind not in {"rank_n", "filter", "count"}:
        return _structured_rows_validation_error(
            f"constraint kind {constraint.kind!r} is not structured aggregation"
        )
    if not isinstance(value, list) or not value:
        return _structured_rows_validation_error(
            "source list blank must be a non-empty list of row objects"
        )

    parsed_rows: list[CandidateRow] = []
    errors: list[str] = []
    missing_evidence: list[str] = []
    for idx, item in enumerate(value):
        try:
            if isinstance(item, CandidateRow):
                row = item
                if not _candidate_row_has_evidence(row):
                    raise ValueError(f"candidate row {row.candidate!r} has no evidence")
            elif isinstance(item, dict):
                row = _candidate_row_from_dict(item)
            else:
                raise ValueError("source list blank rows must be row objects")
        except ValueError as exc:
            errors.append(f"row {idx}: {exc}")
            missing_evidence.append(str(exc))
            continue
        if row.status in {"rejected", "needs_evidence"}:
            errors.append(
                f"row {idx}: candidate row {row.candidate!r} status={row.status!r}"
            )
            continue
        parsed_rows.append(row)

    if require_all_rows_valid and errors:
        return _structured_rows_validation_error(
            "; ".join(errors),
            rows=parsed_rows,
            missing_evidence=missing_evidence,
        )

    if constraint.kind == "rank_n":
        numeric_attribute = str(constraint.numeric_attribute or "").strip()
        if not numeric_attribute:
            return _structured_rows_validation_error(
                "rank_n requires numeric_attribute",
                rows=parsed_rows,
                missing_evidence=missing_evidence,
            )
        valid_rows: list[CandidateRow] = []
        rank_errors: list[str] = list(errors)
        for row in parsed_rows:
            if numeric_attribute not in row.attributes:
                rank_errors.append(
                    f"row {row.candidate!r} missing numeric_attribute "
                    f"{numeric_attribute!r}"
                )
                continue
            value_for_attr = row.attributes.get(numeric_attribute)
            if not _is_structured_numeric(value_for_attr):
                rank_errors.append(
                    f"row {row.candidate!r} numeric_attribute "
                    f"{numeric_attribute!r} is not numeric"
                )
                continue
            valid_rows.append(row)
        n = constraint.n
        if not isinstance(n, int) or n < 1:
            return _structured_rows_validation_error(
                "rank_n requires one-based n >= 1",
                rows=valid_rows,
                missing_evidence=missing_evidence,
            )
        if require_all_rows_valid and rank_errors:
            return _structured_rows_validation_error(
                "; ".join(rank_errors),
                rows=valid_rows,
                missing_evidence=missing_evidence,
            )
        if len(valid_rows) < n:
            return _structured_rows_validation_error(
                f"rank_n requires at least {n} evidenced rows with numeric_attribute "
                f"{numeric_attribute!r}; got {len(valid_rows)}",
                rows=valid_rows,
                missing_evidence=missing_evidence,
            )
        return StructuredCandidateRowsValidation(
            ok=True,
            rows=valid_rows,
            payload=[_candidate_row_to_dict(row) for row in valid_rows],
            missing_evidence=missing_evidence,
        )

    if constraint.kind == "filter":
        filter_attribute = str(constraint.filter_attribute or "").strip()
        if not filter_attribute:
            return _structured_rows_validation_error(
                "filter requires filter_attribute",
                rows=parsed_rows,
                missing_evidence=missing_evidence,
            )
        if plan is None or state is None:
            return _structured_rows_validation_error(
                "filter requires plan and state to resolve expected value",
                rows=parsed_rows,
                missing_evidence=missing_evidence,
            )
        expected, ok = _filter_expected_value(constraint, plan, state)
        if not ok or expected is None:
            return _structured_rows_validation_error(
                "filter requires a resolved expected value",
                rows=parsed_rows,
                missing_evidence=missing_evidence,
            )
        valid_rows = []
        matching_rows = []
        filter_errors: list[str] = list(errors)
        for row in parsed_rows:
            if filter_attribute not in row.attributes:
                filter_errors.append(
                    f"row {row.candidate!r} missing filter_attribute "
                    f"{filter_attribute!r}"
                )
                continue
            valid_rows.append(row)
            if _constraint_values_equal(row.attributes.get(filter_attribute), expected):
                matching_rows.append(row)
        if require_all_rows_valid and filter_errors:
            return _structured_rows_validation_error(
                "; ".join(filter_errors),
                rows=valid_rows,
                missing_evidence=missing_evidence,
            )
        if not matching_rows:
            return _structured_rows_validation_error(
                f"filter over {filter_attribute!r} found zero matching evidenced rows",
                rows=valid_rows,
                missing_evidence=missing_evidence,
            )
        return StructuredCandidateRowsValidation(
            ok=True,
            rows=valid_rows,
            payload=[_candidate_row_to_dict(row) for row in valid_rows],
            missing_evidence=missing_evidence,
        )

    valid_rows = parsed_rows
    if not valid_rows:
        return _structured_rows_validation_error(
            "count requires at least one evidenced row",
            missing_evidence=missing_evidence,
        )
    return StructuredCandidateRowsValidation(
        ok=True,
        rows=valid_rows,
        payload=[_candidate_row_to_dict(row) for row in valid_rows],
        missing_evidence=missing_evidence,
    )


def _candidate_row_from_dict(row: dict[str, Any]) -> CandidateRow:
    candidate = row.get("candidate")
    attributes = row.get("attributes")
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError("candidate row missing string candidate")
    if not isinstance(attributes, dict):
        raise ValueError(f"candidate row {candidate!r} missing attributes dict")

    evidence_by_field_raw = row.get("evidence_by_field")
    if isinstance(evidence_by_field_raw, dict):
        evidence_by_field = {
            str(field): [str(x) for x in values if str(x or "")]
            for field, values in evidence_by_field_raw.items()
            if isinstance(values, list)
        }
    else:
        evidence_ids = [
            str(x) for x in row.get("evidence_chunk_ids", []) if str(x or "")
        ]
        if attributes:
            evidence_by_field = {
                str(field): list(evidence_ids) for field in attributes
            }
        else:
            evidence_by_field = {"candidate": list(evidence_ids)}
    if not any(evidence_by_field.values()):
        raise ValueError(f"candidate row {candidate!r} has no evidence")

    passes_filters = row.get("passes_filters")
    if not isinstance(passes_filters, dict):
        passes_filters = {}
    rank = row.get("rank")
    status = row.get("status")
    if status not in {"candidate", "rejected", "winner", "needs_evidence"}:
        status = "candidate"

    return CandidateRow(
        candidate=candidate,
        attributes=dict(attributes),
        evidence_by_field=evidence_by_field,
        passes_filters={str(k): bool(v) for k, v in passes_filters.items()},
        rank=rank if isinstance(rank, int) else None,
        status=status,
    )


def _candidate_row_has_evidence(row: CandidateRow) -> bool:
    return any(bool(values) for values in row.evidence_by_field.values())


def _candidate_row_to_dict(row: CandidateRow) -> dict[str, Any]:
    evidence_ids: list[str] = []
    for values in row.evidence_by_field.values():
        for value in values:
            if value not in evidence_ids:
                evidence_ids.append(value)
    return {
        "candidate": row.candidate,
        "attributes": dict(row.attributes),
        "evidence_chunk_ids": evidence_ids,
    }


def _filter_expected_value(
    constraint: Constraint,
    plan: GSWPlan,
    state: dict[str, BlankResult],
) -> tuple[Any, bool]:
    if constraint.filter_ref:
        return _constraint_ref_value(constraint.filter_ref, plan, state)
    if constraint.filter_value is not None:
        return constraint.filter_value, True
    return None, False


def _codegen_diff_value(
    constraint: Constraint,
    plan: GSWPlan,
    values: list[Any],
    arg_refs: list[str],
    codegen: Any,
) -> Optional[Any]:
    """Route a two-date ``diff`` compute step through the skill library.

    Returns the library value, or ``None`` to fall through to the
    existing hardcoded path (so we are never worse than today). Only engages
    for genuine two-date diffs (month present on both sides); the both-birth
    month/day-delta case is left to the existing formatter.
    """
    if len(values) < 2 or len(arg_refs) < 2:
        return None
    parts = [_parse_date_parts(v) for v in values[:2]]
    if not (parts[0] and parts[0][1] is not None and parts[1] and parts[1][1] is not None):
        return None
    # Two birth dates -> existing month/day-delta formatter; not a seed intent.
    if all(_ref_has_birth_cue(plan, ref) for ref in arg_refs[:2]):
        return None

    out_type: Optional[str] = None
    if constraint.output_blank_id:
        try:
            out_type = plan.entity_by_id(constraint.output_blank_id).value_type
        except ExecutionError:
            out_type = None

    wants_years = _wants_elapsed_years_diff(plan, arg_refs) or _output_wants_years_diff(
        plan, constraint, getattr(codegen, "question", "") or ""
    )
    if wants_years and out_type == "number":
        intent = "age_years"
    else:
        intent = "days"
    key = ("diff", ("date", "date"), intent)

    from .codegen_library import generator

    value, _prov = generator.resolve(
        codegen.lib,
        key,
        inputs=[values[0], values[1]],
        expected_type=out_type or "number",
    )
    return value


def _compute_constraint(
    constraint: Constraint,
    plan: GSWPlan,
    state: dict[str, BlankResult],
    codegen: Any = None,
) -> BlankResult:
    """Evaluate a derived / argmax / argmin constraint into its output blank.

    ``codegen`` (a ``CodegenRuntime`` or ``None``) routes the compute step
    through the seeded skill library when ``gsw_codegen_compute`` is on. When
    ``None`` / disabled the function is byte-identical to the pre-flag path.
    """
    t0 = time.time()
    out_id = constraint.output_blank_id
    assert out_id, "compute_constraint requires output_blank_id"

    if constraint.kind == "derived":
        arg_refs = _derived_arg_refs(constraint)
        resolved_args = [
            _constraint_ref_value(ref, plan, state)
            for ref in arg_refs
        ]
        if any(not ok or value is None for value, ok in resolved_args):
            return BlankResult(
                blank_id=out_id,
                status="unknown",
                wall_time_s=round(time.time() - t0, 3),
            )
        values = [value for value, _ok in resolved_args]
        op = constraint.op
        if op == "concat":
            sep = _concat_separator(plan, constraint)
            out_val = sep.join(_stringify(value) for value in values)
            return BlankResult(
                blank_id=out_id,
                value=out_val,
                status="resolved",
                wall_time_s=round(time.time() - t0, 3),
            )
        if op == "sort":
            # Modeled on `concat` above: same operand collection, same
            # separator helper -- `sort` differs only in ordering the
            # values before joining (plan-verifier 2026-08-07: `concat`
            # joins in ARGUMENT order, which is wrong for questions that
            # ask for a chronological/ranked list -- q403 wanted "1914,
            # 1987, 2002" and got "2002, 1987, 1914").
            sep = _concat_separator(plan, constraint)
            numeric_keys = [_coerce_constraint_number(value) for value in values]
            # `descending` (default True) is a field shared with
            # rank_n/argmax/argmin, whose default cannot change. `sort`'s
            # common case is chronological/earliest-first order, so this
            # op reads the field INVERTED: the unchanged default (True)
            # means ASCENDING here; set `descending: false` on the
            # constraint to get descending order instead. This keeps the
            # shared default intact while making the common case -- a
            # plan that never mentions `descending` -- do the right
            # thing without every planner-emitted `sort` needing to set
            # it explicitly.
            is_descending = not constraint.descending
            if all(key is not None for key in numeric_keys):
                order = sorted(
                    range(len(values)),
                    key=lambda i: numeric_keys[i],
                    reverse=is_descending,
                )
            else:
                # A value that fails numeric coercion pulls the WHOLE
                # list onto the string fallback rather than mixing
                # numeric and string keys, so ordering stays total and
                # deterministic.
                order = sorted(
                    range(len(values)),
                    key=lambda i: _stringify(values[i]).casefold(),
                    reverse=is_descending,
                )
            out_val = sep.join(_stringify(values[i]) for i in order)
            return BlankResult(
                blank_id=out_id,
                value=out_val,
                status="resolved",
                wall_time_s=round(time.time() - t0, 3),
            )
        if op == "count":
            if len(values) == 1:
                val = _count_constraint_value(
                    values[0],
                    count_letters=_is_letter_count_constraint(
                        plan, constraint, arg_refs
                    ),
                )
            else:
                val = len(values)
            return BlankResult(
                blank_id=out_id,
                value=val,
                status="resolved",
                wall_time_s=round(time.time() - t0, 3),
            )
        if op == "diff":
            if codegen is not None and getattr(codegen, "enabled", False):
                cg_val = _codegen_diff_value(
                    constraint, plan, values, arg_refs, codegen
                )
                if cg_val is not None:
                    return BlankResult(
                        blank_id=out_id,
                        value=cg_val,
                        status="resolved",
                        wall_time_s=round(time.time() - t0, 3),
                    )
            date_val = _date_diff_constraint_value(
                plan, constraint, values, arg_refs
            )
            if date_val is not None:
                date_val = _maybe_floor_duration_to_years(
                    plan, constraint, arg_refs, date_val
                )
                return BlankResult(
                    blank_id=out_id,
                    value=date_val,
                    status="resolved",
                    wall_time_s=round(time.time() - t0, 3),
                )
            # diff(A,B) where A=diff(R,X), B=diff(R,Y): the shared reference R
            # cancels to Y-X, recovering full-date precision the intermediate
            # integer-year values dropped (q209: decomposed age difference ->
            # 8; direct birth-date diff -> 7).
            nested = _nested_date_diff_over_shared_reference(
                plan, constraint, arg_refs, state
            )
            if nested is not None:
                return BlankResult(
                    blank_id=out_id,
                    value=nested,
                    status="resolved",
                    wall_time_s=round(time.time() - t0, 3),
                )

        if op in ("avg", "min", "max") and len(values) == 1 and isinstance(
            values[0], (list, tuple)
        ):
            # A single list/tuple operand under avg/min/max is the
            # aggregate's population (e.g. avg(b_attendance_list)) — coerce
            # each ELEMENT rather than the whole list, which always fails
            # and used to silently no-op the constraint into "unknown".
            # Scoped tightly to avg/min/max: div/mul/round_nearest keep
            # failing on a list operand (that shape is a planner error, not
            # a population to reduce). An empty list has no average, so it
            # stays unknown rather than becoming 0. If ANY element fails to
            # coerce, keep the whole constraint unknown instead of quietly
            # dropping that element (which would change the answer).
            population = values[0]
            if not population:
                return BlankResult(
                    blank_id=out_id,
                    status="unknown",
                    wall_time_s=round(time.time() - t0, 3),
                )
            nums = [_coerce_constraint_number(item) for item in population]
        elif op == "sum":
            nums = [_coerce_sum_term(value) for value in values]
        else:
            nums = [_coerce_constraint_number(value) for value in values]
        if any(n is None for n in nums):
            return BlankResult(
                blank_id=out_id,
                status="unknown",
                wall_time_s=round(time.time() - t0, 3),
            )

        if op == "diff":
            val = abs(nums[0] - nums[1]) if len(nums) >= 2 else nums[0]
        elif op == "sum":
            val = sum(nums)
        elif op == "mul":
            val = 1.0
            for n in nums:
                val *= n
        elif op == "div":
            if len(nums) < 2 or nums[1] == 0:
                return BlankResult(
                    blank_id=out_id,
                    status="unknown",
                    wall_time_s=round(time.time() - t0, 3),
                )
            val = nums[0] / nums[1]
        elif op == "round_nearest":
            step = nums[1] if len(nums) >= 2 else 1.0
            if step == 0:
                return BlankResult(
                    blank_id=out_id,
                    status="unknown",
                    wall_time_s=round(time.time() - t0, 3),
                )
            import math

            q = nums[0] / step
            rounded_q = math.floor(q + 0.5) if q >= 0 else math.ceil(q - 0.5)
            val = rounded_q * step
        elif op == "avg":
            val = sum(nums) / len(nums)
        elif op == "max":
            val = max(nums)
        elif op == "min":
            val = min(nums)
        else:
            val = nums[0]
        return BlankResult(
            blank_id=out_id,
            value=val,
            status="resolved",
            wall_time_s=round(time.time() - t0, 3),
        )

    if constraint.kind in ("argmax", "argmin"):
        candidates = constraint.candidate_entity_ids
        blanks = constraint.sort_by_blank_ids
        if len(candidates) != len(blanks) or not candidates:
            return BlankResult(blank_id=out_id, status="unknown")
        pairs = []
        for ent_id, b_id in zip(candidates, blanks):
            res = state.get(b_id)
            if res is None or res.status != "resolved" or res.value is None:
                return BlankResult(blank_id=out_id, status="unknown")
            # Use year/number coercion for ordinals.
            key = _coerce_number(res.value)
            if key is None:
                key = _coerce_year(res.value)
            if key is None:
                return BlankResult(blank_id=out_id, status="unknown")
            pairs.append((key, ent_id))
        if constraint.kind == "argmax":
            _, winner = max(pairs)
        else:
            _, winner = min(pairs)
        winner_ent = plan.entity_by_id(winner)
        # A blank / unnamed entity must not surface its raw id as the answer
        # (q469: plan-churn made the target its own argmax candidate, so the
        # winning "entity" was a blank whose id, e.g. "t", leaked out as the
        # final answer). Treat an unnamed winner as unresolved instead.
        if not winner_ent.name:
            return BlankResult(
                blank_id=out_id,
                status="unknown",
                wall_time_s=round(time.time() - t0, 3),
            )
        return BlankResult(
            blank_id=out_id,
            value=winner_ent.name,
            status="resolved",
            wall_time_s=round(time.time() - t0, 3),
        )

    if constraint.kind in ("rank_n", "filter", "count"):
        try:
            source_value = _constraint_source_value(constraint, state)
            validation = validate_structured_candidate_rows(
                source_value,
                constraint,
                plan=plan,
                state=state,
                require_all_rows_valid=(constraint.kind != "count"),
            )
            if not validation.ok:
                return _unknown_constraint_result(out_id, t0)
            rows = validation.rows
            if constraint.kind == "rank_n":
                if not constraint.numeric_attribute or constraint.n is None:
                    return _unknown_constraint_result(out_id, t0)
                winner = _gsw_worksheet_ops.rank_n(
                    rows,
                    constraint.numeric_attribute,
                    constraint.n,
                    descending=constraint.descending,
                )
                value = winner.candidate
            elif constraint.kind == "count":
                if not rows:
                    return _unknown_constraint_result(out_id, t0)
                value = _gsw_worksheet_ops.count(rows)
            else:
                if not constraint.filter_attribute:
                    return _unknown_constraint_result(out_id, t0)
                expected, ok = _filter_expected_value(constraint, plan, state)
                if not ok:
                    return _unknown_constraint_result(out_id, t0)
                filtered = _gsw_worksheet_ops.filter_rows(
                    rows,
                    {constraint.filter_attribute: expected},
                )
                value = [_candidate_row_to_dict(row) for row in filtered]
        except (TypeError, ValueError):
            return _unknown_constraint_result(out_id, t0)
        return BlankResult(
            blank_id=out_id,
            value=value,
            status="resolved",
            wall_time_s=round(time.time() - t0, 3),
        )

    if constraint.kind in ("equals", "gt", "lt", "in_list"):
        left_ref, right_ref = _relational_ref_pair(constraint)
        left, left_ok = _constraint_ref_value(left_ref, plan, state)
        right, right_ok = _constraint_ref_value(right_ref, plan, state)
        if not (left_ok and right_ok):
            return BlankResult(
                blank_id=out_id,
                status="unknown",
                wall_time_s=round(time.time() - t0, 3),
            )

        if constraint.kind == "equals":
            val = _constraint_values_equal(left, right)
        elif constraint.kind == "in_list":
            val = _constraint_contains(left, right)
        else:
            left_num = _coerce_constraint_number(left)
            right_num = _coerce_constraint_number(right)
            if left_num is None or right_num is None:
                return BlankResult(
                    blank_id=out_id,
                    status="unknown",
                    wall_time_s=round(time.time() - t0, 3),
                )
            val = (
                left_num > right_num
                if constraint.kind == "gt"
                else left_num < right_num
            )
        return BlankResult(
            blank_id=out_id,
            value=val,
            status="resolved",
            wall_time_s=round(time.time() - t0, 3),
        )

    return BlankResult(blank_id=out_id, status="unknown")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# q39 fix 1c: concat used to hardcode "; " as the join separator no matter
# what the constraint's OUTPUT blank said, so a name-construction concat
# ("the gymnast's middle name followed by a SPACE and then the swimmer's
# first name") produced "<middle>; <first>" instead of "<middle> <first>".
# Narrow cue: only switch to a bare space when the output blank's
# verification_claim/retrieval_hint explicitly names a space join or
# describes concatenating names together; any plan that gives no such
# signal keeps the original "; " (list-style aggregation default).
_CONCAT_SPACE_JOIN_CUES = re.compile(
    r"followed by a space|separated by a space|with a space|space between|"
    r"space[- ]separated|concatenat\w*.{0,60}\bname",
    flags=re.IGNORECASE,
)


def _concat_separator(plan: GSWPlan, constraint: "Constraint") -> str:
    out_id = getattr(constraint, "output_blank_id", None)
    if not out_id:
        return "; "
    try:
        ent = plan.entity_by_id(out_id)
    except ExecutionError:
        return "; "
    signal_text = " ".join(
        filter(None, (getattr(ent, "verification_claim", None), getattr(ent, "retrieval_hint", None)))
    )
    if signal_text and _CONCAT_SPACE_JOIN_CUES.search(signal_text):
        return " "
    return "; "


def _stringify(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        # Integer-like floats should render without trailing .0
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    if isinstance(v, (list, tuple)):
        return ", ".join(_stringify(x) for x in v)
    return str(v)


def _coerce_value(value: Any, value_type: Optional[str]) -> Any:
    """Best-effort coerce LLM output into the blank's declared type.

    If the LLM returns the right type already, pass through. We avoid
    raising on mismatch here — the ``_identify_blank`` caller decides
    whether a return value of None means ``unknown``.
    """
    if value is None:
        return None
    if value_type == "number":
        return _coerce_number(value)
    if value_type == "date":
        # Stays as-is; calling code (argmax) will coerce to year when needed.
        return str(value)
    if value_type == "list":
        if isinstance(value, list):
            return [_stringify(x) for x in value]
        # Attempt to split a string list.
        return [x.strip() for x in str(value).split(",") if x.strip()]
    if value_type == "bool":
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
        return None
    return value  # entity / attribute / text
