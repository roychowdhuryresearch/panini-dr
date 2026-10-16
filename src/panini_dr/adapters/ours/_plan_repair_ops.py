"""How a plan may change. Every mutation the repairer can make lives here.

`reverse_edge` builds and validates its replacement verb phrase through
`_plan_diff`'s `AddVPOp` + `apply_diff`, so there is one implementation of
"add a verb phrase" in this codebase (see `_reverse_edge`'s docstring for
why the stale verb phrase is dropped via a direct list filter instead of
`RemoveVPOp` + a second `apply_diff` call). Every op's result — not just
`reverse_edge`'s — is revalidated to the same standard `apply_diff` uses:
schema (`GSWPlan.model_validate`) AND cycle detection
(`topological_sort_blanks` / `build_dependency_graph`). A retype can create
a new dependency edge (`build_dependency_graph` keys VP-edge direction off
`value_type == "entity"`) that closes a cycle a schema-only check can't see.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from panini_dr.adapters.ours._plan_diff import (
    AddConstraintOp, AddVPOp, DiffApplyError, RemoveBlankOp, _ConstraintSpec,
    _VPSpec, apply_diff,
)
from panini_dr.adapters.ours._planner_exec import (
    Constraint, Entity, ExecutionError, GSWPlan, VerbPhrase,
    build_dependency_graph, topological_sort_blanks,
)


class RepairApplyError(RuntimeError):
    """The op does not apply to this plan (unknown id, illegal target)."""


class ReverseEdgeOp(BaseModel):
    op: Literal["reverse_edge"]
    vp_id: str


class RetypeNodeOp(BaseModel):
    op: Literal["retype_node"]
    node_id: str
    value_type: Literal["date", "number", "entity", "attribute", "list", "text", "bool"]


class SetHintOp(BaseModel):
    op: Literal["set_hint"]
    node_id: str
    text: str


class InlinePlaceholderOp(BaseModel):
    op: Literal["inline_placeholder"]
    node_id: str
    ref: str


class _EdgeSpec(BaseModel):
    phrase: str
    subject_id: str | None = None    # for in_edges
    object_id: str | None = None     # for out_edges


class AddNodeOp(BaseModel):
    op: Literal["add_node"]
    node_id: str
    kind: Literal["blank", "filled"]
    value_type: str
    hint: str | None = None
    claim: str | None = None
    name: str | None = None
    literal_value: Any | None = None
    in_edges: list[_EdgeSpec] = Field(default_factory=list)
    out_edges: list[_EdgeSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _needs_an_edge(self):
        # GSWPlan raises `dangling entities` for a node nothing references, so
        # an edgeless add is unbuildable rather than merely unwise.
        if not self.in_edges and not self.out_edges:
            raise ValueError("add_node requires at least one edge")
        return self


class DeleteNodeOp(BaseModel):
    op: Literal["delete_node"]
    node_id: str


class AddEdgeOp(BaseModel):
    op: Literal["add_edge"]
    subject_id: str
    phrase: str
    object_id: str


class DeleteEdgeOp(BaseModel):
    op: Literal["delete_edge"]
    vp_id: str


class SetTargetOp(BaseModel):
    op: Literal["set_target"]
    node_id: str


class AddConstraintOp2(BaseModel):
    op: Literal["add_constraint"]
    constraint: _ConstraintSpec


class DeleteConstraintOp(BaseModel):
    op: Literal["delete_constraint"]
    constraint_id: str


class ModifyConstraintOp(BaseModel):
    op: Literal["modify_constraint"]
    constraint_id: str
    changes: dict[str, Any]


RepairOp = Annotated[
    Union[
        ReverseEdgeOp, RetypeNodeOp, SetHintOp, InlinePlaceholderOp,
        AddNodeOp, DeleteNodeOp, AddEdgeOp, DeleteEdgeOp, SetTargetOp,
        AddConstraintOp2, DeleteConstraintOp, ModifyConstraintOp,
    ],
    Field(discriminator="op"),
]
OP_NAMES = (
    "reverse_edge", "retype_node", "set_hint", "inline_placeholder",
    "add_node", "delete_node", "add_edge", "delete_edge", "set_target",
    "add_constraint", "delete_constraint", "modify_constraint",
)

_ADAPTER = TypeAdapter(RepairOp)


def parse_repair_op(raw: dict[str, Any]) -> RepairOp:
    return _ADAPTER.validate_python(raw)


def is_destructive(op: RepairOp) -> bool:
    """True when the op removes content. Only these face the deletion test."""
    if isinstance(op, SetHintOp):
        return not (op.text or "").strip()
    return isinstance(op, (DeleteNodeOp, DeleteEdgeOp, DeleteConstraintOp))


def _entity(plan: GSWPlan, node_id: str):
    for e in plan.entities:
        if e.id == node_id:
            return e
    raise RepairApplyError(f"no entity {node_id!r}")


def _vp(plan: GSWPlan, vp_id: str):
    for v in plan.verb_phrases:
        if v.id == vp_id:
            return v
    raise RepairApplyError(f"no verb phrase {vp_id!r}")


def _constraint(plan: GSWPlan, constraint_id: str):
    for c in plan.constraints:
        if c.id == constraint_id:
            return c
    raise RepairApplyError(f"no constraint {constraint_id!r}")


def _duplicate_ids(ids) -> set[str]:
    seen: set[str] = set()
    dup: set[str] = set()
    for i in ids:
        (dup if i in seen else seen).add(i)
    return dup


def _check_unique_ids(plan: GSWPlan) -> None:
    """`GSWPlan` never enforces id uniqueness (entity_ids collapses into a
    ``set`` internally, so a duplicate doesn't even fail the known-refs /
    dangling checks). Left unchecked, a duplicate lets `_entity` / `_vp` /
    `_constraint` (and `GSWPlan.entity_by_id`) silently resolve to whichever
    instance comes first, so a later op can mutate or delete the wrong one.
    Task-3 review finding 2/3.
    """
    if dupes := _duplicate_ids(e.id for e in plan.entities):
        raise RepairApplyError(f"duplicate entity id(s): {sorted(dupes)}")
    if dupes := _duplicate_ids(v.id for v in plan.verb_phrases):
        raise RepairApplyError(f"duplicate verb-phrase id(s): {sorted(dupes)}")
    if dupes := _duplicate_ids(c.id for c in plan.constraints):
        raise RepairApplyError(f"duplicate constraint id(s): {sorted(dupes)}")


def _revalidate(new: GSWPlan) -> GSWPlan:
    """Re-validate a mutated plan to the same standard `apply_diff` applies
    (schema, then cycle detection), plus two invariants neither `apply_diff`
    nor `GSWPlan` itself checks: id uniqueness (`_check_unique_ids`) and
    exactly one target blank. Deleting or redirecting the target away
    doesn't fail `GSWPlan.model_validate` — nothing there reads `is_target`
    — but it silently turns a real `target_producible` violation into zero
    violations for the verifier loop this vocabulary feeds (task-3 review
    finding 1: `target_producible`'s naive "fix" is exactly "delete the
    target"). This function is the ONE gate: the three helpers below that
    delegate to `_plan_diff.apply_diff` (`_reverse_edge`, `_delete_node`,
    `_add_constraint`) route their result back through here too, so every
    op — direct-mutation or delegated — passes through the same checks
    exactly once before returning.
    """
    try:
        revalidated = GSWPlan.model_validate(new.model_dump())
    except Exception as exc:  # noqa: BLE001
        raise RepairApplyError(f"post-apply validation failed: {exc}") from exc
    _check_unique_ids(revalidated)
    try:
        topological_sort_blanks(revalidated)
        build_dependency_graph(revalidated)
    except ExecutionError as exc:
        raise RepairApplyError(
            f"post-apply topology invalid: {exc.kind}: {exc.detail}"
        ) from exc
    try:
        revalidated.target()
    except ExecutionError as exc:
        raise RepairApplyError(
            f"post-apply target invalid: {exc.kind}: {exc.detail}"
        ) from exc
    return revalidated


def _reverse_edge(plan: GSWPlan, vp_id: str) -> GSWPlan:
    """Swap a verb phrase's subject/object; id and phrase are preserved.

    The replacement edge is built and validated through `_plan_diff`'s
    `AddVPOp` + `apply_diff` — the same construction + schema + cycle
    re-validation any other diff op gets, so there is one implementation of
    "add a verb phrase" in this codebase.

    The stale VP is NOT dropped via `RemoveVPOp` + a second `apply_diff`
    call: `apply_diff` fully re-validates the whole plan after every op,
    including the no-dangling-entity check. A VP that is the ONLY edge
    wiring one or both of its endpoints (e.g. this module's own two-entity,
    one-VP test fixture) would transiently dangle the instant it
    disappeared, before its reversed replacement landed — rejecting a
    perfectly good target state because of a doomed intermediate one. So
    the old VP is dropped by a direct list filter on an unvalidated copy,
    and only the single resulting `apply_diff` call re-validates — over the
    real final graph, exactly once.
    """
    vp = next((v for v in plan.verb_phrases if v.id == vp_id), None)
    if vp is None:
        raise RepairApplyError(f"no verb phrase {vp_id!r}")
    without_old = plan.model_copy(deep=True)
    without_old.verb_phrases = [
        v for v in without_old.verb_phrases if v.id != vp_id
    ]
    try:
        reversed_plan = apply_diff(without_old, AddVPOp(op="add_vp", vp=_VPSpec(
            id=vp.id, phrase=vp.phrase,
            subject_id=vp.object_id, object_id=vp.subject_id)))
    except DiffApplyError as exc:
        raise RepairApplyError(str(exc)) from exc
    return _revalidate(reversed_plan)


def _delete_node(plan: GSWPlan, node_id: str) -> GSWPlan:
    """Remove an entity and cascade-remove whatever verb phrase or
    constraint referenced it, via `_plan_diff`'s `RemoveBlankOp` +
    `apply_diff` -- the same cascade + revalidation any other diff-driven
    delete in this codebase gets, so there is one implementation of
    "remove an entity" here too (see module docstring re: `reverse_edge`).
    Cascading a constraint's removal can itself strand a *different*
    entity (e.g. the target it used to feed) -- `apply_diff`'s own
    dangling-entity revalidation catches that.
    """
    _entity(plan, node_id)  # raise RepairApplyError for an unknown id
    try:
        removed = apply_diff(plan, RemoveBlankOp(op="remove_blank", blank_id=node_id))
    except DiffApplyError as exc:
        raise RepairApplyError(str(exc)) from exc
    return _revalidate(removed)


def _add_constraint(plan: GSWPlan, spec: _ConstraintSpec) -> GSWPlan:
    """Route through `_plan_diff`'s `AddConstraintOp` + `apply_diff`, the
    same one true implementation of "add a constraint" `_reverse_edge`
    relies on for verb phrases."""
    try:
        added = apply_diff(plan, AddConstraintOp(op="add_constraint", constraint=spec))
    except DiffApplyError as exc:
        raise RepairApplyError(str(exc)) from exc
    return _revalidate(added)


def apply_op(plan: GSWPlan, op: RepairOp) -> GSWPlan:
    """Return a NEW plan with `op` applied. Never mutates the input."""
    if isinstance(op, ReverseEdgeOp):
        return _reverse_edge(plan, op.vp_id)
    if isinstance(op, DeleteNodeOp):
        return _delete_node(plan, op.node_id)
    if isinstance(op, AddConstraintOp2):
        return _add_constraint(plan, op.constraint)

    new = plan.model_copy(deep=True)
    if isinstance(op, RetypeNodeOp):
        _entity(new, op.node_id).value_type = op.value_type
    elif isinstance(op, SetHintOp):
        _entity(new, op.node_id).retrieval_hint = op.text
    elif isinstance(op, InlinePlaceholderOp):
        target, ref = _entity(new, op.node_id), _entity(new, op.ref)
        ref_hint = (getattr(ref, "retrieval_hint", "") or "").strip().rstrip("?")
        if not ref_hint:
            raise RepairApplyError(f"{op.ref!r} has no hint to inline")
        cur = getattr(target, "retrieval_hint", "") or ""
        target.retrieval_hint = cur.replace("{%s}" % op.ref, ref_hint)
    elif isinstance(op, AddNodeOp):
        new.entities.append(Entity(
            id=op.node_id, kind=op.kind, name=op.name, value_type=op.value_type,
            literal_value=op.literal_value, retrieval_hint=op.hint,
            verification_claim=op.claim,
        ))
        for edge in op.in_edges:
            new.verb_phrases.append(VerbPhrase(
                id=f"vp_{edge.subject_id}_{edge.phrase}_{op.node_id}",
                phrase=edge.phrase, subject_id=edge.subject_id,
                object_id=op.node_id,
            ))
        for edge in op.out_edges:
            new.verb_phrases.append(VerbPhrase(
                id=f"vp_{op.node_id}_{edge.phrase}_{edge.object_id}",
                phrase=edge.phrase, subject_id=op.node_id,
                object_id=edge.object_id,
            ))
    elif isinstance(op, AddEdgeOp):
        new.verb_phrases.append(VerbPhrase(
            id=f"vp_{op.subject_id}_{op.phrase}_{op.object_id}",
            phrase=op.phrase, subject_id=op.subject_id, object_id=op.object_id,
        ))
    elif isinstance(op, DeleteEdgeOp):
        _vp(new, op.vp_id)  # raise RepairApplyError for an unknown id
        new.verb_phrases = [v for v in new.verb_phrases if v.id != op.vp_id]
    elif isinstance(op, SetTargetOp):
        _entity(new, op.node_id)  # raise RepairApplyError for an unknown id
        for e in new.entities:
            e.is_target = e.id == op.node_id
    elif isinstance(op, DeleteConstraintOp):
        _constraint(new, op.constraint_id)  # raise for an unknown id
        new.constraints = [
            c for c in new.constraints if c.id != op.constraint_id
        ]
    elif isinstance(op, ModifyConstraintOp):
        target = _constraint(new, op.constraint_id)
        for key, value in op.changes.items():
            # Renaming isn't a repair, and it's exactly how a duplicate
            # constraint id would sneak past `_check_unique_ids` (review
            # finding 3): change the field, never the identity.
            if key == "id":
                raise RepairApplyError(
                    "modify_constraint cannot rename a constraint's id"
                )
            if key not in Constraint.model_fields:
                raise RepairApplyError(f"constraint has no field {key!r}")
            setattr(target, key, value)
    return _revalidate(new)
