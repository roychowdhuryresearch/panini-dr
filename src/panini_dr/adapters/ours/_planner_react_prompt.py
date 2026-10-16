"""Plan rendering shared by the orchestrator and researcher prompts.

``SYSTEM_PROMPT_LEGEND`` explains the plan schema (what `argmin`,
`bridge-date`, `category=true`, `state="expanded_from_collective:..."` etc.
mean) as a static, cacheable string; ``render_plan_brief`` renders one
question's ``GSWPlan`` (entities + roles + VPs + constraints, then the
topological solve order annotated with retrieval signals).

The module kept its name from the ReAct planner adapter
(``ours_gsw_planner_react_v1``), whose system-prompt assembly
(``build_system_prompt``, the rules block and the topo-only rendering) was
removed with it.
"""

from __future__ import annotations

import textwrap
from collections import defaultdict
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Static prefix: the schema legend
# ---------------------------------------------------------------------------


SYSTEM_PROMPT_LEGEND = textwrap.dedent(
    """
    You are answering a research question using a pre-emitted plan.

    ## Critical protocol

    - Every assistant turn before completion MUST contain at least one
      real tool call.
    - Never reply with prose-only text or inline JSON that imitates a
      tool call.
    - Fill blanks by calling `update_blank(blank_id, value,
      evidence_chunk_ids)`; `evidence_chunk_ids` is REQUIRED on every
      update.
    - The only valid ending is:
      `update_blank(target_id, target_value, evidence_chunk_ids)` then
      `finish(answer=target_value)`.
    - The `finish` tool will REFUSE if the target blank is unresolved.

    ## Plan schema legend — read this before the plan below

    A plan is a small graph describing what the question is asking. It has
    three arrays: `entities`, `verb_phrases`, `constraints`.

    ### Entity

    - `kind="filled"` means the entity is named in the question and has
      a `name`.
    - `kind="blank"` means you must resolve it. Blank `value_type` is one
      of: `date`, `number`, `entity`, `attribute`, `list`, `text`, `bool`.
    - Exactly one blank has `is_target: true`; its resolved value is the
      final answer.

    `role` explains why an entity exists in the plan.
    - Common filled roles: `subject`, `scope-filter`, `year-anchor`,
      `date-anchor`, `as-of-date`, `list-header`, `candidate`.
    - Common blank roles: `target`, `bridge-entity`, `bridge-date`,
      `bridge-number`, `bridge-attribute`, `list-member`,
      `comparison-output`, `aggregate-output`.

    Auxiliary fields:
    - `category: true` means the filled entity is a category or role,
      not a named individual.
    - `literal_value` on a filled `constraint-value` is the typed value
      Python constraints consume; use it instead of researching the
      surface `name`.
    - `state="expanded_from_collective:<group>"` marks a valid candidate
      expanded from a collective named in the question.

    ### VerbPhrase  `(subject, phrase, object)`

    A typed binary relation between two entities. The `phrase` is a short
    snake_case predicate like `died_in`, `has_population`,
    `entered_service_in`. Verb-phrases do two jobs:

    - Identify a blank from a filled anchor plus a predicate.
    - Project from a resolved entity-blank to a downstream blank.

    Temporal predicates like `in_year`, `signed_in_year`,
    `released_in_year`, and `as_of_date` should be used literally as
    retrieval hints.

    ### Constraint

    A constraint computes a blank without retrieval.

    - `derived` applies an operation such as `diff`, `sum`, `avg`,
      `max`, `min`, `count`, `concat`, `mul`, `div`, or
      `round_nearest` to input blanks.
    - `argmax` / `argmin` pick the winning entity from candidates using
      aligned ranking blanks.
    - Relational kinds like `equals`, `in_list`, `gt`, and `lt` produce
      bool outputs when their inputs are resolved.

    Every constraint has an `output_blank_id`. When all of the
    constraint's input blanks are resolved, the adapter AUTOMATICALLY
    computes the output blank — you do NOT need to call `update_blank`
    for constraint outputs. Just make sure the INPUT blanks are resolved
    (by retrieval + `update_blank` on them); the output will appear in
    state as soon as the last input lands.
    """
).strip()


# ---------------------------------------------------------------------------
# Per-question plan rendering
# ---------------------------------------------------------------------------


def _format_entity(e: dict[str, Any]) -> str:
    """One compact line per entity."""
    parts = [f"`{e.get('id','?')}`"]
    parts.append(f"kind={e.get('kind','?')}")
    if e.get("role"):
        parts.append(f"role={e['role']}")
    if e.get("value_type"):
        parts.append(f"value_type={e['value_type']}")
    if e.get("name"):
        parts.append(f"name={e['name']!r}")
    if e.get("category"):
        parts.append("category=true")
    if e.get("is_target"):
        parts.append("**is_target=true**")
    if e.get("state"):
        parts.append(f"state={e['state']!r}")
    if e.get("literal_value") is not None:
        parts.append(f"literal_value={e['literal_value']!r}")
    return "- " + "  ".join(parts)


def _format_vp(vp: dict[str, Any]) -> str:
    return (
        f"- `{vp.get('id','?')}`: "
        f"({vp.get('subject_id','?')}) --[{vp.get('phrase','?')}]--> "
        f"({vp.get('object_id','?')})"
    )


def _format_constraint(c: dict[str, Any]) -> str:
    kind = c.get("kind", "?")
    parts = [f"`{c.get('id','?')}`", f"kind={kind}"]
    if c.get("op"):
        parts.append(f"op={c['op']}")
    if c.get("args_blanks"):
        parts.append(f"args_blanks={c['args_blanks']}")
    if c.get("args_refs"):
        parts.append(f"args_refs={c['args_refs']}")
    if c.get("candidate_entity_ids"):
        parts.append(f"candidates={c['candidate_entity_ids']}")
    if c.get("sort_by_blank_ids"):
        parts.append(f"sort_by={c['sort_by_blank_ids']}")
    if c.get("left_ref") or c.get("right_ref"):
        parts.append(f"left={c.get('left_ref')}  right={c.get('right_ref')}")
    parts.append(f"output={c.get('output_blank_id','?')}")
    return "- " + "  ".join(parts)


def render_plan_brief(
    plan_dict: dict[str, Any],
    order: list[str],
    levels: dict[str, int],
    slice_blank_ids: Optional[list[str]] = None,
) -> str:
    """Render the per-question plan + topological-order section.

    Output is plain markdown. Each blank in the topological order is
    annotated with its retrieval signals (incoming VPs) so the reasoner
    knows what to search for.

    When ``slice_blank_ids`` is given, the output is filtered to the
    subset of entities / VPs / constraints / topo-steps relevant to
    that slice — PLUS any filled entities those slice entities refer
    to (so upstream anchors stay visible for retrieval). Blanks that
    are upstream dependencies of slice blanks are rendered as context
    rows in a separate "upstream bridging values" section (no topo
    step for them; they will already be resolved by the time this
    researcher runs).
    """
    ent_by_id: dict[str, dict[str, Any]] = {
        e.get("id"): e for e in plan_dict.get("entities", [])
    }
    entities = plan_dict.get("entities", [])
    vps = plan_dict.get("verb_phrases", [])
    constraints = plan_dict.get("constraints", [])
    constraint_by_output = {
        c.get("output_blank_id"): c for c in constraints if c.get("output_blank_id")
    }

    vp_by_blank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vp in vps:
        for side in ("subject_id", "object_id"):
            eid = vp.get(side)
            if eid in ent_by_id and ent_by_id[eid].get("kind") == "blank":
                vp_by_blank[eid].append(vp)

    # ----- Slice filtering ------------------------------------------
    slice_set: Optional[set[str]] = set(slice_blank_ids) if slice_blank_ids is not None else None
    if slice_set is not None:
        # Entities to keep: the slice blanks themselves, plus every
        # filled entity connected to a slice blank via a VP (these are
        # the retrieval anchors the researcher needs). Upstream blank
        # entities (already resolved by earlier researchers) are
        # dropped from the primary entity list and exposed separately.
        keep_ent_ids: set[str] = set(slice_set)
        kept_vps: list[dict[str, Any]] = []
        for vp in vps:
            sid = vp.get("subject_id")
            oid = vp.get("object_id")
            if sid in slice_set or oid in slice_set:
                kept_vps.append(vp)
                for eid in (sid, oid):
                    if eid in ent_by_id and ent_by_id[eid].get("kind") == "filled":
                        keep_ent_ids.add(eid)
        # Constraints: keep if their output is in slice OR any input is in slice
        kept_constraints: list[dict[str, Any]] = []
        for c in constraints:
            ins = (
                set(c.get("args_refs") or [])
                | set(c.get("args_blanks") or [])
                | set(c.get("sort_by_blank_ids") or [])
            )
            if c.get("output_blank_id") in slice_set or (ins & slice_set):
                kept_constraints.append(c)
                for ref in c.get("args_refs") or []:
                    if ref in ent_by_id and ent_by_id[ref].get("kind") == "filled":
                        keep_ent_ids.add(ref)
                # Input blanks of a slice-constraint are upstream — we'll render them below.

        entities = [e for e in entities if e.get("id") in keep_ent_ids]
        vps = kept_vps
        constraints = kept_constraints
        constraint_by_output = {
            c.get("output_blank_id"): c for c in constraints if c.get("output_blank_id")
        }
        # Rebuild vp_by_blank over filtered VPs
        vp_by_blank = defaultdict(list)
        for vp in vps:
            for side in ("subject_id", "object_id"):
                eid = vp.get(side)
                if eid in ent_by_id and ent_by_id[eid].get("kind") == "blank" and eid in slice_set:
                    vp_by_blank[eid].append(vp)
        # Filter topo order to slice blanks (preserving original order)
        order = [b for b in order if b in slice_set]

    out: list[str] = []
    out.append("## The plan for this question")
    out.append("")
    out.append("### Entities")
    filled = [e for e in entities if e.get("kind") == "filled"]
    blank = [e for e in entities if e.get("kind") == "blank"]
    out.append("**Filled** (drawn from the question text):")
    for e in filled:
        out.append(_format_entity(e))
    out.append("")
    out.append("**Blanks** (you must fill these):")
    for e in blank:
        out.append(_format_entity(e))
    out.append("")
    if vps:
        out.append("### Verb-phrases")
        for vp in vps:
            out.append(_format_vp(vp))
        out.append("")
    if constraints:
        out.append("### Constraints")
        for c in constraints:
            out.append(_format_constraint(c))
        out.append("")

    out.append("## Topological solve order")
    out.append("")
    out.append(
        "Fill blanks in this order. Blanks at the same level have no "
        "dependencies on each other and can be resolved in any order."
    )
    out.append("")

    if not order:
        out.append("_(no blanks to solve)_")
        out.append("")
    else:
        # Group by level for readability.
        by_level: dict[int, list[str]] = defaultdict(list)
        for b_id in order:
            by_level[levels.get(b_id, 0)].append(b_id)
        step = 1
        for lvl in sorted(by_level):
            ids = by_level[lvl]
            header = f"### Level {lvl + 1}"
            if len(ids) > 1:
                header += "  _(parallel — blanks here have no mutual dependencies)_"
            out.append(header)
            for b_id in ids:
                ent = ent_by_id.get(b_id, {})
                vt = ent.get("value_type") or "text"
                role = ent.get("role") or "—"
                is_target = " **(TARGET)**" if ent.get("is_target") else ""
                out.append(
                    f"**Step {step}** — `{b_id}` (value_type=`{vt}`, role=`{role}`){is_target}"
                )
                if b_id in constraint_by_output:
                    c = constraint_by_output[b_id]
                    kind = c.get("kind")
                    op = c.get("op")
                    method = f"constraint `{kind}`" + (f" (`{op}`)" if op else "")
                    if kind == "derived":
                        method += (
                            f"; inputs: `{c.get('args_refs') or c.get('args_blanks', [])}`"
                        )
                    elif kind in ("argmax", "argmin"):
                        method += (
                            f"; candidates: `{c.get('candidate_entity_ids', [])}`, "
                            f"ranked by: `{c.get('sort_by_blank_ids', [])}`"
                        )
                    out.append(f"    — method: {method}")
                    out.append(
                        "    — no retrieval; auto-computed by the adapter once "
                        "all input blanks above are resolved"
                    )
                else:
                    signals: list[str] = []
                    for vp in vp_by_blank.get(b_id, []):
                        s, o = vp.get("subject_id"), vp.get("object_id")
                        other = o if s == b_id else s
                        other_ent = ent_by_id.get(other, {})
                        other_name = other_ent.get("name") or other
                        signals.append(f"`({other_name}, {vp.get('phrase')})`")
                    sig_text = ", ".join(signals) if signals else "_(no wired VPs — fall back to role+question keywords)_"
                    out.append("    — method: retrieval + LLM extract")
                    out.append(f"    — retrieval signals: {sig_text}")
                step += 1
            out.append("")

    return "\n".join(out).rstrip()
