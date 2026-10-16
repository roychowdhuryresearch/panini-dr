"""Edge taxonomy for connection-evidence verification (CE-v2 spec D1).
'factual' edges demand chunk evidence; scaffold edges are planner artifacts
that are never corpus sentences (the measured CE-v1 reject machine:
instance_of 72%, as_of_date 66% reject-with-evidence). Conservative
structural rule; anything unknown fails CLOSED to 'factual' (strictest).
Pure function over the plan -- consumers gate on --gsw-connection-evidence."""

from __future__ import annotations

from typing import Any

FACTUAL = "factual"
SCAFFOLD_TEMPORAL = "scaffold_temporal"
SCAFFOLD_TYPING = "scaffold_typing"


def classify_edge(plan: Any, edge: dict[str, Any]) -> str:
    phrase = (edge.get("phrase") or "").strip().lower()
    if phrase == "as_of_date":
        return SCAFFOLD_TEMPORAL
    if phrase == "instance_of":
        return SCAFFOLD_TYPING
    try:
        partner = plan.entity_by_id(edge["other_entity_id"])
    except Exception:  # noqa: BLE001 -- fail closed
        return FACTUAL
    role = (getattr(partner, "role", None) or "").strip().lower()
    if role == "as-of-date":
        return SCAFFOLD_TEMPORAL
    if getattr(partner, "category", False):
        return SCAFFOLD_TYPING
    return FACTUAL
