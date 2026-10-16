"""Per-blank connection manifest: the explicit, machine-readable list of a
blank's connections (vp_ids) the researcher must bind evidence to. Derived
from blank_projection_edges + the partner's resolution state. Augments the
narrative plan brief; used only when --gsw-connection-evidence is on."""

from __future__ import annotations

from typing import Any

from panini_dr.adapters.ours._edge_taxonomy import FACTUAL, classify_edge
from panini_dr.adapters.ours._gsw_state import blank_projection_edges


def build_connection_manifest(
    plan: Any, blank_id: str, resolved_values: dict[str, Any]
) -> dict[str, Any]:
    ent = plan.entity_by_id(blank_id)
    connections: list[dict[str, Any]] = []
    for e in blank_projection_edges(plan, blank_id):
        pid = e["other_entity_id"]
        partner_ent = plan.entity_by_id(pid)
        if partner_ent.kind == "filled":
            status = "filled"
            partner = partner_ent.name or partner_ent.id
        elif pid in resolved_values and resolved_values[pid] is not None:
            status = "resolved_bridge"
            partner = str(resolved_values[pid])
        else:
            status = "deferred"
            partner = f"?{pid}"
        connections.append({
            "vp_id": e["vp_id"],
            "relation": e["phrase"],
            "partner": partner,
            "partner_status": status,
            "direction": e["direction"],
            "evidence_required": classify_edge(plan, e) == FACTUAL,
        })
    return {
        "blank_id": blank_id,
        "value_type": ent.value_type or "entity",
        "is_target": bool(ent.is_target),
        "connections": connections,
    }
