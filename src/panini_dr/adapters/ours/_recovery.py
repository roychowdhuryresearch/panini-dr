"""Per-question recovery diagnostics (``traj.extra["gsw_recovery_diag"]``).

The recovery mechanisms these diagnostics describe (M1 re-dispatch of a
cascade-blocked input, M2 bounded overwrite, M3 holistic answer) were off in the
paper configuration and were removed; the record is still written for every
question. Deterministic, no LLM, no I/O."""
from __future__ import annotations


def recovery_diagnostics(*, redispatch_count: int, overwrite_traces: list[dict], holistic_fired: bool,
                         was_blocked_and_unresolved: bool, target_resolved_now: bool) -> dict:
    """Consolidated per-question recovery diagnostics. blocked_then_recovered_correct is the
    headline metric: a pre-recovery blocked_and_unresolved case that recovery resolved."""
    return {
        "recovery_redispatch_count": redispatch_count,
        "recovery_overwrite_count": len(overwrite_traces),
        "overwrites": list(overwrite_traces),
        "holistic_fallback_fired": bool(holistic_fired),
        "blocked_then_recovered_correct": bool(was_blocked_and_unresolved and target_resolved_now),
    }
