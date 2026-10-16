"""The researcher's plan-change proposal, as a value.

`suggest_plan_revision(reason, hint)` ends a researcher's run when no chunk in
the corpus holds a plausible value for its blank. The researcher prompt asks for
a STRUCTURAL hint, not a complaint -- e.g. "add b_address (entity,
role=bridge-attribute) wired by VP (b_hospital, located_at, b_address)".

That proposal already reaches the orchestrator (as `revision_requests` in the
`dispatch_subplan` result), but the orchestrator then writes its OWN prose
`reason`/`evidence` for `request_plan_update`, and the structured hint never
reaches `update_plan`. Measured on FRAMES-824: 63 questions escalated AND
updated, so 63 proposals were paraphrased away.

This module is the vehicle for the last hop. It is deliberately a dumb value
type: no validation of the hint's CONTENT (the author LLM reads it), only that
the fields exist and are non-empty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class PlanRequest:
    """One researcher escalation, carried intact to the plan updater.

    The `assigned_blank_ids` field is stored as a tuple, not a list, to ensure
    the entire object is immutable (including the sequence itself). No code
    downstream may edit the researcher's words.
    """

    request_id: str
    blank_id: Optional[str]
    assigned_blank_ids: tuple[str, ...] = ()
    reason: str = ""
    hint: str = ""

    def __post_init__(self) -> None:
        """Coerce assigned_blank_ids to tuple for immutability."""
        if not isinstance(self.assigned_blank_ids, tuple):
            # frozen=True means we can't rebind directly; use object.__setattr__
            object.__setattr__(
                self,
                "assigned_blank_ids",
                tuple(self.assigned_blank_ids),
            )


def plan_request_from_revision(
    rr: dict[str, Any],
    question_id: str,
    seq: int,
) -> Optional[PlanRequest]:
    """Build a `PlanRequest` from the orchestrator's `revision_request` dict.

    Returns None when the dict cannot support a contract -- a missing or blank
    `reason` or `hint`. The caller falls through to today's behaviour, which is
    always safe: a contract is an ADDITION to the acceptance test, never a
    prerequisite for updating a plan.
    """
    if not isinstance(rr, dict):
        return None
    reason = str(rr.get("reason") or "").strip()
    hint = str(rr.get("hint") or "").strip()
    if not reason or not hint:
        return None
    blank_id = rr.get("blank_id")
    blank_id = str(blank_id) if blank_id else None
    return PlanRequest(
        request_id=f"{question_id}:{blank_id or '_'}:{seq}",
        blank_id=blank_id,
        assigned_blank_ids=tuple(str(b) for b in (rr.get("assigned_blank_ids") or [])),
        reason=reason,
        hint=hint,
    )
