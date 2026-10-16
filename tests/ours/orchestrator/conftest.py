"""Fixtures shared by the orchestrator test modules."""

from __future__ import annotations

import pytest

@pytest.fixture(autouse=True)
def _restore_emit_plan():
    from panini_dr.adapters.ours import gsw_planner_orchestrator_v1 as mod
    original = mod.emit_plan
    yield
    mod.emit_plan = original
