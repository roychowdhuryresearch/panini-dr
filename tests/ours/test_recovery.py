from panini_dr.adapters.ours._recovery import recovery_diagnostics


def test_recovery_diagnostics_fields():
    diag = recovery_diagnostics(
        redispatch_count=1,
        overwrite_traces=[{"blank_id": "b", "old_value": "x", "new_value": "y", "trigger": "partial"}],
        holistic_fired=False,
        was_blocked_and_unresolved=True,   # pre-recovery
        target_resolved_now=True,          # post-recovery
    )
    assert diag["recovery_redispatch_count"] == 1
    assert diag["recovery_overwrite_count"] == 1
    assert diag["holistic_fallback_fired"] is False
    # headline: a blocked_and_unresolved case that recovery turned into a resolved target
    assert diag["blocked_then_recovered_correct"] is True


def test_recovery_diagnostics_not_recovered():
    diag = recovery_diagnostics(redispatch_count=1, overwrite_traces=[], holistic_fired=False,
                                was_blocked_and_unresolved=True, target_resolved_now=False)
    assert diag["blocked_then_recovered_correct"] is False  # still unresolved -> not recovered


def test_recovery_diagnostics_no_block():
    diag = recovery_diagnostics(redispatch_count=0, overwrite_traces=[], holistic_fired=False,
                                was_blocked_and_unresolved=False, target_resolved_now=True)
    assert diag["blocked_then_recovered_correct"] is False  # was never blocked
    assert diag["overwrites"] == []
