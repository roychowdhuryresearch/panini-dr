"""The champion preset (configs/presets/champion_82.json) and the gsw_ extra builder.

Ported from the research repo's tests/ours/test_gsw_champion_layers.py, which
exercised the same functions in playground/run_substitution.py; they now live
in panini_dr.config. The tests of the three champion layers (temporal,
roles-render, actor memory; all off in the paper configuration) and of the old
runner's source text were not ported.
"""
from __future__ import annotations

import json

from panini_dr.config import (
    apply_gsw_champion_preset,
    build_gsw_extra,
    champion_preset_path,
    expand_gsw_champion_preset,
)


def test_preset_expands_from_fixture():
    fixture = json.loads(champion_preset_path().read_text())
    cfg = expand_gsw_champion_preset({})
    for k, v in fixture["gsw_config"].items():
        assert cfg[k] == v, f"preset lost fixture key {k}"
    for k, v in fixture["nongsw_config"].items():
        assert cfg[k] == v, f"preset lost fixture key {k}"
    assert "gsw_codegen_compute" not in cfg
    # explicit user value still survives
    cfg2 = expand_gsw_champion_preset({"gsw_max_plan_updates": 2})
    assert cfg2["gsw_max_plan_updates"] == 2
    cfg3 = expand_gsw_champion_preset({"gsw_edge_grounded_cap": 7})
    assert cfg3["gsw_edge_grounded_cap"] == 7


# The champion preset must not clobber a flag the user typed. A research-repo
# bug overwrote every gsw_ key unconditionally, so an explicitly passed flag was
# silently replaced by the preset's value.


def test_champion_preset_does_not_override_an_explicitly_passed_gsw_flag():
    """A gsw_ key the user set on the command line survives the preset."""
    opts = {"gsw_codegen_compute": True, "gsw_hard_type_filter": True}
    out = apply_gsw_champion_preset(opts, explicit={"gsw_codegen_compute"})
    assert out["gsw_codegen_compute"] is True, "explicit CLI flag was clobbered"
    # a key the user did NOT set still takes the preset's value
    preset_key, preset_val = next(
        (k, v) for k, v in expand_gsw_champion_preset({}).items()
        if k.startswith("gsw_") and k != "gsw_codegen_compute"
    )
    assert out[preset_key] == preset_val


def test_champion_preset_still_applies_when_nothing_is_explicit():
    from panini_dr.config import removed_keys

    preset = expand_gsw_champion_preset({})
    out = apply_gsw_champion_preset({"gsw_lean_prompts": True}, explicit=set())
    for k, v in preset.items():
        if k.startswith("gsw_"):
            if k in removed_keys():
                # the preset file is the research fixture byte for byte; a key
                # whose code was removed is not applied
                assert k not in out, f"{k} was removed and must not be applied"
                continue
            assert out[k] == v, f"{k} should follow the preset when not explicit"
