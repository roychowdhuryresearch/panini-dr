"""configs/paper.json == the expanded preset, and the preset machinery is faithful."""

from __future__ import annotations

import hashlib
import inspect
import json
import re

import pytest

from panini_dr import config as C

# The adapter config that all three paper runs recorded in cell_result.json
# (BCP CF-150 legs, FRAMES-824, MoNaCo-1315), minus the dataset keys. Spelled
# out here so that an edit to configs/paper.json cannot silently move the
# target the identity check compares against.
RECORDED = {
    "gsw_agentic_read": True, "gsw_blank_hints": True, "gsw_blank_qualifier": True,
    "gsw_blank_verifier": True, "gsw_blank_verifier_mode": "enforce",
    "gsw_block_unverified_cascade": False, "gsw_candidate_evidence_matrix": False,
    "gsw_champion": True, "gsw_codegen_compute": True, "gsw_codegen_frozen": True,
    "gsw_context_compaction": True, "gsw_edge_grounded": True, "gsw_edge_grounded_cap": 5,
    "gsw_evidence_gate": "measure", "gsw_evidence_grounder": "llm", "gsw_evidence_state": True,
    "gsw_grounding_holistic_fallback": False, "gsw_grounding_strict_tries": 2,
    "gsw_hard_type_filter": True, "gsw_investigation_memory": True,
    "gsw_investigation_memory_enforce": True, "gsw_investigation_memory_finalshot": True,
    "gsw_max_plan_updates": 5, "gsw_memory_dossier": True, "gsw_memory_scoped_dispatch": True,
    "gsw_orchestrator_live": True, "gsw_plan_issue_surface": "advisory",
    "gsw_plan_quality_checks": True, "gsw_plan_request_contract": True,
    "gsw_plan_update_guard": "enforce", "gsw_plan_verifier": True, "gsw_planner_tool": True,
    "gsw_qualifier_grounding_policy": "lenient", "gsw_react_fallback": True,
    "gsw_read_is_reader": True, "gsw_read_promotion": True, "gsw_read_raw_fallback": True,
    "gsw_recovery": "false", "gsw_recovery_max_overwrites": 1, "gsw_recovery_max_redispatch": 1,
    "gsw_researcher_stateful": False, "gsw_rival_scan": False, "gsw_search_dedup": False,
    "gsw_typed_blank_resolver": True, "gsw_verifier_cap_policy": "accept_with_flag",
    "gsw_verifier_max_retries": 2, "gsw_verifier_reasoning": True, "gsw_verifier_scoped": True,
    "llm_seed": 42, "llm_temperature": 0.0, "max_completion_tokens": 16384,
    "max_prompt_tokens": 32000, "max_turns": 50, "orchestrator_mode": "llm",
    "reasoning_effort": "medium", "reranker": "jina", "retriever_type": "hybrid",
    "synthesis_validator_mode": "repair",
}

# sha256 of tests/fixtures/compaction_config_82_100.json in the research repo,
# which the research runner read at runtime for --gsw-champion.
CHAMPION_PRESET_SHA256 = "27f88e56df6fc54f129a4e88db166335e14e7c13031a99873c2c5cc1b5dcecfd"


def _typed_equal(a: dict, b: dict) -> bool:
    return a == b and all(type(a[k]) is type(b[k]) for k in a)


def test_paper_json_is_the_recorded_config():
    paper = C.load_paper_config()
    assert _typed_equal(paper["config"], RECORDED)
    assert paper["dataset_keys"] == ["corpus", "dense_cache", "lucene_index"]
    assert paper["system"] == "ours_gsw_planner_orchestrator_v1"
    assert paper["concurrency"] == 32
    assert set(paper["inert_keys"]["keys"]) <= set(RECORDED)


def test_expanded_preset_equals_paper_json():
    """resolved ∪ removed == paper.json: the preset expands to the paper's
    recorded config once the removed keys are added back at their paper values."""
    resolved = C.resolve_config()
    assert not set(resolved.flat()) & set(C.removed_keys())
    assert _typed_equal(C.recorded_config(resolved.flat()), C.load_paper_config()["config"])
    assert C.config_drift(resolved.flat()) == {}


def test_removed_features_registry_agrees_with_paper_json():
    """A recorded removed key is in the paper's recorded config at exactly its
    paper value; an unrecorded one is absent from it (the runner omitted it)."""
    paper = C.load_paper_config()["config"]
    removed = C.removed_keys()
    assert removed, "the registry lists the removed keys"
    for key, info in removed.items():
        assert set(info) == {"paper_value", "recorded", "reason"}, key
        assert info["reason"].strip(), key
        if info["recorded"]:
            assert key in paper and paper[key] == info["paper_value"], key
            assert type(paper[key]) is type(info["paper_value"]), key
        else:
            assert key not in paper, key
        # a removed key is no longer a runner option
        assert key not in C.RUNNER_DEFAULTS and key not in C.PASSTHROUGH_KEYS, key
        assert key not in C.AGENTIC_OVERLAY and key not in C.GSW_OMIT_IF_FALSY, key


def test_setting_a_removed_key_or_value_raises():
    for key, info in C.removed_keys().items():
        with pytest.raises(C.RemovedConfigError, match="was removed"):
            C.resolve_config({key: info["paper_value"]})
    for key, info in C.load_removed_features()["values"].items():
        for value in info["removed"]:
            with pytest.raises(C.RemovedConfigError, match="was removed"):
                C.resolve_config({key: value})


def test_removed_env_vars_are_reported_when_set():
    names = list(C.load_removed_features()["env_vars"])
    assert C.removed_env_vars_set({}) == {}
    if names:
        assert set(C.removed_env_vars_set({n: "1" for n in names})) == set(names)


def test_champion_preset_is_byte_identical_to_the_research_fixture():
    data = C.champion_preset_path().read_bytes()
    assert hashlib.sha256(data).hexdigest() == CHAMPION_PRESET_SHA256


def test_paper_values_that_differ_from_adapter_defaults_are_explicit():
    flat = C.resolve_config().flat()
    # the adapter's own default would be 1 / 2 / "off" / bm25 (see audit notes)
    assert flat["gsw_edge_grounded_cap"] == 5
    assert flat["gsw_max_plan_updates"] == 5
    assert flat["synthesis_validator_mode"] == "repair"
    assert flat["retriever_type"] == "hybrid"


def test_overrides_behave_like_typed_flags():
    # an explicit value wins over the agentic overlay and over the champion preset
    r = C.resolve_config({"gsw_plan_request_contract": False, "gsw_max_plan_updates": 2})
    drift = C.config_drift(r.flat())
    assert drift == {"gsw_plan_request_contract": (True, "<absent>"),
                     "gsw_max_plan_updates": (5, 2)}
    r = C.resolve_config({"retriever_type": "bm25", "level_max_turns": 30})
    assert r.extra["retriever_type"] == "bm25" and r.extra["level_max_turns"] == 30
    with pytest.raises(C.UnknownConfigKeyError):
        C.resolve_config({"not_a_key": 1})


def test_without_the_agentic_overlay_the_config_is_the_bare_champion():
    r = C.resolve_config(full_champion_agentic_run_latest=False)
    # gsw_champion is off, so neither the overlay nor the preset applies
    assert "gsw_champion" not in r.extra and "gsw_investigation_memory" not in r.extra
    assert r.extra["gsw_edge_grounded_cap"] == 5


def test_omit_if_falsy_keys_never_reach_extra_when_off():
    extra = C.build_gsw_extra({"gsw_investigation_memory": False,
                               "gsw_codegen_compute": False, "gsw_evidence_gate": "off",
                               "gsw_block_unverified_cascade": False})
    assert extra == {"gsw_evidence_gate": "off", "gsw_block_unverified_cascade": False}


def test_orchestrator_turn_default_literal_matches_the_pin():
    from panini_dr.adapters.ours import gsw_planner_orchestrator_v1 as orch

    src = inspect.getsource(orch.OursGSWPlannerOrchestratorV1Adapter._run_llm_orchestrator)
    m = re.search(r'extra\.get\("orchestrator_max_turns",\s*(\d+)\)', src)
    assert m and int(m.group(1)) == C._ORCHESTRATOR_TURNS_DEFAULT
    assert C.load_paper_config()["pinned_adapter_defaults"]["orchestrator_max_turns"] == 12


def test_paper_json_is_valid_json_with_trailing_newline():
    text = C.paper_config_path().read_text()
    assert text.endswith("\n")
    json.loads(text)
