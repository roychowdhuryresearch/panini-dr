"""Shape-validation tests for the prompt v4 few-shots.

These tests do NOT call an LLM. They take the synthetic few-shot
plans shipped in ``_planner_prompts._FEW_SHOTS`` and push them through
the executor's Pydantic schema to ensure they never silently ship
broken:

- Every plan validates (no dangling entities; all roles present;
  exactly one target blank).
- The literal-offset few-shot uses `literal_value` + `args_refs`, not a
  fake blank for "twelve".
- No few-shot teaches placeholder collective expansion names.

If a future prompt edit breaks a few-shot, these tests fail fast.
"""

from __future__ import annotations

import hashlib
import re

import pytest

from panini_dr.adapters.ours._planner_exec import GSWPlan
from panini_dr.adapters.ours._planner_prompts import (
    _BLANK_HINTS_INSTRUCTION,
    _FEW_SHOT_1_TEMPORAL_BRIDGE,
    _FEW_SHOT_2_AS_OF_DATE_DIFF,
    _FEW_SHOT_3_ENUMERATED_ARGMAX,
    _FEW_SHOT_4_COMPOUND_SCOPE,
    _FEW_SHOT_5_LITERAL_OFFSET,
    _FEW_SHOT_6_LIST_INTERSECTION,
    _FEW_SHOTS,
    PLANNER_SYSTEM,
    build_planner_messages,
    build_repair_messages,
)


_COLLECTIVE_PREFIX = "expanded_from_collective:"


@pytest.mark.parametrize(
    "example",
    [
        _FEW_SHOT_1_TEMPORAL_BRIDGE,
        _FEW_SHOT_2_AS_OF_DATE_DIFF,
        _FEW_SHOT_3_ENUMERATED_ARGMAX,
        _FEW_SHOT_4_COMPOUND_SCOPE,
        _FEW_SHOT_5_LITERAL_OFFSET,
        _FEW_SHOT_6_LIST_INTERSECTION,
    ],
    ids=[
        "temporal_bridge",
        "as_of_date_diff",
        "enumerated_argmax",
        "compound_scope",
        "literal_offset",
        "list_intersection",
    ],
)
def test_few_shot_plan_validates(example: dict) -> None:
    plan = GSWPlan.model_validate(example["plan"])

    # Exactly one target blank.
    targets = [e for e in plan.blank_entities() if e.is_target]
    assert len(targets) == 1, f"expected exactly one target blank, got {len(targets)}"

    # Every entity has a role.
    missing_roles = [e.id for e in plan.entities if not (e.role or "").strip()]
    assert not missing_roles, f"entities missing role: {missing_roles}"

    # Every entity participates in at least one VP or constraint — this is
    # already enforced by the GSWPlan validator, so just confirm we
    # reached here without ValidationError.
    assert len(plan.entities) >= 2


def test_few_shots_list_has_six() -> None:
    """The prompt feeds exactly 6 few-shots into the user message."""
    assert len(_FEW_SHOTS) == 6


def test_no_few_shot_uses_state_breadcrumb() -> None:
    """Few-shots should not teach fake collective expansions."""
    for ex in _FEW_SHOTS:
        plan = GSWPlan.model_validate(ex["plan"])
        leaked = [
            e.id for e in plan.entities
            if (e.state or "").startswith(_COLLECTIVE_PREFIX)
        ]
        assert not leaked, (
            f"few-shot '{ex['question'][:60]}' unexpectedly carries state "
            f"breadcrumb on entities: {leaked}"
        )


def test_literal_offset_shot_uses_literal_value_and_args_refs() -> None:
    plan = GSWPlan.model_validate(_FEW_SHOT_5_LITERAL_OFFSET["plan"])
    literal = plan.entity_by_id("e_nine")
    assert literal.role == "constraint-value"
    assert literal.literal_value == 9

    constraint = plan.constraints[0]
    assert constraint.args_refs == ["e_year", "e_nine"]
    assert constraint.args_blanks == []


def test_list_intersection_shot_uses_intermediate_list_blanks() -> None:
    plan = GSWPlan.model_validate(_FEW_SHOT_6_LIST_INTERSECTION["plan"])
    list_blanks = [e.id for e in plan.blank_entities() if e.value_type == "list"]
    assert set(list_blanks) == {"b_hr_list", "b_sb_list"}

    target_vps = [
        vp for vp in plan.verb_phrases
        if "t" in (vp.subject_id, vp.object_id)
    ]
    assert len(target_vps) == 2
    assert {vp.object_id for vp in target_vps} == {"b_hr_list", "b_sb_list"}


def test_as_of_date_shot_has_date_anchor_role() -> None:
    """Few-shot #2 explicitly demonstrates Hard rule 2 — an
    ``as-of-date`` filled entity should be present."""
    plan = GSWPlan.model_validate(_FEW_SHOT_2_AS_OF_DATE_DIFF["plan"])
    as_of_date_roles = [
        e for e in plan.entities
        if e.kind == "filled" and e.role == "as-of-date"
    ]
    assert as_of_date_roles, "few-shot #2 must carry a filled as-of-date entity"


def test_temporal_bridge_shot_has_year_anchor_role() -> None:
    """Few-shot #1 demonstrates Hard rule 2 via the ``year-anchor`` role."""
    plan = GSWPlan.model_validate(_FEW_SHOT_1_TEMPORAL_BRIDGE["plan"])
    year_anchors = [
        e for e in plan.entities
        if e.kind == "filled" and e.role == "year-anchor"
    ]
    assert year_anchors, "few-shot #1 must carry a filled year-anchor entity"


def test_planner_prompt_defaults_to_v4_target_shape(monkeypatch) -> None:
    monkeypatch.delenv("GSW_PLANNER_TARGET_HUB_MODE", raising=False)
    monkeypatch.delenv("GSW_STRICT_TARGET_HUB_GATE", raising=False)
    messages = build_planner_messages("Which actor appeared in FILM?")
    system = messages[0]["content"]
    user = messages[1]["content"]
    repair = build_repair_messages("Q", "{}", "dangling entity")[3]["content"]
    assert "TARGET is not a junk drawer" in system
    assert "literal answer requested by the question" in system
    assert "full `date` blanks" in system
    assert "Never compute age-at-event as event_year - birth_year" in system
    assert "narrow coverage contract for explicit lists" in system
    assert "filtered-list" in system
    assert "choose the minimal sufficient TARGET shape" not in system
    assert "avoid risky target hubs when intermediate blanks are needed" in user
    assert "use full date blanks for age/elapsed-year questions" in user
    assert "never compute age-at-event from birth_year/event_year alone" in user
    assert "narrow coverage only for explicit multi-item/count questions" in user
    assert "Avoids risky TARGET hubs" in repair
    assert "literal requested answer" in repair
    assert "never event_year - birth_year" in repair
    assert "candidate-list blank" in repair
    assert "filtered-list blank" in repair


def test_planner_prompt_can_opt_into_hybrid_target_shape(monkeypatch) -> None:
    monkeypatch.setenv("GSW_PLANNER_TARGET_HUB_MODE", "hybrid")
    messages = build_planner_messages("Which actor appeared in FILM?")
    system = messages[0]["content"]
    user = messages[1]["content"]
    repair = build_repair_messages("Q", "{}", "dangling entity")[3]["content"]
    assert "choose the minimal sufficient TARGET shape" in system
    assert "literal answer requested by the question" in system
    assert "full `date` blanks" in system
    assert "Never compute age-at-event as event_year - birth_year" in system
    assert "narrow coverage contract for explicit lists" in system
    assert "filtered-list" in system
    assert "numeric thresholds/comparisons" in system
    assert "computed, compared, counted, or formatted" in user
    assert "never compute age-at-event from birth_year/event_year alone" in user
    assert "narrow coverage only for explicit multi-item/count questions" in user
    assert "Uses the minimal sufficient TARGET shape" in repair
    assert "literal\n   answer requested by the question" in repair
    assert "never event_year - birth_year" in repair
    assert "candidate-list blank" in repair
    assert "filtered-list blank" in repair


def test_qualifier_instruction_only_when_enabled():
    base = build_planner_messages("q")[0]["content"]
    on = build_planner_messages("q", blank_qualifier=True)[0]["content"]
    assert "instance_of" not in base
    assert "instance_of" in on and "scope-filter" in on
    # user message unchanged
    assert build_planner_messages("q")[1] == build_planner_messages("q", blank_qualifier=False)[1]


# ---------------------------------------------------------------------------
# Hard rule 7 — unit consistency across combined numeric blanks
# ---------------------------------------------------------------------------
#
# Cross-unit arithmetic bug fix: the planner must declare a unit in the
# PLAN (via retrieval_hint) rather than the executor reconciling units
# after the fact. See _planner_emit._reject_unit_mismatch_constraints for
# the deterministic-check half of this fix.


def test_hard_rule_7_present_in_planner_system_unconditionally():
    """Hard rule 7 is a base v4 rule -- always present, not gated behind
    blank_hints, matching Hard rules 1-6."""
    system_off = build_planner_messages("Who founded Nike?", blank_hints=False)[0]["content"]
    system_on = build_planner_messages("Who founded Nike?", blank_hints=True)[0]["content"]
    for system in (system_off, system_on):
        assert "Hard rule 7" in system
        assert "unit consistency" in system.lower()
        assert "diff, sum, avg, max, min, mul, div" in system
        assert "same unit" in system.lower()


def test_blank_hints_instruction_operationalizes_unit_rule_only_when_on():
    """The concrete retrieval_hint-level unit guidance (with the fictional
    Ashgrove Spire / Ferrymoor Lighthouse example) only appears when
    blank_hints=True, since retrieval_hint itself is only introduced then."""
    system_on = build_planner_messages("Q?", blank_hints=True)[0]["content"]
    system_off = build_planner_messages("Q?", blank_hints=False)[0]["content"]
    assert "UNIT CONSISTENCY (Hard rule 7)" in system_on
    assert "Ashgrove Spire" in system_on
    assert "Ferrymoor Lighthouse" in system_on
    assert "UNIT CONSISTENCY" not in system_off
    assert "Ashgrove Spire" not in system_off


def test_unit_rule_few_shot_example_uses_fictional_entities():
    """The illustrative unit-consistency example must not leak a real FRAMES
    eval entity (same rule the module header states for the six few-shots)."""
    assert "Ashgrove Spire" in _BLANK_HINTS_INSTRUCTION
    assert "Ferrymoor Lighthouse" in _BLANK_HINTS_INSTRUCTION
    # Spot-check against a couple of the real entities from the verified
    # bug traces (q16/q692/q771) -- these must NEVER appear in the prompt.
    for leaked in ("New Britain Trench", "Mount Katahdin", "Mount Kilimanjaro", "Alabama"):
        assert leaked not in _BLANK_HINTS_INSTRUCTION


# Pre-Hard-rule-7 baseline hashes, captured from the exact PLANNER_SYSTEM /
# _BLANK_HINTS_INSTRUCTION content on this branch BEFORE the unit-rule was
# added. These are regression guards, not literal "unchanged" assertions
# (Hard rule 7 is an always-on addition to the base prompt, per the fix
# design -- see the module's PR/report): they prove the unit-rule text is
# the ONLY diff from the pre-fix prompt, i.e. splicing it back out
# reproduces the old prompt byte-for-byte.
#
# Updated 2026-08-07 (plan-verifier sort-op report) for the `sort` derived
# op: PLANNER_SYSTEM gained `sort` in both op enumerations (Step 3 and the
# Field spec) plus its one-line usage note, all OUTSIDE the Hard rule 7
# span this test strips, so the stripped-prompt hash moved. Recomputed by
# hashing the current PLANNER_SYSTEM with the same strip regex below.
_PRE_UNIT_RULE_PLANNER_SYSTEM_SHA256 = (
    "652a48fcd375d2ed82751c8987a034eb333f6e25f939be8129438bacfd15792f"
)
_PRE_UNIT_RULE_BLANK_HINTS_INSTRUCTION_SHA256 = (
    "904fcfa2538fadc4f70a376d3ee1f7ab8ab6077f7f3a60bceaa097e9e48e9871"
)


def test_planner_system_byte_identical_to_pre_unit_rule_baseline_modulo_rule_7():
    stripped = re.sub(
        r"\n\n[ \t]*## Hard rule 7.*?(?=\n\n[ \t]*## Field spec \(strict\))",
        "",
        PLANNER_SYSTEM,
        flags=re.DOTALL,
    )
    assert stripped != PLANNER_SYSTEM, "Hard rule 7 text was not found to strip"
    assert hashlib.sha256(stripped.encode()).hexdigest() == _PRE_UNIT_RULE_PLANNER_SYSTEM_SHA256


def test_blank_hints_instruction_byte_identical_to_pre_unit_rule_baseline_modulo_rule_7():
    """Modulo BOTH Hard rule 7 (unit consistency) and Hard rule 8 (qualifier
    preservation) — both are always-on additions to the base blank-hints
    instruction, per their respective fix designs; stripping both bullets
    and their illustrative examples reproduces the pre-Hard-rule-7 text."""
    stripped = re.sub(
        r"  - UNIT CONSISTENCY \(Hard rule 7\):.*?constraint\.\n",
        "",
        _BLANK_HINTS_INSTRUCTION,
    )
    stripped = re.sub(
        r"  - QUALIFIER PRESERVATION \(Hard rule 8\):.*?\)\.\n",
        "",
        stripped,
    )
    stripped = re.sub(r" Unit example:.*?unit-safe\.", "", stripped)
    stripped = re.sub(r" Qualifier example:.*?wrongly pass\.", "", stripped)
    assert stripped != _BLANK_HINTS_INSTRUCTION, "unit/qualifier bullet/example text was not found to strip"
    assert (
        hashlib.sha256(stripped.encode()).hexdigest()
        == _PRE_UNIT_RULE_BLANK_HINTS_INSTRUCTION_SHA256
    )


def test_build_planner_messages_user_message_unaffected_by_unit_rule():
    """The unit rule lives entirely in the system prompt; the user message
    (few-shots + question) must be byte-identical to before."""
    user = build_planner_messages("Who founded Nike?", blank_hints=True)[1]
    expected_sha = "9b227fea36b3f64e113dd892222866b2fc5982b92361b7ac53730e0925612910"
    assert hashlib.sha256(user["content"].encode()).hexdigest() == expected_sha


# ---------------------------------------------------------------------------
# Hard rule 8 — qualifier words ("full", "exact", "official", ...) must
# survive verbatim into the blank's retrieval_hint / verification_claim.
# q39 fix 1b: the planner dropped "full" from "the FULL first name of the
# swimmer", so a nickname satisfied the (unqualified) hint. See
# _planner_emit._reject_dropped_qualifier_hints for the deterministic
# backstop half of this fix.
# ---------------------------------------------------------------------------


def test_hard_rule_8_present_in_planner_system_unconditionally():
    """Hard rule 8 is a base v4 rule -- always present, not gated behind
    blank_hints, matching Hard rules 1-7."""
    system_off = build_planner_messages("Who founded Nike?", blank_hints=False)[0]["content"]
    system_on = build_planner_messages("Who founded Nike?", blank_hints=True)[0]["content"]
    for system in (system_off, system_on):
        assert "Hard rule 8" in system
        assert "qualifier" in system.lower()
        for word in ("full", "exact", "official", "original", "legal", "formal", "complete", "maiden"):
            assert word in system.lower()


def test_blank_hints_instruction_operationalizes_qualifier_rule_only_when_on():
    """The concrete retrieval_hint-level qualifier guidance only appears
    when blank_hints=True, since retrieval_hint itself is only introduced
    then (same pattern as the Hard rule 7 unit guidance)."""
    system_on = build_planner_messages("Q?", blank_hints=True)[0]["content"]
    system_off = build_planner_messages("Q?", blank_hints=False)[0]["content"]
    assert "QUALIFIER PRESERVATION (Hard rule 8)" in system_on
    assert "QUALIFIER PRESERVATION" not in system_off


def test_qualifier_rule_few_shot_example_uses_fictional_entities():
    """The illustrative qualifier-preservation example must not leak a real
    FRAMES eval entity."""
    assert "Ada Voss" in _BLANK_HINTS_INSTRUCTION
    for leaked in ("Katie", "Kathleen", "Arianne"):
        assert leaked not in _BLANK_HINTS_INSTRUCTION
        assert leaked not in PLANNER_SYSTEM
