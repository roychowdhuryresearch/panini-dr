"""Unit tests for the planner-emit helper (multi-retry repair + exhausted)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from panini_dr.adapters.ours._planner_emit import (
    MAX_REPAIR_ATTEMPTS,
    PlanEmitError,
    _reject_dropped_qualifier_hints,
    _reject_unit_mismatch_constraints,
    emit_plan,
)
from panini_dr.adapters.ours._planner_exec import GSWPlan


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubResp:
    text: str = ""
    prompt_tokens: int = 5
    completion_tokens: int = 7
    reasoning_content: str = ""


class _ScriptedLLM:
    """Returns scripted responses one at a time."""

    def __init__(self, script: list[_StubResp]):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, messages, *, max_tokens=None, **kwargs):
        self.calls.append({"messages": messages, "max_tokens": max_tokens, **kwargs})
        if not self.script:
            return _StubResp(text="(out of script)")
        return self.script.pop(0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _valid_plan_json(question: str = "What is X?") -> str:
    """A minimal grounded GSWPlan that passes validation.

    Filled name "X" is a substring of the question text so the plan
    grounding test passes if any extra prompt-side checks are layered
    in. (The pydantic-level validators don't actually enforce
    grounding — the planner-side prompt does — so a tiny plan suffices.)
    """
    return json.dumps({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "X", "role": "subject"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "text",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "is", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    })


def _valid_number_diff_plan_json() -> str:
    return json.dumps({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "A", "role": "subject"},
            {"id": "e2", "kind": "filled", "name": "B", "role": "subject"},
            {"id": "b1", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "b2", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "number", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_value", "subject_id": "e1", "object_id": "b1"},
            {"id": "vp2", "phrase": "has_value", "subject_id": "e2", "object_id": "b2"},
        ],
        "constraints": [
            {"id": "c1", "kind": "derived", "op": "diff", "args_refs": ["b1", "b2"], "output_blank_id": "t"},
        ],
    })


def _valid_age_date_plan_json() -> str:
    return json.dumps({
        "entities": [
            {"id": "e_film", "kind": "filled", "name": "The Dark Knight", "role": "subject"},
            {"id": "e_fifty", "kind": "filled", "name": "50", "role": "constraint-value", "literal_value": 50},
            {"id": "b_actor", "kind": "blank", "role": "bridge-entity", "value_type": "entity"},
            {"id": "b_child", "kind": "blank", "role": "bridge-entity", "value_type": "entity"},
            {"id": "b_release_date", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "b_child_birth_date", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "b_anniversary_date", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "number", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "played_by", "subject_id": "e_film", "object_id": "b_actor"},
            {"id": "vp2", "phrase": "has_child", "subject_id": "b_actor", "object_id": "b_child"},
            {"id": "vp3", "phrase": "released_on", "subject_id": "e_film", "object_id": "b_release_date"},
            {"id": "vp4", "phrase": "born_on", "subject_id": "b_child", "object_id": "b_child_birth_date"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "derived",
                "op": "sum",
                "args_refs": ["b_release_date", "e_fifty"],
                "output_blank_id": "b_anniversary_date",
            },
            {
                "id": "c2",
                "kind": "derived",
                "op": "diff",
                "args_refs": ["b_anniversary_date", "b_child_birth_date"],
                "output_blank_id": "t",
            },
        ],
    })


def _valid_threshold_entity_plan_json() -> str:
    return json.dumps({
        "entities": [
            {"id": "e_threshold", "kind": "filled", "name": "15", "role": "constraint-value", "literal_value": 15},
            {"id": "e_club", "kind": "filled", "name": "Auxerre", "role": "scope-filter"},
            {"id": "b_goals", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "b_gt", "kind": "blank", "role": "comparison-output", "value_type": "bool"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "played_for", "subject_id": "t", "object_id": "e_club"},
            {"id": "vp2", "phrase": "goals_scored", "subject_id": "t", "object_id": "b_goals"},
        ],
        "constraints": [
            {"id": "c1", "kind": "gt", "left_ref": "b_goals", "right_ref": "e_threshold", "output_blank_id": "b_gt"},
        ],
    })


def _valid_letter_count_plan_json() -> str:
    return json.dumps({
        "entities": [
            {"id": "e_team", "kind": "filled", "name": "The Defenders", "role": "subject"},
            {"id": "b_member", "kind": "blank", "role": "bridge-entity", "value_type": "entity"},
            {"id": "b_name", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "number", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "non_doctor_member", "subject_id": "e_team", "object_id": "b_member"},
            {"id": "vp2", "phrase": "has_name", "subject_id": "b_member", "object_id": "b_name"},
        ],
        "constraints": [
            {"id": "c1", "kind": "derived", "op": "count", "args_refs": ["b_name"], "output_blank_id": "t"},
        ],
    })


def _valid_argmin_filled_candidates_plan_json() -> str:
    return json.dumps({
        "entities": [
            {"id": "e_jerry", "kind": "filled", "name": "Jerry", "role": "candidate"},
            {"id": "e_kramer", "kind": "filled", "name": "Kramer", "role": "candidate"},
            {"id": "b_birth_jerry", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "b_birth_kramer", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "born_in", "subject_id": "e_jerry", "object_id": "b_birth_jerry"},
            {"id": "vp2", "phrase": "born_in", "subject_id": "e_kramer", "object_id": "b_birth_kramer"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "argmin",
                "candidate_entity_ids": ["e_jerry", "e_kramer"],
                "sort_by_blank_ids": ["b_birth_jerry", "b_birth_kramer"],
                "output_blank_id": "t",
            },
        ],
    })


def _valid_rank_n_candidate_set_plan_json() -> str:
    return json.dumps({
        "entities": [
            {"id": "b_rows", "kind": "blank", "role": "candidate-set", "value_type": "list"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [],
        "constraints": [
            {
                "id": "c1",
                "kind": "rank_n",
                "args_blanks": ["b_rows"],
                "numeric_attribute": "record_sales",
                "n": 2,
                "descending": True,
                "output_blank_id": "t",
            },
        ],
    })

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_emit_plan_first_attempt_success():
    llm = _ScriptedLLM([_StubResp(text=_valid_plan_json())])
    plan, meta = emit_plan("What is X?", llm)
    assert plan.target().id == "t"
    assert meta.repair_used is False
    assert meta.repair_attempts == []
    assert len(llm.calls) == 1


def test_emit_plan_repairs_dormant_row_aggregation_constraints():
    llm = _ScriptedLLM([
        _StubResp(text=_valid_rank_n_candidate_set_plan_json()),
        _StubResp(text=_valid_plan_json()),
    ])

    plan, meta = emit_plan("Which song had the second-highest record sales?", llm)

    assert plan.target().id == "t"
    assert meta.repair_used is True
    assert "row_aggregation_constraints_unavailable" in meta.parse_error


def test_emit_plan_repairs_after_one_failure():
    """First call returns junk; first repair returns valid JSON."""
    llm = _ScriptedLLM([
        _StubResp(text="not json"),
        _StubResp(text=_valid_plan_json()),
    ])
    plan, meta = emit_plan("What is X?", llm)
    assert plan.target().id == "t"
    assert meta.repair_used is True
    assert len(meta.repair_attempts) == 1
    assert meta.repair_attempts[0]["error"] == ""
    assert meta.parse_error  # captured first failure
    assert len(llm.calls) == 2


def test_emit_plan_passes_seed_to_initial_and_repair_calls():
    llm = _ScriptedLLM([
        _StubResp(text="not json"),
        _StubResp(text=_valid_plan_json()),
    ])
    plan, _meta = emit_plan("What is X?", llm, llm_seed=123)
    assert plan.target().id == "t"
    assert [c.get("seed") for c in llm.calls] == [123, 123]


def test_emit_plan_repairs_on_third_attempt():
    """First call + 2 bad repairs, third repair succeeds."""
    llm = _ScriptedLLM([
        _StubResp(text="not json"),
        _StubResp(text="still not json"),
        _StubResp(text="more garbage"),
        _StubResp(text=_valid_plan_json()),
    ])
    plan, meta = emit_plan("What is X?", llm)
    assert plan.target().id == "t"
    assert meta.repair_used is True
    assert len(meta.repair_attempts) == 3
    assert meta.repair_attempts[0]["error"]
    assert meta.repair_attempts[1]["error"]
    assert meta.repair_attempts[2]["error"] == ""
    assert len(llm.calls) == 4


def test_emit_plan_raises_exhausted_after_all_retries():
    """First + 3 bad repairs → PlanEmitError(kind='exhausted')."""
    llm = _ScriptedLLM([
        _StubResp(text="garbage 0"),
        _StubResp(text="garbage 1"),
        _StubResp(text="garbage 2"),
        _StubResp(text="garbage 3"),
    ])
    with pytest.raises(PlanEmitError) as exc_info:
        emit_plan("What is X?", llm)
    assert exc_info.value.kind == "exhausted"
    assert "3 repair attempts" in exc_info.value.detail
    assert len(llm.calls) == 1 + MAX_REPAIR_ATTEMPTS  # 4 total


def test_emit_plan_repair_messages_sharpen_each_round():
    """The repair messages on attempts 2 and 3 must mention the attempt
    index so the LLM knows it has already failed earlier rounds."""
    llm = _ScriptedLLM([
        _StubResp(text="g0"),
        _StubResp(text="g1"),
        _StubResp(text="g2"),
        _StubResp(text=_valid_plan_json()),
    ])
    emit_plan("What is X?", llm)

    # Repair-call messages are at call indices 1, 2, 3.
    repair2_user_msg = llm.calls[2]["messages"][-1]["content"]
    repair3_user_msg = llm.calls[3]["messages"][-1]["content"]

    assert "attempt 2" in repair2_user_msg
    assert "FINAL repair attempt" in repair3_user_msg


def test_emit_plan_first_repair_uses_legacy_preface():
    """Attempt 1 of repair preserves the original (non-sharpened) tone."""
    llm = _ScriptedLLM([
        _StubResp(text="garbage"),
        _StubResp(text=_valid_plan_json()),
    ])
    emit_plan("What is X?", llm)
    repair1_user_msg = llm.calls[1]["messages"][-1]["content"]
    # First repair must NOT carry the "attempt N" preface.
    assert "attempt 2" not in repair1_user_msg
    assert "FINAL repair attempt" not in repair1_user_msg
    # But it must still carry the structural rules.
    assert "no dangling entities" in repair1_user_msg


def test_emit_plan_disable_repair_raises_parse_failure():
    """enable_repair=False short-circuits to parse_failure on first bad JSON."""
    llm = _ScriptedLLM([_StubResp(text="not json")])
    with pytest.raises(PlanEmitError) as exc_info:
        emit_plan("What is X?", llm, enable_repair=False)
    assert exc_info.value.kind == "parse_failure"
    assert len(llm.calls) == 1


def test_emit_plan_repairs_literal_constant_blank():
    bad_plan = json.dumps({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "X", "role": "subject"},
            {"id": "e_twelve", "kind": "filled", "name": "twelve", "role": "constraint-value"},
            {"id": "b_offset", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "number", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "value", "subject_id": "e_twelve", "object_id": "b_offset"},
            {"id": "vp2", "phrase": "has_value", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    })
    llm = _ScriptedLLM([
        _StubResp(text=bad_plan),
        _StubResp(text=_valid_number_diff_plan_json()),
    ])
    plan, meta = emit_plan("What is twelve years earlier than X?", llm)
    assert plan.target().id == "t"
    assert meta.repair_used is True
    assert "literal_constant_blank" in meta.parse_error


def test_emit_plan_repairs_numeric_literal_ref_without_literal_value():
    bad_plan = json.dumps({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "X", "role": "subject"},
            {"id": "e_twelve", "kind": "filled", "name": "twelve", "role": "constraint-value"},
            {"id": "b_year", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "date", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "won_in_year", "subject_id": "e1", "object_id": "b_year"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "derived",
                "op": "diff",
                "args_refs": ["b_year", "e_twelve"],
                "output_blank_id": "t",
            }
        ],
    })
    llm = _ScriptedLLM([
        _StubResp(text=bad_plan),
        _StubResp(text=_valid_number_diff_plan_json()),
    ])
    plan, meta = emit_plan("What is twelve years earlier than X?", llm)
    assert plan.target().id == "t"
    assert meta.repair_used is True
    assert "literal_value_missing" in meta.parse_error


def test_emit_plan_repairs_age_from_year_only_blanks():
    bad_plan = json.dumps({
        "entities": [
            {"id": "e_film", "kind": "filled", "name": "The Dark Knight", "role": "subject"},
            {"id": "e_year", "kind": "filled", "name": "2008", "role": "constraint-value", "literal_value": 2008},
            {"id": "e_fifty", "kind": "filled", "name": "50", "role": "constraint-value", "literal_value": 50},
            {"id": "b_actor", "kind": "blank", "role": "bridge-entity", "value_type": "entity"},
            {"id": "b_child", "kind": "blank", "role": "bridge-entity", "value_type": "entity"},
            {"id": "b_child_birth_year", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "b_anniversary_year", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "number", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "played_by", "subject_id": "e_film", "object_id": "b_actor"},
            {"id": "vp2", "phrase": "has_child", "subject_id": "b_actor", "object_id": "b_child"},
            {"id": "vp3", "phrase": "born_in_year", "subject_id": "b_child", "object_id": "b_child_birth_year"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "derived",
                "op": "sum",
                "args_refs": ["e_year", "e_fifty"],
                "output_blank_id": "b_anniversary_year",
            },
            {
                "id": "c2",
                "kind": "derived",
                "op": "diff",
                "args_refs": ["b_anniversary_year", "b_child_birth_year"],
                "output_blank_id": "t",
            },
        ],
    })
    llm = _ScriptedLLM([
        _StubResp(text=bad_plan),
        _StubResp(text=_valid_age_date_plan_json()),
    ])
    plan, meta = emit_plan(
        'If the person who played the joker in "The Dark Knight" had children, how old will they be on the film\'s 50th anniversary?',
        llm,
    )
    assert plan.target().id == "t"
    assert meta.repair_used is True
    assert "date_granularity_missing" in meta.parse_error


def test_emit_plan_allows_full_date_age_with_years_target_name():
    plan_dict = json.loads(_valid_age_date_plan_json())
    for ent in plan_dict["entities"]:
        if ent["id"] == "t":
            ent["id"] = "t_age_years"
            break
    for constraint in plan_dict["constraints"]:
        if constraint.get("output_blank_id") == "t":
            constraint["output_blank_id"] = "t_age_years"

    llm = _ScriptedLLM([_StubResp(text=json.dumps(plan_dict))])
    plan, meta = emit_plan(
        'If the person who played the joker in "The Dark Knight" had children, how old will they be on the film\'s 50th anniversary?',
        llm,
    )
    assert plan.target().id == "t_age_years"
    assert meta.repair_used is False
    assert "date_granularity_missing" not in meta.extras["plan_audit_tags"]


def test_emit_plan_audits_risky_target_hub_without_repair():
    plan_json = json.dumps({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Kessington", "role": "subject"},
            {"id": "e2", "kind": "filled", "name": "Raleford", "role": "subject"},
            {"id": "e_date", "kind": "filled", "name": "2047", "role": "year-anchor"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "number", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_population", "subject_id": "e1", "object_id": "t"},
            {"id": "vp2", "phrase": "has_population", "subject_id": "e2", "object_id": "t"},
            {"id": "vp3", "phrase": "as_of_date", "subject_id": "t", "object_id": "e_date"},
        ],
        "constraints": [],
    })
    llm = _ScriptedLLM([_StubResp(text=plan_json)])
    plan, meta = emit_plan(
        "What is the difference in population between Kessington and Raleford in 2047?",
        llm,
    )
    assert plan.target().id == "t"
    assert meta.repair_used is False
    assert "under_decomposed" in meta.extras["plan_audit_tags"]
    assert "ambiguous_target" not in meta.extras["plan_audit_tags"]


def test_emit_plan_allows_direct_entity_intersection_target_hub():
    plan_json = json.dumps({
        "entities": [
            {"id": "e_film", "kind": "filled", "name": "FILM", "role": "scope-filter"},
            {"id": "e_play", "kind": "filled", "name": "PLAY", "role": "scope-filter"},
            {"id": "e_city", "kind": "filled", "name": "CITY", "role": "scope-filter"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "appeared_in", "subject_id": "t", "object_id": "e_film"},
            {"id": "vp2", "phrase": "directed", "subject_id": "t", "object_id": "e_play"},
            {"id": "vp3", "phrase": "born_in", "subject_id": "t", "object_id": "e_city"},
        ],
        "constraints": [],
    })
    llm = _ScriptedLLM([_StubResp(text=plan_json)])
    plan, meta = emit_plan(
        "Which actor appeared in FILM, directed PLAY, and was born in CITY?",
        llm,
    )
    assert plan.target().id == "t"
    assert meta.repair_used is False
    assert len(llm.calls) == 1


def test_emit_plan_audits_threshold_target_hub_without_comparison():
    plan_json = json.dumps({
        "entities": [
            {"id": "e_goals", "kind": "filled", "name": "15", "role": "constraint-value", "literal_value": 15},
            {"id": "e_league", "kind": "filled", "name": "Eredivisie", "role": "scope-filter"},
            {"id": "e_season", "kind": "filled", "name": "21-22", "role": "year-anchor"},
            {"id": "e_club", "kind": "filled", "name": "Auxerre", "role": "scope-filter"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "scored_more_than", "subject_id": "t", "object_id": "e_goals"},
            {"id": "vp2", "phrase": "played_in_league", "subject_id": "t", "object_id": "e_league"},
            {"id": "vp3", "phrase": "in_season", "subject_id": "t", "object_id": "e_season"},
            {"id": "vp4", "phrase": "previously_played_for", "subject_id": "t", "object_id": "e_club"},
        ],
        "constraints": [],
    })
    llm = _ScriptedLLM([_StubResp(text=plan_json)])
    plan, meta = emit_plan(
        "Which player scored more than 15 goals in Eredivisie during the 21-22 season and had previously played for Auxerre?",
        llm,
    )
    assert plan.target().id == "t"
    assert meta.repair_used is False
    assert "missing_operation" in meta.extras["plan_audit_tags"]


def test_emit_plan_audits_numeric_question_targeting_entity():
    plan_json = json.dumps({
        "entities": [
            {"id": "e_team", "kind": "filled", "name": "The Defenders", "role": "subject"},
            {"id": "e_doctor", "kind": "filled", "name": "doctor", "role": "scope-filter", "category": True},
            {"id": "b_member", "kind": "blank", "role": "target", "value_type": "entity", "is_target": True},
            {"id": "b_count", "kind": "blank", "role": "bridge-number", "value_type": "number"},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_founding_member", "subject_id": "e_team", "object_id": "b_member"},
            {"id": "vp2", "phrase": "not_profession", "subject_id": "b_member", "object_id": "e_doctor"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "derived",
                "op": "count",
                "args_refs": ["b_member"],
                "output_blank_id": "b_count",
            }
        ],
    })
    llm = _ScriptedLLM([_StubResp(text=plan_json)])
    plan, meta = emit_plan(
        "Only one of the founding members of The Defenders was not a doctor. How many letters are in his name?",
        llm,
    )
    assert plan.target().id == "b_member"
    assert meta.repair_used is False
    assert "ambiguous_target" in meta.extras["plan_audit_tags"]


def test_emit_plan_does_not_reject_entity_answer_with_numeric_context():
    plan_json = json.dumps({
        "entities": [
            {"id": "e_award", "kind": "filled", "name": "2011 Dorothy Arzner Directors award", "role": "scope-filter"},
            {"id": "e_started", "kind": "filled", "name": "2005", "role": "year-anchor"},
            {"id": "e_count", "kind": "filled", "name": "20", "role": "constraint-value", "literal_value": 20},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "directed_by_award_winner", "subject_id": "t", "object_id": "e_award"},
            {"id": "vp2", "phrase": "series_started_in", "subject_id": "t", "object_id": "e_started"},
            {"id": "vp3", "phrase": "season_episode_count", "subject_id": "t", "object_id": "e_count"},
        ],
        "constraints": [],
    })
    llm = _ScriptedLLM([_StubResp(text=plan_json)])
    plan, meta = emit_plan(
        'The series started in 2005. The season that contains my episode included 20 episodes in total. The episode is one word that starts with an "M."',
        llm,
    )
    assert plan.target().id == "t"
    assert meta.repair_used is False
    assert "ambiguous_target" in meta.extras["plan_audit_tags"]


def test_emit_plan_qualifier_edge_when_flag_on():
    plan_json = (
        '{"entities":['
        '{"id":"e1","kind":"filled","name":"Netflix","role":"subject"},'
        '{"id":"env","kind":"filled","name":"environmental group",'
        '"category":true,"role":"scope-filter"},'
        '{"id":"b_environmental_group","kind":"blank","value_type":"entity",'
        '"role":"bridge-entity"},'
        '{"id":"t","kind":"blank","value_type":"entity","is_target":true,'
        '"role":"target"}],'
        '"verb_phrases":['
        '{"id":"vpq","subject_id":"b_environmental_group","phrase":"instance_of",'
        '"object_id":"env"},'
        '{"id":"vt","subject_id":"t","phrase":"co_founded",'
        '"object_id":"b_environmental_group"},'
        '{"id":"ve1","subject_id":"t","phrase":"joined",'
        '"object_id":"e1"}],'
        '"constraints":[]}'
    )
    llm = _ScriptedLLM([_StubResp(text=plan_json)])
    plan, _meta = emit_plan(
        "environmental group co-founded by a businessman who joined Netflix's cofounder",
        llm, blank_qualifier=True)
    phrases = [(vp.subject_id, vp.phrase, vp.object_id) for vp in plan.verb_phrases]
    assert ("b_environmental_group", "instance_of", "env") in phrases
    b = plan.entity_by_id("b_environmental_group")
    assert b.kind == "blank" and (b.name is None)


def test_emit_plan_does_not_treat_what_song_did_as_yes_no():
    plan_json = json.dumps({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Christina Aguilera", "role": "subject"},
            {"id": "e2", "kind": "filled", "name": "...Baby One More Time", "role": "scope-filter"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "released_after", "subject_id": "e1", "object_id": "t"},
            {"id": "vp2", "phrase": "after_release", "subject_id": "t", "object_id": "e2"},
        ],
        "constraints": [],
    })
    llm = _ScriptedLLM([_StubResp(text=plan_json)])
    plan, meta = emit_plan(
        'What song did Christina Aguilera release after Britney Spears released "...Baby One More Time."?',
        llm,
    )
    assert plan.target().id == "t"
    assert meta.repair_used is False
    assert meta.extras["plan_audit_tags"] == ["ok_minimal"]


def test_emit_plan_repairs_argmin_over_blank_candidates():
    bad_plan = json.dumps({
        "entities": [
            {"id": "e_show", "kind": "filled", "name": "Seinfeld", "role": "subject"},
            {"id": "b_actor_jerry", "kind": "blank", "role": "bridge-entity", "value_type": "entity"},
            {"id": "b_actor_kramer", "kind": "blank", "role": "bridge-entity", "value_type": "entity"},
            {"id": "b_birth_jerry", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "b_birth_kramer", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "actor", "subject_id": "e_show", "object_id": "b_actor_jerry"},
            {"id": "vp2", "phrase": "actor", "subject_id": "e_show", "object_id": "b_actor_kramer"},
            {"id": "vp3", "phrase": "born_in", "subject_id": "b_actor_jerry", "object_id": "b_birth_jerry"},
            {"id": "vp4", "phrase": "born_in", "subject_id": "b_actor_kramer", "object_id": "b_birth_kramer"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "argmin",
                "candidate_entity_ids": ["b_actor_jerry", "b_actor_kramer"],
                "sort_by_blank_ids": ["b_birth_jerry", "b_birth_kramer"],
                "output_blank_id": "t",
            }
        ],
    })
    llm = _ScriptedLLM([
        _StubResp(text=bad_plan),
        _StubResp(text=_valid_argmin_filled_candidates_plan_json()),
    ])
    plan, meta = emit_plan("Of the actors on Seinfeld, which actor is the oldest?", llm)
    assert plan.target().id == "t"
    assert meta.repair_used is True
    assert "arg_selection_blank_candidates" in meta.parse_error


def test_emit_plan_strict_env_keeps_semantic_target_hubs_audit_only(monkeypatch):
    monkeypatch.setenv("GSW_STRICT_TARGET_HUB_GATE", "1")
    plan_json = json.dumps({
        "entities": [
            {"id": "e_goals", "kind": "filled", "name": "15", "role": "constraint-value", "literal_value": 15},
            {"id": "e_league", "kind": "filled", "name": "Eredivisie", "role": "scope-filter"},
            {"id": "e_season", "kind": "filled", "name": "21-22", "role": "year-anchor"},
            {"id": "e_club", "kind": "filled", "name": "Auxerre", "role": "scope-filter"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "scored_more_than", "subject_id": "t", "object_id": "e_goals"},
            {"id": "vp2", "phrase": "played_in_league", "subject_id": "t", "object_id": "e_league"},
            {"id": "vp3", "phrase": "in_season", "subject_id": "t", "object_id": "e_season"},
            {"id": "vp4", "phrase": "previously_played_for", "subject_id": "t", "object_id": "e_club"},
        ],
        "constraints": [],
    })
    llm = _ScriptedLLM([_StubResp(text=plan_json)])
    plan, meta = emit_plan(
        "Which player scored more than 15 goals in Eredivisie during the 21-22 season and had previously played for Auxerre?",
        llm,
    )
    assert plan.target().id == "t"
    assert meta.repair_used is False
    assert "missing_operation" in meta.extras["plan_audit_tags"]


def test_emit_plan_llm_exception_during_repair():
    """LLM-level exception during a repair call surfaces as llm_error."""

    class _BombingLLM:
        def __init__(self):
            self.calls: list[Any] = []

        def chat(self, messages, **kwargs):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return _StubResp(text="not json")
            raise RuntimeError("transport boom")

    llm = _BombingLLM()
    with pytest.raises(PlanEmitError) as exc_info:
        emit_plan("What is X?", llm)
    assert exc_info.value.kind == "llm_error"
    assert "boom" in exc_info.value.detail


def test_a_valid_first_response_is_exactly_one_call():
    llm = _ScriptedLLM([_StubResp(text=_valid_plan_json())])
    emit_plan("What is X?", llm)
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# Hard rule 7 / _reject_unit_mismatch_constraints — cross-unit arithmetic
# bug fix. See _planner_prompts.py Hard rule 7 for the prompt half; this
# is the deterministic backstop that drives the existing repair loop when
# the planner doesn't comply.
# ---------------------------------------------------------------------------


def _div_plan(
    *, hint_a: str | None, hint_b: str | None, op: str = "div", question: str = "Q?"
) -> GSWPlan:
    """A minimal two-numeric-blank derived-constraint plan, mirroring the
    q16/q692/q771 shape: two bridge-number blanks feed one arithmetic op."""
    return GSWPlan.model_validate({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "A", "role": "subject"},
            {"id": "e2", "kind": "filled", "name": "B", "role": "subject"},
            {
                "id": "b1", "kind": "blank", "role": "bridge-number", "value_type": "number",
                "retrieval_hint": hint_a,
            },
            {
                "id": "b2", "kind": "blank", "role": "bridge-number", "value_type": "number",
                "retrieval_hint": hint_b,
            },
            {"id": "t", "kind": "blank", "role": "target", "value_type": "number", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_value", "subject_id": "e1", "object_id": "b1"},
            {"id": "vp2", "phrase": "has_value", "subject_id": "e2", "object_id": "b2"},
        ],
        "constraints": [
            {"id": "c1", "kind": "derived", "op": op, "args_refs": ["b1", "b2"], "output_blank_id": "t"},
        ],
    })


def test_unit_check_fires_on_mismatched_units():
    """q16-shape: one hint asks for feet, the other for metres."""
    plan = _div_plan(
        hint_a="What is the depth of the New Britain Trench in feet?",
        hint_b="What is the height of the tallest SF building in metres?",
        question="how many times deeper is the trench than the tallest SF building is tall?",
    )
    with pytest.raises(ValueError, match="unit_mismatch"):
        _reject_unit_mismatch_constraints(
            "how many times deeper is the trench than the tallest SF building is tall?",
            plan,
        )


def test_unit_check_fires_when_one_operand_names_a_unit_and_the_other_does_not():
    """Partial compliance -- one blank declares a unit, its sibling is silent.
    This must still fire even with no magnitude cue in the question, since
    the operand-declared unit alone is signal enough."""
    plan = _div_plan(hint_a="What is the depth of X, in feet?", hint_b="What is the height of Y?")
    with pytest.raises(ValueError, match="unit_mismatch"):
        _reject_unit_mismatch_constraints("Q?", plan)


def test_unit_check_fires_on_no_unit_plan_that_needs_one():
    """q771-shape: neither blank's hint (nor the question) names any unit at
    all, but the question's magnitude-comparison phrasing ('makes up the
    height of') signals a physical measurement is being compared."""
    plan = _div_plan(hint_a="What is the height of Mount Kilimanjaro?", hint_b="What is the height of Mount Katahdin?")
    question = "How many Mount Katahdins makes up the height of Mount Kilimanjaro?"
    with pytest.raises(ValueError, match="unit_missing"):
        _reject_unit_mismatch_constraints(question, plan)


@pytest.mark.parametrize(
    "question",
    [
        "How many Emmy nominations does A have compared to B?",
        "What year was A born?",
        "How many more Grammy awards has A won than B?",
        "What is the difference in population between Kessington and Raleford?",
        "What is twelve years earlier than X?",
    ],
)
def test_unit_check_does_not_fire_on_count_year_or_ranking_plans(question):
    """False-positive guard: no unit anywhere, no magnitude-comparison cue
    -- this must be treated as a dimensionless count/year/rank diff, not
    forced into a pointless repair."""
    plan = _div_plan(hint_a=None, hint_b=None, op="diff", question=question)
    _reject_unit_mismatch_constraints(question, plan)  # must not raise


def test_unit_check_does_not_fire_when_units_match():
    plan = _div_plan(
        hint_a="What is the depth of X, in feet?",
        hint_b="What is the height of Y, in feet?",
    )
    _reject_unit_mismatch_constraints(
        "How many times deeper is X than Y is tall?", plan
    )  # must not raise


def test_unit_check_fires_on_centimetres_vs_metres():
    """Vocabulary-extension coverage: centimetres was added to _UNIT_VOCAB
    (previously excluded); this must now catch a cm-vs-metres mismatch the
    same way it already catches feet-vs-metres."""
    plan = _div_plan(
        hint_a="What is the width of X, in cm?",
        hint_b="What is the height of Y, in metres?",
    )
    with pytest.raises(ValueError, match="unit_mismatch"):
        _reject_unit_mismatch_constraints("Q?", plan)


def test_unit_check_does_not_fire_when_both_declare_centimetres():
    plan = _div_plan(
        hint_a="What is the width of X, in centimeters?",
        hint_b="What is the height of Y, in cm?",
    )
    _reject_unit_mismatch_constraints("Q?", plan)  # must not raise


def test_unit_check_ignores_round_nearest():
    """round_nearest is unary-ish (value + optional rounding interval), not
    a combine-two-measurements op -- must never trip this check."""
    plan = GSWPlan.model_validate({
        "entities": [
            {"id": "b1", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "e_ten", "kind": "filled", "name": "ten", "role": "constraint-value", "literal_value": 10},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "number", "is_target": True},
        ],
        "verb_phrases": [],
        "constraints": [
            {
                "id": "c1", "kind": "derived", "op": "round_nearest",
                "args_refs": ["b1", "e_ten"], "output_blank_id": "t",
            },
        ],
    })
    _reject_unit_mismatch_constraints("how many times deeper is A than B?", plan)  # must not raise


def test_unit_check_ignores_date_valued_blanks():
    """Age/date diffs (value_type='date') are out of scope for this check --
    they're covered separately by _reject_date_granularity_missing."""
    plan = GSWPlan.model_validate({
        "entities": [
            {"id": "b1", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "b2", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "number", "is_target": True},
        ],
        "verb_phrases": [],
        "constraints": [
            {"id": "c1", "kind": "derived", "op": "diff", "args_refs": ["b1", "b2"], "output_blank_id": "t"},
        ],
    })
    _reject_unit_mismatch_constraints("how many times deeper is A than B?", plan)  # must not raise


def _valid_matched_unit_div_plan_json() -> str:
    return json.dumps({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "A", "role": "subject"},
            {"id": "e2", "kind": "filled", "name": "B", "role": "subject"},
            {
                "id": "b1", "kind": "blank", "role": "bridge-number", "value_type": "number",
                "retrieval_hint": "What is the depth of A, in feet?",
            },
            {
                "id": "b2", "kind": "blank", "role": "bridge-number", "value_type": "number",
                "retrieval_hint": "What is the height of B, in feet?",
            },
            {"id": "t", "kind": "blank", "role": "target", "value_type": "number", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_value", "subject_id": "e1", "object_id": "b1"},
            {"id": "vp2", "phrase": "has_value", "subject_id": "e2", "object_id": "b2"},
        ],
        "constraints": [
            {"id": "c1", "kind": "derived", "op": "div", "args_refs": ["b1", "b2"], "output_blank_id": "t"},
        ],
    })


def test_emit_plan_repairs_unit_mismatched_arithmetic_constraint():
    """End-to-end: a mismatched-unit plan is rejected on the first parse and
    the existing repair loop recovers a matched-unit plan on retry, mirroring
    test_emit_plan_repairs_literal_constant_blank's shape."""
    bad_plan = json.dumps({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "A", "role": "subject"},
            {"id": "e2", "kind": "filled", "name": "B", "role": "subject"},
            {
                "id": "b1", "kind": "blank", "role": "bridge-number", "value_type": "number",
                "retrieval_hint": "What is the depth of A, in feet?",
            },
            {
                "id": "b2", "kind": "blank", "role": "bridge-number", "value_type": "number",
                "retrieval_hint": "What is the height of B, in metres?",
            },
            {"id": "t", "kind": "blank", "role": "target", "value_type": "number", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_value", "subject_id": "e1", "object_id": "b1"},
            {"id": "vp2", "phrase": "has_value", "subject_id": "e2", "object_id": "b2"},
        ],
        "constraints": [
            {"id": "c1", "kind": "derived", "op": "div", "args_refs": ["b1", "b2"], "output_blank_id": "t"},
        ],
    })
    llm = _ScriptedLLM([
        _StubResp(text=bad_plan),
        _StubResp(text=_valid_matched_unit_div_plan_json()),
    ])
    plan, meta = emit_plan("how many times deeper is A than B is tall?", llm)
    assert plan.target().id == "t"
    assert meta.repair_used is True
    assert "unit_mismatch" in meta.parse_error


# ---------------------------------------------------------------------------
# Hard rule 8 / _reject_dropped_qualifier_hints — q39 fix 1b. See
# _planner_prompts.py Hard rule 8 for the prompt half; this is the
# deterministic backstop that drives the existing repair loop when the
# planner drops a qualifier word ("full", "exact", ...) from a blank hint
# that otherwise clearly covers the qualified noun phrase.
# ---------------------------------------------------------------------------


def _qualifier_plan(*, hint: str | None, claim: str | None = None) -> GSWPlan:
    """Fictional q39-shape plan: a blank resolving "the full first name of
    Ada Voss's understudy", used as a middle name in a concat target."""
    return GSWPlan.model_validate({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Ada Voss", "role": "subject"},
            {
                "id": "b_first", "kind": "blank", "role": "bridge-attribute", "value_type": "text",
                "retrieval_hint": hint,
                "verification_claim": claim,
            },
            {"id": "t", "kind": "blank", "role": "target", "value_type": "text", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "understudy_first_name", "subject_id": "e1", "object_id": "b_first"},
            {"id": "vp2", "phrase": "used_as_middle_name", "subject_id": "b_first", "object_id": "t"},
        ],
        "constraints": [],
    })


def test_qualifier_check_fires_when_hint_drops_full():
    plan = _qualifier_plan(hint="What is the first name of Ada Voss's understudy?")
    question = "What is the full first name of Ada Voss's understudy?"
    with pytest.raises(ValueError, match="dropped_qualifier"):
        _reject_dropped_qualifier_hints(question, plan)


def test_qualifier_check_does_not_fire_when_hint_retains_full():
    plan = _qualifier_plan(hint="What is the full first name of Ada Voss's understudy?")
    question = "What is the full first name of Ada Voss's understudy?"
    _reject_dropped_qualifier_hints(question, plan)  # must not raise


def test_qualifier_check_does_not_fire_when_verification_claim_retains_qualifier():
    """The qualifier may live in verification_claim instead of retrieval_hint."""
    plan = _qualifier_plan(
        hint="What is the first name of Ada Voss's understudy?",
        claim="The candidate must be the FULL first name of Ada Voss's understudy.",
    )
    question = "What is the full first name of Ada Voss's understudy?"
    _reject_dropped_qualifier_hints(question, plan)  # must not raise


def test_qualifier_check_does_not_fire_without_qualifier_in_question():
    plan = _qualifier_plan(hint="What is the first name of Ada Voss's understudy?")
    _reject_dropped_qualifier_hints(
        "What is the first name of Ada Voss's understudy?", plan
    )  # must not raise -- no qualifier word in the question at all


def test_qualifier_check_does_not_fire_when_no_blank_covers_the_noun():
    """Conservative gate: a qualifier elsewhere in the question with no
    blank hint text overlapping the qualified noun phrase must not fire."""
    plan = _qualifier_plan(hint="Which club did Ada Voss captain?")
    question = "What is the full first name of Ada Voss's understudy, who captained which club?"
    _reject_dropped_qualifier_hints(question, plan)  # must not raise


def test_qualifier_check_ignores_unrelated_full_usage():
    """'full' attached to a noun outside the closed vocabulary (not
    name/date/title/...) must not false-fire."""
    plan = _qualifier_plan(hint="What is the first name of Ada Voss's understudy?")
    question = "Ada Voss's understudy played a full match before being subbed off."
    _reject_dropped_qualifier_hints(question, plan)  # must not raise


def test_emit_plan_repairs_dropped_qualifier_hint():
    """End-to-end: a plan that drops 'full' is rejected on first parse and
    the existing repair loop recovers a qualifier-preserving plan on retry."""
    bad_plan = json.dumps({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Ada Voss", "role": "subject"},
            {
                "id": "b_first", "kind": "blank", "role": "bridge-attribute", "value_type": "text",
                "retrieval_hint": "What is the first name of Ada Voss's understudy?",
            },
            {"id": "t", "kind": "blank", "role": "target", "value_type": "text", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "understudy_first_name", "subject_id": "e1", "object_id": "b_first"},
            {"id": "vp2", "phrase": "used_as_middle_name", "subject_id": "b_first", "object_id": "t"},
        ],
        "constraints": [],
    })
    good_plan = json.dumps({
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Ada Voss", "role": "subject"},
            {
                "id": "b_first", "kind": "blank", "role": "bridge-attribute", "value_type": "text",
                "retrieval_hint": "What is the full first name of Ada Voss's understudy?",
            },
            {"id": "t", "kind": "blank", "role": "target", "value_type": "text", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "understudy_first_name", "subject_id": "e1", "object_id": "b_first"},
            {"id": "vp2", "phrase": "used_as_middle_name", "subject_id": "b_first", "object_id": "t"},
        ],
        "constraints": [],
    })
    llm = _ScriptedLLM([
        _StubResp(text=bad_plan),
        _StubResp(text=good_plan),
    ])
    plan, meta = emit_plan(
        "What is the full first name of Ada Voss's understudy?", llm
    )
    assert plan.target().id == "t"
    assert meta.repair_used is True
    assert "dropped_qualifier" in meta.parse_error
