import json

import pytest

from panini_dr.adapters.ours._plan_contract import Violation
from panini_dr.adapters.ours._plan_repair_prompt import (
    REPAIRER_SYSTEM, build_repair_tools, make_proposer,
)
from panini_dr.adapters.ours._planner_exec import GSWPlan


class _Resp:
    def __init__(self, name, args):
        self.text = ""
        self.reasoning_content = ""
        self.tool_calls = [type("TC", (), {
            "function": type("F", (), {"name": name, "arguments": json.dumps(args)})()
        })()]


class _LLM:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        return self._resp


def _plan() -> GSWPlan:
    return GSWPlan.model_validate({
        "entities": [
            {"id": "b_year", "kind": "blank", "role": "attribute",
             "value_type": "date", "retrieval_hint": "When?"},
            {"id": "t", "kind": "blank", "role": "target", "value_type": "text",
             "is_target": True, "retrieval_hint": "Monarch on {b_year}?"},
        ],
        "verb_phrases": [{"id": "vp1", "subject_id": "b_year",
                          "phrase": "monarch_on", "object_id": "t"}],
        "constraints": [],
    })


_V = Violation(clause="ordering", binding=True, blank_id="t",
               evidence="t's hint interpolates {b_year}, unordered")


def test_the_proposer_returns_the_tool_call_as_a_raw_op():
    llm = _LLM(_Resp("reverse_edge", {"vp_id": "vp1"}))
    op = make_proposer(llm, max_tokens=2048, seed=42)(_plan(), _V, None)
    assert op == {"op": "reverse_edge", "vp_id": "vp1"}


def test_the_prompt_carries_the_violation_and_the_reject_reason():
    llm = _LLM(_Resp("reverse_edge", {"vp_id": "vp1"}))
    make_proposer(llm, 2048, 42)(_plan(), _V, "would create a cycle")
    user = llm.calls[0][0][-1]["content"]
    assert "ordering" in user and "{b_year}" in user
    assert "would create a cycle" in user


def test_every_op_is_offered_as_a_tool():
    from panini_dr.adapters.ours._plan_repair_ops import OP_NAMES
    names = {t["function"]["name"] for t in build_repair_tools()}
    assert names == set(OP_NAMES)


def test_the_system_prompt_forbids_emitting_a_whole_plan():
    assert "one op" in REPAIRER_SYSTEM.lower()
    assert "plan" in REPAIRER_SYSTEM.lower()


def test_a_model_that_returns_no_tool_call_gives_up():
    class _Empty:
        text = ""
        reasoning_content = ""
        tool_calls = []
    assert make_proposer(_LLM(_Empty()), 2048, 42)(_plan(), _V, None) is None


class _MalformedResp:
    """Mimics a tool call whose `arguments` is unusable -- not necessarily
    even a string. Covers three distinct ways `propose` must not crash:
    malformed JSON text (Bedrock's documented streaming corruption,
    llm_client.py:707-709, or truncation from a tight max_tokens),
    syntactically valid JSON that isn't an object (`"null"`, `"[1,2]"`),
    and `arguments` never having been JSON-encoded at all (a raw Python
    dict/list, reachable only through a raw-SDK-shaped tool call)."""
    def __init__(self, name, raw_arguments):
        self.text = ""
        self.reasoning_content = ""
        self.tool_calls = [type("TC", (), {
            "function": type("F", (), {"name": name, "arguments": raw_arguments})()
        })()]


def test_malformed_tool_call_arguments_gives_up_instead_of_raising():
    llm = _LLM(_MalformedResp("reverse_edge", "{vp_id: vp1"))
    assert make_proposer(llm, 2048, 42)(_plan(), _V, None) is None


@pytest.mark.parametrize("raw_arguments", ["null", "[1,2]"], ids=["null", "list"])
def test_syntactically_valid_non_object_json_gives_up_instead_of_raising(raw_arguments):
    """`json.loads` succeeds on these -- `"null"` -> `None`, `"[1,2]"` -> a
    list -- but neither is a JSON *object*, so unpacking it with `**args`
    would crash just as hard as malformed JSON does. Reachable: nothing
    about a confused model, or Bedrock's corrupted tool-use streaming,
    guarantees the parsed value is an object, only that `json.loads`
    doesn't raise.
    """
    llm = _LLM(_MalformedResp("reverse_edge", raw_arguments))
    assert make_proposer(llm, 2048, 42)(_plan(), _V, None) is None


@pytest.mark.parametrize(
    "raw_arguments", [{"vp_id": "vp1"}, [1, 2]], ids=["dict", "list-object"]
)
def test_arguments_already_a_python_object_gives_up_instead_of_raising(raw_arguments):
    """`_tool_call_name_and_args` accepts the raw-SDK tool-call shape,
    which in principle could hand back `.function.arguments` as a Python
    object rather than a JSON string. Not reachable through the real
    `LLMClient` today -- all four `LLMResponse.tool_calls` construction
    sites in llm_client.py always supply a `str` -- but nothing in
    `_tool_call_name_and_args`'s own contract rules it out, so this pins
    the same give-up behavior instead of leaving it to accident.
    """
    llm = _LLM(_MalformedResp("reverse_edge", raw_arguments))
    assert make_proposer(llm, 2048, 42)(_plan(), _V, None) is None
