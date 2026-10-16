"""One LLM call turns the researcher's hint into a frozen contract.

This is the ONLY place semantic judgment happens. After this returns, every
accept/reject is mechanical -- which is what bounds the loop. So the author's
output must be constrained hard: clauses from the five-kind vocabulary, nothing
else, and any failure degrades to BINDING-only rather than to something
permissive.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from panini_dr.adapters.ours._plan_request import PlanRequest
from panini_dr.adapters.ours._plan_request_author import author_contract
from panini_dr.adapters.ours._plan_request_clauses import Clause
from panini_dr.adapters.ours._planner_exec import GSWPlan

# GSWPlan._check_no_dangling rejects any entity unreferenced by a verb-phrase
# or constraint -- the brief's fixture named both blanks and no edge, so one
# minimal edge is added here to wire b_postcode to t without touching entity
# ids or the assertions below (`known` is built from entity ids, not edges).
PLAN = GSWPlan.model_validate({
    "entities": [
        {"id": "b_postcode", "kind": "blank", "value_type": "entity",
         "retrieval_hint": "What is the postcode of Wren Clinic?"},
        {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True,
         "retrieval_hint": "Which district?"},
    ],
    "verb_phrases": [
        {"id": "vp1", "subject_id": "b_postcode", "phrase": "located_in", "object_id": "t"},
    ],
    "constraints": [],
})

REQ = PlanRequest(
    request_id="q1:b_postcode:0",
    blank_id="b_postcode",
    assigned_blank_ids=["b_postcode"],
    reason="no chunk lists a postcode for Wren Clinic",
    hint="add b_address wired by VP (b_hospital, located_at, b_address)",
)


class _Stub:
    """Minimal chat client: returns scripted tool calls, records the calls.

    `message_log` keeps every call's messages (not just the last), so a test
    can tell the first call's prompt apart from a retry's. `last_kwargs`
    keeps the non-`messages` kwargs of the last call, so a test can pin
    things like `tool_choice`/`temperature` that never show up in the
    messages themselves.
    """

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.calls = 0
        self.last_messages = None
        self.message_log = []
        self.last_kwargs = None

    def chat(self, messages=None, **kw):
        self.last_messages = messages
        self.message_log.append(messages)
        self.last_kwargs = kw
        s = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        return s


def _resp(clauses):
    return SimpleNamespace(
        text="", reasoning_content="", prompt_tokens=0, completion_tokens=0,
        tool_calls=[SimpleNamespace(function=SimpleNamespace(
            name="emit_contract",
            arguments=json.dumps({"clauses": clauses})))])


def _resp_flat(clauses):
    """Mimic the REAL `LLMClient.chat()` shape: `tool_calls` as flat dicts
    (`{"id":..., "name":..., "arguments":...}`), never nested `.function`
    objects. `_resp` above is the nested SDK-mimicking shape; production
    traffic is always this one."""
    return SimpleNamespace(
        text="", reasoning_content="", prompt_tokens=0, completion_tokens=0,
        tool_calls=[{"id": "call_1", "name": "emit_contract",
                     "arguments": json.dumps({"clauses": clauses})}])


def test_it_returns_the_clauses_the_model_asked_for():
    llm = _Stub(_resp([
        {"kind": "blank_added"},
        {"kind": "depends_on", "blank_id": "b_postcode"},
    ]))
    out = author_contract(REQ, PLAN, llm, 2048, 42)
    assert out == [Clause("blank_added"), Clause("depends_on", blank_id="b_postcode")]


def test_the_hint_and_reason_reach_the_prompt():
    """If the researcher's words don't get into the messages, the whole feature
    is a no-op dressed up as a contract."""
    llm = _Stub(_resp([{"kind": "blank_added"}]))
    author_contract(REQ, PLAN, llm, 2048, 42)
    blob = json.dumps(llm.last_messages)
    assert REQ.hint in blob
    assert REQ.reason in blob


def test_a_clause_outside_the_vocabulary_is_dropped():
    """Restricted means restricted. A smuggled kind must not reach the checker,
    where `satisfied` would return False for it forever and stall the loop."""
    llm = _Stub(_resp([
        {"kind": "blank_added"},
        {"kind": "delete_everything", "blank_id": "t"},
    ]))
    assert author_contract(REQ, PLAN, llm, 2048, 42) == [Clause("blank_added")]


def test_a_clause_naming_an_unknown_blank_is_dropped():
    """An unsatisfiable clause can never be satisfied by any op, so it would
    make the contract permanently incomplete."""
    llm = _Stub(_resp([{"kind": "hint_changed", "blank_id": "b_ghost"}]))
    assert author_contract(REQ, PLAN, llm, 2048, 42) == []


def test_blank_added_is_allowed_to_name_no_blank():
    """It is the one kind with no blank_id -- it must not be dropped by the
    unknown-blank filter."""
    llm = _Stub(_resp([{"kind": "blank_added"}]))
    assert author_contract(REQ, PLAN, llm, 2048, 42) == [Clause("blank_added")]


def test_malformed_output_is_retried_once():
    bad = SimpleNamespace(text="not json", reasoning_content="", prompt_tokens=0,
                          completion_tokens=0, tool_calls=[])
    llm = _Stub(bad, _resp([{"kind": "blank_added"}]))
    assert author_contract(REQ, PLAN, llm, 2048, 42) == [Clause("blank_added")]
    assert llm.calls == 2


def test_two_failures_fall_back_to_an_empty_contract():
    """Empty means BINDING-only. That is still strictly better than today's
    liveness-only test, so the fallback is safe."""
    bad = SimpleNamespace(text="nope", reasoning_content="", prompt_tokens=0,
                          completion_tokens=0, tool_calls=[])
    llm = _Stub(bad, bad)
    assert author_contract(REQ, PLAN, llm, 2048, 42) == []
    assert llm.calls == 2


def test_an_llm_exception_never_escapes():
    """Replan is on the default path. A transient outage must not cost the
    question -- the caller keeps today's behaviour."""
    class Boom:
        def chat(self, **kw):
            raise RuntimeError("bedrock is having a day")
    assert author_contract(REQ, PLAN, Boom(), 2048, 42) == []


def test_duplicate_clauses_are_collapsed():
    """The satisfied-set comparison is a set operation; duplicates would make
    the strict-superset rule behave oddly."""
    llm = _Stub(_resp([{"kind": "blank_added"}, {"kind": "blank_added"}]))
    assert author_contract(REQ, PLAN, llm, 2048, 42) == [Clause("blank_added")]


def test_it_handles_the_real_clients_flat_dict_tool_calls():
    """`LLMClient.chat()` (the real client) returns `tool_calls` as flat
    dicts, not nested `.function` objects -- every one of its four
    normalizers builds them that way. A bespoke extractor that only walked
    the nested shape returned None forever against real responses, so this
    pins the flat-dict path directly rather than trusting the nested-shape
    tests (which is all the original 9 exercised) to stand in for it."""
    llm = _Stub(_resp_flat([{"kind": "blank_added"}]))
    assert author_contract(REQ, PLAN, llm, 2048, 42) == [Clause("blank_added")]


def test_an_unhashable_kind_as_a_list_does_not_raise():
    """The model can emit `"kind": [...]` where a string belongs -- valid
    JSON, invalid vocabulary. `kind not in CLAUSE_KINDS` hashes `kind`
    against a frozenset; an unhashable `kind` must degrade to [], not raise
    `TypeError` out of a default-on path."""
    llm = _Stub(_resp([{"kind": ["blank_added"]}]))
    assert author_contract(REQ, PLAN, llm, 2048, 42) == []


def test_an_unhashable_kind_as_a_dict_does_not_raise():
    llm = _Stub(_resp([{"kind": {"a": 1}}]))
    assert author_contract(REQ, PLAN, llm, 2048, 42) == []


def test_non_iterable_tool_calls_does_not_raise():
    """`tool_calls` is meant to be a list; a response that sets it to
    something else entirely (e.g. a bare int) must not raise iterating over
    it in `_extract`."""
    resp = SimpleNamespace(text="", reasoning_content="", prompt_tokens=0,
                            completion_tokens=0, tool_calls=42)
    llm = _Stub(resp)
    assert author_contract(REQ, PLAN, llm, 2048, 42) == []


def test_a_none_plan_does_not_raise():
    """`known = {e.id for e in plan.entities}` runs before the LLM is even
    called; a `None` plan must still degrade to [] rather than raising
    `AttributeError`."""
    llm = _Stub(_resp([{"kind": "blank_added"}]))
    assert author_contract(REQ, None, llm, 2048, 42) == []


def test_blank_added_ignores_a_stray_blank_id_for_deduping():
    """`blank_added` names no blank. If the model attaches a `blank_id`
    anyway (even one naming an unknown blank -- `blank_added` skips the
    known-blank filter entirely), two such clauses that differ only in that
    ignored field must still collapse to one."""
    llm = _Stub(_resp([
        {"kind": "blank_added", "blank_id": "b_postcode"},
        {"kind": "blank_added", "blank_id": "b_ghost"},
    ]))
    assert author_contract(REQ, PLAN, llm, 2048, 42) == [Clause("blank_added")]


def test_tool_choice_and_temperature_are_pinned():
    """Without `tool_choice='required'` a prose reply burns both retry
    attempts for nothing; without `temperature=0.0` the one semantic
    judgment in the whole loop would be nondeterministic. The sibling
    proposer (`_plan_repair_prompt.make_proposer`) pins both -- match it."""
    llm = _Stub(_resp([{"kind": "blank_added"}]))
    author_contract(REQ, PLAN, llm, 2048, 42)
    assert llm.last_kwargs["tool_choice"] == "required"
    assert llm.last_kwargs["temperature"] == 0.0


def test_the_retry_prompt_only_appears_on_the_second_call():
    """The retry framing must differentiate the second call from the first --
    otherwise a model that ignores the vocabulary once will ignore it twice
    for exactly the same reason."""
    bad = SimpleNamespace(text="not json", reasoning_content="", prompt_tokens=0,
                          completion_tokens=0, tool_calls=[])
    llm = _Stub(bad, _resp([{"kind": "blank_added"}]))
    author_contract(REQ, PLAN, llm, 2048, 42)
    assert llm.calls == 2
    first_blob = json.dumps(llm.message_log[0])
    second_blob = json.dumps(llm.message_log[1])
    retry_text = "previous response was not a usable"
    assert retry_text not in first_blob
    assert retry_text in second_blob


def test_the_prompt_tells_the_author_which_side_of_depends_on_to_name():
    """`depends_on` was satisfied 4/33 in a measured 824 run, and the cause was
    direction, not reachability.

    `depends_on(B)` needs B to gain an UPSTREAM dependency -- a new VP running
    `(new) -> B`, or a constraint OUTPUTTING into B. Researchers describe the
    opposite: "add b_new linked by (b_X -> b_new)" feeds the NEW blank FROM
    b_X, so b_new is the one that waits. The author kept naming b_X, the blank
    the researcher happened to mention, and the clause could never be met --
    burning up to three repair attempts per update.

    This pins the guidance rather than the wording: the prompt must state the
    direction rule, and must offer `blank_added` as the safe fallback when the
    author cannot tell which side gains the dependency.
    """
    llm = _Stub(_resp([{"kind": "blank_added"}]))
    author_contract(REQ, PLAN, llm, 2048, 42)
    system = llm.last_messages[0]["content"]
    # the direction rule, in whatever words
    assert "depends_on" in system
    assert "UPSTREAM" in system or "upstream" in system
    # the researcher-describes-the-opposite trap
    assert "b_new" in system and "b_X" in system
    # the safe fallback
    assert "blank_added" in system.split("### `depends_on`")[1]
