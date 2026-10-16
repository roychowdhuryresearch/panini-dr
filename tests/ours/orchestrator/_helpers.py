"""Shared stubs, fixture plans and adapter builders for the orchestrator tests.

Split out of the research repo's single test_planner_orchestrator.py; the
definitions are unchanged.
"""

from __future__ import annotations

import json
import inspect
import threading
from dataclasses import dataclass
from typing import Any

import pytest

from panini_dr.adapters.base import AdapterContext
from panini_dr.adapters.ours._planner_exec import BlankResult, GSWPlan
from panini_dr.adapters.ours import gsw_planner_orchestrator_v1 as gmod
from panini_dr.adapters.ours.gsw_planner_orchestrator_v1 import (
    OursGSWPlannerOrchestratorV1Adapter,
    ReActResearcher,
    _resolved_target_answer,
    _windowed_article_text,
)
from panini_dr.models.trace import Trajectory


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubChunk:
    chunk_id: str
    title: str
    text: str


@dataclass
class _StubHit:
    chunk: _StubChunk
    score: float = 1.0


class _StubCorpus:
    def __init__(self, chunks: dict[str, _StubChunk]) -> None:
        self._chunks = dict(chunks)
        self._titles = {c.title: c.text for c in chunks.values()}

    def get_chunk(self, chunk_id: str):
        return self._chunks.get(chunk_id)

    def article_text(self, title: str) -> str:
        return self._titles.get(title, "")


class _AnyIdCorpus(_StubCorpus):
    """Default corpus of the adapter helpers: every chunk id resolves (to an
    empty placeholder chunk), so orchestration tests can cite arbitrary ids
    past the commit's unknown-chunk-id gate. Tests of that gate pass their
    own `_StubCorpus`."""

    def __init__(self) -> None:
        super().__init__({})

    def get_chunk(self, chunk_id: str):
        if not chunk_id:
            return None
        return self._chunks.get(chunk_id) or _StubChunk(chunk_id, "T", "")


class _StubRetriever:
    def __init__(self, mapping: dict[str, list[_StubChunk]]) -> None:
        self.mapping = mapping

    def search(self, query: str, *, top_k: int = 5):
        for key, chunks in self.mapping.items():
            if key.lower() in query.lower():
                return [_StubHit(chunk=c) for c in chunks[:top_k]]
        return []


@dataclass
class _StubResponse:
    text: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    prompt_tokens: int = 10
    completion_tokens: int = 20
    reasoning_content: str = ""
    finish_reason: str = "stop"

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


def _planning_phase_opening(messages, tools):
    """The paper's orchestrator starts every question with a planning phase
    whose only tool is ``call_planner`` (``_plan_via_tool``). Its opening call
    (no tool result in the conversation yet) is answered here with
    ``call_planner``, so a test scripts only what happens once a plan exists.
    The next planning-phase call reaches the script: its first tool call ends
    the planning phase and is executed as the main loop's first turn.
    Returns None for every other call."""
    if not tools or len(tools) != 1:
        return None
    if tools[0].get("function", {}).get("name") != "call_planner":
        return None
    if any(m.get("role") == "tool" for m in messages):
        return None
    return _StubResponse(
        tool_calls=[_tc("plan_0", "call_planner", {})],
        prompt_tokens=0, completion_tokens=0,
    )


class _ScriptedLLM:
    """Returns a pre-scripted sequence on each .chat() call."""

    def __init__(self, script: list[_StubResponse]):
        self.script = list(script)
        self.calls: list[dict] = []
        self.planning_calls: list[dict] = []

    def chat(self, messages, *, tools=None, max_tokens=None, **kwargs):
        opening = _planning_phase_opening(messages, tools)
        if opening is not None:
            self.planning_calls.append({"messages": list(messages), "tools": tools})
            return opening
        self.calls.append(
            {"messages": list(messages), "tools": tools, **kwargs}
        )
        if not self.script:
            return _StubResponse(text="(script exhausted)", tool_calls=[])
        return self.script.pop(0)


def _tc(tid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"id": tid, "name": name, "arguments": json.dumps(args)}


def _dispatch_then_submit(blank_ids, answer: str = "") -> list:
    """The orchestrator turns of a one-dispatch question: dispatch ``blank_ids``,
    then submit. An empty ``answer`` submits the worksheet's resolved target
    value (``submit_answer`` reads it when the model passes nothing)."""
    return [
        _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                     {"blank_ids": list(blank_ids)})]),
        _StubResponse(tool_calls=[_tc("o2", "submit_answer", {"answer": answer})]),
    ]


def _orchestrated_llm(researcher_script, *, blank_ids=("t",), answer: str = ""):
    """A routed stub for one dispatch of ``blank_ids`` whose researcher follows
    ``researcher_script``, then a submit (see ``_dispatch_then_submit``)."""
    return _RoutedLLM({
        "You are the **Orchestrator**": _dispatch_then_submit(blank_ids, answer),
        "resolve exactly these blanks": list(researcher_script),
    })


# Phase-3.6: the plan-updater LLM emits a single PlanDiffOp JSON.
# Helpers to construct the most common ops in the tests.

def _add_blank_op(blank_id: str, vp_id: str, *, subject_id: str = "e1",
                  phrase: str = "associated_with",
                  value_type: str = "text",
                  role: str = "bridge-attribute") -> str:
    return json.dumps({
        "op": "add_blank",
        "blank": {"id": blank_id, "value_type": value_type, "role": role},
        "wire_via_vp": {
            "id": vp_id, "phrase": phrase,
            "subject_id": subject_id, "object_id": blank_id,
        },
    })


def _change_target_op(from_id: str, to_id: str) -> str:
    return json.dumps({"op": "change_target", "from_id": from_id, "to_id": to_id})


def _trivial_op() -> str:
    """An op that ``is_trivial_op`` always rejects (no-op shape)."""
    return json.dumps({"op": "change_target", "from_id": "t", "to_id": "t"})


def _invalid_op_remove_target_wiring() -> str:
    """Removing the only VP wiring the target — apply_diff rejects."""
    return json.dumps({"op": "remove_vp", "vp_id": "vp1"})


# ---------------------------------------------------------------------------
# Fixture plan — Picasso / Pink Floyd bridge (2 levels)
# ---------------------------------------------------------------------------


def _two_level_plan() -> dict[str, Any]:
    """Bridge-entity plan: a filled book → bridge author (blank entity)
    → target award (blank entity). ``build_dependency_graph`` adds a
    projection edge from b_author to t because b_author is a
    blank-entity subject in vp2, which makes t depend on b_author and
    lands them in distinct topological levels."""
    return {
        "entities": [
            {"id": "e_book", "kind": "filled", "name": "The Old Book", "role": "subject"},
            {
                "id": "b_author",
                "kind": "blank",
                "role": "bridge-entity",
                "value_type": "entity",
            },
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "written_by", "subject_id": "e_book", "object_id": "b_author"},
            {"id": "vp2", "phrase": "awarded", "subject_id": "b_author", "object_id": "t"},
        ],
        "constraints": [],
    }


def _one_level_plan() -> dict[str, Any]:
    """Single blank (target) at level 0."""
    return {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pink Floyd", "role": "subject"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "released", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    }


def _concat_plan() -> dict[str, Any]:
    """Two parallel blanks at level 0, target at level 1 via concat."""
    return {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Alice", "role": "subject"},
            {"id": "b_a", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {"id": "b_b", "kind": "blank", "role": "bridge-attribute", "value_type": "text"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "text",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_a", "subject_id": "e1", "object_id": "b_a"},
            {"id": "vp2", "phrase": "has_b", "subject_id": "e1", "object_id": "b_b"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "derived",
                "op": "concat",
                "args_blanks": ["b_a", "b_b"],
                "output_blank_id": "t",
            }
        ],
    }


# Every commit runs the edge-grounded check (bind_connection per factual
# edge, then one holistic verify). The orchestration tests script only the
# researcher's update_blank, so they set the shared re-attempt budget to 0:
# a commit that fails either gate is admitted at once, flagged
# refuted_best_effort, instead of being bounced back for another turn.
# Tests of the gates themselves override the key.
_GATES_NON_BLOCKING: dict[str, Any] = {"gsw_edge_grounded_cap": 0}


def _mk_adapter(
    scripted: _ScriptedLLM,
    *,
    plan_dict: dict[str, Any],
    corpus=None,
    retriever=None,
    extra: dict[str, Any] | None = None,
):
    extra_cfg = {
        "corpus": corpus or _AnyIdCorpus(),
        "retriever": retriever or _StubRetriever({}),
        "level_max_turns": 6,
        **_GATES_NON_BLOCKING,
    }
    if extra:
        extra_cfg.update(extra)
    ctx = AdapterContext(
        system_id="ours_gsw_planner_orchestrator_v1",
        model_id="stub",
        max_turns=30,
        max_completion_tokens=2000,
        extra=extra_cfg,
    )
    adapter = OursGSWPlannerOrchestratorV1Adapter(ctx)
    adapter.llm = scripted
    # Patch emit_plan so we don't need to mock the planner conversation.
    from panini_dr.adapters.ours import gsw_planner_orchestrator_v1 as mod
    from panini_dr.adapters.ours._planner_emit import PlanEmitMeta

    def _fake_emit(
        question,
        llm_client,
        *,
        max_tokens=4096,
        enable_repair=True,
        llm_seed=None,
        blank_qualifier=False,
        blank_hints=False,
        prose_plan_mode="off",
        directives=None,
    ):
        return GSWPlan.model_validate(plan_dict), PlanEmitMeta(
            prompt_tokens=1, completion_tokens=1, raw_response="{}"
        )
    mod.emit_plan = _fake_emit  # local monkey-patch; auto-restored next test by fixture
    return adapter


# ---------------------------------------------------------------------------
# q101 fix 2 — a well-evidenced empty list is a valid TERMINAL resolution,
# not something to keep re-dispatching. Before this fix
# `_invalid_update_value_error` unconditionally bounced `value=[]`, so a
# researcher who correctly determined a list-type blank is empty (e.g. "did
# her costar ALSO win Best Actor?" -> no one in the costar-who-won list) had
# no way to ever commit it and burned its whole turn budget retrying.
# ---------------------------------------------------------------------------


def _one_level_list_plan(*, verification_claim: str | None = None) -> dict[str, Any]:
    return {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Ada Voss", "role": "subject"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "list",
                "is_target": True,
                "verification_claim": verification_claim,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "costars_who_also_won", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    }


def _mk_researcher_spy(calls: list[dict[str, Any]]):
    """A fake ReActResearcher: records the kwargs it was constructed with that
    the dispatch tests inspect, resolves its assigned blank on ``solve()``, and
    records an ACCEPTED real-evidence proposal into its investigation-memory
    episode (``self._invmem_episode``, set by the real dispatch code exactly as
    for a live researcher) -- what a genuine ``update_blank`` commit records."""
    from panini_dr.adapters.ours._investigation_memory import (
        InvestigationMemory,
    )

    class _SpyResearcher:
        def __init__(self, *args, state=None, allowed_blank_ids=None,
                     prior_investigation=None, max_turns=None, **kwargs):
            self._state = state
            self._blank_id = (allowed_blank_ids or [None])[0]
            calls.append({
                "blank_id": self._blank_id,
                "prior_investigation": prior_investigation,
                "max_turns": max_turns,
            })
            self._react_attempts = []
            self._verifier_flagged = {}
            self._cascade_blocks = []
            self._verifier_rejects = {}

        def solve(self):
            messages = [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "Q"},
            ]
            if self._state is not None and self._blank_id in self._state:
                br = self._state[self._blank_id]
                br.status = "resolved"
                br.value = "Dark Side of the Moon"
                br.evidence_chunk_ids = ["chunk_pf_0"]
            ep = getattr(self, "_invmem_episode", None)
            if ep is not None:
                InvestigationMemory.add_proposal(
                    ep, self._blank_id, "Dark Side of the Moon",
                    ["chunk_pf_0"], "accepted", None)
            return {"tool_calls": [], "messages": messages}

    return _SpyResearcher


def _seed_open_dossier_memory(adapter, plan_dict: dict[str, Any]) -> None:
    from panini_dr.adapters.ours._investigation_memory import (
        InvestigationMemory,
    )

    im = InvestigationMemory(plan_dict)
    ep = im.open_episode(1, "researcher", ["t"])
    im.add_attempt(ep, "search", ["chunk_pf_0"], query="pink floyd released album")
    im.add_proposal(
        ep,
        "t",
        "Wish You Were Here",
        ["chunk_pf_0"],
        verdict="rejected",
        reject_reason="release connection was unconfirmed",
    )
    im.add_binding(ep, "t", "vp1", ["chunk_pf_0"], ok=True)
    im.fold(ep)
    im.close_round(1)
    adapter._invmem = im
    adapter._invmem_round = 1
    adapter._dossier_log = []


# ---------------------------------------------------------------------------
# Connection evidence: the bind_connection tool and the edge-grounded commit
# ---------------------------------------------------------------------------


def _mk_researcher(
    allowed,
    *,
    blank_verifier=None,
    verifier_reasoning: bool = False,
):
    """Minimal ReActResearcher over the 2-edge `b_author` plan (vp1, vp2).

    `b_author` is the object of vp1 and the subject of vp2, so
    blank_projection_edges returns both vp_ids. No LLM call is made — the
    handlers are exercised directly. C1/C2 are real chunks so the cited-chunk
    read path works. Passing `blank_verifier` lets a test exercise the
    edge-grounded commit check's holistic verify.
    """
    plan = GSWPlan.model_validate(_two_level_plan())
    state = {
        "e_book": BlankResult(blank_id="e_book", status="resolved"),
        "b_author": BlankResult(blank_id="b_author", status="unknown"),
        "t": BlankResult(blank_id="t", status="unknown"),
    }
    corpus = _StubCorpus({
        "C1": _StubChunk("C1", "Doc1", "Judy Blume wrote The Old Book."),
        "C2": _StubChunk("C2", "Doc2", "The Old Book won an award."),
    })
    kwargs = {
        "llm": _ScriptedLLM([]),
        "corpus": corpus,
        "retriever": _StubRetriever({}),
        "plan": plan,
        "state": state,
        "allowed_blank_ids": list(allowed),
        "blank_verifier": blank_verifier,
        "system_prompt": "SYS",
        "question": "Q",
        "max_turns": 5,
    }
    if verifier_reasoning:
        kwargs["verifier_reasoning"] = True
    return ReActResearcher(**kwargs)


def _tool_name(t):
    return t["function"]["name"]


def _update_blank_schema(r: ReActResearcher) -> dict[str, Any]:
    return next(
        t["function"]["parameters"]
        for t in r._tools
        if t["function"]["name"] == "update_blank"
    )


# ---------------------------------------------------------------------------
# LLM orchestrator mode tests
# ---------------------------------------------------------------------------


import threading


class _RoutedLLM:
    """Content-routed stub LLM for LLM-orchestrator tests.

    Thread-safe. Picks a response based on markers in the system
    prompt's content (e.g. a slice of a specific blank_id) so parallel
    researchers each receive the matching scripted response.
    """

    def __init__(self, routes: dict[str, list[_StubResponse]], *, auto_plan: bool = True):
        # routes: key = marker substring to look for in system msg,
        # value = scripted responses in order for matching calls.
        # auto_plan=False: the script answers the planning phase itself.
        self.routes = {k: list(v) for k, v in routes.items()}
        self.calls: list[dict] = []
        self.planning_calls: list[dict] = []
        self.auto_plan = auto_plan
        self._lock = threading.Lock()

    def chat(self, messages, *, tools=None, max_tokens=None, **kwargs):
        opening = _planning_phase_opening(messages, tools) if self.auto_plan else None
        if opening is not None:
            with self._lock:
                self.planning_calls.append({"messages": list(messages), "tools": tools})
            return opening
        sys_content = ""
        for m in messages:
            if m.get("role") == "system":
                sys_content = m.get("content", "") or ""
                break
        with self._lock:
            self.calls.append(
                {"messages": list(messages), "tools": tools, **kwargs}
            )
            chosen_key = None
            for key in self.routes:
                if key in sys_content:
                    chosen_key = key
                    break
            if chosen_key is None or not self.routes[chosen_key]:
                return _StubResponse(text="(no route)", tool_calls=[])
            return self.routes[chosen_key].pop(0)


def _mk_llm_adapter(
    routed: _RoutedLLM,
    *,
    plan_dict: dict[str, Any],
    corpus=None,
    retriever=None,
    orchestrator_max_turns: int = 6,
    extra: dict[str, Any] | None = None,
):
    extra_cfg = {
        "corpus": corpus or _AnyIdCorpus(),
        "retriever": retriever or _StubRetriever({}),
        "level_max_turns": 6,
        "orchestrator_max_turns": orchestrator_max_turns,
        **_GATES_NON_BLOCKING,
    }
    if extra:
        extra_cfg.update(extra)
    ctx = AdapterContext(
        system_id="ours_gsw_planner_orchestrator_v1",
        model_id="stub",
        max_turns=30,
        max_completion_tokens=2000,
        extra=extra_cfg,
    )
    adapter = OursGSWPlannerOrchestratorV1Adapter(ctx)
    adapter.llm = routed
    from panini_dr.adapters.ours import gsw_planner_orchestrator_v1 as mod
    from panini_dr.adapters.ours._planner_emit import PlanEmitMeta

    def _fake_emit(
        question,
        llm_client,
        *,
        max_tokens=4096,
        enable_repair=True,
        llm_seed=None,
        blank_qualifier=False,
        blank_hints=False,
        prose_plan_mode="off",
        directives=None,
    ):
        return GSWPlan.model_validate(plan_dict), PlanEmitMeta(
            prompt_tokens=1, completion_tokens=1, raw_response="{}"
        )
    mod.emit_plan = _fake_emit
    return adapter


# ---------------------------------------------------------------------------
# Phase-1 GSW evidence state (logging only)
# ---------------------------------------------------------------------------


def _one_level_script() -> list:
    """Resolve the single target blank `t` with one update_blank call."""
    return [
        _StubResponse(tool_calls=[_tc("c1", "update_blank", {
            "blank_id": "t",
            "value": "Wish You Were Here",
            "evidence_chunk_ids": ["chunk_1"],
        })]),
    ]


# ---------------------------------------------------------------------------
# Phase 2 — evidence gate in update_blank (off / measure / enforce)
# ---------------------------------------------------------------------------

from panini_dr.adapters.ours._planner_exec import GSWPlan as _GSWPlan


def _grounding_corpus():
    return _StubCorpus({
        "chunk_1": _StubChunk(
            chunk_id="chunk_1", title="Pink Floyd",
            text="Pink Floyd released The Wall in 1979.",
        ),
    })


# ---------------------------------------------------------------------------
# Task 4 — TypedBlankResolver pre-pass in deterministic orchestrator mode
# ---------------------------------------------------------------------------


class _AlwaysHitRetriever:
    """Retriever stub that returns a single hit regardless of query."""

    def __init__(self, chunk):
        self._chunk = chunk

    def search(self, query, *, top_k=5):
        return [_StubHit(chunk=self._chunk)]


def _date_plan_for_orch() -> dict:
    """Single date blank — minimal plan for resolver pre-pass test."""
    return {
        "entities": [
            {"id": "e", "kind": "filled", "name": "Serbia", "role": "subject"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "date",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {
                "id": "vp1",
                "phrase": "became_independent_in_year",
                "subject_id": "e",
                "object_id": "t",
            }
        ],
        "constraints": [],
    }


def _number_plan_for_orch() -> dict:
    """Single numeric target — minimal plan for the deterministic finishing
    test (review finding 2)."""
    return {
        "entities": [
            {"id": "e", "kind": "filled", "name": "Somebody", "role": "subject"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "number",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {
                "id": "vp1",
                "phrase": "has_age_in_days",
                "subject_id": "e",
                "object_id": "t",
            }
        ],
        "constraints": [],
    }


class _ContentLLM:
    """Content/role-addressed stub LLM.

    Unlike ``_ScriptedLLM`` (order-addressed), this inspects the SYSTEM
    prompt of each ``.chat()`` call and returns a fixed response per ROLE:

      * a VERIFIER prompt  -> a canned REJECTING verdict,
      * a RESOLVER prompt  -> the canned resolver value,
      * anything else      -> an empty stub.

    Because the stub is keyed on prompt CONTENT (not call order), verifier
    calls never shift a resolver or orchestrator response.
    """

    def __init__(self, *, resolver_json: str, verifier_json: str, orchestrator=None):
        self._resolver_json = resolver_json
        self._verifier_json = verifier_json
        self._orchestrator = list(orchestrator or [])
        self.calls: list[dict] = []
        self.verify_calls = 0
        self._lock = threading.Lock()

    def chat(self, messages, *, tools=None, max_tokens=None, **kwargs):
        opening = _planning_phase_opening(messages, tools)
        if opening is not None:
            return opening
        sys_content = ""
        for m in messages:
            if m.get("role") == "system":
                sys_content = m.get("content", "") or ""
                break
        if "You are the **Orchestrator**" in sys_content:
            if self._orchestrator:
                return self._orchestrator.pop(0)
            return _StubResponse(text="(no route)", tool_calls=[])
        with self._lock:
            self.calls.append({"messages": list(messages), "tools": tools, **kwargs})
            if "You verify whether a proposed VALUE" in sys_content:
                self.verify_calls += 1
                return _StubResponse(text=self._verifier_json)
            if "You resolve exactly ONE typed slot" in sys_content:
                return _StubResponse(text=self._resolver_json)
            return _StubResponse(text="(no route)", tool_calls=[])


def _committed_values(traj) -> dict[str, Any]:
    """Committed blank id -> value mapping from the trajectory's gsw_state."""
    blanks = traj.extra["gsw_state"]["blanks"]
    return {bid: b.get("value") for bid, b in blanks.items()}


def _run_fixture_orchestrator(extra_overrides: dict[str, Any]):
    """Run ONE question on a fixed resolver-path fixture with the
    content-addressed LLM, returning the resulting Trajectory. The
    content-addressed verifier stub REJECTS every resolver candidate.
    """
    chunk = _StubChunk(chunk_id="c1", title="T", text="body")
    retriever = _AlwaysHitRetriever(chunk)
    corpus = _StubCorpus({"c1": chunk})
    resolver_json = json.dumps({
        "value": "Yvon Chouinard", "object_type": "entity", "source_span": "s",
        "supported": True, "confidence": 0.9, "evidence_chunk_ids": ["c1"]})
    # Verifier REJECTS this value (satisfies_all=false).
    verifier_json = json.dumps({
        "satisfies_all": False, "unique": True,
        "missing_constraints": ["cofounder_of Netflix"], "deferred_constraints": [],
        "ambiguous_candidates": [], "reason": "not a Netflix cofounder"})
    llm = _ContentLLM(resolver_json=resolver_json, verifier_json=verifier_json,
                      orchestrator=_dispatch_then_submit(["t"]))
    extra = {
        "gsw_typed_blank_resolver": True,
        "gsw_react_fallback": False,
        "gsw_evidence_state": True,
    }
    extra.update(extra_overrides)
    adapter = _mk_adapter(
        llm, plan_dict=_date_plan_for_orch(), corpus=corpus, retriever=retriever,
        extra=extra)
    traj = adapter.run_question("q?", question_id="q_obs_eq_off")
    # Stash the stub so callers can assert on verify-call counts.
    traj.extra["_llm_verify_calls"] = llm.verify_calls
    return traj


# ---------------------------------------------------------------------------
# Minimal single-blank researcher
# ---------------------------------------------------------------------------


def _make_min_researcher(**over):
    """Build a minimal ReActResearcher for a single writable blank "t" that
    reaches the verifier `not accepted` path in `_tool_update_blank`.

    The single-blank `_one_level_plan` gives a target blank "t" with
    value_type "entity" (permissive type filter). The corpus has chunk "c1"
    so `_read_cited_chunks(["c1"])` returns one chunk dict — the verifier runs
    the `not accepted` branch (not the "no usable chunks" branch).
    """
    plan = GSWPlan.model_validate(_one_level_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    corpus = _StubCorpus({"c1": _StubChunk("c1", "Title", "body text")})
    kwargs = dict(
        llm=_ScriptedLLM([]),
        corpus=corpus,
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=5,
        hard_type_filter=False,
    )
    kwargs.update(over)
    return ReActResearcher(**kwargs)


# ---------------------------------------------------------------------------
# Phase 4 Task 7 (Part 2) — gsw_cascade_diag written to traj.extra
# ---------------------------------------------------------------------------


class _CascadeBlockLLM:
    """Content-addressed stub for the deterministic resolver pre-pass.

    Routes by SYSTEM/USER prompt content so call ORDER is irrelevant:
      * resolver prompt mentioning ``has_a`` -> value "alpha"
      * resolver prompt mentioning ``has_b`` -> value "beta"
      * verifier prompt for ``Blank: b_a``   -> accepted but PARTIAL
                                                 (complete=False)
      * verifier prompt for ``Blank: b_b``   -> accepted & complete
    A partial b_a makes the ``(concat b_a b_b) -> t`` cascade BLOCK when
    ``gsw_block_unverified_cascade`` is ON (and fire normally when OFF).
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._orchestrator = _dispatch_then_submit(["b_a", "b_b"])
        self._lock = threading.Lock()

    def chat(self, messages, *, tools=None, max_tokens=None, **kwargs):
        opening = _planning_phase_opening(messages, tools)
        if opening is not None:
            return opening
        sys_content = ""
        user_content = ""
        for m in messages:
            if m.get("role") == "system" and not sys_content:
                sys_content = m.get("content", "") or ""
            if m.get("role") == "user":
                user_content = m.get("content", "") or ""
        if "You are the **Orchestrator**" in sys_content:
            if self._orchestrator:
                return self._orchestrator.pop(0)
            return _StubResponse(text="(no route)", tool_calls=[])
        with self._lock:
            self.calls.append({"messages": list(messages)})
        if "You verify whether a proposed VALUE" in sys_content:
            partial = "Blank: b_a" in user_content  # only b_a is partial
            return _StubResponse(text=json.dumps({
                "satisfies_all": True, "unique": True,
                "complete": (not partial),
                "missing_constraints": [], "deferred_constraints": [],
                "ambiguous_candidates": [],
                "reason": "partial" if partial else "ok",
            }))
        if "You resolve exactly ONE typed slot" in sys_content:
            value = "alpha" if "has_a" in user_content else "beta"
            return _StubResponse(text=json.dumps({
                "value": value, "object_type": "text", "source_span": "s",
                "supported": True, "confidence": 0.9,
                "evidence_chunk_ids": ["c1"],
            }))
        return _StubResponse(text="(no route)", tool_calls=[])


def _mk_cascade_block_adapter(extra_overrides: dict[str, Any]):
    chunk = _StubChunk(chunk_id="c1", title="T", text="alpha beta body")
    retriever = _AlwaysHitRetriever(chunk)
    corpus = _StubCorpus({"c1": chunk})
    extra = {
        "gsw_typed_blank_resolver": True,
        "gsw_react_fallback": False,
        "gsw_evidence_state": True,
    }
    extra.update(extra_overrides)
    return _mk_adapter(
        _CascadeBlockLLM(), plan_dict=_concat_plan(), corpus=corpus,
        retriever=retriever, extra=extra)


from panini_dr.adapters.ours._gsw_state import initialize_query_gsw_state
from panini_dr.adapters.ours._planner_exec import build_dependency_graph


# ---------------------------------------------------------------------------
# W6 Task 3 — gsw_plan_update_guard gate + hook in request_plan_update
# ---------------------------------------------------------------------------


def _make_w6_test_plans():
    """Two-level plan where b_country has a retrieval_hint, and t depends on it.
    old_plan has hint='highest in 2023 Democracy Index';
    new_plan has hint='highest in 2024 Democracy Index' (identity changed)."""
    old_dict = {
        "entities": [
            {"id": "b_country", "kind": "blank", "value_type": "entity",
             "role": "bridge-entity", "retrieval_hint": "highest in 2023 Democracy Index"},
            {"id": "t", "kind": "blank", "value_type": "entity",
             "role": "target", "is_target": True,
             "retrieval_hint": "largest religion in {b_country}"},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "biggest_religion",
             "subject_id": "b_country", "object_id": "t"},
        ],
        "constraints": [],
    }
    new_dict = {
        "entities": [
            {"id": "b_country", "kind": "blank", "value_type": "entity",
             "role": "bridge-entity", "retrieval_hint": "highest in 2024 Democracy Index"},
            {"id": "t", "kind": "blank", "value_type": "entity",
             "role": "target", "is_target": True,
             "retrieval_hint": "largest religion in {b_country}"},
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "biggest_religion",
             "subject_id": "b_country", "object_id": "t"},
        ],
        "constraints": [],
    }
    return old_dict, new_dict


def _make_w6_adapter_and_workspace(mode: str):
    """Build adapter with gsw_plan_update_guard=mode and a workspace
    where both b_country and t are resolved."""
    from panini_dr.adapters.ours._gsw_workspace import GSWWorkspace
    from panini_dr.adapters.ours._planner_exec import (
        BlankResult, build_dependency_graph, topological_sort_blanks,
    )
    from panini_dr.adapters.ours.gsw_planner_react_v1 import _compute_levels
    from panini_dr.adapters.ours._plan_updater import PlanReconcileDiff

    old_dict, new_dict = _make_w6_test_plans()
    old_plan = GSWPlan.model_validate(old_dict)
    new_plan = GSWPlan.model_validate(new_dict)

    state = {
        "b_country": BlankResult(blank_id="b_country", status="resolved", value="Norway",
                                 resolved_by="resolver"),
        "t": BlankResult(blank_id="t", status="resolved", value="Church of Norway",
                         resolved_by="resolver"),
    }
    deps = build_dependency_graph(old_plan)
    levels = _compute_levels(deps)
    order = topological_sort_blanks(old_plan)

    adapter = _mk_llm_adapter(
        _RoutedLLM({}),
        plan_dict=old_dict,
        extra={"gsw_plan_update_guard": mode},
    )

    ws = GSWWorkspace(question="Q", plan=old_plan, state=state, deps=deps,
                      levels=levels, order=order, qgsw_state=None)

    diff = PlanReconcileDiff(
        preserved_ids=["b_country", "t"],
        added_ids=[],
        dropped_ids=[],
        invalidated_ids=[],
        summary="hint changed",
        op_summary="update_hint",
        attempts=1,
    )
    return adapter, ws, old_plan, new_plan, diff, state


# ---------------------------------------------------------------------------
# W6a: gsw_plan_issue_surface — flag-off parity + advisory wiring
# ---------------------------------------------------------------------------


def _plan_with_high_det_issue() -> dict[str, Any]:
    """One-blank plan whose target has a retrieval_hint referencing an unknown
    blank id {bad_ref}. This triggers PlanIssue severity=HIGH (C6 placeholder
    error) in the deterministic checker without breaking topological ordering
    or plan execution (the plan still has exactly one target, no cycles)."""
    return {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pink Floyd", "role": "subject"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
                "retrieval_hint": "album released by {bad_ref}",  # unknown placeholder
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "released", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    }


class _CapturingRoutedLLM:
    """Routes like _RoutedLLM but also records every system prompt seen."""

    def __init__(self, routes: dict[str, list[_StubResponse]]) -> None:
        self._routes = {k: list(v) for k, v in routes.items()}
        self.calls: list[dict] = []
        self.system_prompts: list[str] = []

    def chat(self, messages, *, tools=None, max_tokens=None, **kwargs):
        opening = _planning_phase_opening(messages, tools)
        if opening is not None:
            return opening
        self.calls.append({"messages": list(messages), "tools": tools, **kwargs})
        sys_content = ""
        for m in messages:
            if m.get("role") == "system":
                sys_content = m.get("content", "") or ""
                break
        self.system_prompts.append(sys_content)
        for key, script in self._routes.items():
            if key in sys_content:
                if script:
                    return script.pop(0)
        return _StubResponse(text="(no route match)", tool_calls=[])


# --- per-role model routing tests (ported from 5974c8a) ---
def _routing_ctx(extra_overrides: dict | None = None, model_id: str = "stub"):
    extra = {"corpus": _StubCorpus({}), "retriever": _StubRetriever({})}
    if extra_overrides:
        extra.update(extra_overrides)
    return AdapterContext(
        system_id="ours_gsw_planner_orchestrator_v1",
        model_id=model_id,
        max_turns=30,
        max_completion_tokens=2000,
        extra=extra,
    )


def _gillis_plan() -> GSWPlan:
    return GSWPlan.model_validate({
        "entities": [
            {"id": "e_show", "kind": "filled", "name": "Hot Ones"},
            {"id": "e_snl", "kind": "filled", "name": "SNL"},
            {"id": "t", "kind": "blank", "value_type": "entity", "is_target": True},
        ],
        "verb_phrases": [
            {"id": "vp1", "subject_id": "t", "phrase": "fired_from", "object_id": "e_snl"},
            {"id": "vp2", "subject_id": "t", "phrase": "appeared_on", "object_id": "e_show"},
        ],
        "constraints": [],
    })


# ---------------------------------------------------------------------------
# Version (b): edge-grounded verification (gsw_edge_grounded)
# ---------------------------------------------------------------------------


class _VerifierLLM:
    """Returns a fixed verifier JSON payload on every chat()."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    def chat(self, *, messages=None, **kwargs):
        self.calls += 1
        return _StubResponse(text=self.payload)


_ACCEPT = ('{"satisfies_all": true, "unique": true, "missing_constraints": [],'
           ' "deferred_constraints": [], "ambiguous_candidates": [],'
           ' "reason": "ok"}')
_REJECT = ('{"satisfies_all": false, "unique": true,'
           ' "missing_constraints": ["released t"], "deferred_constraints": [],'
           ' "ambiguous_candidates": [], "reason": "no support"}')


def _edge_grounded_researcher(
    script, *, verifier_payload, corpus, verifier_llm=None, **researcher_kwargs
):
    from panini_dr.adapters.ours._blank_verifier import BlankVerifier
    plan = GSWPlan.model_validate(_one_level_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    return ReActResearcher(
        llm=_ScriptedLLM(script),
        corpus=corpus,
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=6,
        blank_verifier=BlankVerifier(
            llm=verifier_llm if verifier_llm is not None else _VerifierLLM(verifier_payload)
        ),
        **researcher_kwargs,
    ), state


# ---------------------------------------------------------------------------
# `read` tool: truncation notice + offset pagination.
#
# Bug: `read` silently hard-truncated every article at 12000 chars with
# no signal anything was cut, so the model believed it saw the whole
# article (e.g. a table continuing past the cut was undercounted). Fix:
# append a `[TRUNCATED: ...]` notice naming the next offset, and accept
# an `offset` param so the model can page through the rest. Short
# articles must be byte-identical to the pre-fix behavior.
# ---------------------------------------------------------------------------


def _read_researcher(chunk: _StubChunk, **kwargs: Any) -> ReActResearcher:
    plan = GSWPlan.model_validate(_one_level_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    return ReActResearcher(
        llm=_ScriptedLLM([]),
        corpus=_StubCorpus({chunk.chunk_id: chunk}),
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        **kwargs,
    )


class _StubReaderLLM:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def chat(self, messages, **kw):
        from types import SimpleNamespace
        self.calls.append((messages, kw))
        text = self.texts.pop(0) if self.texts else ""
        return SimpleNamespace(text=text, reasoning_content="", prompt_tokens=1,
                               completion_tokens=1, finish_reason="stop", tool_calls=[])


# --- read() honours the retrieved chunk's position (2026-07-29) -------------
# `search` selects a chunk by its own text, but `_tool_read` fetched the article
# from offset 0 regardless, so the chunk that was retrieved could be absent from
# the response. Measured on MoNaCo: 63% of read calls asked for a non-first
# chunk (mean index 20). Real `Chunk` carries char_start/char_end; stubs may not,
# hence the getattr default in the implementation.


class _PositionedCorpus:
    """Full article text, with a chunk positioned inside it (like the real corpus)."""

    def __init__(self, chunk, article: str) -> None:
        self._chunk = chunk
        self._article = article

    def get_chunk(self, chunk_id: str):
        return self._chunk if chunk_id == self._chunk.chunk_id else None

    def article_text(self, title: str) -> str:
        return self._article if title == self._chunk.title else ""


def _positioned_researcher(article: str, char_start: int) -> ReActResearcher:
    from panini_dr.retrieval.corpus import Chunk

    chunk = Chunk(
        chunk_id="c1",
        title="Long Article",
        text=article[char_start:char_start + 1200],
        char_start=char_start,
        char_end=min(len(article), char_start + 1200),
    )
    plan = GSWPlan.model_validate(_one_level_plan())
    return ReActResearcher(
        llm=_ScriptedLLM([]),
        corpus=_PositionedCorpus(chunk, article),
        retriever=_StubRetriever({}),
        plan=plan,
        state={"t": BlankResult(blank_id="t", status="unknown")},
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
    )


def _marked_article() -> tuple[str, int]:
    article = ("intro " * 4000) + "UNIQUE_TABLE_MARKER" + (" tail" * 8000)
    return article, article.index("UNIQUE_TABLE_MARKER")


# --- read() returns an article outline (2026-07-29) -------------------------
# A 12k window is a fragment, and the model was never told what it was a fragment
# OF -- it received a byte count, not a map. Measured cost of that opacity on
# FRAMES-200: 6 wrong counts (q479 gold 7 -> answered 2) and 3 empties, because a
# model that cannot see it holds a partial view still emits an aggregate from it.
# The outline names every section and its offset, so a deep table is one targeted
# read away and a partial view is visibly partial.


def _outline_article() -> str:
    return (
        "Lead paragraph about the topic.\n\n"
        + "## History\n" + ("h" * 400) + "\n"
        + "## Defunct parties\n" + ("d" * 13000) + "\n"
        + "## Parties represented in Parliament\nTABLE_WITH_LEADERS\n"
        + "## References\nrefs\n"
    )


def _read_schema(researcher):
    return next(t for t in researcher._tools if t["function"]["name"] == "read")


# ---------------------------------------------------------------------------
# Candidate-evidence matrix (gsw_candidate_evidence_matrix)
# ---------------------------------------------------------------------------


def _cem_plan() -> dict[str, Any]:
    """Target with TWO factual, checkable edges (both partners filled), so a
    candidate row can be PARTIALLY covered (vp1 bound, vp2 not) — which is
    what the coverage table has to render."""
    return {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Pink Floyd", "role": "subject"},
            {"id": "e2", "kind": "filled", "name": "Harvest Records", "role": "subject"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "released", "subject_id": "e1", "object_id": "t"},
            {"id": "vp2", "phrase": "distributed", "subject_id": "e2", "object_id": "t"},
        ],
        "constraints": [],
    }


class _CemVerifier:
    """Duck-typed blank verifier: records the chunk_ids of every holistic
    verify and returns a scripted verdict. Cheaper than BlankVerifier+LLM and
    it makes gate-2's evidence set directly assertable."""

    def __init__(self, verdict_for):
        self.verdict_for = verdict_for
        self.seen: list[list[str]] = []

    def verify(self, plan, blank_id, value, chunks, dep_values, **kwargs):
        self.seen.append([c["chunk_id"] for c in chunks])
        return self.verdict_for(value)


def _cem_accept(_value):
    from panini_dr.adapters.ours._blank_verifier import VerifierResult
    return VerifierResult(satisfies_all=True, unique=True)


def _cem_researcher(*, verifier=None, **kwargs):
    """Edge-grounded researcher over `_cem_plan` (two checkable factual edges)."""
    plan = GSWPlan.model_validate(_cem_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    corpus = _StubCorpus({
        "c1": _StubChunk("c1", "Pink Floyd", "Wish You Were Here, 1975."),
        "c2": _StubChunk("c2", "Pink Floyd", "Animals, 1977."),
        "c3": _StubChunk("c3", "Harvest", "Harvest Records distributed it."),
    })
    r = ReActResearcher(
        llm=_ScriptedLLM([]),
        corpus=corpus,
        retriever=_StubRetriever({}),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=6,
        blank_verifier=verifier,
        **kwargs,
    )
    return r, state


def _bind_schema(r) -> dict[str, Any]:
    for t in r._tools:
        if t["function"]["name"] == "bind_connection":
            return t["function"]["parameters"]
    raise AssertionError("bind_connection tool not exposed")


# ---------------------------------------------------------------------------
# Rival scan (gsw_rival_scan)
# ---------------------------------------------------------------------------


class _RivalVerifier:
    """Duck-typed blank verifier: records each verify call's cited pile, the
    rival block it was (or was not) handed, and every other kwarg, then returns
    a scripted verdict."""

    def __init__(self, verdict_for):
        self.verdict_for = verdict_for
        self.calls: list[dict[str, Any]] = []

    def verify(self, plan, blank_id, value, chunks, dep_values, **kwargs):
        self.calls.append({
            "cited": [c["chunk_id"] for c in chunks],
            "rivals": [c["chunk_id"] for c in (kwargs.get("rival_chunks") or [])],
            "kwargs": dict(kwargs),
        })
        return self.verdict_for(value)


def _rs_accept(_value):
    from panini_dr.adapters.ours._blank_verifier import VerifierResult
    return VerifierResult(satisfies_all=True, unique=True)


def _rs_ambiguous(_value):
    from panini_dr.adapters.ours._blank_verifier import VerifierResult
    return VerifierResult(satisfies_all=True, unique=False,
                          ambiguous_candidates=["Album 4", "Album 5"],
                          reason="two albums fit the same evidence")


def _rs_chunks() -> dict[str, _StubChunk]:
    """Eight retrievable chunks with distinct titles: c1-c6 answer one query,
    c7/c8 a second, so a test can grow the ledger across two searches."""
    return {
        f"c{i}": _StubChunk(f"c{i}", f"Album {i}",
                            f"Album {i} was released by Pink Floyd.")
        for i in range(1, 9)
    }


def _rs_researcher(*, verifier=None, **kwargs):
    """Researcher over `_cem_plan` (two checkable factual edges). The retriever
    answers 'floyd' with c1-c6 and 'harvest' with c7/c8."""
    plan = GSWPlan.model_validate(_cem_plan())
    state = {"t": BlankResult(blank_id="t", status="unknown")}
    chunks = _rs_chunks()
    r = ReActResearcher(
        llm=_ScriptedLLM([]),
        corpus=_StubCorpus(chunks),
        retriever=_StubRetriever({
            "floyd": [chunks[f"c{i}"] for i in range(1, 7)],
            "harvest": [chunks["c7"], chunks["c8"]],
        }),
        plan=plan,
        state=state,
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
        max_turns=6,
        blank_verifier=verifier,
        **kwargs,
    )
    return r, state


def _rs_ground(r) -> None:
    """Satisfy gate 1: one bound chunk per checkable factual edge."""
    r._tool_bind_connection(blank_id="t", vp_id="vp1", evidence_chunk_ids=["c1"])
    r._tool_bind_connection(blank_id="t", vp_id="vp2", evidence_chunk_ids=["c2"])


def _rs_verdict_response() -> _StubResponse:
    return _StubResponse(text='{"satisfies_all": true, "unique": true}')


__all__ = [
    'AdapterContext',
    'Any',
    'BlankResult',
    'GSWPlan',
    'OursGSWPlannerOrchestratorV1Adapter',
    'ReActResearcher',
    'Trajectory',
    '_ACCEPT',
    '_AlwaysHitRetriever',
    '_AnyIdCorpus',
    '_dispatch_then_submit',
    '_planning_phase_opening',
    '_orchestrated_llm',
    '_CapturingRoutedLLM',
    '_CascadeBlockLLM',
    '_CemVerifier',
    '_ContentLLM',
    '_GSWPlan',
    '_PositionedCorpus',
    '_REJECT',
    '_RivalVerifier',
    '_RoutedLLM',
    '_ScriptedLLM',
    '_StubChunk',
    '_StubCorpus',
    '_StubHit',
    '_StubReaderLLM',
    '_StubResponse',
    '_StubRetriever',
    '_VerifierLLM',
    '_add_blank_op',
    '_bind_schema',
    '_cem_accept',
    '_cem_plan',
    '_cem_researcher',
    '_change_target_op',
    '_committed_values',
    '_concat_plan',
    '_date_plan_for_orch',
    '_edge_grounded_researcher',
    '_gillis_plan',
    '_grounding_corpus',
    '_invalid_op_remove_target_wiring',
    '_make_min_researcher',
    '_make_w6_adapter_and_workspace',
    '_make_w6_test_plans',
    '_marked_article',
    '_mk_adapter',
    '_mk_cascade_block_adapter',
    '_mk_llm_adapter',
    '_mk_researcher',
    '_mk_researcher_spy',
    '_number_plan_for_orch',
    '_one_level_list_plan',
    '_one_level_plan',
    '_one_level_script',
    '_outline_article',
    '_plan_with_high_det_issue',
    '_positioned_researcher',
    '_read_researcher',
    '_read_schema',
    '_resolved_target_answer',
    '_routing_ctx',
    '_rs_accept',
    '_rs_ambiguous',
    '_rs_chunks',
    '_rs_ground',
    '_rs_researcher',
    '_rs_verdict_response',
    '_run_fixture_orchestrator',
    '_seed_open_dossier_memory',
    '_tc',
    '_tool_name',
    '_trivial_op',
    '_two_level_plan',
    '_update_blank_schema',
    '_windowed_article_text',
    'build_dependency_graph',
    'dataclass',
    'gmod',
    'initialize_query_gsw_state',
    'inspect',
    'json',
    'pytest',
    'threading',
]
