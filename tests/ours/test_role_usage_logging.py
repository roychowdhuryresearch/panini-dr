"""Per-role model usage on the FULL arm (`extra["role_usage"]`).

The 2026-09-07 GSW-off ablation protocol requires both arms instrumented
identically at the model-client boundary: prompt and completion tokens by role,
plus the actual number of calls. The control emits exactly that shape; these
tests pin the full system's copy of it.

Two properties matter more than the totals themselves.

*Per question, never cumulative.* The harness fans `run_question` over a thread
pool against ONE adapter instance, and each question's ledger is a fresh object
reachable only from that question's thread — the same discipline
`_checker_sink` and `_read_ledger` already follow in this module. A ledger
living on `self` would silently sum two concurrent questions into both.

*Pool-dispatched workers meter into their own question's ledger.* The
researcher, its reader and its verifier are constructed INSIDE
`_dispatch_subplan_parallel`'s pool worker, where a thread-local lookup mints a
fresh empty ledger and every researcher token is lost. The metered clients are
therefore hoisted onto the question thread before `pool.map`, exactly as
`read_ledger` and `checker_sink` are, and the multi-blank test below is what
holds that hoist in place.

Nothing here may change what the full system DOES: the wrapper only counts.
"""
from __future__ import annotations

import threading

import pytest

from panini_dr.adapters.ours import gsw_planner_orchestrator_v1 as mod
from tests.ours.orchestrator._helpers import (
    _mk_llm_adapter,
    _one_level_plan,
    _planning_phase_opening,
    _RoutedLLM,
    _StubResponse,
    _tc,
    _two_level_plan,
)

#: every role name the adapter meters, so a new `_metered(..., "x")` call site
#: has to be declared rather than appearing anonymously in a run's telemetry
REQUIRED_ROLES = ("orchestrator", "researcher", "verifier", "resolver",
                  "reader", "planner", "finalizer")


@pytest.fixture(autouse=True)
def _restore_emit_plan():
    """`_mk_llm_adapter` monkey-patches the module-level `emit_plan`."""
    original = mod.emit_plan
    yield
    mod.emit_plan = original


def _single_dispatch_routes(answer: str = "Dark Side of the Moon"):
    return {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan", {"blank_ids": ["t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "submit_answer", {"answer": answer})]),
        ],
        "resolve exactly these blanks": [
            _StubResponse(tool_calls=[_tc("r1", "update_blank", {
                "blank_id": "t", "value": answer, "evidence_chunk_ids": ["c"]})]),
        ],
    }


# --- the ledger ------------------------------------------------------------

def test_role_usage_has_the_controls_shape():
    adapter = _mk_llm_adapter(_RoutedLLM(_single_dispatch_routes()), plan_dict=_one_level_plan())
    traj = adapter.run_question("Q", question_id="q1")
    usage = traj.extra["role_usage"]
    assert isinstance(usage, dict) and usage
    for role, row in usage.items():
        assert isinstance(role, str)
        assert set(row) == {"prompt_tokens", "completion_tokens", "calls"}
        assert all(isinstance(v, int) for v in row.values())


def test_role_usage_counts_the_orchestrators_and_researchers_calls():
    adapter = _mk_llm_adapter(_RoutedLLM(_single_dispatch_routes()), plan_dict=_one_level_plan())
    traj = adapter.run_question("Q", question_id="q1")
    usage = traj.extra["role_usage"]
    # call_planner, the call that ends planning by dispatching, then submit
    assert usage["orchestrator"]["calls"] == 3
    assert usage["researcher"]["calls"] == 1
    # _StubResponse defaults: 10 prompt / 20 completion per call; the stub's
    # call_planner opening costs nothing.
    assert usage["orchestrator"]["prompt_tokens"] == 20
    assert usage["orchestrator"]["completion_tokens"] == 40
    assert usage["researcher"]["prompt_tokens"] == 10


def test_role_usage_totals_match_the_stubs_own_call_count():
    """The ledger's calls sum to the number of `.chat()` calls the stub saw."""
    llm = _RoutedLLM(_single_dispatch_routes())
    adapter = _mk_llm_adapter(llm, plan_dict=_one_level_plan())
    traj = adapter.run_question("Q", question_id="q1")
    n_calls = len(llm.calls) + len(llm.planning_calls)
    assert sum(r["calls"] for r in traj.extra["role_usage"].values()) == n_calls


def test_role_usage_is_per_question_not_cumulative():
    adapter = _mk_llm_adapter(_RoutedLLM(_single_dispatch_routes()), plan_dict=_one_level_plan())
    first = adapter.run_question("Q", question_id="q1")
    adapter.llm = _RoutedLLM(_single_dispatch_routes())
    second = adapter.run_question("Q", question_id="q2")
    assert second.extra["role_usage"]["orchestrator"]["calls"] == \
        first.extra["role_usage"]["orchestrator"]["calls"]


class _PerQuestionLLM:
    """One stub, two questions in flight, each with its own script.

    Routes on the question id (which appears in both the orchestrator's and the
    researcher's prompt) and then on the role marker, so neither question can
    consume the other's scripted turns — the test is about the ledgers, and a
    shared script would make a short ledger look like a leak. Every question
    parks at the barrier on its first call, so both are provably mid-run on the
    same adapter at the same time.
    """

    def __init__(self, barrier):
        self._lock = threading.Lock()
        self._barrier = barrier
        self._scripts: dict[tuple, list] = {}
        self._parked: set = set()
        self.calls: list[dict] = []

    def _script(self, qid: str, role: str) -> list:
        if role == "orch":
            return [_StubResponse(tool_calls=[_tc("o1", "dispatch_subplan", {"blank_ids": ["t"]})]),
                    _StubResponse(tool_calls=[_tc("o2", "submit_answer", {"answer": qid})])]
        return [_StubResponse(tool_calls=[_tc("r1", "update_blank", {
            "blank_id": "t", "value": qid, "evidence_chunk_ids": ["c"]})])]

    def chat(self, messages, *, tools=None, max_tokens=None, **kw):
        opening = _planning_phase_opening(messages, tools)
        if opening is not None:
            return opening
        blob = "\n".join(str(m.get("content") or "") for m in messages)
        qid = "qa" if "qa" in blob else "qb"
        role = "orch" if "You are the **Orchestrator**" in blob else "res"
        with self._lock:
            self.calls.append({"qid": qid, "role": role})
            script = self._scripts.setdefault((qid, role), self._script(qid, role))
            park = qid not in self._parked
            self._parked.add(qid)
        if park:
            self._barrier.wait()
        with self._lock:
            return script.pop(0) if script else _StubResponse(text="(exhausted)", tool_calls=[])


def test_role_usage_ledgers_do_not_mix_across_concurrent_questions():
    """Two questions in flight on ONE adapter, forced to interleave."""
    barrier = threading.Barrier(2, timeout=20)
    adapter = _mk_llm_adapter(_RoutedLLM({}), plan_dict=_one_level_plan())
    adapter.llm = _PerQuestionLLM(barrier)
    out: dict[str, dict] = {}
    lock = threading.Lock()

    def _run(qid: str) -> None:
        traj = adapter.run_question(qid, question_id=qid)
        with lock:
            out[qid] = traj.extra["role_usage"]

    threads = [threading.Thread(target=_run, args=(q,)) for q in ("qa", "qb")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert set(out) == {"qa", "qb"}
    # Each question sees ONLY its own calls: on a counter shared through `self`
    # both sides would read the sum of the two runs.
    for qid, usage in out.items():
        assert usage["orchestrator"]["calls"] == 3, f"{qid}: {usage}"
        assert usage["researcher"]["calls"] == 1, f"{qid}: {usage}"


def test_pool_dispatched_researchers_meter_into_the_questions_ledger():
    """Two blanks in ONE dispatch run in the researcher pool (see the hoist)."""
    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(tool_calls=[_tc("o1", "dispatch_subplan",
                                          {"blank_ids": ["b_author", "t"]})]),
            _StubResponse(tool_calls=[_tc("o2", "submit_answer", {"answer": "A"})]),
        ],
        "resolve exactly these blanks": [
            _StubResponse(tool_calls=[_tc("r1", "update_blank", {
                "blank_id": "b_author", "value": "Alice", "evidence_chunk_ids": ["c"]})]),
            _StubResponse(tool_calls=[_tc("r2", "update_blank", {
                "blank_id": "t", "value": "A", "evidence_chunk_ids": ["c"]})]),
        ],
    }
    adapter = _mk_llm_adapter(_RoutedLLM(routes), plan_dict=_two_level_plan())
    traj = adapter.run_question("Q", question_id="qpool")
    assert traj.extra["role_usage"]["researcher"]["calls"] >= 2, traj.extra["role_usage"]


# --- the wrapper itself ----------------------------------------------------

class _RecordingLLM:
    supports_seed_called = False

    def __init__(self):
        self.seen = []
        self.default_temperature = 0.25

    def chat(self, messages, *, tools=None, **kw):
        self.seen.append((messages, tools, kw))
        return _StubResponse(text="ok", prompt_tokens=7, completion_tokens=3)

    def supports_seed(self):
        return True


def test_metered_llm_forwards_the_call_verbatim_and_returns_the_response():
    ledger = mod._RoleUsage()
    inner = _RecordingLLM()
    wrapped = mod._MeteredLLM(inner, "researcher", ledger)
    resp = wrapped.chat([{"role": "user", "content": "x"}], tools=["t"], max_tokens=11, seed=3)
    assert resp.text == "ok"
    messages, tools, kw = inner.seen[0]
    assert messages == [{"role": "user", "content": "x"}]
    assert tools == ["t"] and kw == {"max_tokens": 11, "seed": 3}
    assert ledger.to_dict() == {
        "researcher": {"prompt_tokens": 7, "completion_tokens": 3, "calls": 1}}


def test_metered_llm_delegates_every_other_attribute():
    inner = _RecordingLLM()
    wrapped = mod._MeteredLLM(inner, "reader", mod._RoleUsage())
    assert wrapped.supports_seed() is True
    assert wrapped.default_temperature == 0.25
    assert wrapped.llm is inner


def test_metered_llm_lets_a_provider_error_through_untouched():
    class _Boom:
        def chat(self, *a, **k):
            raise RuntimeError("provider down")

    ledger = mod._RoleUsage()
    with pytest.raises(RuntimeError, match="provider down"):
        mod._MeteredLLM(_Boom(), "verifier", ledger).chat([])
    assert ledger.to_dict() == {}                  # a failed call is not a call


def test_role_usage_add_is_atomic_under_threads():
    ledger = mod._RoleUsage()
    def _hammer():
        for _ in range(500):
            ledger.add("researcher", 1, 2)
    threads = [threading.Thread(target=_hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert ledger.to_dict() == {
        "researcher": {"prompt_tokens": 4000, "completion_tokens": 8000, "calls": 4000}}


# --- role coverage ---------------------------------------------------------

def test_every_metered_call_site_declares_a_known_role():
    """No anonymous role names: the set is declared once and asserted here."""
    import re
    src = open(mod.__file__).read()
    used = set(re.findall(r'_metered\([^,]+,\s*"([a-z_]+)"\)', src))
    assert used, "no metered call sites found — the regex, not the code, is stale"
    assert used <= set(mod.ROLE_USAGE_ROLES), sorted(used - set(mod.ROLE_USAGE_ROLES))
    for role in REQUIRED_ROLES:
        assert role in mod.ROLE_USAGE_ROLES, f"protocol role {role!r} is not metered"


def test_the_seven_protocol_roles_have_a_call_site():
    import re
    src = open(mod.__file__).read()
    used = set(re.findall(r'_metered\([^,]+,\s*"([a-z_]+)"\)', src))
    used |= set(re.findall(r'_metered_reader_llm', src)) and {"reader"} or set()
    missing = [r for r in REQUIRED_ROLES if r not in used]
    assert not missing, f"roles with no metering call site: {missing}"
