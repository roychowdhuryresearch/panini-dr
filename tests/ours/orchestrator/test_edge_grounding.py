"""Edge-grounded commits: bind_connection, grounding gates, re-attempt cap, sentinel citations, and the evidence store / verify call shape.

Split from the research repo's test_planner_orchestrator.py."""

from __future__ import annotations

from tests.ours.orchestrator._helpers import *  # noqa: F401,F403


def test_bind_connection_accumulates_and_rejects_unknown_vp():
    r = _mk_researcher(allowed={"b_author"})
    assert r._tool_bind_connection(blank_id="b_author", vp_id="vp1",
                                   evidence_chunk_ids=["C1"])["ok"] is True
    r._tool_bind_connection(blank_id="b_author", vp_id="vp2", evidence_chunk_ids=["C2"])
    assert r._connection_evidence["b_author"] == {"vp1": ["C1"], "vp2": ["C2"]}
    bad = r._tool_bind_connection(blank_id="b_author", vp_id="vpX", evidence_chunk_ids=["C9"])
    assert bad["ok"] is False


def test_bind_connection_rejects_unknown_chunk_ids():
    r = _mk_researcher(allowed={"b_author"})
    out = r._tool_bind_connection(
        blank_id="b_author", vp_id="vp1", evidence_chunk_ids=["not_a_real_chunk"]
    )
    assert out["ok"] is False
    assert "unknown chunk id" in out["error"]
    assert r._connection_evidence == {}


def test_bind_connection_rejects_empty_evidence():
    r = _mk_researcher(allowed={"b_author"})
    out = r._tool_bind_connection(
        blank_id="b_author", vp_id="vp1", evidence_chunk_ids=[]
    )
    assert out["ok"] is False
    assert "empty" in out["error"].lower()
    assert r._connection_evidence == {}


def test_update_blank_ce_mode_rejects_fake_evidence_ids():
    # 2026-08-19: this test previously pinned the SENTINEL as a fake id --
    # which deadlocked the give-up contract (see the give-up tests below).
    # Fake ids remain rejected; the sentinel is the documented terminal form.
    r = _mk_researcher(allowed={"b_author"})
    out = r._tool_update_blank(
        blank_id="b_author",
        value="Norm Macdonald",
        evidence_chunk_ids=["definitely_fake__0"],
    )
    assert out["ok"] is False
    assert "unknown chunk id" in out["error"]
    assert r.state["b_author"].status == "unknown"


def test_update_blank_ce_mode_allows_empty_evidence_list():
    r = _mk_researcher(allowed={"b_author"})
    out = r._tool_update_blank(
        blank_id="b_author", value="Some Value", evidence_chunk_ids=[]
    )
    assert out["ok"] is True
    assert "unknown chunk id" not in str(out.get("error", ""))


def test_verifier_reasoning_adds_optional_update_blank_schema_only_when_on():
    r_off = _mk_researcher(allowed={"b_author"})
    off_schema = _update_blank_schema(r_off)
    assert "reasoning" not in off_schema["properties"]
    assert off_schema["required"] == ["blank_id", "value", "evidence_chunk_ids"]

    r_on = _mk_researcher(
        allowed={"b_author"},
        verifier_reasoning=True,
    )
    on_schema = _update_blank_schema(r_on)
    assert on_schema["properties"]["reasoning"]["type"] == "string"
    assert "reasoning" not in on_schema["required"]

    r_off_again = _mk_researcher(allowed={"b_author"})
    assert "reasoning" not in _update_blank_schema(r_off_again)["properties"]


def test_update_blank_reasoning_reaches_verifier_prompt_and_blank_result():
    from panini_dr.adapters.ours._blank_verifier import BlankVerifier

    reasoning = "Chunk C1 says Judy Blume wrote The Old Book."
    sink: list[dict[str, Any]] = []
    r = _mk_researcher(
        allowed={"b_author"},
        verifier_reasoning=True,
        blank_verifier=BlankVerifier(llm=_VerifierLLM(_ACCEPT), sink=sink),
    )
    # ground the checkable factual edge first (vp2's partner t is unresolved)
    assert r._tool_bind_connection(blank_id="b_author", vp_id="vp1",
                                   evidence_chunk_ids=["C1"])["ok"] is True

    out = r._tool_update_blank(
        blank_id="b_author",
        value="Judy Blume",
        evidence_chunk_ids=["C1"],
        reasoning=reasoning,
    )

    assert out["ok"] is True
    assert r.state["b_author"].reasoning == reasoning
    assert "Researcher's reasoning (GUIDANCE ONLY)" in sink[-1]["prompt"]
    assert reasoning in sink[-1]["prompt"]


def test_edge_grounded_requires_factual_edge_binding_then_accepts():
    from panini_dr.adapters.ours._blank_verifier import BlankVerifier
    corpus = _StubCorpus({"c1": _StubChunk("c1", "Pink Floyd", "Wish You Were Here")})
    # 1) commit with no bound edge -> rejected; 2) bind vp1; 3) commit -> accepted.
    script = [
        _StubResponse(tool_calls=[_tc("a", "update_blank", {
            "blank_id": "t", "value": "Wish You Were Here",
            "evidence_chunk_ids": ["c1"]})]),
        _StubResponse(tool_calls=[_tc("b", "bind_connection", {
            "blank_id": "t", "vp_id": "vp1", "evidence_chunk_ids": ["c1"]})]),
        _StubResponse(tool_calls=[_tc("c", "update_blank", {
            "blank_id": "t", "value": "Wish You Were Here",
            "evidence_chunk_ids": ["c1"]})]),
    ]
    r, state = _edge_grounded_researcher(
        script, verifier_payload=_ACCEPT, corpus=corpus)
    # patch BlankVerifier in the closure-built researcher uses _VerifierLLM already
    trace = r.solve()
    assert state["t"].status == "resolved"
    assert state["t"].value == "Wish You Were Here"
    assert state["t"].verifier_status is None              # clean accept
    assert state["t"].connection_evidence == {"vp1": ["c1"]}
    # the first commit was bounced with the grounding instruction
    assert any(tc.name == "update_blank"
               and "Ground each factual connection" in (tc.error or "")
               for tc in trace["tool_calls"])


def test_edge_grounded_caps_at_one_reattempt_then_best_effort():
    from panini_dr.adapters.ours._blank_verifier import BlankVerifier
    corpus = _StubCorpus({"c1": _StubChunk("c1", "Pink Floyd", "Some text")})
    # bind first (grounding passes), then TWO holistic-rejected commits.
    script = [
        _StubResponse(tool_calls=[_tc("a", "bind_connection", {
            "blank_id": "t", "vp_id": "vp1", "evidence_chunk_ids": ["c1"]})]),
        _StubResponse(tool_calls=[_tc("b", "update_blank", {
            "blank_id": "t", "value": "Wrong", "evidence_chunk_ids": ["c1"]})]),
        _StubResponse(tool_calls=[_tc("c", "update_blank", {
            "blank_id": "t", "value": "Wrong", "evidence_chunk_ids": ["c1"]})]),
    ]
    r, state = _edge_grounded_researcher(
        script, verifier_payload=_REJECT, corpus=corpus)
    trace = r.solve()
    # second reject exceeds the 1-retry cap -> committed best-effort, not dropped.
    assert state["t"].status == "resolved"
    assert state["t"].value == "Wrong"
    assert state["t"].verifier_status == "refuted_best_effort"
    # exactly one commit was bounced (the first reject); the second committed.
    bounced = [tc for tc in trace["tool_calls"]
               if tc.name == "update_blank" and (tc.error or "")]
    assert len(bounced) == 1


# ---------------------------------------------------------------------------
# NO give-up contract. `evidence_chunk_ids=["insufficient_evidence"]` is
# just an unknown chunk id and is rejected like any other.
#
# HISTORY: 219f20ec (2026-08-19) made a pure-sentinel list a terminal
# give-up -- evidence normalized to none, verifier skipped, value committed
# -- because the BCP-200 run showed hopeless blanks could not terminate
# (83% of losses were exhaustion modes at a median of 214 turns).
#
# MEASURED (CF-150, 2026-09-03/04): that path admitted 223 of 674
# model-resolved bindings (33.1%) with no evidence at all, behind 47 of 150
# answers. Two paired ablations refusing the sentinel scored 68.0% and 68.7%
# against the give-up run's 65.3%, with ZERO evidence-free bindings and no
# exhaustion blow-up. The path was not load-bearing, so it is gone: every
# admitted binding now cites evidence. The prompts no longer instruct the
# sentinel either.


def test_update_blank_pure_sentinel_is_rejected_not_a_giveup():
    r = _mk_researcher(allowed={"b_author"})
    out = r._tool_update_blank(
        blank_id="b_author",
        value="Norm Macdonald",
        evidence_chunk_ids=["insufficient_evidence"],
    )
    assert out["ok"] is False, f"sentinel must not admit a binding: {out}"
    assert "b_author" not in r.state or r.state["b_author"].status != "resolved"


def test_update_blank_sentinel_is_rejected_before_the_verifier_runs():
    class _RejectingVerifier:
        def verify(self, *a, **k):
            raise AssertionError("verifier must not run on an uncitable commit")

    r = _mk_researcher(allowed={"b_author"})
    r.blank_verifier = _RejectingVerifier()
    out = r._tool_update_blank(
        blank_id="b_author",
        value="Norm Macdonald",
        evidence_chunk_ids=["insufficient_evidence"],
    )
    # rejected at the citation gate, so the verifier is never reached
    assert out["ok"] is False


def test_update_blank_mixed_sentinel_is_rejected_like_any_unknown_id():
    # The sentinel is no longer stripped out of a mixed list: it is an
    # unknown chunk id, so a list carrying it is rejected even when a real
    # id sits beside it. A worker that read C1 should cite only C1.
    r = _mk_researcher(allowed={"b_author"})
    out = r._tool_update_blank(
        blank_id="b_author",
        value="Norm Macdonald",
        evidence_chunk_ids=["C1", "insufficient_evidence"],
    )
    assert out["ok"] is False
    assert "unknown chunk id" in out["error"]
    # the real id alone still works
    out_ok = r._tool_update_blank(
        blank_id="b_author",
        value="Norm Macdonald",
        evidence_chunk_ids=["C1"],
    )
    assert out_ok["ok"] is True


def test_update_blank_fake_ids_still_rejected():
    r = _mk_researcher(allowed={"b_author"})
    out = r._tool_update_blank(
        blank_id="b_author",
        value="Norm Macdonald",
        evidence_chunk_ids=["not_a_real_chunk"],
    )
    assert out["ok"] is False
    assert "unknown chunk id" in out["error"]


def test_bind_connection_schema_and_store():
    """The tool schema the LLM sees and the evidence store's shape (one
    {vp_id: [chunk_id]} map per blank)."""
    r, _ = _cem_researcher()
    assert _bind_schema(r) == {
        "type": "object",
        "properties": {
            "blank_id": {"type": "string"},
            "vp_id": {"type": "string"},
            "evidence_chunk_ids": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
        "required": ["blank_id", "vp_id", "evidence_chunk_ids"],
    }
    out = r._tool_bind_connection(
        blank_id="t", vp_id="vp1", evidence_chunk_ids=["c1"])
    assert out == {"ok": True, "bound": {"vp1": ["c1"]}}
    assert r._connection_evidence == {"t": {"vp1": ["c1"]}}


def test_ambiguous_rejection_seeds_no_candidate_rows():
    from panini_dr.adapters.ours._blank_verifier import VerifierResult

    def _ambiguous(_value):
        return VerifierResult(satisfies_all=True, unique=False,
                              ambiguous_candidates=["Animals", "Meddle"],
                              reason="two records match")

    r, _ = _cem_researcher(verifier=_CemVerifier(_ambiguous))
    r._tool_bind_connection(blank_id="t", vp_id="vp1", evidence_chunk_ids=["c1"])
    r._tool_bind_connection(blank_id="t", vp_id="vp2", evidence_chunk_ids=["c3"])
    r._tool_update_blank(
        blank_id="t", value="Wish You Were Here", evidence_chunk_ids=["c1"])
    assert r._connection_evidence == {"t": {"vp1": ["c1"], "vp2": ["c3"]}}


def test_bind_connection_reports_no_coverage_table():
    r, _ = _cem_researcher()
    r._tool_bind_connection(blank_id="t", vp_id="vp1", evidence_chunk_ids=["c1"])
    out = r._tool_bind_connection(blank_id="t", vp_id="vp2", evidence_chunk_ids=["c3"])
    assert "coverage" not in out


def test_gate2_union_is_the_whole_blank():
    verifier = _CemVerifier(_cem_accept)
    r, _ = _cem_researcher(verifier=verifier)
    r._tool_bind_connection(blank_id="t", vp_id="vp1", evidence_chunk_ids=["c1"])
    r._tool_bind_connection(blank_id="t", vp_id="vp2", evidence_chunk_ids=["c3"])
    r._tool_update_blank(blank_id="t", value="Animals", evidence_chunk_ids=["c2"])
    assert verifier.seen == [["c2", "c1", "c3"]]


def test_connection_evidence_telemetry_is_a_flat_map():
    r, state = _cem_researcher(verifier=_CemVerifier(_cem_accept))
    r._tool_bind_connection(blank_id="t", vp_id="vp1", evidence_chunk_ids=["c1"])
    r._tool_bind_connection(blank_id="t", vp_id="vp2", evidence_chunk_ids=["c3"])
    r._tool_update_blank(blank_id="t", value="Animals", evidence_chunk_ids=["c1"])
    assert state["t"].connection_evidence == {"vp1": ["c1"], "vp2": ["c3"]}


def test_verify_call_carries_no_rival_scan():
    """No rival kwarg, and the verifier's prompt is the strict one (the rival
    scan was removed)."""
    from panini_dr.adapters.ours._blank_verifier import BlankVerifier
    from panini_dr.adapters.ours._evidence_prompts import VERIFIER_SYSTEM

    verifier = _RivalVerifier(_rs_accept)
    r, _ = _rs_researcher(verifier=verifier)
    r._tool_search(query="floyd", top_k=6)
    _rs_ground(r)
    out = r._tool_update_blank(blank_id="t", value="Album 1",
                               evidence_chunk_ids=["c1"])
    assert out["ok"] is True
    assert verifier.calls[0]["kwargs"] == {}

    llm = _ScriptedLLM([_rs_verdict_response()])
    r2, _ = _rs_researcher(verifier=BlankVerifier(llm=llm))
    r2._tool_search(query="floyd", top_k=6)
    _rs_ground(r2)
    r2._tool_update_blank(blank_id="t", value="Album 1", evidence_chunk_ids=["c1"])
    system, user = (llm.calls[0]["messages"][0]["content"],
                    llm.calls[0]["messages"][1]["content"])
    assert system == VERIFIER_SYSTEM
    assert user.endswith("Return the JSON object now.")
    assert "OTHER RETRIEVED TEXT" not in user
