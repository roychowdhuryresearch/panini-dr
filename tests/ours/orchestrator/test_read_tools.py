"""Search snippets, raw read windows, the agentic reader (read -> ask), read ledger and reader telemetry.

Split from the research repo's test_planner_orchestrator.py."""

from __future__ import annotations

from tests.ours.orchestrator._helpers import *  # noqa: F401,F403


def test_search_snippet_centers_on_first_non_stopword_query_match_default_budget():
    plan = GSWPlan.model_validate(_one_level_plan())
    text = ("prefixword " * 90) + "targetneedle answer is here " + ("suffixword " * 90)
    chunk = _StubChunk("c1", "Needle Doc", text)
    r = ReActResearcher(
        llm=_ScriptedLLM([]),
        corpus=_StubCorpus({"c1": chunk}),
        retriever=_StubRetriever({"needle": [chunk]}),
        plan=plan,
        state={"t": BlankResult(blank_id="t", status="unknown")},
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
    )

    result = r._tool_search(query="the targetneedle", top_k=1)
    snippet = result["results"][0]["text"]

    assert len(snippet) <= 820
    assert "targetneedle answer is here" in snippet
    assert snippet != text[:800]
    assert 300 <= snippet.index("targetneedle") <= 500


def test_search_snippet_falls_back_to_prefix_when_no_query_term_matches():
    plan = GSWPlan.model_validate(_one_level_plan())
    text = "prefix " + ("body " * 220)
    chunk = _StubChunk("c1", "No Match Doc", text)
    r = ReActResearcher(
        llm=_ScriptedLLM([]),
        corpus=_StubCorpus({"c1": chunk}),
        retriever=_StubRetriever({"unmatched": [chunk]}),
        plan=plan,
        state={"t": BlankResult(blank_id="t", status="unknown")},
        allowed_blank_ids=["t"],
        system_prompt="SYS",
        question="Q",
    )

    result = r._tool_search(query="unmatched phrase", top_k=1)

    assert result["results"][0]["text"] == text[:800]


def test_tool_read_short_article_is_byte_identical_no_notice():
    """Guard rail: articles under the read window come back exactly as
    they did before this fix — no marker text, no notice appended."""
    short_text = "x" * 11999
    researcher = _read_researcher(_StubChunk("c1", "Short Article", short_text))

    result = researcher._tool_read_raw(title="Short Article")

    assert result["article_text"] == short_text
    assert "TRUNCATED" not in result["article_text"]
    assert "END OF ARTICLE" not in result["article_text"]


def test_tool_read_short_article_exactly_at_window_is_byte_identical():
    """total == window (not > window) must also be untouched."""
    exact_text = "y" * 12000
    researcher = _read_researcher(_StubChunk("c1", "Exact Article", exact_text))

    result = researcher._tool_read_raw(title="Exact Article")

    assert result["article_text"] == exact_text


def test_tool_read_default_offset_matches_no_offset_call():
    """A pre-existing call site that never passes offset behaves exactly
    like offset=0 (only the notice is new for long articles)."""
    long_text = "a" * 20000
    researcher = _read_researcher(_StubChunk("c1", "Long Article", long_text))

    implicit = researcher._tool_read_raw(title="Long Article")
    explicit = researcher._tool_read_raw(title="Long Article", offset=0)

    assert implicit == explicit


def test_tool_read_long_article_offset_zero_truncates_with_notice():
    long_text = "a" * 20000
    researcher = _read_researcher(_StubChunk("c1", "Long Article", long_text))

    result = researcher._tool_read_raw(title="Long Article")

    body, marker, notice = result["article_text"].partition("\n\n[TRUNCATED")
    assert body == long_text[:12000]
    assert marker  # notice was appended, not substituted in
    assert "showing characters 0-12000 of 20000" in notice
    assert "offset=12000" in notice


def test_tool_read_offset_returns_next_window():
    long_text = "a" * 12000 + "b" * 8000  # total 20000, second window is final
    researcher = _read_researcher(_StubChunk("c1", "Long Article", long_text))

    result = researcher._tool_read_raw(title="Long Article", offset=12000)

    assert "TRUNCATED" not in result["article_text"]
    body, marker, _ = result["article_text"].partition("\n\n[END OF ARTICLE]")
    assert body == "b" * 8000
    assert marker


def test_tool_read_middle_window_of_three_keeps_truncation_notice():
    long_text = "a" * 12000 + "b" * 12000 + "c" * 5000  # total 29000
    researcher = _read_researcher(_StubChunk("c1", "Long Article", long_text))

    result = researcher._tool_read_raw(title="Long Article", offset=12000)

    body, _, notice = result["article_text"].partition("\n\n[TRUNCATED")
    assert body == "b" * 12000
    assert "showing characters 12000-24000 of 29000" in notice
    assert "offset=24000" in notice


def test_tool_read_offset_past_end_clamps_and_signals_end_of_article():
    long_text = "a" * 20000
    researcher = _read_researcher(_StubChunk("c1", "Long Article", long_text))

    result = researcher._tool_read_raw(title="Long Article", offset=999_999)

    assert result["article_text"].endswith("[END OF ARTICLE]")
    assert "call read with offset=" not in result["article_text"]


def test_tool_read_negative_offset_clamps_to_zero():
    long_text = "a" * 20000
    researcher = _read_researcher(_StubChunk("c1", "Long Article", long_text))

    result = researcher._tool_read_raw(title="Long Article", offset=-500)

    assert result["article_text"].startswith("a" * 12000)
    assert "characters 0-12000 of 20000" in result["article_text"]


def test_tool_read_raw_supports_offset_and_notice():
    long_text = "z" * 20000
    researcher = _read_researcher(_StubChunk("c1", "Long Article", long_text))

    first = researcher._tool_read_raw(title="Long Article")
    assert "TRUNCATED: showing characters 0-12000 of 20000" in first["article_text"]

    second = researcher._tool_read_raw(title="Long Article", offset=12000)
    assert second["article_text"].startswith("z" * 8000)
    assert second["article_text"].endswith("[END OF ARTICLE]")


def test_tool_read_short_article_has_no_outline_noise():
    """Articles with no headings need no map -- keep the response clean."""
    researcher = _read_researcher(_StubChunk("c1", "Short", "no headings here"))

    result = researcher._tool_read_raw(title="Short")

    assert "outline" not in result


def test_tool_read_outline_reports_window_span_for_fragment_awareness():
    """The response must say which slice of the article it is, so the model can
    tell a partial view from a whole one before emitting a count."""
    article = _outline_article()
    researcher = _read_researcher(_StubChunk("c1", "Long Article", article))

    result = researcher._tool_read_raw(title="Long Article")

    assert result["article_chars"] == len(article)
    assert result["window"]["start"] == 0
    assert result["window"]["end"] == 12000


def test_normalise_question_collapses_case_and_punctuation():
    from panini_dr.adapters.ours.gsw_planner_orchestrator_v1 import (
        _normalise_question,
    )

    assert _normalise_question("Who LEADS  the party?") == "who leads the party"
    assert _normalise_question("who leads the party") == "who leads the party"


def test_normalise_question_does_not_conflate_different_questions():
    """Normalisation is textual only — a false cache hit is worse than a miss."""
    from panini_dr.adapters.ours.gsw_planner_orchestrator_v1 import (
        _normalise_question,
    )

    assert _normalise_question("who leads it") != _normalise_question("who led it")


def test_read_ledger_is_isolated_per_thread():
    import threading

    from panini_dr.adapters.ours.gsw_planner_orchestrator_v1 import (
        OursGSWPlannerOrchestratorV1Adapter as A,
    )

    adapter = A.__new__(A)          # bypass __init__; the accessor must self-heal
    adapter._read_ledger()[("c1", "q")] = "main-thread-value"
    seen = {}

    def worker():
        seen["len"] = len(adapter._read_ledger())

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert seen["len"] == 0, "a worker thread must not see the main thread's ledger"
    assert len(adapter._read_ledger()) == 1


def test_tool_read_agentic_returns_the_reader_answer():
    import json as _json

    article = ("x" * 30) + "Marcel Ciolacu" + ("y" * 30)
    researcher = _read_researcher(
        _StubChunk("c1", "Romania", article),
    )
    researcher.llm = _StubReaderLLM([_json.dumps({
        "answer": "Marcel Ciolacu",
        "citations": [{"span": "Marcel Ciolacu", "offset": 30}],
    })])

    out = researcher._tool_ask(chunk_id="c1", question="who leads it?")

    assert out["answer"] == "Marcel Ciolacu"
    assert out["citations"] == [{"span": "Marcel Ciolacu", "offset": 30}]
    assert "article_text" not in out, "the parent no longer receives raw text"


def test_tool_read_agentic_second_identical_call_is_cached_with_no_llm_call():
    import json as _json

    article = ("x" * 30) + "Marcel Ciolacu" + ("y" * 30)
    ledger = {}
    researcher = _read_researcher(
        _StubChunk("c1", "Romania", article),
        read_ledger=ledger,
    )
    stub = _StubReaderLLM([_json.dumps({
        "answer": "Marcel Ciolacu",
        "citations": [{"span": "Marcel Ciolacu", "offset": 30}],
    })])
    researcher.llm = stub

    first = researcher._tool_ask(chunk_id="c1", question="Who leads it?")
    second = researcher._tool_ask(chunk_id="c1", question="who leads  it")

    assert first["answer"] == "Marcel Ciolacu"
    assert second["answer"] == "Marcel Ciolacu"
    assert second["cached"] is True
    assert len(stub.calls) == 1, "normalised repeat must not call the LLM again"


def test_tool_read_agentic_different_question_same_page_runs_fresh():
    import json as _json

    article = ("x" * 30) + "Marcel Ciolacu" + ("y" * 30) + "PSD"
    ledger = {}
    researcher = _read_researcher(
        _StubChunk("c1", "Romania", article),
        read_ledger=ledger,
    )
    stub = _StubReaderLLM([
        _json.dumps({"answer": "Marcel Ciolacu",
                     "citations": [{"span": "Marcel Ciolacu", "offset": 30}]}),
        _json.dumps({"answer": "PSD",
                     "citations": [{"span": "PSD", "offset": article.index("PSD")}]}),
    ])
    researcher.llm = stub

    researcher._tool_ask(chunk_id="c1", question="who leads it?")
    out = researcher._tool_ask(chunk_id="c1", question="which parties exist?")

    assert out["answer"] == "PSD"
    assert len(stub.calls) == 2


def test_tool_read_agentic_error_is_not_cached():
    """A transient failure must not poison this (page, question) for the run."""
    ledger = {}
    researcher = _read_researcher(
        _StubChunk("c1", "Romania", "text"),
        read_ledger=ledger,
    )

    class _Boom:
        def chat(self, *a, **k):
            raise RuntimeError("timeout")

    researcher.llm = _Boom()
    out = researcher._tool_ask(chunk_id="c1", question="q")

    assert "error" in out
    assert ledger == {}, "errors are never cached"


def test_tool_read_agentic_two_chunks_of_the_same_article_share_one_reader_call():
    """The reader reads the whole ARTICLE, so a chunk_id-keyed ledger paid twice
    for byte-identical text. MoNaCo averages 6.0 reads over 3.0 distinct articles
    per question, so that forfeited most of the dedup the cache exists for."""
    import json as _json

    article = ("x" * 30) + "Marcel Ciolacu" + ("y" * 30)
    ledger: dict = {}
    researcher = _read_researcher(
        _StubChunk("ro__0", "Romania", article),
        read_ledger=ledger,
    )
    # a SECOND chunk of the same article, as `search` would return it
    researcher.corpus._chunks["ro__7"] = _StubChunk("ro__7", "Romania", article)
    stub = _StubReaderLLM([_json.dumps({
        "answer": "Marcel Ciolacu",
        "citations": [{"span": "Marcel Ciolacu", "offset": 30}],
    })])
    researcher.llm = stub

    first = researcher._tool_ask(chunk_id="ro__0", question="who leads it?")
    second = researcher._tool_ask(chunk_id="ro__7", question="who leads it?")

    assert first["answer"] == "Marcel Ciolacu"
    assert second["answer"] == "Marcel Ciolacu"
    assert second["cached"] is True
    assert len(stub.calls) == 1, "same article + same question is one reader call"
    assert list(ledger) == [("Romania", "who leads it")], "keyed on the title"


def test_tool_read_agentic_same_chunk_different_article_is_not_conflated():
    """Title-keying must not collapse two DIFFERENT pages into one answer."""
    import json as _json

    researcher = _read_researcher(
        _StubChunk("ro__0", "Romania", "Romania is led by Marcel Ciolacu"),
        read_ledger={},
    )
    researcher.corpus._chunks["bg__0"] = _StubChunk(
        "bg__0", "Bulgaria", "Bulgaria is led by Rosen Zhelyazkov")
    researcher.corpus._titles["Bulgaria"] = "Bulgaria is led by Rosen Zhelyazkov"
    stub = _StubReaderLLM([
        _json.dumps({"answer": "Marcel Ciolacu",
                     "citations": [{"span": "Marcel Ciolacu"}]}),
        _json.dumps({"answer": "Rosen Zhelyazkov",
                     "citations": [{"span": "Rosen Zhelyazkov"}]}),
    ])
    researcher.llm = stub

    a = researcher._tool_ask(chunk_id="ro__0", question="who leads it?")
    b = researcher._tool_ask(chunk_id="bg__0", question="who leads it?")

    assert a["answer"] == "Marcel Ciolacu"
    assert b["answer"] == "Rosen Zhelyazkov"
    assert len(stub.calls) == 2


def test_dispatch_parallel_shares_one_read_ledger_across_pool_workers(monkeypatch):
    """`_read_ledger()` is THREAD-LOCAL, so evaluating it inside `_solve_one`
    (a ThreadPoolExecutor worker) gave every sibling its own empty dict and
    silently reduced the cache to per-researcher — the scoping spec §3 rejects.
    This IS the live path."""
    import threading as _t

    ledgers: list[dict] = []
    threads: set = set()
    # Both siblings must be alive at once, so the ledger really is shared across
    # two concurrent workers rather than reused by one that drained the queue.
    both_alive = _t.Barrier(2, timeout=10)

    class _LedgerSpy:
        def __init__(self, *a, read_ledger=None, state=None,
                     allowed_blank_ids=None, **kw):
            ledgers.append(read_ledger)
            threads.add(_t.get_ident())
            both_alive.wait()
            self._state = state
            self._blank_id = (allowed_blank_ids or [None])[0]
            self._react_attempts = []
            self._verifier_flagged = {}
            self._cascade_blocks = []
            self._verifier_rejects = {}

        def solve(self):
            if self._state is not None and self._blank_id in self._state:
                br = self._state[self._blank_id]
                br.status = "resolved"
                br.value = "v"
                br.evidence_chunk_ids = ["c0"]
            return {"tool_calls": [], "messages": []}

    plan_dict = _concat_plan()      # b_a, b_b: two independent level-0 blanks
    plan = GSWPlan.model_validate(plan_dict)
    state = {
        "b_a": BlankResult(blank_id="b_a", status="unknown"),
        "b_b": BlankResult(blank_id="b_b", status="unknown"),
    }
    adapter = OursGSWPlannerOrchestratorV1Adapter(
        _routing_ctx({"gsw_agentic_read": True})
    )
    question_ledger = adapter._read_ledger()
    monkeypatch.setattr(gmod, "ReActResearcher", _LedgerSpy)

    adapter._dispatch_subplan_parallel(
        plan=plan, plan_dict=plan_dict, state=state,
        order=["b_a", "b_b"], levels={"b_a": 0, "b_b": 0}, question="q?",
        blank_ids=["b_a", "b_b"], hints="", level_tag=0, qgsw_state=None,
    )

    assert len(ledgers) == 2
    assert len(threads) == 2, "two concurrent pool workers"
    assert _t.get_ident() not in threads, "neither ran on the question thread"
    assert ledgers[0] is ledgers[1], "siblings must share ONE ledger"
    assert ledgers[0] is question_ledger, "and it is the question thread's"


def test_dispatch_parallel_shares_the_question_thread_checker_sink_across_pool_workers(monkeypatch):
    """`_checker_sink` is THREAD-LOCAL (per-question isolation under the
    harness's ThreadPoolExecutor). Building each sibling's BlankVerifier with
    ``sink=self._checker_sink`` INSIDE `_solve_one` (a pool worker) therefore
    handed every sibling a fresh per-thread list that nothing ever merged, so
    `traj.extra["gsw_checker_calls"]` silently lost every Verifier verdict from
    multi-blank dispatches. MEASURED on the BCP run of record: worker-admitted
    bindings kept their verdict 51/59 times in single-worker dispatches and
    35/387 times in multi-worker ones. Same hoist as `read_ledger` above."""
    import threading as _t

    sinks: list = []
    threads: set = set()
    both_alive = _t.Barrier(2, timeout=10)

    class _SinkSpy:
        def __init__(self, *a, blank_verifier=None, state=None,
                     allowed_blank_ids=None, **kw):
            sinks.append(getattr(blank_verifier, "sink", None))
            threads.add(_t.get_ident())
            both_alive.wait()
            self._state = state
            self._blank_id = (allowed_blank_ids or [None])[0]
            self._react_attempts = []
            self._verifier_flagged = {}
            self._cascade_blocks = []
            self._verifier_rejects = {}

        def solve(self):
            if self._state is not None and self._blank_id in self._state:
                br = self._state[self._blank_id]
                br.status = "resolved"
                br.value = "v"
                br.evidence_chunk_ids = ["c0"]
            return {"tool_calls": [], "messages": []}

    plan_dict = _concat_plan()
    plan = GSWPlan.model_validate(plan_dict)
    state = {
        "b_a": BlankResult(blank_id="b_a", status="unknown"),
        "b_b": BlankResult(blank_id="b_b", status="unknown"),
    }
    adapter = OursGSWPlannerOrchestratorV1Adapter(
        _routing_ctx()
    )
    question_sink = adapter._checker_sink
    monkeypatch.setattr(gmod, "ReActResearcher", _SinkSpy)

    adapter._dispatch_subplan_parallel(
        plan=plan, plan_dict=plan_dict, state=state,
        order=["b_a", "b_b"], levels={"b_a": 0, "b_b": 0}, question="q?",
        blank_ids=["b_a", "b_b"], hints="", level_tag=0, qgsw_state=None,
    )

    assert len(sinks) == 2 and len(threads) == 2
    assert _t.get_ident() not in threads, "neither ran on the question thread"
    assert sinks[0] is not None and sinks[0] is sinks[1], "siblings must share ONE sink"
    assert sinks[0] is question_sink, "and it is the question thread's, so verdicts reach the trace"


# --- Fix wave F4: the reader's token cost must be recorded --------------------


def test_agentic_read_telemetry_totals_the_reader_token_cost():
    """`ReadResult.prompt_tokens`/`completion_tokens` were populated and read by
    nothing, so spec §3's cost trade (one extra call per (page, question) against
    12k windows leaving the parent's context) was not computable from the logs."""
    import json as _json

    article = ("x" * 30) + "Marcel Ciolacu" + ("y" * 30)
    researcher = _read_researcher(
        _StubChunk("c1", "Romania", article),
    )
    researcher.llm = _StubReaderLLM([_json.dumps({
        "answer": "Marcel Ciolacu",
        "citations": [{"span": "Marcel Ciolacu", "offset": 30}],
    })])

    researcher._tool_ask(chunk_id="c1", question="who leads it?")
    after_fresh = researcher.agentic_read_telemetry()
    researcher._tool_ask(chunk_id="c1", question="who leads it?")   # cached
    after_cached = researcher.agentic_read_telemetry()

    # _StubReaderLLM reports prompt_tokens=1, completion_tokens=1 per call.
    assert after_fresh["prompt_tokens"] == 1
    assert after_fresh["completion_tokens"] == 1
    assert after_cached["prompt_tokens"] == 1, "a cache hit adds ZERO prompt tokens"
    assert after_cached["completion_tokens"] == 1
    cached_rec = next(r for r in after_cached["records"] if r["outcome"] == "cached")
    assert cached_rec["prompt_tokens"] == 0
    assert cached_rec["completion_tokens"] == 0


def test_traj_extra_agentic_read_sums_reader_token_cost_across_dispatches():
    """A question spawns ~2.5 researchers, so the merged totals must SUM the
    reader's token cost, not overwrite it — otherwise the run is un-analysable
    against spec §3's cost trade even though the per-trace numbers exist."""
    from types import SimpleNamespace

    adapter = OursGSWPlannerOrchestratorV1Adapter.__new__(
        OursGSWPlannerOrchestratorV1Adapter)
    adapter.gsw_agentic_read = True
    adapter.gsw_context_compaction = False
    traj = SimpleNamespace(extra={})
    traces = [
        {"metadata": {"agentic_read_telemetry": {
            "calls": 2, "cached_hits": 1, "answers": 1, "not_found": 0,
            "errors": 0, "prompt_tokens": 900, "completion_tokens": 40,
            "records": [{"outcome": "answer"}, {"outcome": "cached"}]}}},
        {"metadata": {"agentic_read_telemetry": {
            "calls": 1, "cached_hits": 0, "answers": 0, "not_found": 1,
            "errors": 0, "prompt_tokens": 1_100, "completion_tokens": 12,
            "records": [{"outcome": "not_found"}]}}},
    ]

    adapter._attach_gsw_trace_metadata(traj, traces)

    tel = traj.extra["agentic_read"]
    assert tel["calls"] == 3
    assert tel["cached_hits"] == 1
    assert tel["prompt_tokens"] == 2_000
    assert tel["completion_tokens"] == 52
    assert len(tel["records"]) == 3


# --- Fix wave F10: an empty article is an error, never a cached not_found -----


def test_tool_read_agentic_empty_article_errors_without_calling_the_reader():
    """Spec §4 lists "article missing/empty" as an error return. Reading an empty
    article yields a not_found that would then be CACHED, so the rest of the
    question inherits a corpus gap as if the page had been read."""
    ledger: dict = {}
    researcher = _read_researcher(
        _StubChunk("c1", "Ghost Page", ""), read_ledger=ledger,
    )
    stub = _StubReaderLLM(["{}"])
    researcher.llm = stub

    out = researcher._tool_ask(chunk_id="c1", question="who leads it?")

    assert "error" in out
    assert "empty" in out["error"]
    assert stub.calls == [], "the reader is never called on an empty article"
    assert ledger == {}


def test_agentic_read_records_every_call_for_offline_audit():
    import json as _json

    article = ("x" * 30) + "Marcel Ciolacu" + ("y" * 30)
    researcher = _read_researcher(
        _StubChunk("c1", "Romania", article),
    )
    researcher.llm = _StubReaderLLM([
        _json.dumps({"answer": "Marcel Ciolacu",
                     "citations": [{"span": "Marcel Ciolacu", "offset": 30}]}),
        _json.dumps({"not_found": True, "reason": "not_stated_on_page"}),
    ])

    researcher._tool_ask(chunk_id="c1", question="who leads it?")
    researcher._tool_ask(chunk_id="c1", question="who leads it?")   # cached
    researcher._tool_ask(chunk_id="c1", question="what is its budget?")

    tel = researcher.agentic_read_telemetry()
    assert tel["calls"] == 3
    assert tel["cached_hits"] == 1
    assert tel["answers"] == 1
    assert tel["not_found"] == 1
    nf = [r for r in tel["records"] if r["outcome"] == "not_found"]
    # A not_found must stay SCOPED — the parent has to see what was actually
    # examined. The briefed reader is not handed sections, so it reports the
    # channels it read (infobox, retrieved passage) plus any tool it called,
    # rather than a section list it never enumerated.
    assert nf and nf[0]["sections_checked"]
    assert "retrieved passage" in nf[0]["sections_checked"]


def test_agentic_read_telemetry_carries_the_coverage_count():
    """Coverage must be measurable from the logs, not only visible to the model.

    Whether the narrowing signal actually fires on list questions is an empirical
    question (monaco_88 is one data point), and it cannot be answered from traces
    unless the count is recorded per read.
    """
    import json as _json

    article = ("x" * 30) + "Marcel Ciolacu" + ("y" * 30)
    researcher = _read_researcher(
        _StubChunk("c1", "Romania", article),
    )
    researcher.llm = _StubReaderLLM([_json.dumps({
        "answer": "Marcel Ciolacu",
        "citations": [{"span": "Marcel Ciolacu", "offset": 30}],
        "candidates_considered": 42,
    })])

    out = researcher._tool_ask(chunk_id="c1", question="who leads it?")

    assert out["coverage"] == {"matched": 1, "considered": 42}
    rec = researcher.agentic_read_telemetry()["records"][-1]
    assert rec["candidates_considered"] == 42


def test_agentic_read_cached_record_keeps_the_coverage_count():
    """A cache hit reports the same coverage as the read it replays."""
    import json as _json

    article = ("x" * 30) + "Marcel Ciolacu" + ("y" * 30)
    ledger = {}
    researcher = _read_researcher(
        _StubChunk("c1", "Romania", article), read_ledger=ledger,
    )
    researcher.llm = _StubReaderLLM([_json.dumps({
        "answer": "Marcel Ciolacu",
        "citations": [{"span": "Marcel Ciolacu", "offset": 30}],
        "candidates_considered": 42,
    })])

    researcher._tool_ask(chunk_id="c1", question="Who leads it?")
    second = researcher._tool_ask(chunk_id="c1", question="who leads  it")

    assert second["cached"] is True
    assert second["coverage"] == {"matched": 1, "considered": 42}
    assert researcher.agentic_read_telemetry()["records"][-1]["candidates_considered"] == 42


def test_agentic_read_telemetry_count_is_none_when_not_reported():
    """Point questions stay unchanged: no count reported, none recorded."""
    import json as _json

    article = ("x" * 30) + "Marcel Ciolacu" + ("y" * 30)
    researcher = _read_researcher(
        _StubChunk("c1", "Romania", article),
    )
    researcher.llm = _StubReaderLLM([_json.dumps({
        "answer": "Marcel Ciolacu",
        "citations": [{"span": "Marcel Ciolacu", "offset": 30}],
    })])

    researcher._tool_ask(chunk_id="c1", question="who leads it?")

    assert researcher.agentic_read_telemetry()["records"][-1]["candidates_considered"] is None


# --- read + ask: the two channels are separate and additive (2026-07-31) ----
# `read` windows from the ARTICLE HEAD on purpose. Measured on FRAMES-824:
# offset-0 head reaches [INFOBOX] on 74% of reads and scores 667/824; re-aimed
# at the retrieved chunk it reaches 40% and scores 651; replaced by a reader
# that returns no raw text it reaches 0% and scores 648. `search` already
# returns the retrieved chunk's text (overlap between read and prior search is
# 0.9% MoNaCo / 3.0% FRAMES), so the head is the parent's ONLY infobox channel.


def test_read_payload_carries_no_outline():
    """The section map goes to the reader, not the researcher: the researcher
    does not navigate an article, so a map is tokens it cannot act on."""
    article = "T\n\n## Early life\nx\n\n## Career\ny\n\n" + ("q" * 20_000)
    out = _read_researcher(_StubChunk("c1", "T", article))._tool_read_raw(title="T")
    assert "outline" not in out


def test_read_is_the_reader_and_no_ask_exists():
    """ONE tool: `read` takes a question and routes to the reader."""
    r = _read_researcher(_StubChunk("c1", "T", "text"))
    by = {t.get("function", {}).get("name"): t for t in r._tools}
    assert "ask" not in by, "there is no separate ask tool"
    assert "question" in by["read"]["function"]["parameters"]["properties"]


def test_ask_hands_the_reader_the_retrieved_chunk_and_its_position():
    """The single-shot reader was never told which chunk matched — 24% of its
    reads concerned a passage >12k chars in, and it went looking for something
    the retriever had already located."""
    import json as _json

    article = ("A" * 20_000) + "TARGET_PASSAGE marker" + ("B" * 5_000)
    chunk = _StubChunk("c1", "T", article)
    chunk.char_start = 20_000
    chunk.text = article[20_000:20_400]
    researcher = _read_researcher(chunk)
    stub = _StubReaderLLM([_json.dumps(
        {"answer": "marker", "citations": [{"span": "TARGET_PASSAGE marker"}]})])
    researcher.llm = stub

    out = researcher._tool_ask(chunk_id="c1", question="what is the marker?")

    assert out["answer"] == "marker"
    briefing = stub.calls[0][0][-1]["content"]   # calls hold (messages, kw)
    assert "TARGET_PASSAGE" in briefing, "the matched chunk must be handed over"
    assert "20,000" in briefing, "its char position must be stated"
    assert len(briefing) < len(article) // 2, "briefing, not a whole-article dump"


def test_repeat_not_found_falls_back_to_the_raw_window():
    """Measured on FRAMES-824: questions where the reader returns not_found twice
    or more are 126 of 824, and there the arm loses 15 net (8 fixed / 23 broken,
    McNemar p=0.011). Everywhere else it is exactly level (386 vs 386). Champion
    wins those by reading the raw window and mining it with find; the reader just
    keeps coming back empty. After the second empty read of an article, hand the
    parent the window instead of a third nothing."""
    art = "Long Article\n\n[INFOBOX]\nBorn | 1959\n\n" + ("prose here. " * 900)
    chunk = _StubChunk("c1", "Long Article", art)
    ledger: dict = {}
    r = _read_researcher(chunk, read_ledger=ledger)
    r.llm = _ScriptedLLM([])

    import panini_dr.adapters.ours._agentic_read as ar
    calls = {"n": 0}

    def _always_not_found(**kw):
        calls["n"] += 1
        return ar.ReadResult(not_found_reason="not on page",
                             sections_checked=["infobox", "retrieved passage"],
                             infobox="Born | 1959")
    orig = ar.read_article_briefed
    ar.read_article_briefed = _always_not_found
    try:
        first = r._tool_ask(chunk_id="c1", question="who was born in 1959?")
        second = r._tool_ask(chunk_id="c1", question="what year was the birth?")
    finally:
        ar.read_article_briefed = orig

    assert first.get("not_found") is True, "first empty read is still a not_found"
    assert "article_text" not in first
    assert second.get("article_text"), "second empty read hands over the raw window"
    assert "Born | 1959" in second["article_text"], "which carries the infobox"
    assert second.get("reader_exhausted") is True, "and says why it fell back"
