# tests/ours/test_agentic_read.py
from panini_dr.adapters.ours._agentic_read import ReadResult, verify_citations


def test_verify_citations_keeps_a_span_present_at_its_offset():
    full = ("x" * 100) + "Marcel Ciolacu" + ("y" * 100)
    kept, dropped = verify_citations(full, [{"span": "Marcel Ciolacu", "offset": 100}])
    assert kept == [{"span": "Marcel Ciolacu", "offset": 100}]
    assert dropped == 0


def test_verify_citations_keeps_a_span_within_slack_of_its_offset():
    """Models mis-count offsets by a little; the span existing nearby is enough."""
    full = ("x" * 300) + "Marcel Ciolacu" + ("y" * 100)
    kept, dropped = verify_citations(full, [{"span": "Marcel Ciolacu", "offset": 250}])
    assert len(kept) == 1
    assert kept[0]["offset"] == 300, "offset is corrected to the real position"
    assert dropped == 0


def test_verify_citations_drops_a_span_absent_from_the_article():
    full = "nothing relevant here"
    kept, dropped = verify_citations(full, [{"span": "Marcel Ciolacu", "offset": 0}])
    assert kept == []
    assert dropped == 1


# --- Fix wave F1: an ABSENT offset must not be read as "offset 0" ---------
# `int(cit.get("offset", 0))` made a citation with no `offset` key search only
# [0, len(span)+200]. Measured: a verbatim-correct span at char 35,000 with no
# offset key was dropped, the answer demoted to
# not_found(reason="citations_unverifiable") -- and `_tool_read` CACHES that, so
# one absent key poisoned the (page, question) for the rest of the run.


def test_verify_citations_keeps_a_span_with_no_offset_key_anywhere_in_the_article():
    full = ("x" * 35_000) + "Marcel Ciolacu" + ("y" * 20_000)
    kept, dropped = verify_citations(full, [{"span": "Marcel Ciolacu"}])
    assert kept == [{"span": "Marcel Ciolacu", "offset": 35_000}]
    assert dropped == 0


def test_verify_citations_keeps_a_span_whose_offset_drifted_beyond_the_slack():
    """A verbatim span with a badly drifted offset is drift, not confabulation.
    The window stays the PREFERRED path (see the multiple-occurrence test); a
    miss falls back to the whole text and reports the span's real offset."""
    full = ("x" * 35_000) + "Marcel Ciolacu" + ("y" * 20_000)
    kept, dropped = verify_citations(full, [{"span": "Marcel Ciolacu", "offset": 35_500}])
    assert kept == [{"span": "Marcel Ciolacu", "offset": 35_000}]
    assert dropped == 0


def test_verify_citations_prefers_the_occurrence_near_a_supplied_offset():
    """The windowed search must still run FIRST when an offset is supplied, so a
    span occurring many times resolves to the occurrence the model pointed at
    rather than collapsing to the first one."""
    full = "Ciolacu" + ("x" * 20_000) + "Ciolacu" + ("y" * 20_000)
    second = 7 + 20_000
    kept, _dropped = verify_citations(full, [{"span": "Ciolacu", "offset": second + 50}])
    assert kept == [{"span": "Ciolacu", "offset": second}]


def test_verify_citations_still_drops_a_span_absent_from_the_whole_article():
    """The confabulation guard survives the fallback: a span that occurs nowhere
    is still dropped, however its offset is (mis)stated."""
    full = "Marcel Ciolacu" + ("y" * 5000)
    kept, dropped = verify_citations(
        full, [{"span": "Klaus Iohannis", "offset": 4000}]
    )
    assert kept == []
    assert dropped == 1


def test_read_result_defaults_are_all_empty():
    r = ReadResult()
    assert r.answer is None
    assert r.citations == []
    assert r.not_found_reason is None
    assert r.error is None


def test_read_result_has_no_unused_cached_field():
    """Fix wave F8: `cached` was two mechanisms for one job -- a never-set
    dataclass field plus `_tool_read`'s `payload["cached"] = True` stamp. Only
    the stamp was ever used; the dead field is gone."""
    assert not hasattr(ReadResult(), "cached")
    assert "cached" not in ReadResult().to_dict()


import json

import pytest

from panini_dr.adapters.ours import _agentic_read
from panini_dr.adapters.ours._agentic_read import (
    MAX_READ_CHARS,
    MAX_READ_TOKENS,
    MODEL_CONTEXT_TOKENS,
    READ_COMPLETION_TOKENS,
    effective_char_limit,
    estimate_tokens,
    parse_read_response,
)


def test_parse_answer_with_verified_citation():
    full = ("x" * 50) + "Marcel Ciolacu" + ("y" * 50)
    raw = json.dumps({
        "answer": "Marcel Ciolacu",
        "citations": [{"span": "Marcel Ciolacu", "offset": 50}],
    })
    r = parse_read_response(raw, full, ["All"], [])
    assert r.answer == "Marcel Ciolacu"
    assert r.citations == [{"span": "Marcel Ciolacu", "offset": 50}]
    assert r.not_found_reason is None


def test_parse_demotes_to_not_found_when_every_citation_is_bogus():
    """An answer whose evidence does not exist is confabulated, not an answer."""
    raw = json.dumps({
        "answer": "Marcel Ciolacu",
        "citations": [{"span": "Marcel Ciolacu", "offset": 0}],
    })
    r = parse_read_response(raw, "unrelated article text", ["All"], [])
    assert r.answer is None
    assert r.not_found_reason == "citations_unverifiable"


def test_parse_keeps_derivation_and_marks_inputs_grounded():
    full = "the show ran 19 seasons" + ("z" * 200) + "he joined in season 5"
    raw = json.dumps({
        "answer": "14",
        "derivation": "19 - 5",
        "citations": [
            {"span": "19 seasons", "offset": 13},
            {"span": "season 5", "offset": full.index("season 5")},
        ],
    })
    r = parse_read_response(raw, full, ["All"], [])
    assert r.answer == "14"
    assert r.derivation == "19 - 5"
    assert r.inputs_grounded is True
    assert len(r.citations) == 2


def test_parse_not_found_carries_sections_checked():
    raw = json.dumps({"not_found": True, "reason": "not_stated_on_page"})
    r = parse_read_response(raw, "text", ["Career", "Statistics"], ["References"])
    assert r.answer is None
    assert r.not_found_reason == "not_stated_on_page"
    assert r.sections_checked == ["Career", "Statistics"]
    assert r.sections_skipped == ["References"]


def test_parse_unparseable_response_is_an_error_not_an_answer():
    r = parse_read_response("I could not find anything, sorry.", "text", ["All"], [])
    assert r.error is not None
    assert r.answer is None


# --- Fix round 1: malformed `citations` must not raise (Finding A) ---
# `citations` is unvetted model JSON. Six syntactically-valid JSON shapes must
# each degrade to a demoted not_found ReadResult, never propagate an exception.

def test_verify_citations_bare_string_yields_no_kept_no_drop():
    kept, dropped = verify_citations("some article text", "abc")
    assert kept == []
    assert dropped == 0


def test_verify_citations_int_yields_no_kept_no_drop():
    kept, dropped = verify_citations("some article text", 5)
    assert kept == []
    assert dropped == 0


def test_verify_citations_dict_yields_no_kept_no_drop():
    kept, dropped = verify_citations("some article text", {"span": "x"})
    assert kept == []
    assert dropped == 0


def test_verify_citations_skips_non_dict_list_entries_and_counts_them_dropped():
    kept, dropped = verify_citations("some article text", [None, "x", ["span", "x"]])
    assert kept == []
    assert dropped == 3


def test_parse_citations_bare_string_demotes_to_not_found_without_raising():
    raw = json.dumps({"answer": "Marcel Ciolacu", "citations": "abc"})
    r = parse_read_response(raw, "some article text", ["All"], [])
    assert r.answer is None
    assert r.not_found_reason == "citations_unverifiable"


def test_parse_citations_int_demotes_to_not_found_without_raising():
    raw = json.dumps({"answer": "Marcel Ciolacu", "citations": 5})
    r = parse_read_response(raw, "some article text", ["All"], [])
    assert r.answer is None
    assert r.not_found_reason == "citations_unverifiable"


def test_parse_citations_dict_demotes_to_not_found_without_raising():
    raw = json.dumps({"answer": "Marcel Ciolacu", "citations": {"span": "x"}})
    r = parse_read_response(raw, "some article text", ["All"], [])
    assert r.answer is None
    assert r.not_found_reason == "citations_unverifiable"


def test_parse_citations_list_with_none_entry_demotes_without_raising():
    raw = json.dumps({"answer": "Marcel Ciolacu", "citations": [None]})
    r = parse_read_response(raw, "some article text", ["All"], [])
    assert r.answer is None
    assert r.not_found_reason == "citations_unverifiable"


def test_parse_citations_list_with_string_entry_demotes_without_raising():
    raw = json.dumps({"answer": "Marcel Ciolacu", "citations": ["x"]})
    r = parse_read_response(raw, "some article text", ["All"], [])
    assert r.answer is None
    assert r.not_found_reason == "citations_unverifiable"


def test_parse_citations_list_with_list_entry_demotes_without_raising():
    raw = json.dumps({"answer": "Marcel Ciolacu", "citations": [["span", "x"]]})
    r = parse_read_response(raw, "some article text", ["All"], [])
    assert r.answer is None
    assert r.not_found_reason == "citations_unverifiable"


# --- Fix round 1: inputs_grounded must reflect drops, not just kept-nonempty (Finding B) ---

def test_parse_inputs_grounded_is_false_when_a_citation_is_dropped():
    """Must be able to fail against a hardcoded True: one good + one bogus citation
    keeps the answer (kept is non-empty) but must NOT be reported fully grounded."""
    full = ("x" * 50) + "Marcel Ciolacu" + ("y" * 50)
    raw = json.dumps({
        "answer": "Marcel Ciolacu",
        "citations": [
            {"span": "Marcel Ciolacu", "offset": 50},
            {"span": "not anywhere in this text", "offset": 9999},
        ],
    })
    r = parse_read_response(raw, full, ["All"], [])
    assert r.answer == "Marcel Ciolacu"
    assert r.inputs_grounded is False


def test_parse_inputs_grounded_is_true_when_every_offered_citation_verifies():
    full = ("x" * 50) + "Marcel Ciolacu" + ("y" * 50)
    raw = json.dumps({
        "answer": "Marcel Ciolacu",
        "citations": [{"span": "Marcel Ciolacu", "offset": 50}],
    })
    r = parse_read_response(raw, full, ["All"], [])
    assert r.inputs_grounded is True


# --- coverage signal (2026-07-30) -------------------------------------------
# monaco_88: the parent asked "collaborations (multiple primary artists)"; the
# reader honoured that qualifier exactly (15 items, precision 1.00) while gold
# also counted "X featuring Y" — 30 of 45 items. Under the old path the parent
# got article text and could notice; under the reader it gets {answer,citations}
# and cannot. The failure even presents as success (outcome=answer, 0 errors),
# so `not_found` never fires and the raw-read fallback never triggers.
#
# The fix is not a better reader prompt — the reader was right. It is telling the
# parent HOW MUCH it filtered out, so "15 matched of 100 considered" is visible
# and re-askable. Metadata only: a line, not the article, so the token win holds.


def test_coverage_reported_when_reader_counts_candidates():
    full = ("x" * 50) + "Marcel Ciolacu" + ("y" * 50)
    raw = json.dumps({
        "answer": "Marcel Ciolacu",
        "citations": [{"span": "Marcel Ciolacu", "offset": 50}],
        "candidates_considered": 100,
    })
    r = parse_read_response(raw, full, ["All"], [])
    assert r.candidates_considered == 100
    assert r.to_dict()["coverage"] == {"matched": 1, "considered": 100}


def test_no_coverage_key_when_reader_omits_it():
    """Non-list reads must be unchanged: absent count -> no coverage key at all."""
    full = ("x" * 50) + "Marcel Ciolacu" + ("y" * 50)
    raw = json.dumps({
        "answer": "Marcel Ciolacu",
        "citations": [{"span": "Marcel Ciolacu", "offset": 50}],
    })
    r = parse_read_response(raw, full, ["All"], [])
    assert r.candidates_considered is None
    assert "coverage" not in r.to_dict()


@pytest.mark.parametrize("bad", ["many", None, -3, 1.5, [100], {"n": 100}])
def test_malformed_candidate_count_is_ignored_not_raised(bad):
    """Unvetted model JSON: a bad count drops the signal, never breaks the read."""
    full = ("x" * 50) + "Marcel Ciolacu" + ("y" * 50)
    raw = json.dumps({
        "answer": "Marcel Ciolacu",
        "citations": [{"span": "Marcel Ciolacu", "offset": 50}],
        "candidates_considered": bad,
    })
    r = parse_read_response(raw, full, ["All"], [])
    assert r.candidates_considered is None
    assert r.answer == "Marcel Ciolacu"
    assert "coverage" not in r.to_dict()


def test_incoherent_coverage_is_dropped():
    """considered < matched cannot be true; a nonsense ratio is worse than none."""
    full = ("x" * 50) + "Marcel Ciolacu" + ("y" * 30) + "Klaus Iohannis"
    raw = json.dumps({
        "answer": "both",
        "citations": [
            {"span": "Marcel Ciolacu", "offset": 50},
            {"span": "Klaus Iohannis", "offset": 94},
        ],
        "candidates_considered": 1,
    })
    r = parse_read_response(raw, full, ["All"], [])
    assert len(r.citations) == 2
    assert "coverage" not in r.to_dict()


def test_coverage_reported_on_not_found():
    """A scoped negative is exactly when 'considered N, matched 0' is informative."""
    raw = json.dumps({
        "not_found": True, "reason": "no collaborations listed",
        "candidates_considered": 100,
    })
    r = parse_read_response(raw, "irrelevant", ["Year-end list"], [])
    d = r.to_dict()
    assert d["not_found"] is True
    assert d["coverage"] == {"matched": 0, "considered": 100}


def test_reader_system_prompt_requests_the_candidate_count():
    assert "candidates_considered" in _agentic_read.BRIEFED_READER_SYSTEM


# --- token budget (2026-07-30) ----------------------------------------------
# monaco_613 died on: "Input length (71074) exceeds model's maximum context
# length (65536)". MAX_READ_CHARS is a CHARACTER budget, but the constraint is
# TOKENS. The article was 121,028 chars — under the 150,000 char cap, so the
# whole thing was sent — yet it tokenised to 70,766 (1.71 chars/token) because
# polling tables are numeric. Prose runs ~4 chars/token, so one char budget
# cannot serve both: it either wastes 60% of context on prose or overflows on
# tables. The read is lost entirely (hard 400), on exactly the table-heavy list
# pages the reader exists for.


def test_dense_text_is_costed_higher_per_char_than_prose():
    prose = "the quick brown fox jumped over the lazy dog and kept running " * 200
    table = "| 23 | 27.5 | 12.1 | 4.8 | 19.3 | 8.2 | 31.7 | 2.9 | 15.4 |\n" * 200
    assert estimate_tokens(table) / len(table) > estimate_tokens(prose) / len(prose)


def test_estimate_tokens_survives_without_a_tokeniser(monkeypatch):
    """tiktoken is a convenience, not a dependency: the fallback must still bound."""
    monkeypatch.setattr(_agentic_read, "_ENCODER", None)
    n = estimate_tokens("| 23 | 27.5 | 12.1 |\n" * 100)
    assert n > 0


def test_prose_article_under_budget_keeps_the_full_char_limit():
    prose = "the quick brown fox jumped over the lazy dog " * 500
    assert effective_char_limit(prose) == MAX_READ_CHARS


def test_dense_article_over_budget_gets_a_smaller_char_limit():
    """The whole point: a token-dense article must be cut on TOKENS, not chars."""
    dense = "| 23 | 27.5 | 12.1 | 4.8 | 19.3 | 8.2 |\n" * 20_000
    assert estimate_tokens(dense) > MAX_READ_TOKENS
    limit = effective_char_limit(dense)
    assert limit < MAX_READ_CHARS
    assert estimate_tokens(dense[:limit]) <= MAX_READ_TOKENS


def test_effective_limit_never_exceeds_the_char_cap():
    """A very sparse article must not be allowed to blow past MAX_READ_CHARS."""
    sparse = "a" + (" " * 400_000)
    assert effective_char_limit(sparse) <= MAX_READ_CHARS


def test_estimate_tokens_is_fast_on_a_long_repeated_run():
    """BPE merging is quadratic in run length: 400k spaces took 81s unguarded, and
    the reader estimates on EVERY read, so one padded article stalls a question."""
    import time as _time

    t0 = _time.time()
    estimate_tokens("a" + (" " * 400_000))
    assert _time.time() - t0 < 5.0


def test_collapsed_run_is_still_charged_for():
    """The collapse is a speed device — the characters are still SENT, so dropping
    them from the estimate would under-count and reproduce the 400 it prevents."""
    assert estimate_tokens(" " * 400_000) > 400_000 // 8


# --- completion budget (2026-07-30) -----------------------------------------
# Fixing the INPUT budget exposed the OUTPUT one. With the whole 100-row table
# now visible, the reader tries to cite every row; all four errors in the re-run
# had completion_tokens == 4000 exactly (the max_tokens cap) and surfaced as
# "unparseable reader response: unbalanced braces". A truncated reply lost the
# entire read — including the answer, which the model had already finished
# emitting before the citations ran out of room.


def test_completion_budget_fits_inside_the_model_context():
    """The two budgets are one constraint; independent magic numbers drift apart."""
    assert MAX_READ_TOKENS + READ_COMPLETION_TOKENS < MODEL_CONTEXT_TOKENS


def test_salvage_recovers_answer_and_whole_citations_from_a_truncated_reply():
    truncated = (
        '{"answer": "a, b, c", "citations": ['
        '{"span": "alpha", "offset": 1}, '
        '{"span": "beta", "offset": 2}, '
        '{"span": "gam'
    )
    obj = _agentic_read._salvage_truncated_json(truncated)
    assert obj["answer"] == "a, b, c"
    assert obj["citations"] == [
        {"span": "alpha", "offset": 1},
        {"span": "beta", "offset": 2},
    ], "the incomplete trailing citation is dropped, the complete ones survive"


def test_salvage_returns_none_for_text_that_is_not_truncated_json():
    assert _agentic_read._salvage_truncated_json("I could not find anything.") is None
    assert _agentic_read._salvage_truncated_json('{"answer": "x"}') is None


def test_parse_salvages_a_truncated_response_instead_of_losing_the_read():
    full = "alpha" + ("x" * 100) + "beta"
    truncated = (
        '{"answer": "alpha and beta", "citations": ['
        '{"span": "alpha", "offset": 0}, '
        '{"span": "beta", "offset": 105}, '
        '{"span": "gam'
    )
    r = parse_read_response(truncated, full, ["All"], [])
    assert r.error is None, "a truncated reply is partial evidence, not no evidence"
    assert r.answer == "alpha and beta"
    assert len(r.citations) == 2
    assert r.truncated is True
    assert r.to_dict()["truncated"] is True, "the parent must know the list is partial"


def test_salvaged_result_is_not_marked_fully_grounded():
    """Citations were dropped, so `inputs_grounded` would be a lie."""
    full = "alpha" + ("x" * 100) + "beta"
    truncated = (
        '{"answer": "alpha", "derivation": "d", "citations": ['
        '{"span": "alpha", "offset": 0}, '
        '{"span": "be'
    )
    r = parse_read_response(truncated, full, ["All"], [])
    assert r.answer == "alpha"
    assert r.inputs_grounded is False


def test_genuinely_unparseable_output_is_still_an_error():
    r = parse_read_response("sorry, no idea", "text", ["All"], [])
    assert r.error is not None
    assert r.answer is None


def test_prompt_asks_for_short_citation_spans():
    assert "short" in _agentic_read.BRIEFED_READER_SYSTEM.lower()


# --- A: the reader returns the INFOBOX (2026-07-31) -------------------------
# Measured cause of every failed read change: the parent loses its infobox.
# FRAMES-824, infobox reach vs score — offset-0 head 74% -> 667; chunk window
# 40% -> 651; reader-only 0% -> 648. The reader replaced raw text with
# {answer, citations}, so the parent saw no infobox on ANY of 2,477 reads.
# `ask` therefore returns the infobox verbatim alongside its answer: a few
# hundred structured chars that carry birth dates, founding dates and counts.


def test_extract_infobox_pulls_the_head_table():
    art = ("Marie Curie\n\n[INFOBOX]\nBorn | 7 November 1867\n"
           "Died | 4 July 1934\nAwards | Nobel Prize\n\n"
           "Marie Curie was a physicist and chemist. " + ("x" * 500))
    ib = _agentic_read.extract_infobox(art)
    assert "7 November 1867" in ib
    assert "Nobel Prize" in ib
    assert "physicist and chemist" not in ib, "body prose is not the infobox"


def test_extract_infobox_returns_empty_when_absent():
    assert _agentic_read.extract_infobox("Some article with no infobox at all.") == ""


def test_read_result_emits_infobox_when_present():
    full = ("T\n\n[INFOBOX]\nBorn | 1867\n\n" + ("y" * 40) + "Marcel Ciolacu")
    raw = json.dumps({"answer": "Marcel Ciolacu",
                      "citations": [{"span": "Marcel Ciolacu"}]})
    r = parse_read_response(raw, full, ["All"], [], infobox=_agentic_read.extract_infobox(full))
    d = r.to_dict()
    assert "Born | 1867" in d["infobox"]


def test_no_infobox_key_when_the_article_has_none():
    full = ("y" * 40) + "Marcel Ciolacu"
    raw = json.dumps({"answer": "Marcel Ciolacu",
                      "citations": [{"span": "Marcel Ciolacu"}]})
    r = parse_read_response(raw, full, ["All"], [], infobox=_agentic_read.extract_infobox(full))
    assert "infobox" not in r.to_dict()


# --- briefed reader: opening context (2026-07-31) ---------------------------
# The single-shot reader got the whole article and the section map but was never
# told WHICH chunk matched the search. 24% of its reads concerned a chunk >12k
# chars into the article, and 26% came back not_found. The briefing hands over
# what we already know — infobox, map, and the retrieved passage with its char
# range — instead of a ~30k-char dump, and the reader asks for more via tools.


def test_briefing_carries_infobox_map_and_the_retrieved_passage():
    art = ("T\n\n[INFOBOX]\nSpouse | Suzanne (m. 2001)\n\n"
           + ("a" * 1000) + "\n\n## Personal life\n" + ("PASSAGE_MARKER " * 20)
           + "\n\n## Discography\n" + ("z" * 400))
    start = art.index("PASSAGE_MARKER")
    b = _agentic_read.build_reader_briefing(
        title="T", question="when did he marry?", full_text=art,
        chunk_text=art[start:start + 200], chunk_start=start,
        outline=[{"section": "Personal life", "start": start - 18, "shown": True},
                 {"section": "Discography", "start": len(art) - 410, "shown": True}],
    )
    user = b[-1]["content"]
    assert "Spouse | Suzanne (m. 2001)" in user, "infobox must be handed over"
    assert "Personal life" in user, "section map must be handed over"
    assert "PASSAGE_MARKER" in user, "the retrieved chunk must be handed over"
    assert f"{start:,}" in user, "the chunk's char position must be stated"
    assert "when did he marry?" in user


def test_briefing_is_far_smaller_than_the_whole_article():
    art = "T\n\n[INFOBOX]\nBorn | 1959\n\n" + ("x" * 60_000)
    b = _agentic_read.build_reader_briefing(
        title="T", question="q?", full_text=art,
        chunk_text="x" * 800, chunk_start=30_000, outline=[],
    )
    assert len(b[-1]["content"]) < len(art) // 4


def test_briefing_says_so_when_there_is_no_infobox():
    """Explicit, so the reader does not assume the fact table was withheld."""
    b = _agentic_read.build_reader_briefing(
        title="T", question="q?", full_text="no infobox here " * 50,
        chunk_text="no infobox here", chunk_start=0, outline=[],
    )
    assert "no infobox" in b[-1]["content"].lower()


def test_briefing_system_prompt_documents_both_reader_tools():
    b = _agentic_read.build_reader_briefing(
        title="T", question="q?", full_text="x" * 100,
        chunk_text="x" * 10, chunk_start=0, outline=[],
    )
    sysmsg = b[0]["content"]
    assert "find(" in sysmsg and "read_full(" in sysmsg


class _StubReaderLLM:
    """Replays scripted reader replies; records what it was asked."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def chat(self, messages, **kw):
        from types import SimpleNamespace
        self.calls.append(messages)
        text = self.texts.pop(0) if self.texts else ""
        return SimpleNamespace(text=text, reasoning_content="", prompt_tokens=1,
                               completion_tokens=1, finish_reason="stop", tool_calls=[])


# --- briefed reader: tools and the turn cap ---------------------------------


def test_reader_find_returns_offsets_and_context():
    art = ("x" * 5000) + "Suzanne Krajewski married in 2001" + ("y" * 5000)
    out = _agentic_read.reader_find(art, "Krajewski")
    assert out["match_count"] == 1
    m = out["matches"][0]
    assert m["offset"] == art.index("Krajewski")
    assert "married in 2001" in m["text"]


def test_reader_find_reports_zero_without_raising_on_a_bad_regex():
    out = _agentic_read.reader_find("some text", "(unclosed")
    assert out["match_count"] == 0
    assert "error" in out


def test_reader_read_full_is_token_budgeted_and_pages():
    art = "H" * 300_000
    out = _agentic_read.reader_read_full(art, offset=0)
    assert out["window"]["start"] == 0
    assert len(out["text"]) < len(art), "must not dump a 300k-char article"
    assert out["window"]["end"] < len(art)


def test_reader_answers_from_the_briefing_without_any_tool_call():
    """The point of the briefing: most questions end on turn one."""
    art = "T\n\n[INFOBOX]\nSpouse | Suzanne (m. 2001)\n\n" + ("z" * 900)
    llm = _StubReaderLLM([json.dumps(
        {"answer": "2001", "citations": [{"span": "Suzanne (m. 2001)"}]})])
    r = _agentic_read.read_article_briefed(
        title="T", full_text=art, question="when did he marry?",
        chunk_text="z" * 100, chunk_start=60, outline=[], llm=llm)
    assert r.answer == "2001"
    assert r.turns == 1
    assert r.tool_calls == 0


def test_reader_escalates_with_find_then_answers():
    art = ("q" * 4000) + "the winner was Rafer Johnson" + ("w" * 4000)
    llm = _StubReaderLLM([
        json.dumps({"tool": "find", "pattern": "winner was"}),
        json.dumps({"answer": "Rafer Johnson",
                    "citations": [{"span": "the winner was Rafer Johnson"}]}),
    ])
    r = _agentic_read.read_article_briefed(
        title="T", full_text=art, question="who won?",
        chunk_text="q" * 50, chunk_start=0, outline=[], llm=llm)
    assert r.answer == "Rafer Johnson"
    assert r.tool_calls == 1
    assert r.turns == 2
    assert r.prompt_tokens == 2
    assert r.completion_tokens == 2


def test_reader_stops_at_the_turn_cap():
    """A reader that keeps calling tools must be stopped; this project has
    produced a 581-turn runaway."""
    art = "nothing useful here " * 500
    llm = _StubReaderLLM([json.dumps({"tool": "find", "pattern": "zzz"})] * 12)
    r = _agentic_read.read_article_briefed(
        title="T", full_text=art, question="q?",
        chunk_text="nothing", chunk_start=0, outline=[], llm=llm)
    assert r.turns <= _agentic_read.MAX_READER_TURNS
    assert r.answer is None and r.not_found_reason is not None


def test_reader_tells_the_model_when_its_reply_did_not_parse():
    """Measured: 13 of 46 reads burned all 4 turns and errored, every one with
    'no opening brace in response'. The loop was re-sending IDENTICAL messages,
    so the model repeated itself four times. A retry has to differ from the
    attempt that failed."""
    art = "T\n\n[INFOBOX]\nBorn | 1959\n\n" + ("z" * 500)
    llm = _StubReaderLLM([
        "I think the answer is probably 1959.",            # prose, no JSON
        json.dumps({"answer": "1959", "citations": [{"span": "Born | 1959"}]}),
    ])
    r = _agentic_read.read_article_briefed(
        title="T", full_text=art, question="when born?",
        chunk_text="z" * 50, chunk_start=30, outline=[], llm=llm)
    assert r.answer == "1959"
    assert len(llm.calls) == 2
    second = llm.calls[1][-1]["content"]
    assert "JSON" in second, "the retry must tell the model what went wrong"


def test_reader_gives_up_after_repeated_unparseable_replies():
    art = "T\n\n[INFOBOX]\nBorn | 1959\n\n" + ("z" * 500)
    llm = _StubReaderLLM(["not json at all"] * 10)
    r = _agentic_read.read_article_briefed(
        title="T", full_text=art, question="q?",
        chunk_text="z", chunk_start=0, outline=[], llm=llm)
    assert r.turns <= _agentic_read.MAX_READER_TURNS
    assert len(llm.calls) <= _agentic_read.MAX_READER_TURNS


def test_reader_budget_survives_four_tool_calls():
    """Measured on the 200Q run: every read that hit the cap had spent two turns
    on find and had real hits back — the budget ran out mid-investigation, not on
    a runaway. Four searches then an answer must fit."""
    art = ("q" * 4000) + "the winner was Rafer Johnson" + ("w" * 4000)
    llm = _StubReaderLLM(
        [json.dumps({"tool": "find", "pattern": f"p{i}"}) for i in range(4)]
        + [json.dumps({"answer": "Rafer Johnson",
                       "citations": [{"span": "the winner was Rafer Johnson"}]})]
    )
    r = _agentic_read.read_article_briefed(
        title="T", full_text=art, question="who won?",
        chunk_text="q" * 50, chunk_start=0, outline=[], llm=llm)
    assert r.answer == "Rafer Johnson"
    assert r.tool_calls == 4
    assert r.turns == 5


def test_exhausted_budget_is_not_found_even_after_a_parse_failure():
    """Measured: 3 reads reported outcome=error with turns=4, tool_calls=2 and a
    real find trail. Budget exhaustion is a not_found, not an error — and the
    parent gets NO article text on the error path, so the infobox has to come
    back with it."""
    art = "T\n\n[INFOBOX]\nBorn | 1959\n\n" + ("z" * 500)
    llm = _StubReaderLLM(
        [json.dumps({"tool": "find", "pattern": "zz"})] + ["not json at all"] * 12)
    r = _agentic_read.read_article_briefed(
        title="T", full_text=art, question="q?",
        chunk_text="z" * 50, chunk_start=30, outline=[], llm=llm)
    assert r.error is None, "an exhausted budget is not an error"
    assert r.not_found_reason is not None
    assert r.infobox and "1959" in r.infobox


def test_absence_claim_from_the_passage_alone_must_search_first():
    """q185: asked which actors in Home Alone 2 later became US President, the
    reader answered "None", citing the main cast list, having never searched.
    Trump has a cameo. You cannot establish absence from a partial passage."""
    art = ("Home Alone 2\n\nCast: Macaulay Culkin, Joe Pesci, Daniel Stern.\n"
           + "z" * 600 + "\nDonald Trump appears in a cameo.\n")
    llm = _StubReaderLLM([
        json.dumps({"answer": "None of the actors later became President",
                    "citations": [{"span": "Macaulay Culkin, Joe Pesci, Daniel Stern"}]}),
        json.dumps({"tool": "find", "pattern": "Trump"}),
        json.dumps({"answer": "Donald Trump",
                    "citations": [{"span": "Donald Trump appears in a cameo"}]}),
    ])
    r = _agentic_read.read_article_briefed(
        title="Home Alone 2", full_text=art,
        question="Which actors later became U.S. President?",
        chunk_text="Cast: Macaulay Culkin, Joe Pesci, Daniel Stern.",
        chunk_start=14, outline=[], llm=llm)
    assert r.answer == "Donald Trump"
    assert r.tool_calls == 1
    pushback = llm.calls[1][-1]["content"]
    assert "find" in pushback


def _table(first, last):
    return "".join(f"{i} | row {i} | value {i}\n" for i in range(first, last + 1))


def test_a_cut_table_is_completed_forwards():
    """Measured on MoNaCo: 27% of list/table runs corpus-wide are cut by a chunk
    boundary, 36% on the articles behind our under-emission losses, and 37% of
    all missed gold items sat in a retrieved article OUTSIDE the briefed chunk.
    A chunk that stops mid-table hands the reader half a list."""
    art = "T\n\nintro\n\n" + _table(1, 60) + "\nafterwards prose\n"
    start = art.index("1 | row 1")
    cut = art.index("21 | row 21")            # chunk stops a third of the way in
    msgs = _agentic_read.build_reader_briefing(
        title="T", question="list them", full_text=art,
        chunk_text=art[start:cut], chunk_start=start, outline=[])
    passage = msgs[-1]["content"]
    assert "21 | row 21" in passage, "the rest of the cut table must be delivered"
    assert "60 | row 60" in passage
    assert "afterwards prose" not in passage, "completion stops at the run's end"


def test_a_table_entered_mid_run_is_completed_backwards():
    art = "T\n\nintro\n\n" + _table(1, 40) + "\nprose\n"
    start = art.index("15 | row 15")           # chunk starts mid-table
    end = art.index("30 | row 30")
    msgs = _agentic_read.build_reader_briefing(
        title="T", question="q", full_text=art,
        chunk_text=art[start:end], chunk_start=start, outline=[])
    passage = msgs[-1]["content"]
    assert "1 | row 1" in passage, "rows before the chunk are part of the same run"
    assert "40 | row 40" in passage


def test_an_infobox_is_not_a_list_to_complete():
    """Measured on the first list-run run: the completion fired on 60% of reads,
    median +7 rows. The row pattern matched ANY pipe-bearing line, so the
    infobox's own `key | value` rows counted as a table — and point-fact reads
    got padded with fact-table fragments. Single-answer F1 fell to 0.846 against
    champion's 0.892 while multi-value rose. The gate has to want a real list."""
    art = ("Marvin Gaye\n\n[INFOBOX]\nBorn | April 2, 1939\nDied | April 1, 1984\n"
           "Genres | Soul\nLabels | Motown\nYears | 1958-1984\n\n"
           + "He was a singer. " * 60)
    # the chunk boundary falls INSIDE the infobox — the case that fired in prod
    start, cut = 0, art.index("Genres | Soul")
    s, e, added, _ = _agentic_read.complete_cut_run(art, start, cut)
    assert added == 0, "a five-row infobox is not a list worth completing"


def test_a_short_run_is_not_worth_completing():
    art = "T\n\nprose\n\n" + _table(1, 5) + "\nmore prose here\n"
    start = art.index("prose\n\n") + 7
    s, e, added, _ = _agentic_read.complete_cut_run(art, start, start + 30)
    assert added == 0, "5 rows is a fragment, not a list"


def test_a_long_run_still_completes():
    art = "T\n\n" + _table(1, 200) + "\nprose\n"
    start = art.index("1 | row 1")
    s, e, added, _ = _agentic_read.complete_cut_run(art, start, start + 40)
    assert added >= 100, "a 200-row table is exactly what this exists for"


def test_a_prose_chunk_is_left_alone():
    """No run, no completion — this must not tax the ordinary point-fact read."""
    art = "T\n\n" + ("Marvin Gaye was born in 1939. " * 40)
    body = art[20:300]
    msgs = _agentic_read.build_reader_briefing(
        title="T", question="when born?", full_text=art,
        chunk_text=body, chunk_start=20, outline=[])
    passage = msgs[-1]["content"].split("RETRIEVED PASSAGE")[1]
    assert body.strip() in passage
    assert "rows" not in passage.lower()


def test_completion_is_capped_and_says_where_to_resume():
    """List of Roman emperors is 334 rows cut into 16 chunks; delivering a whole
    run unbounded would reinstate the whole-article dump the briefing replaced."""
    art = "T\n\n" + _table(1, 4000)
    start = art.index("1 | row 1")
    cut = art.index("5 | row 5")
    msgs = _agentic_read.build_reader_briefing(
        title="T", question="q", full_text=art,
        chunk_text=art[start:cut], chunk_start=start, outline=[])
    passage = msgs[-1]["content"]
    assert len(passage) < 40_000, "the completion must stay bounded"
    assert "read_full" in passage, "a truncated completion must say how to continue"


def test_a_fabricated_tool_result_is_never_echoed_back():
    """Measured: 167 replies in one run, 58 in another, contained BOTH the tool
    request and an invented result — usually {"match_count": 0, "matches": []},
    once "Tool execution failed: read_full not supported in this environment"
    (our code emits no such string). The loop executes only the first object, but
    it echoed the whole reply back, so the history held the model's invented
    "0 matches" next to the real result and it had to arbitrate between them."""
    art = "T\n\nalpha\n" + ("z" * 600) + "\nthe winner was Rafer Johnson\n"
    fabricated = (json.dumps({"tool": "find", "pattern": "winner"})
                  + json.dumps({"match_count": 0, "matches": []}))
    llm = _StubReaderLLM([
        fabricated,
        json.dumps({"answer": "Rafer Johnson",
                    "citations": [{"span": "the winner was Rafer Johnson"}]}),
    ])
    r = _agentic_read.read_article_briefed(
        title="T", full_text=art, question="who won?",
        chunk_text="alpha", chunk_start=5, outline=[], llm=llm)
    assert r.answer == "Rafer Johnson"
    echoed = [m for m in llm.calls[1] if m["role"] == "assistant"][-1]["content"]
    assert '"match_count"' not in echoed, "the invented result must not survive"
    assert '"tool"' in echoed and "winner" in echoed, "the request itself must"


def test_the_wrong_attribute_case_is_handled_in_the_prompt_not_by_a_gate():
    """q318 answered the SPIKE height when asked the FLOWER height, citing the
    spike verbatim. The mechanical form of this check — "no citation contains an
    attribute term from the question" — was built and REFUTED: replayed over the
    run it fired on 62% of answers, and all 13 inspected fires were correct
    citations of bare values ("1929", "5:31"). It stays in the prompt, where a
    false positive costs nothing."""
    s = _agentic_read.BRIEFED_READER_SYSTEM
    assert "property that was ASKED FOR" in s
    assert "wrong property is still a wrong answer" in s
    assert not hasattr(_agentic_read, "_attribute_terms"), \
        "the refuted citation-term gate must not come back"


def test_the_prompt_quotes_the_real_turn_budget():
    """It said "at most 4 turns" for a whole run after the cap became 8."""
    assert (f"at most {_agentic_read.MAX_READER_TURNS} turns"
            in _agentic_read.BRIEFED_READER_SYSTEM)


def test_a_grounded_answer_is_not_made_to_search():
    """The gate must not tax the 78 passage-only answers that are already right —
    an extra turn on every read is the whole cost of the reader."""
    art = "Marvin Gaye\n\n[INFOBOX]\nBorn | April 2, 1939\n\n" + ("z" * 400)
    llm = _StubReaderLLM([
        json.dumps({"answer": "April 2, 1939",
                    "citations": [{"span": "Born | April 2, 1939"}]}),
    ])
    r = _agentic_read.read_article_briefed(
        title="Marvin Gaye", full_text=art, question="When was he born?",
        chunk_text="z" * 50, chunk_start=44, outline=[], llm=llm)
    assert r.answer == "April 2, 1939"
    assert r.turns == 1 and r.tool_calls == 0


def test_the_search_gate_fires_at_most_once():
    """A reader that keeps answering off the same citation must not be pushed
    back forever — the budget is for investigating, not for arguing."""
    art = "T\n\nCast: A, B, C.\n" + ("z" * 600)
    llm = _StubReaderLLM([json.dumps(
        {"answer": "None", "citations": [{"span": "Cast: A, B, C"}]})] * 10)
    r = _agentic_read.read_article_briefed(
        title="T", full_text=art, question="Which one became President?",
        chunk_text="Cast: A, B, C.", chunk_start=5, outline=[], llm=llm)
    assert r.answer == "None"
    assert len(llm.calls) == 2, "one pushback, then take the answer"


def test_reply_schema_shows_the_model_HOW_to_request_a_tool():
    """Measured: 0 tool calls in 46 reads. The tools were described in prose but
    the reply schema listed only answer/not_found, so the escalation path was
    unreachable — the model did the only two things it was shown."""
    s = _agentic_read.BRIEFED_READER_SYSTEM
    schema = s[s.find("Reply with ONE JSON"):]
    assert '"tool": "find"' in schema, "the find request shape must be in the schema"
    assert '"tool": "read_full"' in schema
    assert '"pattern"' in schema and '"offset"' in schema


# ---- defects found in the 2026-08-01 arm A postmortem --------------------------

def test_find_reports_the_true_match_count():
    """`_MAX_FIND_MATCHES` truncated the match LIST and the reported COUNT
    together. Measured on MoNaCo: 27.1% of find calls returned the saturated
    value 8, so a reader enumerating a 40-row table was told there were 8 —
    and 33% of list-intent asks came back with a single item."""
    art = "\n".join(f"{i} | Team {i} | 19{i:02d}" for i in range(40))
    r = _agentic_read.reader_find(art, r"Team \d+")
    assert r["match_count"] == 40, "the count must be the truth, not the cap"
    assert len(r["matches"]) == _agentic_read._MAX_FIND_MATCHES
    assert r["truncated"] is True, "and the reader must be told it is seeing a slice"


def test_find_is_unchanged_when_under_the_cap():
    art = "alpha\nbeta\ngamma\n"
    r = _agentic_read.reader_find(art, "alpha|beta")
    assert r["match_count"] == 2 and len(r["matches"]) == 2
    assert r["truncated"] is False


def test_a_tool_result_is_always_valid_json():
    """Results were injected as json.dumps(res)[:20000] — cut mid-string with no
    closing brace. 41% of FRAMES and 50% of MoNaCo read_full windows exceeded
    that cap, so the reader's escalation path was corrupted half the time."""
    payload = _agentic_read.render_tool_result({"text": "x" * 80_000, "window": {"start": 0, "end": 80_000}})
    assert len(payload) < 40_000
    body = json.loads(payload)                      # must parse
    assert len(body["text"]) < 80_000
    assert body.get("truncated_for_context") is True


def test_a_small_tool_result_is_passed_through_whole():
    res = {"match_count": 2, "matches": [{"span": "a"}, {"span": "b"}]}
    assert json.loads(_agentic_read.render_tool_result(res)) == res


def test_an_exhausted_read_returns_what_find_actually_matched():
    """86-89% of budget-exhausted reads HAD a find that returned matches, and
    returned not_found with citations: []. The spans sat in the reader's own
    message history and were thrown away — this was 100% of the FRAMES deficit
    (the not_found slice went 21 vs champion's 27; every other slice was tied)."""
    art = ("T\n\n" + "z" * 400 + "\nthe winner was Rafer Johnson\n" + "z" * 400)
    llm = _StubReaderLLM([json.dumps({"tool": "find", "pattern": "winner was"})] * 12)
    r = _agentic_read.read_article_briefed(
        title="T", full_text=art, question="who won?",
        chunk_text="z" * 50, chunk_start=4, outline=[], llm=llm)
    assert r.answer is None and r.not_found_reason is not None
    spans = " ".join(c.get("span", "") for c in r.citations)
    assert "Rafer Johnson" in spans, "hand back the evidence it did find"
