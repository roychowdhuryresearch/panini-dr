"""Smoke tests for the article corpus loader + BM25 retriever.

The research repo ran these against the full FRAMES article cache; here a tiny
synthetic {title: text} corpus is written to a temporary directory.
"""

from __future__ import annotations

import json

import pytest

from panini_dr.retrieval import (
    BM25Retriever,
    CorpusNotConfiguredError,
    load_frames_corpus,
    require_corpus,
)

_TOY_ARTICLES = {
    "Brannoc Vell": (
        "Brannoc Vell\n\n[INFOBOX]\nsport | boxing\ndivision | heavyweight\n\n"
        "Brannoc Vell is a fictional heavyweight boxer from the town of Ostrel. "
        + "He trained at the Ostrel harbour gym for many years. " * 40
        + "\n\n## Career\n\nVell won the Harbour Belt in his ninth professional bout."
    ),
    "Ostrel": "Ostrel\n\nOstrel is a fictional port town on the Merrow estuary. " * 30,
    "Merrow estuary": "Merrow estuary\n\nThe Merrow estuary drains the Cald hills. " * 30,
    "Harbour Belt": "Harbour Belt\n\nThe Harbour Belt is a regional boxing title first contested in 1931.",
    "Cald hills": "Cald hills\n\nThe Cald hills are low granite uplands. " * 30,
    "Empty page": "",
}


@pytest.fixture(scope="module")
def corpus_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("corpus") / "corpus.json"
    path.write_text(json.dumps(_TOY_ARTICLES))
    return path


@pytest.fixture(scope="module")
def corpus(corpus_path):
    return load_frames_corpus(cache_path=corpus_path)


def test_corpus_loads_and_has_chunks(corpus):
    assert len(corpus) > 0
    titles = corpus.titles()
    assert len(titles) == len(_TOY_ARTICLES)
    sample = titles[0]
    assert len(corpus.article_text(sample)) > 0
    # empty articles produce no chunks; long ones are split (2000/200 chunker)
    assert not [c for c in corpus.chunks if c.title == "Empty page"]
    assert len([c for c in corpus.chunks if c.title == "Brannoc Vell"]) > 1
    assert all(c.chunk_id.startswith(c.title + "__") for c in corpus.chunks)


def test_bm25_retrieves_relevant_titles(corpus):
    retriever = BM25Retriever(corpus)
    hits = retriever.search("Brannoc Vell heavyweight boxing", top_k=5)
    assert hits, "expected at least one hit for boxer query"
    assert any("Brannoc Vell" in h.chunk.title for h in hits)


def test_bm25_empty_query_returns_nothing(corpus):
    retriever = BM25Retriever(corpus)
    assert retriever.search("", top_k=5) == []


def test_whole_document_chunking_gives_one_chunk_per_article(corpus_path):
    from panini_dr.eval.bcp_dense import BCP_SINGLE_DOCUMENT_CHARS

    whole = load_frames_corpus(cache_path=corpus_path, chunk_chars=BCP_SINGLE_DOCUMENT_CHARS,
                               chunk_overlap=0)
    non_empty = [t for t, x in _TOY_ARTICLES.items() if x]
    assert sorted(c.chunk_id for c in whole.chunks) == sorted(f"{t}__0" for t in non_empty)


# ---- no default corpus -------------------------------------------------------


def test_load_without_a_path_is_an_explicit_error():
    with pytest.raises(CorpusNotConfiguredError):
        load_frames_corpus()


def test_load_of_a_missing_file_is_an_explicit_error(tmp_path):
    with pytest.raises(CorpusNotConfiguredError, match="not found"):
        load_frames_corpus(cache_path=tmp_path / "nope.json")


def test_require_corpus_refuses_an_empty_extra(corpus):
    with pytest.raises(CorpusNotConfiguredError, match="--articles-cache"):
        require_corpus({})
    assert require_corpus({"corpus": corpus}) is corpus


# ---- list-aware chunking -----------------------------------------------------
#
# Instrumented on MoNaCo-200: 27% of list/table runs corpus-wide are cut by a
# chunk boundary and 36% on the articles behind our under-emission losses, where
# 37% of every missed gold item sat in a retrieved article OUTSIDE the briefed
# chunk. Patching the reader's briefing after the fact cannot help retrieval —
# a half-table scores worse against a query than a whole one. This fixes it at
# the chunker, which is the only place that changes BOTH what search matches and
# what the reader is shown.

def _rows(first, last):
    return "".join(f"{i} | row {i} | value {i}\n" for i in range(first, last + 1))


def test_list_aware_keeps_a_table_whole():
    from panini_dr.retrieval.corpus import _split_spans
    art = ("Intro paragraph.\n\n" + "prose. " * 200 + "\n\n"
           + _rows(1, 60) + "\nClosing paragraph.\n")
    spans = _split_spans(art, 2000, 200, list_aware=True)
    tbl_start, tbl_end = art.index("1 | row 1"), art.index("60 | row 60")
    holding = [(s, e) for s, e in spans if s <= tbl_start and e >= tbl_end]
    assert holding, "the whole 60-row table must live in one chunk"


def test_list_aware_splits_a_huge_table_on_row_boundaries():
    from panini_dr.retrieval.corpus import _split_spans, MAX_TABLE_CHARS
    art = "T\n\n" + _rows(1, 4000)
    spans = _split_spans(art, 2000, 200, list_aware=True)
    for s, e in spans:
        assert e - s <= MAX_TABLE_CHARS + 200, "a table chunk must stay bounded"
        body = art[s:e].strip("\n")
        if "|" in body:
            assert body.endswith(tuple(str(d) for d in range(10))), \
                "a table chunk must not stop mid-row"


def test_prose_chunking_is_unchanged_by_the_flag():
    """The flag must be inert on articles with no lists — champion's corpus
    fingerprint depends on it, and a changed fingerprint re-embeds 150k chunks."""
    from panini_dr.retrieval.corpus import _split_spans
    art = "\n\n".join("Paragraph %d. %s" % (i, "word " * 80) for i in range(30))
    assert (_split_spans(art, 2000, 200, list_aware=True)
            == _split_spans(art, 2000, 200, list_aware=False))


def test_default_chunking_is_byte_identical():
    from panini_dr.retrieval.corpus import _split_spans, _split_on_paragraphs
    art = "T\n\n" + "prose. " * 300 + "\n\n" + _rows(1, 80) + "\ntail\n"
    assert _split_spans(art, 2000, 200) == _split_on_paragraphs(art, 2000, 200)


def test_fingerprint_separates_list_aware_chunking():
    """The dense cache is keyed by (titles, text lengths) — NOT by chunking. A
    list-aware corpus has 134,384 chunks where the legacy one has 150,675, and
    the loader does no length check, so it would have loaded the stale index and
    returned chunk_ids that no longer exist. Legacy fingerprints must not move
    (that would re-embed every existing cache); the new chunking needs its own
    bucket."""
    from panini_dr.retrieval.corpus import ArticleCorpus, Chunk
    from panini_dr.retrieval.dense import DenseRetriever
    by_title = {"A": "x" * 100}
    legacy = ArticleCorpus([Chunk("A__0", "A", "x" * 100, 0, 100)], by_title)
    aware = ArticleCorpus([Chunk("A__0", "A", "x" * 100, 0, 100)], by_title)
    aware.list_aware = True
    f_legacy = DenseRetriever.__dict__["_fingerprint"](
        type("S", (), {"corpus": legacy, "model_name": "m"})())
    f_aware = DenseRetriever.__dict__["_fingerprint"](
        type("S", (), {"corpus": aware, "model_name": "m"})())
    assert f_legacy != f_aware, "list-aware chunking needs its own cache bucket"
    assert not f_legacy.endswith("listaware"), "legacy key must not change"


def test_corpus_label_follows_the_dataset(corpus_path):
    """A user's dataset is not labelled as a FRAMES-derived control corpus."""
    import logging

    class _Collect(logging.Handler):
        def __init__(self):
            super().__init__(logging.DEBUG)
            self.lines: list[str] = []

        def emit(self, record):
            self.lines.append(record.getMessage())

    # A handler on the module's own logger: "panini_dr" may not propagate to the
    # root logger (a run sets propagate=False), so caplog can miss these lines.
    logger = logging.getLogger("panini_dr.retrieval.corpus")
    handler, level = _Collect(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        mine = load_frames_corpus(cache_path=corpus_path, benchmark="byod", dataset_name="toy")
        text = "\n".join(handler.lines)
        handler.lines.clear()
        frames = load_frames_corpus(cache_path=corpus_path)
        frames_text = "\n".join(handler.lines)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    assert mine.manifest.corpus_id.startswith("dataset:toy:")
    assert mine.manifest.scope.value == "custom" and mine.manifest.benchmark_name is None
    assert "your own dataset" in text
    assert "frames-derived" not in text and "control runs only" not in text

    # the benchmark corpora keep their label and their log line
    assert frames.manifest.corpus_id.startswith("frames-derived:corpus.json:")
    assert frames.manifest.scope.value == "benchmark_derived" and frames.manifest.benchmark_name == "FRAMES"
    assert "has scope=benchmark_derived" in frames_text and "control runs only" in frames_text
    monaco = load_frames_corpus(cache_path=corpus_path, benchmark="monaco")
    assert monaco.manifest.corpus_id.startswith("monaco-derived:") and monaco.manifest.benchmark_name == "MoNaCo"
    bcp = load_frames_corpus(cache_path=corpus_path, benchmark="browsecomp_plus")
    assert bcp.manifest.corpus_id.startswith("browsecomp_plus-derived:")
    # the label never changes the chunks
    assert [c.chunk_id for c in mine.chunks] == [c.chunk_id for c in frames.chunks]
