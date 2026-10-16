"""Tevatron's published BrowseComp-Plus BM25 index, behind our retriever API.

Every test injects a fake Lucene searcher. The real one needs a JVM and a 2.1 GB
index, and none of the logic under test is Lucene's — it is the docid -> our-chunk
mapping, which is exactly where a silent misalignment would hide (the same failure
class ``bcp_dense.validate_prebuilt_dense_cache`` guards on the dense side).
"""

import pytest

from panini_dr.retrieval.corpus import ArticleCorpus, Chunk
from panini_dr.retrieval.lucene_bm25 import (
    LuceneBM25Retriever,
    build_docid_index,
    docid_from_cache_key,
)


class _FakeHit:
    def __init__(self, docid: str, score: float):
        self.docid = docid
        self.score = score


class _FakeSearcher:
    """Stands in for pyserini's LuceneSearcher: ``.num_docs`` + ``.search``."""

    def __init__(self, results, num_docs=None):
        self._results = results
        self.num_docs = len(results.get("_docids", [])) if num_docs is None else num_docs

    def search(self, query, k=10):
        return list(self._results.get(query, []))[:k]


def _corpus(keys_and_text):
    """One chunk per document — the BCP chunking (``chunk_chars`` > longest doc)."""
    chunks = [
        Chunk(chunk_id=f"{key}__0", title=key, text=text, char_start=0, char_end=len(text))
        for key, text in keys_and_text
    ]
    return ArticleCorpus(chunks, {k: t for k, t in keys_and_text})


@pytest.fixture
def corpus():
    return _corpus(
        [
            ("Alexander Fleming [98474]", "Fleming discovered penicillin in 1928."),
            ("Discovery of Penicillin [67914]", "Penicillin reached mass production in 1944."),
            ("5412", "An untitled scrape with no front-matter title."),
        ]
    )


# --- the docid <-> cache-key inversion ------------------------------------


def test_docid_parses_out_of_a_titled_cache_key():
    assert docid_from_cache_key("Alexander Fleming [98474]") == "98474"


def test_docid_is_the_whole_key_when_the_document_had_no_title():
    # browsecomp_keys.cache_key falls back to the bare docid for the 1,176
    # documents that carry no front-matter title.
    assert docid_from_cache_key("5412") == "5412"


def test_a_title_that_itself_ends_in_brackets_yields_the_appended_docid():
    # cache_key appends "[docid]" LAST, so the final bracket is always the docid.
    assert docid_from_cache_key("Report [1999] [82002]") == "82002"


def test_a_bracketed_suffix_that_is_not_a_docid_is_not_mistaken_for_one():
    # Only a numeric final bracket is a docid; anything else means the whole key
    # is the id (the untitled form), not a title we may strip.
    assert docid_from_cache_key("Chapter [draft]") == "Chapter [draft]"


# --- the docid -> chunk index ----------------------------------------------


def test_build_docid_index_maps_every_document(corpus):
    index = build_docid_index(corpus)
    assert set(index) == {"98474", "67914", "5412"}
    assert index["98474"].chunk_id == "Alexander Fleming [98474]__0"


def test_duplicate_docid_is_refused(corpus):
    dupe = _corpus(
        [("Fleming [98474]", "one"), ("Fleming, again [98474]", "two")]
    )
    with pytest.raises(ValueError, match="duplicate docid"):
        build_docid_index(dupe)


# --- the retriever ----------------------------------------------------------


def test_search_returns_our_chunks_in_lucene_rank_order(corpus):
    searcher = _FakeSearcher(
        {"penicillin": [_FakeHit("67914", 7.26), _FakeHit("98474", 7.49)]}, num_docs=3
    )
    r = LuceneBM25Retriever(corpus, searcher=searcher)
    hits = r.search("penicillin", top_k=5)

    # Lucene's order is authoritative — we relabel, never re-sort.
    assert [h.chunk.chunk_id for h in hits] == [
        "Discovery of Penicillin [67914]__0",
        "Alexander Fleming [98474]__0",
    ]
    assert hits[0].score == pytest.approx(7.26)
    assert hits[0].chunk.text.startswith("Penicillin reached")


def test_search_asks_lucene_for_exactly_top_k(corpus):
    seen = {}

    class _Recording(_FakeSearcher):
        def search(self, query, k=10):
            seen["k"] = k
            return super().search(query, k)

    searcher = _Recording(
        {"q": [_FakeHit("98474", 1.0), _FakeHit("67914", 0.5), _FakeHit("5412", 0.1)]},
        num_docs=3,
    )
    r = LuceneBM25Retriever(corpus, searcher=searcher)
    assert len(r.search("q", top_k=2)) == 2
    assert seen["k"] == 2


def test_unmapped_docid_raises_rather_than_dropping_a_hit(corpus):
    # Dropping it would shorten the result list silently and make an index built
    # over a different corpus look like a merely weak retriever.
    searcher = _FakeSearcher({"q": [_FakeHit("999999", 3.0)]}, num_docs=3)
    r = LuceneBM25Retriever(corpus, searcher=searcher)
    with pytest.raises(ValueError, match="999999"):
        r.search("q", top_k=5)


def test_corpus_size_mismatch_is_refused_at_construction(corpus):
    searcher = _FakeSearcher({}, num_docs=100_195)
    with pytest.raises(ValueError, match="index/corpus mismatch"):
        LuceneBM25Retriever(corpus, searcher=searcher)


def test_title_bias_reweights_like_our_own_bm25(corpus):
    searcher = _FakeSearcher(
        {"q": [_FakeHit("98474", 2.0), _FakeHit("67914", 1.0)]}, num_docs=3
    )
    r = LuceneBM25Retriever(corpus, searcher=searcher)
    hits = r.search("q", top_k=5, title_bias={"Discovery of Penicillin [67914]": 10.0})
    assert [h.chunk.title for h in hits][0] == "Discovery of Penicillin [67914]"
    assert hits[0].score == pytest.approx(10.0)


def test_empty_query_returns_nothing_without_calling_lucene(corpus):
    class _Exploding(_FakeSearcher):
        def search(self, query, k=10):
            raise AssertionError("Lucene must not be queried for an empty string")

    r = LuceneBM25Retriever(corpus, searcher=_Exploding({}, num_docs=3))
    assert r.search("   ", top_k=5) == []


# --- routing ----------------------------------------------------------------


def test_build_retriever_routes_bm25_to_lucene_when_an_index_is_given(corpus, monkeypatch):
    from panini_dr.retrieval import dense, lucene_bm25

    monkeypatch.setattr(
        lucene_bm25, "open_lucene_searcher", lambda path: _FakeSearcher({}, num_docs=3)
    )
    r = dense.build_retriever("bm25", corpus, lucene_index="/some/index")
    assert isinstance(r, LuceneBM25Retriever)


def test_build_retriever_without_a_lucene_index_still_uses_our_own_bm25(corpus):
    from panini_dr.retrieval.bm25 import BM25Retriever
    from panini_dr.retrieval.dense import build_retriever

    assert isinstance(build_retriever("bm25", corpus), BM25Retriever)


def test_hybrid_uses_lucene_as_the_lexical_leg(corpus, monkeypatch):
    from panini_dr.retrieval import dense, lucene_bm25

    monkeypatch.setattr(
        lucene_bm25, "open_lucene_searcher", lambda path: _FakeSearcher({}, num_docs=3)
    )

    class _StubDense:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr(dense, "DenseRetriever", _StubDense)
    r = dense.build_retriever("hybrid", corpus, lucene_index="/some/index")
    assert isinstance(r.bm25, LuceneBM25Retriever)
    assert isinstance(r.dense, _StubDense)
