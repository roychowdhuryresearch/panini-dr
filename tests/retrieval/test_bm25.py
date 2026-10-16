

# --- Vectorised scoring must be EXACTLY the reference scorer -----------------
# BM25 scored every chunk in a Python loop: at 616,096 chunks (MoNaCo v2) that
# is ~1.5s per query AND it holds the GIL, so the gateway's threadpool gave only
# 1.9x on 4 concurrent searches instead of 4x. Every arm shares this retriever,
# so the vectorised version has to return byte-identical rankings or it silently
# changes what every system retrieves.

import math

import pytest
from panini_dr.retrieval.bm25 import BM25Retriever, _tokenize
from panini_dr.retrieval.corpus import ArticleCorpus, Chunk


def _corpus(texts):
    chunks = [Chunk(chunk_id=f"d{i}", title=f"T{i}", text=t, char_start=0, char_end=len(t))
              for i, t in enumerate(texts)]
    return ArticleCorpus(chunks=chunks, by_title={c.title: c.text for c in chunks})


def _reference_search(r, query, top_k, title_bias=None):
    """The original loop, kept here as the oracle."""
    terms = _tokenize(query)
    if not terms:
        return []
    scored = []
    for idx, chunk in enumerate(r.corpus.chunks):
        s = r._score_doc(idx, terms)
        if title_bias:
            s *= title_bias.get(chunk.title, 1.0)
        if s > 0:
            scored.append((s, idx))
    scored.sort(reverse=True)
    return [(r.corpus.chunks[i].chunk_id, round(s, 10)) for s, i in scored[:top_k]]


_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "quick quick brown fox",                       # repeated term in doc
    "lazy dog sleeps",
    "a completely unrelated sentence about turtles",
    "brown fox brown fox brown fox",               # skewed tf
    "dog",                                          # very short doc
    "the " * 200 + "fox",                          # very long doc (length norm)
]


def test_vectorised_matches_reference_ranking():
    r = BM25Retriever(_corpus(_TEXTS))
    for q in ["quick brown fox", "dog", "lazy dog", "turtles", "fox fox", "nothing matches here"]:
        got = [(c.chunk.chunk_id, round(c.score, 10)) for c in r.search(q, top_k=5)]
        assert got == _reference_search(r, q, 5), f"ranking diverged for {q!r}"


def test_vectorised_honours_repeated_query_terms():
    """`fox fox` scores double `fox` -- the loop iterated raw tokens, not a set."""
    r = BM25Retriever(_corpus(_TEXTS))
    one = {c.chunk.chunk_id: c.score for c in r.search("fox", top_k=9)}
    two = {c.chunk.chunk_id: c.score for c in r.search("fox fox", top_k=9)}
    assert one and two
    for cid in one:
        assert two[cid] == pytest.approx(one[cid] * 2), cid


def test_vectorised_honours_title_bias_and_tie_order():
    r = BM25Retriever(_corpus(_TEXTS))
    bias = {"T0": 2.0, "T4": 0.5}
    got = [(c.chunk.chunk_id, round(c.score, 10)) for c in r.search("brown fox", top_k=7, title_bias=bias)]
    assert got == _reference_search(r, "brown fox", 7, bias)


def test_vectorised_drops_zero_scores():
    r = BM25Retriever(_corpus(_TEXTS))
    assert r.search("turtles", top_k=9)[0].chunk.chunk_id == "d3"
    assert all(c.score > 0 for c in r.search("dog", top_k=9))
