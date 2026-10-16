"""The rerank fallback is loud: failures are counted, strict mode aborts.

A failed rerank call still degrades to fusion order (the behaviour of the paper
runs, which recorded zero failures), but RERANK_HEALTH counts every call and
every fallback so a run can report them, and strict mode raises instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from panini_dr.retrieval import rerank
from panini_dr.retrieval.rerank import (
    RERANK_HEALTH,
    RerankedRetriever,
    RerankUnavailableError,
    set_strict,
)


@dataclass
class _Chunk:
    chunk_id: str
    text: str


@dataclass
class _Hit:
    chunk: _Chunk
    score: float = 1.0


class _Base:
    def search(self, query, *, top_k=5):
        return [_Hit(_Chunk(f"c{i}", f"text {i}")) for i in range(top_k)][:4]


class _GoodReranker:
    def rerank(self, query, documents, *, top_k):
        return list(reversed(range(len(documents))))[:top_k]


class _DownReranker:
    def rerank(self, query, documents, *, top_k):
        raise ConnectionError("reranker unreachable")


@pytest.fixture(autouse=True)
def _clean_health():
    RERANK_HEALTH.reset()
    set_strict(False)
    yield
    RERANK_HEALTH.reset()
    set_strict(False)


def test_successful_calls_are_counted():
    r = RerankedRetriever(_Base(), _GoodReranker())
    hits = r.search("q", top_k=2)
    assert [h.chunk.chunk_id for h in hits] == ["c3", "c2"]
    snap = RERANK_HEALTH.snapshot()
    assert snap["calls"] == 1 and snap["failures"] == 0


def test_failure_falls_back_to_fusion_order_and_is_counted():
    r = RerankedRetriever(_Base(), _DownReranker())
    hits = r.search("q", top_k=2)
    assert [h.chunk.chunk_id for h in hits] == ["c0", "c1"]  # fusion order, as before
    r.search("q2", top_k=2)
    snap = RERANK_HEALTH.snapshot()
    assert snap["calls"] == 2
    assert snap["failures"] == 2
    assert snap["fallback_to_fusion_order"] == 2
    assert snap["first_errors"][0].startswith("ConnectionError")


def test_strict_mode_raises_and_runs_the_hook_first():
    seen = []
    set_strict(True, on_failure=lambda exc, snap: seen.append((type(exc).__name__, snap["failures"])))
    r = RerankedRetriever(_Base(), _DownReranker())
    with pytest.raises(RerankUnavailableError):
        r.search("q", top_k=2)
    assert seen == [("ConnectionError", 1)]


def test_strict_mode_off_again_restores_the_fallback():
    set_strict(True)
    set_strict(False)
    assert not rerank.is_strict()
    r = RerankedRetriever(_Base(), _DownReranker())
    assert len(r.search("q", top_k=2)) == 2


def test_single_candidate_skips_the_reranker_and_counts_nothing():
    class _One:
        def search(self, query, *, top_k=5):
            return [_Hit(_Chunk("c0", "x"))]

    r = RerankedRetriever(_One(), _DownReranker())
    assert len(r.search("q", top_k=5)) == 1
    assert RERANK_HEALTH.snapshot()["calls"] == 0
