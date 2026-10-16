"""Fixtures for the end-to-end smoke tests (fake LLM, dense encoder, reranker)."""

from __future__ import annotations

import pytest

from tests.e2e.fakes import (
    FakeDenseRetriever,
    FakeLLMService,
    FakeReranker,
    write_toy_corpus,
)


@pytest.fixture
def fake_services(monkeypatch):
    """Install the fakes; yields ``(llm, reranker)``.

    Everything else is real: BM25, RRF fusion, RerankedRetriever, the adapter.
    """
    from panini_dr.retrieval import dense, rerank

    for var in ("GSW_READ_TITLE_FALLBACK", "GSW_PLANNER_TARGET_HUB_MODE",
                "GSW_STRICT_TARGET_HUB_GATE", "PANINI_STRICT_RERANK"):
        monkeypatch.delenv(var, raising=False)
    FakeDenseRetriever.instances.clear()
    monkeypatch.setattr(dense, "DenseRetriever", FakeDenseRetriever)
    reranker = FakeReranker()
    monkeypatch.setattr(rerank.httpx, "post", reranker.post)
    monkeypatch.setenv("JINA_RERANK_URL", "http://reranker.invalid/v1/rerank")
    rerank.RERANK_HEALTH.reset()
    rerank.set_strict(False)
    llm = FakeLLMService().install(monkeypatch)
    monkeypatch.setattr(rerank.httpx, "get", llm.get)  # the run's LLM preflight
    yield llm, reranker
    rerank.set_strict(False)
    rerank.RERANK_HEALTH.reset()


@pytest.fixture
def toy_corpus_path(tmp_path):
    return write_toy_corpus(tmp_path / "corpus.json")
