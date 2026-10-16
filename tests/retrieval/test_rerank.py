"""Reranker spec parsing and the jina/vLLM client.

jina-reranker-v3.5 is listwise: it ranks the query against ALL candidates in one
forward pass. So the client must send one request carrying every document, never
one request per document — batching per document would silently turn it into a
pointwise ranker and cost hit@5 (measured 100.0% -> 97.9% when the candidate
list is split).
"""
import pytest

from panini_dr.retrieval.rerank import (
    JinaReranker,
    RerankedRetriever,
    _bounded_rerank_documents,
    maybe_wrap_reranker,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_jina_sends_every_document_in_one_request(monkeypatch):
    seen = []

    def fake_post(url, **kw):
        seen.append((url, kw["json"]))
        docs = kw["json"]["documents"]
        return _FakeResponse({"results": [{"index": i} for i in reversed(range(len(docs)))]})

    monkeypatch.setattr("panini_dr.retrieval.rerank.httpx.post", fake_post)
    r = JinaReranker(url="http://x/v1/rerank")
    order = r.rerank("q", ["a", "b", "c", "d"], top_k=2)

    assert len(seen) == 1, "listwise ranking must be ONE call, not one per document"
    assert seen[0][1]["documents"] == ["a", "b", "c", "d"]
    assert order == [3, 2], "best-first, truncated to top_k"


def test_jina_bounds_large_listwise_request_and_keeps_query_window(monkeypatch):
    marker = "decisive telescope evidence"
    documents = [("header\n" + "x" * 20_000 + marker) for _ in range(40)]
    seen = {}

    def fake_post(url, **kw):
        seen.update(kw["json"])
        return _FakeResponse({"results": [{"index": 0}]})

    monkeypatch.setattr("panini_dr.retrieval.rerank.httpx.post", fake_post)

    JinaReranker(url="http://x/v1/rerank").rerank(
        "find decisive telescope evidence", documents, top_k=1
    )

    assert sum(map(len, seen["documents"])) <= 96_000
    assert marker in seen["documents"][0]
    assert seen["documents"][0].startswith("header")


def test_bounded_rerank_documents_leaves_small_inputs_unchanged():
    documents = ["alpha", "beta"]
    assert _bounded_rerank_documents("q", documents, request_chars=96_000) == documents


def test_jina_asks_for_every_candidate_back():
    """top_n must cover the whole list: the model ranks all of them anyway, and
    asking for fewer discards ordering the caller may slice differently."""
    captured = {}

    def fake_post(url, **kw):
        captured.update(kw["json"])
        return _FakeResponse({"results": [{"index": 0}]})

    import panini_dr.retrieval.rerank as mod
    orig, mod.httpx.post = mod.httpx.post, fake_post
    try:
        JinaReranker(url="http://x/v1/rerank").rerank("q", ["a", "b", "c"], top_k=1)
    finally:
        mod.httpx.post = orig
    assert captured["top_n"] == 3


def test_spec_jina_builds_a_jina_reranker(monkeypatch):
    monkeypatch.setenv("JINA_RERANK_URL", "http://reranker.invalid/v1/rerank")
    wrapped = maybe_wrap_reranker(object(), "jina")
    assert isinstance(wrapped, RerankedRetriever)
    assert isinstance(wrapped.reranker, JinaReranker)
    assert wrapped.reranker.url == "http://reranker.invalid/v1/rerank"


def test_spec_jina_without_an_endpoint_is_an_explicit_error(monkeypatch):
    """No built-in host: the endpoint must come from JINA_RERANK_URL or the spec."""
    monkeypatch.delenv("JINA_RERANK_URL", raising=False)
    with pytest.raises(ValueError, match="JINA_RERANK_URL"):
        maybe_wrap_reranker(object(), "jina")


def test_spec_jina_carries_a_custom_url():
    wrapped = maybe_wrap_reranker(object(), "jina:http://reranker.invalid/v1/rerank")
    assert wrapped.reranker.url == "http://reranker.invalid/v1/rerank"


@pytest.mark.parametrize("spec", ["voyage", "voyage:rerank-2.5"])
def test_spec_voyage_is_refused_as_removed(spec):
    with pytest.raises(ValueError, match="Voyage reranker was removed"):
        maybe_wrap_reranker(object(), spec)


def test_spec_empty_leaves_the_retriever_alone():
    base = object()
    assert maybe_wrap_reranker(base, "") is base
    assert maybe_wrap_reranker(base, None) is base


def test_unknown_provider_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="jina"):
        maybe_wrap_reranker(object(), "cohere")
