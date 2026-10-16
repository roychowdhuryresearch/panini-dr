"""Cross-encoder reranking over a base retriever's candidates.

The base (hybrid) retriever returns fusion-ordered candidates; a reranker
reads each ``(query, chunk)`` pair jointly and reorders by true relevance.
Targets the failure where the answer-bearing chunk is retrieved but
out-ranked by a lexically-similar-but-wrong chunk (e.g. a review-scores
table beating a release-date row).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Callable

import httpx

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rerank health: a failed rerank call degrades to fusion order (the behaviour
# the paper runs had; they recorded zero failures). That fallback silently
# changes what the agent sees, so every failure is counted, the run prints a
# summary, and strict mode turns the first failure into an abort.
# ---------------------------------------------------------------------------


class RerankUnavailableError(RuntimeError):
    """Raised on a rerank failure when strict mode is on."""


class RerankHealth:
    """Thread-safe counters for rerank calls and fallbacks to fusion order."""

    MAX_RECORDED_ERRORS = 20

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.calls = 0
            self.failures = 0
            self.errors: list[str] = []

    def record_ok(self) -> None:
        with self._lock:
            self.calls += 1

    def record_failure(self, exc: BaseException) -> None:
        with self._lock:
            self.calls += 1
            self.failures += 1
            if len(self.errors) < self.MAX_RECORDED_ERRORS:
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "failures": self.failures,
                "fallback_to_fusion_order": self.failures,
                "first_errors": list(self.errors),
            }


RERANK_HEALTH = RerankHealth()

# Strict mode: raise instead of falling back. ``PANINI_STRICT_RERANK=1`` or
# ``set_strict(True)``; ``panini-dr run --strict-rerank`` also installs an abort hook.
_STRICT = os.environ.get("PANINI_STRICT_RERANK", "").strip() in {"1", "true", "yes"}
_FAILURE_HOOK: Callable[[BaseException, dict[str, Any]], None] | None = None


def set_strict(strict: bool, *, on_failure: Callable[[BaseException, dict[str, Any]], None] | None = None) -> None:
    """Turn strict mode on/off. ``on_failure`` runs (before raising) on a failure."""
    global _STRICT, _FAILURE_HOOK
    _STRICT = bool(strict)
    _FAILURE_HOOK = on_failure if strict else None


def is_strict() -> bool:
    return _STRICT

DEFAULT_JINA_REQUEST_CHARS = 96_000


def jina_rerank_url() -> str:
    """The jina-reranker endpoint, from ``$JINA_RERANK_URL`` (resolved at call time).

    Point it at the vLLM server started by ``serve/jina_reranker.sh``, e.g.
    ``http://<host>:<port>/v1/rerank``. There is no built-in default host.
    """
    url = os.environ.get("JINA_RERANK_URL", "").strip()
    if not url:
        raise ValueError(
            "reranker 'jina' needs JINA_RERANK_URL (e.g. http://<host>:<port>/v1/rerank, "
            "the vLLM server from serve/jina_reranker.sh), or pass the spec 'jina:<url>'."
        )
    return url


def _query_focused_excerpt(query: str, document: str, *, max_chars: int) -> str:
    """Bound one reranker document while retaining its header and best query window."""
    if len(document) <= max_chars:
        return document
    if max_chars < 200:
        return document[:max_chars]

    folded = document.casefold()
    terms = sorted(
        {
            term
            for term in re.findall(r"[\w-]+", query.casefold())
            if len(term) >= 3
        },
        key=lambda term: (-len(term), term),
    )
    positions = [folded.find(term) for term in terms]
    positions = [position for position in positions if position >= 0]
    if not positions:
        return document[:max_chars]

    header_chars = min(400, max_chars // 5)
    separator = "\n...\n"
    window_chars = max_chars - header_chars - len(separator)
    best_start = 0
    best_score = -1
    for position in positions:
        start = max(0, min(position - window_chars // 3, len(document) - window_chars))
        end = start + window_chars
        window = folded[start:end]
        score = sum((len(term) + 1) ** 2 for term in terms if term in window)
        if score > best_score or (score == best_score and start < best_start):
            best_score = score
            best_start = start
    if best_start <= header_chars:
        return document[:max_chars]
    return document[:header_chars] + separator + document[best_start : best_start + window_chars]


def _bounded_rerank_documents(
    query: str, documents: list[str], *, request_chars: int
) -> list[str]:
    if not documents:
        return []
    available = max(1_000, request_chars - len(query))
    per_document = max(200, available // len(documents))
    return [
        _query_focused_excerpt(query, document, max_chars=per_document)
        for document in documents
    ]


class JinaReranker:
    """jina-reranker-v3.5 served by vLLM. Self-hosted, no API key, no per-call cost.

    The model is LISTWISE: it ranks the query against every candidate in one
    forward pass, so the whole document list goes in a single request. Splitting
    it into batches would turn it into a pointwise ranker -- measured at
    hit@5 100.0% -> 97.9% on the FRAMES pool, for no speedup, since the work
    follows total token count either way.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        model: str = "jina-rerank",
        timeout_s: float = 120.0,
        max_request_chars: int = DEFAULT_JINA_REQUEST_CHARS,
    ) -> None:
        self.url = url or jina_rerank_url()
        self.model = model
        self._timeout_s = timeout_s
        self._max_request_chars = max_request_chars

    def rerank(self, query: str, documents: list[str], *, top_k: int) -> list[int]:
        """Return document indices ordered best-first (length <= top_k)."""
        bounded_documents = _bounded_rerank_documents(
            query,
            documents,
            request_chars=self._max_request_chars,
        )
        resp = httpx.post(
            self.url,
            json={
                "model": self.model,
                "query": query,
                "documents": bounded_documents,
                # Ask for the full ranking: the model orders all of them in the
                # same pass, and the caller may want to slice differently.
                "top_n": len(documents),
            },
            timeout=self._timeout_s,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [item["index"] for item in results][:top_k]


class RerankedRetriever:
    """Wrap a base retriever: fetch ``candidate_k`` then rerank to ``top_k``."""

    def __init__(
        self,
        base: Any,
        reranker: JinaReranker,
        *,
        candidate_k: int = 40,
    ) -> None:
        self.base = base
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search(self, query: str, *, top_k: int = 5) -> list[Any]:
        candidates = self.base.search(query, top_k=self.candidate_k)
        if len(candidates) <= 1:
            return candidates[:top_k]
        docs = [c.chunk.text for c in candidates]
        try:
            order = self.reranker.rerank(query, docs, top_k=top_k)
        except Exception as exc:  # noqa: BLE001 — degrade to fusion order
            RERANK_HEALTH.record_failure(exc)
            if _STRICT:
                if _FAILURE_HOOK is not None:
                    _FAILURE_HOOK(exc, RERANK_HEALTH.snapshot())
                raise RerankUnavailableError(
                    f"rerank failed in strict mode: {exc}") from exc
            _log.warning(
                f"rerank failed ({exc}); using fusion order "
                f"[{RERANK_HEALTH.failures} rerank fallback(s) so far in this run]"
            )
            return candidates[:top_k]
        RERANK_HEALTH.record_ok()
        return [candidates[i] for i in order if 0 <= i < len(candidates)][:top_k]


def maybe_wrap_reranker(retriever: Any, spec: str | None) -> Any:
    """Wrap ``retriever`` per a spec: ``jina`` or ``jina:<url>``.

    Empty/None spec returns the retriever unchanged (default off). The paper ran
    ``jina``; the Voyage API reranker was removed.
    """
    if not spec:
        return retriever
    provider, _, rest = spec.partition(":")
    if provider != "jina":
        raise ValueError(
            f"unknown reranker provider {provider!r}; expected 'jina' "
            "(the Voyage reranker was removed, see configs/removed_features.json)"
        )
    return RerankedRetriever(retriever, JinaReranker(url=rest or None))
