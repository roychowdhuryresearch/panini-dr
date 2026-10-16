"""Record a toy cassette for the replay end-to-end test (run as a subprocess, never collected).

    python tests/replay/_record_toy.py WORK_DIR

WORK_DIR must hold ``data/frames/{frames_split.json,corpus.json}`` and ``subset.json`` (the test
writes them). This script

1. builds the toy corpus's dense index where ``panini-dr run`` looks for it
   (``$PANINI_DATA_DIR/retrieval_cache/<fingerprint>``), with the stub encoder,
2. puts the CPU fakes behind the cassette hooks: the scripted LLM of tests/e2e (wrapped so
   the hook sees ``LLMClient.chat``'s real signature, exactly as in a replay) and a jina-style
   reranker that answers ``httpx.post`` with a real ``httpx.Response``,
3. runs ``panini-dr run`` (paper configuration) under ``python -m panini_dr.replay --mode record``.

The caller starts it with PYTHONHASHSEED=0 / HF offline already set, so the launcher does not
re-exec (which would drop the fakes), and with tests/replay/_stubs first on PYTHONPATH.
"""

from __future__ import annotations

import functools
import os
import re
import sys
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


def fake_rerank_post(url, *args, **kwargs):
    """jina listwise API: rank documents by token overlap with the query."""
    import httpx

    body = kwargs.get("json") or {}
    docs = body.get("documents") or []
    q = _tokens(body.get("query") or "")
    order = sorted(range(len(docs)), key=lambda i: (-len(q & _tokens(docs[i])), i))
    results = [{"index": i, "relevance_score": round(1.0 - 0.01 * r, 4)} for r, i in enumerate(order)]
    return httpx.Response(200, json={"results": results}, request=httpx.Request("POST", str(url)))


def main(work: Path) -> int:
    assert os.environ.get("PYTHONHASHSEED") == "0", "start this helper with PYTHONHASHSEED=0"
    import httpx

    from panini_dr.models.llm_client import LLMClient
    from panini_dr.replay import launcher
    from panini_dr.retrieval.dense import DenseRetriever
    from panini_dr.runner import DatasetSpec, load_corpus
    from tests.e2e.fakes import FakeLLMService

    corpus_path = work / "data" / "frames" / "corpus.json"
    corpus = load_corpus(DatasetSpec(benchmark="frames", subset_id="toy", questions=[], corpus_path=corpus_path))
    DenseRetriever(corpus, allow_build=True)  # stub encoder + stub faiss

    service = FakeLLMService()
    real_chat = LLMClient.chat

    def chat(client_self, messages=None, *args, **kwargs):
        if messages is None:
            messages = kwargs.pop("messages", None)
        return service.chat(list(messages or []), **kwargs)

    functools.update_wrapper(chat, real_chat)  # the LLM hook keys on the real signature
    LLMClient.chat = chat
    httpx.post = fake_rerank_post

    return launcher.main([
        "--mode", "record", "--cassette", str(work / "cassette"), "--no-probe", "--",
        "run", "--subset", str(work / "subset.json"), "--articles-cache", str(corpus_path),
        "--base-url", "http://llm.invalid/v1", "--api-key", "dummy", "--concurrency", "2",
        "--skip-rerank-preflight", "--output-dir", str(work / "record_run"),
    ])


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
