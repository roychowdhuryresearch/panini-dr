"""Tevatron's published BrowseComp-Plus BM25 index, behind our retriever API.

WHY THIS EXISTS. ``--retriever bm25`` builds our own Okapi implementation
(``retrieval/bm25.py``) over the loaded corpus. That is a *different* BM25 than
the one BrowseComp-Plus publishes numbers against: theirs is Anserini/Lucene with
an EnglishAnalyzer (Porter stemming, stopword list) and Anserini's default
parameters, ours is a whitespace/alphanumeric tokenizer with k1=1.5, b=0.75 and
no stemming. Both are "BM25"; only one is comparable to their table. This module
opens THEIR index so the lexical leg is theirs, exactly as ``eval/bcp_dense.py``
already reuses their Qwen3-Embedding-8B vectors for the dense leg.

    ours (retrieval/bm25.py)         theirs (indexes/bm25)
    alphanumeric tokens, no stem     EnglishAnalyzer (Porter + stopwords)
    k1=1.5, b=0.75                   Anserini defaults (k1=0.9, b=0.4)
    built at load from the corpus    prebuilt Lucene index, 2.1 GB, 100,195 docs

PARAMETERS ARE DELIBERATELY UNSET. Their ``searcher/searchers/bm25_searcher.py``
constructs ``LuceneSearcher(index_path)`` and never calls ``set_bm25``, so the
faithful configuration is the default one. Setting k1/b here would silently make
our numbers incomparable to the paper's, which is the whole point of the module.

THE ONE HARD PART is the identifier join. Their hits carry a BCP docid; every
consumer downstream of us (the reader, the dossier, gold-coverage matching) is
keyed on our chunk_id. ``eval/browsecomp_keys.cache_key`` builds our key as
``"<title> [<docid>]"`` — or the bare docid for the 1,176 documents with no
front-matter title — and the BCP chunking gives exactly one chunk per document,
so the chunk_id is that key plus ``__0``. That makes the join invertible from the
corpus alone: no sidecar mapping file to drift out of date, and no second copy of
the key logic to disagree with the first.

FAIL CLOSED, BOTH ENDS. A doc-count mismatch at construction catches an index
built over a different corpus; an unmappable docid at query time raises instead
of being dropped from the result list. A dropped hit is invisible — it just looks
like a weak retriever — and that is the failure mode that cost half a day on
2026-08-12 when a wrong-corpus run mimicked an architecture collapse.

JAVA. pyserini runs Lucene in-process through pyjnius, so a JDK 21 must be
findable before the first query. See ``_ensure_java_home`` — it needs a full JDK,
not a jlink-trimmed one: pyserini starts the JVM with
``--add-modules=jdk.incubator.vector``, and a runtime image without that
incubator module dies at boot-layer initialisation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from panini_dr.retrieval.bm25 import RetrievedChunk
from panini_dr.retrieval.corpus import ArticleCorpus, Chunk

# cache_key appends the docid LAST, so the final bracketed group is the docid
# even for a title that itself ends in brackets ("Report [1999] [82002]").
# Restricted to digits because BCP docids are numeric (measured: 100,195 ids,
# 1-6 chars, all digits) — an alphabetic bracket is part of a title, not an id.
_DOCID_SUFFIX_RE = re.compile(r"^(?P<title>.*) \[(?P<docid>\d+)\]$", re.DOTALL)

_JDK_HINT = (
    "pyserini needs a full JDK 21 on JAVA_HOME. Install one without root via:\n"
    "    .venv/bin/python -c \"import jdk; print(jdk.install('21', jre=False))\"\n"
    "then export JAVA_HOME to the printed path. A jlink-trimmed runtime (jdk4py) "
    "will NOT work: pyserini boots the JVM with --add-modules=jdk.incubator.vector."
)


def docid_from_cache_key(key: str) -> str:
    """The BrowseComp-Plus docid behind one of our cache keys."""
    m = _DOCID_SUFFIX_RE.match(key)
    return m.group("docid") if m else key


def build_docid_index(corpus: ArticleCorpus) -> dict[str, Chunk]:
    """docid -> the single chunk of that document.

    Raises on a duplicate docid: two chunks claiming one id means the corpus was
    not loaded one-chunk-per-document (or the keys collided), and every hit for
    that id would then resolve to an arbitrary one of them.
    """
    index: dict[str, Chunk] = {}
    for chunk in corpus.chunks:
        docid = docid_from_cache_key(chunk.title)
        previous = index.get(docid)
        if previous is not None:
            raise ValueError(
                f"duplicate docid {docid!r} in the corpus: "
                f"{previous.chunk_id!r} and {chunk.chunk_id!r}; "
                "load BrowseComp-Plus with chunk_chars > the longest document "
                "(BCP_SINGLE_DOCUMENT_CHARS) so each document is exactly one chunk"
            )
        index[docid] = chunk
    return index


def _ensure_java_home() -> None:
    """Point JAVA_HOME at a full JDK, or explain how to get one."""
    current = os.environ.get("JAVA_HOME")
    if current and (Path(current) / "bin" / "java").is_file():
        return
    # install-jdk's default location, newest first. This is where the JDK lands
    # for `jdk.install("21")`, which is the no-root path documented in _JDK_HINT.
    candidates = sorted(Path.home().glob(".jdk/jdk-*"), reverse=True)
    for candidate in candidates:
        if (candidate / "bin" / "java").is_file():
            os.environ["JAVA_HOME"] = str(candidate)
            return
    raise RuntimeError(_JDK_HINT)


def open_lucene_searcher(index_dir: Path | str) -> Any:
    """A pyserini ``LuceneSearcher`` over a local index directory."""
    path = Path(index_dir)
    if not (path / "segments_3").exists() and not any(path.glob("segments*")):
        raise FileNotFoundError(f"{path} does not look like a Lucene index (no segments file)")
    _ensure_java_home()
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "pyserini is not installed. BrowseComp-Plus pins pyserini==1.2.0 "
            "(their uv.lock); newer releases require transformers>=5 and would "
            "upgrade the pin every other benchmark number depends on.\n" + _JDK_HINT
        ) from exc
    return LuceneSearcher(str(path))


class LuceneBM25Retriever:
    """Their Lucene BM25, returning our chunks.

    Same shape as ``BM25Retriever`` / ``DenseRetriever`` — ``search(query, *,
    top_k, title_bias) -> list[RetrievedChunk]`` — so adapters and
    ``HybridRetriever`` take it unchanged.
    """

    def __init__(
        self,
        corpus: ArticleCorpus,
        index_dir: Path | str | None = None,
        *,
        searcher: Any | None = None,
    ) -> None:
        if searcher is None and index_dir is None:
            raise ValueError("LuceneBM25Retriever needs either index_dir or searcher")
        self.corpus = corpus
        self._by_docid = build_docid_index(corpus)
        self.searcher = searcher if searcher is not None else open_lucene_searcher(index_dir)

        num_docs = int(getattr(self.searcher, "num_docs", 0) or 0)
        if num_docs != len(corpus.chunks):
            raise ValueError(
                f"index/corpus mismatch: the Lucene index holds {num_docs:,} documents "
                f"but the corpus has {len(corpus.chunks):,} chunks; refusing to search "
                "an index built over different text"
            )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        title_bias: dict[str, float] | None = None,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []
        hits = self.searcher.search(query, top_k)
        out: list[RetrievedChunk] = []
        for hit in hits:
            docid = str(hit.docid)
            chunk = self._by_docid.get(docid)
            if chunk is None:
                raise ValueError(
                    f"Lucene returned docid {docid!r}, which is absent from the corpus; "
                    "the index and the article cache describe different corpora"
                )
            score = float(hit.score)
            if title_bias:
                score *= title_bias.get(chunk.title, 1.0)
            out.append(RetrievedChunk(chunk=chunk, score=score))
        if title_bias:
            # Lucene's order is authoritative until a bias changes the scores;
            # then, and only then, we re-sort — matching BM25Retriever, which
            # applies the bias before its own sort.
            out.sort(key=lambda rc: rc.score, reverse=True)
        return out
