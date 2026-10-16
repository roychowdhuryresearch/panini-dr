"""Retrieval stack used by the harness: corpus, BM25, dense (Qwen3-Embedding-8B),
hybrid RRF fusion, reranking and the optional Lucene lexical leg (BrowseComp-Plus).

Only the lightweight pieces are re-exported here; the dense, rerank and Lucene
modules are imported where they are used so that their optional dependencies
(torch, sentence-transformers, pyserini) stay optional.
"""
from panini_dr.retrieval.bm25 import BM25Retriever
from panini_dr.retrieval.corpus import (
    ArticleCorpus,
    Chunk,
    CorpusNotConfiguredError,
    load_frames_corpus,
    require_corpus,
)
from panini_dr.retrieval.manifest import (
    CorpusEligibilityError,
    CorpusManifest,
    CorpusScope,
    assert_official_evaluation_eligible,
)

__all__ = [
    "ArticleCorpus",
    "BM25Retriever",
    "Chunk",
    "CorpusEligibilityError",
    "CorpusManifest",
    "CorpusNotConfiguredError",
    "CorpusScope",
    "assert_official_evaluation_eligible",
    "load_frames_corpus",
    "require_corpus",
]
