"""Load Tevatron's BrowseComp-Plus dense index into our DenseRetriever cache.

WHY WE REUSE THEIR VECTORS. Verified equivalent, not assumed:

    ours (retrieval/dense.py)          theirs (indexes/qwen3-embedding-8b)
    Qwen/Qwen3-Embedding-8B            Qwen3-Embedding-8B
    4096-dim                           4096-dim            (measured)
    normalize_embeddings=True          norm = 1.0000       (measured)
    FAISS IndexFlatIP                  IndexFlatIP -- cosine over unit vectors

Same tower, same geometry. Reusing their matrix removes the only expensive part
of the integration: encoding 100,195 documents with an 8B model.

WHY THIS NEEDS NO CHANGE TO DenseRetriever. Its cached path already loads
``chunk_embeddings.npy`` + ``chunk_ids.pkl`` + ``faiss.index`` from an explicit
``cache_dir``, pairing embeddings to ids BY POSITION. So the conversion is purely
"write those three files", and the live FRAMES/MoNaCo retrieval path is untouched
— which matters, because that path is carrying every number we have.

THE ONE CONSTRAINT. Their vectors are one-per-DOCUMENT; our chunk ids are
``f"{key}__{i}"``. So the corpus must be loaded with ``chunk_chars`` larger than
the longest document (9,962,379 chars) to give exactly one chunk per document,
``__0``. DenseRetriever's own guard (`len(ids) != len(corpus.chunks)`) then
passes, and a mismatched chunking is caught rather than silently mis-retrieving.

WHAT THEIR TRUNCATION COSTS. They encoded with ``--passage_max_len 4096``, so
41.5% of documents are truncated and 72.6% of corpus text never influences
retrieval. It is bounded where it matters: gold documents have median 5,140 chars
against the corpus median of 10,241, and only 22.9% exceed the window. And it is
a FIRST-STAGE constraint only — we hold full text in the cache, so reranking and
reading are untruncated.
"""
from __future__ import annotations

import pickle
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

BCP_SINGLE_DOCUMENT_CHARS = 10_000_000


def validate_prebuilt_dense_cache(corpus: Any, cache_dir: Path | str) -> None:
    """Fail closed when a prebuilt row/id mapping does not match the corpus."""
    root = Path(cache_dir)
    required = [root / name for name in ("chunk_embeddings.npy", "chunk_ids.pkl", "faiss.index")]
    missing_files = [path for path in required if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "prebuilt dense cache is incomplete: "
            + ", ".join(str(path) for path in missing_files)
        )
    ids_path = root / "chunk_ids.pkl"
    with ids_path.open("rb") as handle:
        chunk_ids = pickle.load(handle)  # noqa: S301 - trusted local cache artifact
    if len(chunk_ids) != len(corpus.chunks):
        raise ValueError(
            "prebuilt dense cache/corpus mismatch: "
            f"{len(chunk_ids):,} cached ids versus {len(corpus.chunks):,} corpus chunks; "
            "refusing an implicit embedding rebuild"
        )
    missing = next((chunk_id for chunk_id in chunk_ids if corpus.get_chunk(chunk_id) is None), None)
    if missing is not None:
        raise ValueError(
            f"prebuilt dense cache id {missing!r} is absent from the corpus; "
            "refusing a silently misaligned index"
        )


def align_chunk_ids(
    lookup: Iterable[Any], docid_to_key: Mapping[str, str]
) -> list[str]:
    """Our chunk id for each row of their matrix, in their row order.

    ``reps[i]`` embeds document ``lookup[i]``, and DenseRetriever pairs
    embeddings to ids by position — so the matrix is never reordered, only
    labelled. A docid with no document in our cache raises: dropping it would
    shift every subsequent row and mis-map the entire index, a failure that is
    invisible at query time.
    """
    ids: list[str] = []
    for docid in lookup:
        key = docid_to_key[str(docid)]
        ids.append(f"{key}__0")
    return ids


def load_shards(index_dir: Path | str) -> tuple[Any, list[str]]:
    """Concatenate Tevatron's ``corpus.shard*_of_*.pkl`` into ``(reps, lookup)``."""
    import numpy as np

    files = sorted(Path(index_dir).glob("corpus.shard*.pkl"))
    if not files:
        raise FileNotFoundError(f"no corpus.shard*.pkl under {index_dir}")
    all_reps, all_lookup = [], []
    for f in files:
        with f.open("rb") as fh:
            reps, lookup = pickle.load(fh)
        all_reps.append(np.asarray(reps, dtype="float32"))
        all_lookup.extend(lookup)
    return np.concatenate(all_reps, axis=0), all_lookup
