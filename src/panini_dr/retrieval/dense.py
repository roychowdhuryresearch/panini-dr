"""Dense retrieval + hybrid (BM25 + dense RRF) for the FRAMES corpus.

The adapters expect a retriever with::

    .search(query: str, *, top_k: int) -> list[RetrievedChunk]

where ``RetrievedChunk`` has ``.chunk`` (with ``chunk_id``, ``title``,
``text``) and ``.score``. ``BM25Retriever`` already matches this shape;
this module adds two drop-in alternatives:

- :class:`DenseRetriever` — Qwen3-Embedding-8B (default) encoded chunks,
  FAISS ``IndexFlatIP`` over L2-normalized vectors, disk-cached so the
  ~3-minute index build only happens once per corpus.
- :class:`HybridRetriever` — reciprocal-rank-fusion of BM25 + dense.

Both preserve ``RetrievedChunk`` so no adapter code changes beyond
choosing which retriever to instantiate.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from panini_dr.retrieval.bm25 import BM25Retriever, RetrievedChunk
from panini_dr.retrieval.corpus import ArticleCorpus

_log = logging.getLogger(__name__)


# Default embedding model — the top-tier Qwen3 family member the user used
# before. ~16GB FP16 on GPU. Swap via DenseRetriever(model_name=...).
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


class DenseCacheMissingError(RuntimeError):
    """The dense index for this corpus is not on disk (or does not match it).

    Building it means encoding every chunk with Qwen3-Embedding-8B on a GPU,
    which can take hours, so a run never does it implicitly: build it once with
    ``panini-dr prepare`` (or a benchmark's ``datasets/*/prepare.py``).
    """


def _default_cache_root() -> Path:
    # Resolved at call time from $PANINI_DATA_DIR (see panini_dr.paths).
    from panini_dr.paths import dense_cache_root

    return dense_cache_root()


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------


class DenseRetriever:
    """Encode corpus chunks with Qwen3-Embedding, search via FAISS.

    The encoder is a :class:`sentence_transformers.SentenceTransformer`.
    Chunks are encoded once at init and cached to disk; subsequent
    constructor calls with the same ``(corpus fingerprint, model_name)``
    load the cached index in <1s.
    """

    def __init__(
        self,
        corpus: ArticleCorpus,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
        batch_size: int = 16,
        query_prompt: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        allow_build: bool = False,
    ) -> None:
        self.corpus = corpus
        self.model_name = model_name
        self.batch_size = batch_size
        # Qwen3 embedding models use an instruction-style query prompt.
        self.query_prompt = (
            query_prompt
            if query_prompt is not None
            else (
                "Instruct: Given a web search query, retrieve relevant "
                "passages that answer the query\nQuery: "
            )
        )

        cache_dir = cache_dir or _default_cache_root() / self._fingerprint()
        if allow_build:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir = cache_dir

        self._device = device or _pick_device()
        # Opt-in: run the exhaustive search on a GPU instead of FAISS-on-CPU.
        # Off by default so the shipped path stays byte-identical; set e.g.
        # GSW_DENSE_GPU=cuda:1 to move it. Deliberately a separate device from
        # the query encoder's -- the encoder already occupies cuda:0.
        self._gpu_device = os.environ.get("GSW_DENSE_GPU", "").strip() or None
        self._gpu_emb = None
        self._gpu_lock = threading.Lock()
        # Auto-pick torch_dtype: FP16 on GPU (2× faster + half memory),
        # FP32 on CPU (half-precision on CPU is usually slower).
        if torch_dtype is None:
            torch_dtype = "float16" if self._device.startswith("cuda") else "float32"
        self._torch_dtype = torch_dtype
        _log.info(
            f"DenseRetriever: model={model_name} device={self._device} "
            f"dtype={torch_dtype} batch_size={batch_size}"
        )

        emb_path = cache_dir / "chunk_embeddings.npy"
        ids_path = cache_dir / "chunk_ids.pkl"
        index_path = cache_dir / "faiss.index"

        # Lazy-import so this module stays importable even if torch is missing.
        import numpy as np

        self._embeddings: np.ndarray
        self._chunk_ids: list[str]

        cached = emb_path.exists() and ids_path.exists() and index_path.exists()
        if cached:
            with ids_path.open("rb") as f:
                _ids = pickle.load(f)
            # A cache whose chunk count disagrees with the corpus is a DIFFERENT
            # chunking, and its ids point at text that no longer exists. Nothing
            # checked this before: retrieval would have returned dead ids and
            # looked like a quality problem rather than a stale cache.
            if len(_ids) != len(corpus.chunks):
                if not allow_build:
                    raise DenseCacheMissingError(
                        f"dense cache {cache_dir} has {len(_ids)} chunk ids but the "
                        f"corpus has {len(corpus.chunks)} chunks (a different chunking "
                        "or corpus). Rebuild it with `panini-dr prepare`."
                    )
                _log.warning(
                    "DenseRetriever: cached index has %d chunk ids but the corpus "
                    "has %d — stale chunking, rebuilding.",
                    len(_ids), len(corpus.chunks),
                )
                cached = False
        if cached:
            _log.info(f"DenseRetriever: loading cached index from {cache_dir}")
            self._embeddings = np.load(emb_path)
            self._chunk_ids = _ids
            if self._gpu_device:
                # FAISS ``IndexFlatIP`` is exhaustive: every query streams the
                # whole matrix (10.1 GB at 616k x 4096 fp32) through the CPU,
                # and it does so on every OpenMP thread it can grab. Under 32+
                # concurrent questions that oversubscribes the box — measured at
                # 79 of 128 cores busy and ~166 core-seconds per search. The
                # same exhaustive inner product is a single matrix-vector
                # product on an idle GPU. Scores are identical fp32 dot products
                # over the same L2-normalized rows; only the hardware moves.
                self._index = None
            else:
                import faiss

                self._index = faiss.read_index(str(index_path))
                # Cap FAISS's OpenMP pool. IndexFlatIP is memory-bandwidth
                # bound, not compute bound -- a 1-thread search costs the same
                # as a 128-thread one serially (549 vs 617 ms at 616k x 4096) --
                # so the default pool only oversubscribes the box once several
                # questions search at once: 32 callers x 128 threads on 128
                # cores measured 2.7 q/s against 18.4 q/s capped, a 6.8x loss.
                # Results are thread-count invariant (top-50 order identical on
                # 200/200 queries, 1 thread vs 128), so this is a scheduling
                # fix and changes nothing that is retrieved. Opt-in, so the
                # shipped path stays exactly as measured.
                _threads = os.environ.get("GSW_FAISS_THREADS", "").strip()
                if _threads:
                    faiss.omp_set_num_threads(int(_threads))
                    _log.info("DenseRetriever: faiss omp threads capped at %s", _threads)
        else:
            if not allow_build:
                raise DenseCacheMissingError(
                    f"no dense index at {cache_dir} (expected chunk_embeddings.npy, "
                    "chunk_ids.pkl and faiss.index) for this corpus "
                    f"({len(corpus.chunks)} chunks, fingerprint {self._fingerprint()}). "
                    "A run never builds it implicitly: build it with `panini-dr prepare` "
                    "(GPU), pass --dense-cache <dir>, or use the downloaded cache."
                )
            _log.info(
                f"DenseRetriever: building index for {len(corpus.chunks)} chunks "
                f"(this takes a few minutes for 8B on first run)"
            )
            t0 = time.time()
            self._build_index(emb_path, ids_path, index_path)
            _log.info(f"DenseRetriever: index built in {time.time() - t0:.1f}s")

        # The encoder is only needed for queries after the build.
        self._model = None  # lazy — loaded on first search
        # Lock around the lazy load so concurrent search() calls (e.g.,
        # ThreadPoolExecutor over questions) don't race to materialize
        # the SentenceTransformer — that race triggers a meta-tensor
        # error on torch 2.x.
        self._model_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Index build
    # ------------------------------------------------------------------

    def _build_index(self, emb_path: Path, ids_path: Path, index_path: Path) -> None:
        import faiss
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        model_kwargs = {"torch_dtype": dtype_map.get(self._torch_dtype, torch.float32)}
        self._model = SentenceTransformer(
            self.model_name, device=self._device, model_kwargs=model_kwargs
        )
        chunks = self.corpus.chunks
        texts = [c.text for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]

        # Encode in batches to stay within GPU memory.
        embs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)

        np.save(emb_path, embs)
        with ids_path.open("wb") as f:
            pickle.dump(chunk_ids, f)
        faiss.write_index(index, str(index_path))

        self._embeddings = embs
        self._chunk_ids = chunk_ids
        self._index = index

    def _fingerprint(self) -> str:
        """A stable key for (corpus, model) cache bucketing.

        Includes a hash of article TEXT lengths so the cache
        auto-invalidates when the corpus content changes (e.g.,
        Phase-3.3 table-aware re-parse). Lengths are cheap and
        sufficient — the chance of a coincidental length collision
        across all 366 articles is effectively zero.
        """
        sorted_titles = sorted(self.corpus._by_title.keys())
        titles_hash = hashlib.sha256(
            "\n".join(sorted_titles).encode()
        ).hexdigest()[:12]
        # Include text-length signature so a re-parse with same titles
        # but different body text triggers a fresh embedding build.
        text_sig = "\n".join(
            f"{t}:{len(self.corpus._by_title[t])}" for t in sorted_titles
        )
        text_hash = hashlib.sha256(text_sig.encode()).hexdigest()[:12]
        model_slug = self.model_name.replace("/", "__")
        key = f"{model_slug}__{titles_hash}__{text_hash}"
        # The key above covers article TEXT but not how it was CHUNKED, and the
        # loader below does no length check — so a re-chunked corpus would load
        # the stale index and return chunk_ids that no longer exist. Legacy keys
        # must stay byte-identical (changing them re-embeds every cache we have),
        # so a non-default chunking gets its own suffix instead.
        if getattr(self.corpus, "list_aware", False):
            key += "__listaware"
        return key

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        # Double-checked locking — only one thread loads the model.
        with self._model_lock:
            if self._model is not None:
                return self._model
            import torch
            from sentence_transformers import SentenceTransformer

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            model_kwargs = {
                "torch_dtype": dtype_map.get(self._torch_dtype, torch.float32)
            }
            self._model = SentenceTransformer(
                self.model_name, device=self._device, model_kwargs=model_kwargs
            )
        return self._model

    def _gpu_search(self, q_emb, top_k: int):
        """Exhaustive inner product on the GPU: same arithmetic, different chip.

        Returns ``(scores, idxs)`` shaped exactly like ``faiss.Index.search`` so
        the caller cannot tell the two apart. The embedding matrix is uploaded
        once and reused; each query is then one matrix-vector product over
        L2-normalized rows, which is what ``IndexFlatIP`` computes.
        """
        import numpy as np
        import torch

        if self._gpu_emb is None:
            with self._gpu_lock:
                if self._gpu_emb is None:
                    t0 = time.time()
                    self._gpu_emb = torch.from_numpy(self._embeddings).to(
                        self._gpu_device, non_blocking=False
                    )
                    _log.info(
                        "DenseRetriever: embeddings on %s (%.2f GB) in %.1fs",
                        self._gpu_device,
                        self._gpu_emb.numel() * self._gpu_emb.element_size() / 1e9,
                        time.time() - t0,
                    )
        q = torch.from_numpy(np.ascontiguousarray(q_emb[0])).to(self._gpu_emb.device)
        scores_t = self._gpu_emb.mv(q)
        k = min(top_k, scores_t.numel())
        top = torch.topk(scores_t, k)
        return (
            top.values.unsqueeze(0).cpu().numpy(),
            top.indices.unsqueeze(0).cpu().numpy(),
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        import numpy as np

        model = self._ensure_model()
        q_emb = model.encode(
            [self.query_prompt + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        if self._gpu_device:
            scores, idxs = self._gpu_search(q_emb, top_k)
        else:
            scores, idxs = self._index.search(q_emb, top_k)
        out: list[RetrievedChunk] = []
        id_to_chunk = self.corpus._by_id  # ArticleCorpus maintains this
        for s, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i < 0:
                continue
            chunk_id = self._chunk_ids[i]
            chunk = id_to_chunk.get(chunk_id)
            if chunk is None:
                continue
            out.append(RetrievedChunk(chunk=chunk, score=float(s)))
        return out


# ---------------------------------------------------------------------------
# HybridRetriever — reciprocal rank fusion
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Reciprocal Rank Fusion (RRF) over BM25 + dense.

    RRF score for a chunk c is::

        score(c) = sum_r (1 / (k + rank_r(c)))

    where r ranges over the member retrievers and ``rank_r(c)`` is c's
    1-based rank in retriever r (or infinity if not in r's top_k).
    Industry-standard fusion; no hyperparameter tuning beyond ``k``.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        *,
        k: int = 60,
        per_retriever_top_k: int = 50,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.k = k
        self.per_top_k = per_retriever_top_k

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        bm25_hits = self.bm25.search(query, top_k=self.per_top_k)
        dense_hits = self.dense.search(query, top_k=self.per_top_k)

        scores: dict[str, float] = {}
        chunk_by_id: dict[str, Any] = {}

        for rank, h in enumerate(bm25_hits, start=1):
            cid = h.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.k + rank)
            chunk_by_id[cid] = h.chunk
        for rank, h in enumerate(dense_hits, start=1):
            cid = h.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.k + rank)
            chunk_by_id.setdefault(cid, h.chunk)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        out: list[RetrievedChunk] = []
        for cid, s in ranked[:top_k]:
            out.append(RetrievedChunk(chunk=chunk_by_id[cid], score=s))
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_retriever(
    retriever_type: str,
    corpus: ArticleCorpus,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    cache_dir: Optional[Path] = None,
    device: Optional[str] = None,
    lucene_index: Optional[Path | str] = None,
    allow_dense_build: bool = False,
) -> Any:
    """Construct the requested retriever. Shared entry point for adapters.

    ``lucene_index`` swaps the lexical leg for a prebuilt Anserini/Lucene index —
    BrowseComp-Plus publishes one, and their BM25 numbers are only comparable to
    ours if we query theirs rather than our own Okapi implementation. It is the
    lexical counterpart of ``cache_dir``, which already lets the dense leg reuse
    their published embeddings. Unset (the FRAMES/MoNaCo path) nothing changes.
    """
    rt = (retriever_type or "bm25").lower()

    def _lexical() -> Any:
        if lucene_index:
            from panini_dr.retrieval.lucene_bm25 import LuceneBM25Retriever

            return LuceneBM25Retriever(corpus, lucene_index)
        return BM25Retriever(corpus)

    if rt == "bm25":
        return _lexical()
    if rt == "dense":
        return DenseRetriever(
            corpus, model_name=model_name, cache_dir=cache_dir, device=device,
            allow_build=allow_dense_build,
        )
    if rt == "hybrid":
        bm25 = _lexical()
        dense = DenseRetriever(
            corpus, model_name=model_name, cache_dir=cache_dir, device=device,
            allow_build=allow_dense_build,
        )
        return HybridRetriever(bm25, dense)
    raise ValueError(
        f"unknown retriever_type {retriever_type!r}; expected bm25|dense|hybrid"
    )


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            # Pick the GPU with the most free memory.
            free_mems = []
            for i in range(torch.cuda.device_count()):
                try:
                    free, _total = torch.cuda.mem_get_info(i)
                except Exception:  # noqa: BLE001, PERF203
                    # Probing a device needs a CUDA context on it, so a GPU that
                    # another tenant has filled raises here. Skip it and keep
                    # looking. This used to abort the whole probe via the outer
                    # handler and silently put an 8B encoder on the CPU -- on a
                    # shared box that cost ~27 cores per run and starved the
                    # LLM, with nothing in the logs to say so.
                    _log.warning(
                        "DenseRetriever: cuda:%d is not probeable (full or "
                        "unavailable); skipping it when picking a device.", i
                    )
                    continue
                free_mems.append((free, i))
            if free_mems:
                free_mems.sort(reverse=True)
                return f"cuda:{free_mems[0][1]}"
        _log.warning(
            "DenseRetriever: no usable CUDA device — the embedding model will "
            "run on CPU, which is far slower. This is rarely what you want."
        )
        return "cpu"
    except Exception:  # noqa: BLE001
        _log.warning("DenseRetriever: device probe failed; falling back to CPU.")
        return "cpu"
