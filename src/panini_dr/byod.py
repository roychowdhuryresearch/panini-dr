"""Bring your own dataset: ``panini-dr prepare``.

Inputs
------
``questions.jsonl``
    ``{"id", "question", "answer" (string or list), "answer_type"?}`` per line
    (see :mod:`panini_dr.eval.byod_dataset`).
``docs.jsonl``
    ``{"id"?, "title"?, "text"}`` per line; at least one of ``id``/``title``.
``dataset.yaml`` (optional)
    ``name``, ``questions``, ``docs`` (paths relative to the yaml), and
    ``chunking: paragraph | whole_doc``. Command-line options win.

Output directory
----------------
``corpus.json``      the ``{title: text}`` corpus the harness reads (``--articles-cache``)
``questions.jsonl``  validated copy of the questions
``subset.json``      all question ids, ``benchmark: byod``
``dense_cache/``     Qwen3-Embedding-8B index (``chunk_embeddings.npy``,
                     ``chunk_ids.pkl``, ``faiss.index``) unless ``--no-dense``,
                     and ``built_from.json``: the sha256 of the corpus and of the
                     chunk-id list the index was encoded from
``dataset.json``     manifest: counts, sha256 of every file, chunking, dense info

Re-running ``prepare`` into the same directory reuses the dense index only when
``built_from.json`` matches the new corpus and chunking; otherwise the index is
rebuilt. ``panini-dr run --dataset`` refuses an index whose ``built_from.json``
does not match the corpus it is about to search.

Document keys (the article title the agent sees and cites) are the document
title when it is present and unique, the id when there is no title, and
``"<title> [<id>]"`` when a title is shared by several documents (the rule the
BrowseComp-Plus corpus uses).

Lexical retrieval (our Okapi BM25) is built in memory at the start of every run
from ``corpus.json``; ``prepare`` builds it once to report vocabulary
statistics and catch documents that yield no ASCII tokens (the tokenizer is
``[A-Za-z0-9]+``). The dense index needs the ``dense`` extra and a GPU with
about 16 GB free; ``--no-dense`` skips it, and such a dataset can only be run
BM25-only (``--set retriever_type=bm25 --allow-config-drift``), which is not
the paper configuration.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from panini_dr.eval.byod_dataset import ByodSchemaError, read_questions_jsonl

CHUNKING_MODES = ("paragraph", "whole_doc")
MANIFEST_NAME = "dataset.json"
FORMAT_VERSION = 1
#: Written next to the dense index: what it was encoded from.
DENSE_STAMP_NAME = "built_from.json"
DENSE_FILES = ("chunk_embeddings.npy", "chunk_ids.pkl", "faiss.index")


class PreparedDatasetError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def read_docs_jsonl(path: str | Path) -> list[dict[str, str]]:
    """Read docs.jsonl into ``[{"id", "title", "text"}]`` (validated)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"docs file not found: {path}")
    docs: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path.name}:{lineno}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ByodSchemaError(f"{where}: not valid JSON ({exc})") from exc
            if not isinstance(raw, dict):
                raise ByodSchemaError(f"{where}: each line must be a JSON object")
            text = raw.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ByodSchemaError(f"{where}: missing or empty 'text'")
            doc_id = str(raw.get("id", raw.get("doc_id", "")) or "").strip()
            title = re.sub(r"\s+", " ", str(raw.get("title") or "")).strip()
            if not doc_id and not title:
                raise ByodSchemaError(f"{where}: a document needs an 'id' or a 'title'")
            if doc_id:
                if doc_id in seen_ids:
                    raise ByodSchemaError(f"{where}: duplicate id {doc_id!r}")
                seen_ids.add(doc_id)
            docs.append({"id": doc_id, "title": title, "text": text})
    if not docs:
        raise ByodSchemaError(f"{path}: no documents")
    return docs


def build_corpus(docs: list[dict[str, str]]) -> dict[str, str]:
    """``{key: text}`` with unique keys (see module docstring for the rule)."""
    title_counts = Counter(d["title"] for d in docs if d["title"])
    corpus: dict[str, str] = {}
    for d in docs:
        if d["title"] and title_counts[d["title"]] == 1:
            key = d["title"]
        elif d["title"]:
            if not d["id"]:
                raise ByodSchemaError(
                    f"title {d['title']!r} is shared by several documents; give them ids"
                )
            key = f"{d['title']} [{d['id']}]"
        else:
            key = d["id"]
        if key in corpus:
            raise ByodSchemaError(f"two documents map to the same key {key!r}")
        corpus[key] = d["text"]
    return corpus


# ---------------------------------------------------------------------------
# dataset.yaml
# ---------------------------------------------------------------------------


def load_dataset_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ByodSchemaError(f"{path}: expected a mapping")
    base = path.parent
    out: dict[str, Any] = {}
    for key in ("name", "chunking"):
        if key in data:
            out[key] = data[key]
    for key in ("questions", "docs"):
        if key in data and data[key]:
            out[key] = (base / str(data[key])).resolve()
    unknown = set(data) - {"name", "chunking", "questions", "docs", "description"}
    if unknown:
        raise ByodSchemaError(f"{path}: unknown keys {sorted(unknown)}")
    return out


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chunk_ids_sha256(chunk_ids) -> str:
    """sha256 of a chunk-id list (one id per line, in order)."""
    h = hashlib.sha256()
    for cid in chunk_ids:
        h.update(str(cid).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _dense_stamp(*, corpus_sha256: str, chunk_ids: list[str], chunking: str, model: str) -> dict[str, Any]:
    return {
        "corpus_sha256": corpus_sha256,
        "chunk_ids_sha256": chunk_ids_sha256(chunk_ids),
        "n_chunks": len(chunk_ids),
        "chunking": chunking,
        "model": model,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _cached_chunk_ids_sha256(cache_dir: Path) -> str | None:
    import pickle

    try:
        with (cache_dir / "chunk_ids.pkl").open("rb") as f:
            return chunk_ids_sha256(pickle.load(f))
    except Exception:  # noqa: BLE001 - a missing or unreadable file is a mismatch
        return None


def _dense_stale_reason(cache_dir: Path, want: dict[str, Any]) -> str | None:
    """Why the index in ``cache_dir`` is not the one ``want`` describes (None: it is)."""
    if not all((cache_dir / n).is_file() for n in DENSE_FILES):
        return "the index files are incomplete"
    have = _read_json(cache_dir / DENSE_STAMP_NAME)
    if have is None:
        return f"no {DENSE_STAMP_NAME} records what it was built from"
    if have.get("corpus_sha256") != want["corpus_sha256"]:
        return "it was built from a different corpus.json"
    for key in ("chunking", "model", "chunk_ids_sha256"):
        if have.get(key) != want[key]:
            return f"it was built with a different {key.replace('_sha256', '')}"
    if _cached_chunk_ids_sha256(cache_dir) != want["chunk_ids_sha256"]:
        return "its chunk_ids.pkl does not hold the corpus's chunk ids"
    return None


def chunking_kwargs(chunking: str) -> dict[str, Any]:
    """``load_frames_corpus`` keyword arguments for a chunking mode."""
    if chunking == "paragraph":
        return {}  # the paper's FRAMES/MoNaCo chunker: 2000 chars, 200 overlap
    if chunking == "whole_doc":
        from panini_dr.eval.bcp_dense import BCP_SINGLE_DOCUMENT_CHARS

        return {"chunk_chars": BCP_SINGLE_DOCUMENT_CHARS, "chunk_overlap": 0}
    raise ValueError(f"chunking must be one of {CHUNKING_MODES}, got {chunking!r}")


def prepare_dataset(
    *,
    questions: str | Path,
    docs: str | Path,
    out: str | Path,
    name: str | None = None,
    chunking: str = "paragraph",
    dense: bool = True,
    device: str | None = None,
    batch_size: int = 16,
    log=print,
) -> dict[str, Any]:
    """Build a prepared dataset directory; returns the manifest."""
    if chunking not in CHUNKING_MODES:
        raise ValueError(f"chunking must be one of {CHUNKING_MODES}, got {chunking!r}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    name = name or out.name

    qs = read_questions_jsonl(questions)
    docs_list = read_docs_jsonl(docs)
    corpus = build_corpus(docs_list)

    corpus_path = out / "corpus.json"
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False))
    questions_path = out / "questions.jsonl"
    with questions_path.open("w", encoding="utf-8") as f:
        for q in qs:
            row = dict(q.raw)
            row["answer_type"] = q.answer_type
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    subset_path = out / "subset.json"
    subset_path.write_text(
        json.dumps(
            {
                "subset_id": name,
                "benchmark": "byod",
                "split": "all",
                "question_ids": [q.id for q in qs],
                "dataset_path": "questions.jsonl",
            },
            indent=2,
        )
    )
    log(f"wrote {len(qs)} questions, {len(corpus)} documents -> {out}")

    # Chunk with the exact loader the run uses, and build BM25 once.
    from panini_dr.retrieval.bm25 import BM25Retriever
    from panini_dr.retrieval.corpus import load_frames_corpus

    article_corpus = load_frames_corpus(cache_path=corpus_path, **chunking_kwargs(chunking),
                                        benchmark="byod", dataset_name=name)
    BM25Retriever(article_corpus)
    tokenless = [
        t for t, text in corpus.items() if not re.search(r"[A-Za-z0-9]", text)
    ]
    if tokenless:
        log(
            f"WARNING: {len(tokenless)} document(s) have no ASCII letters/digits; the BM25 "
            f"tokenizer finds no terms in them (first: {tokenless[:3]})"
        )
    log(f"chunking={chunking}: {len(article_corpus.chunks)} chunks; BM25 built in memory")

    dense_info: dict[str, Any] = {"built": False}
    if dense:
        from panini_dr.retrieval.dense import DEFAULT_EMBEDDING_MODEL, DenseRetriever

        cache_dir = out / "dense_cache"
        stamp = _dense_stamp(
            corpus_sha256=_sha256(corpus_path),
            chunk_ids=[c.chunk_id for c in article_corpus.chunks],
            chunking=chunking,
            model=DEFAULT_EMBEDDING_MODEL,
        )
        stale = _dense_stale_reason(cache_dir, stamp)
        if stale is None:
            log(f"reusing the dense index in {cache_dir} (built from this corpus)")
        else:
            if any((cache_dir / n).exists() for n in (*DENSE_FILES, DENSE_STAMP_NAME)):
                # The retriever itself reuses any index with the right number of
                # chunks, so a stale one is removed first.
                log(f"the dense index in {cache_dir} is stale ({stale}); rebuilding it")
                for n in (*DENSE_FILES, DENSE_STAMP_NAME):
                    (cache_dir / n).unlink(missing_ok=True)
            log(f"encoding {len(article_corpus.chunks)} chunks with {DEFAULT_EMBEDDING_MODEL} -> {cache_dir}")
        DenseRetriever(
            article_corpus,
            cache_dir=cache_dir,
            device=device,
            batch_size=batch_size,
            allow_build=True,
        )
        (cache_dir / DENSE_STAMP_NAME).write_text(json.dumps(stamp, indent=2) + "\n")
        dense_info = {
            "built": True,
            "model": DEFAULT_EMBEDDING_MODEL,
            "dir": "dense_cache",
            "n_chunks": len(article_corpus.chunks),
            "corpus_sha256": stamp["corpus_sha256"],
            "chunk_ids_sha256": stamp["chunk_ids_sha256"],
        }

    manifest = {
        "format_version": FORMAT_VERSION,
        "name": name,
        "benchmark": "byod",
        "chunking": chunking,
        "n_questions": len(qs),
        "answer_types": dict(Counter(q.answer_type for q in qs)),
        "n_documents": len(corpus),
        "n_chunks": len(article_corpus.chunks),
        "files": {
            "corpus": {"path": "corpus.json", "sha256": _sha256(corpus_path)},
            "questions": {"path": "questions.jsonl", "sha256": _sha256(questions_path)},
            "subset": {"path": "subset.json", "sha256": _sha256(subset_path)},
        },
        "dense": dense_info,
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ---------------------------------------------------------------------------
# Reading a prepared directory back (for `panini-dr run --dataset`)
# ---------------------------------------------------------------------------


@dataclass
class PreparedDataset:
    root: Path
    manifest: dict[str, Any]

    @property
    def corpus_path(self) -> Path:
        return self.root / self.manifest["files"]["corpus"]["path"]

    @property
    def questions_path(self) -> Path:
        return self.root / self.manifest["files"]["questions"]["path"]

    @property
    def subset_path(self) -> Path:
        return self.root / self.manifest["files"]["subset"]["path"]

    @property
    def chunking(self) -> str:
        return self.manifest.get("chunking", "paragraph")

    @property
    def dense_cache(self) -> Path | None:
        d = self.manifest.get("dense") or {}
        return self.root / d["dir"] if d.get("built") else None


def load_prepared(root: str | Path, *, verify: bool = True) -> PreparedDataset:
    root = Path(root)
    mpath = root / MANIFEST_NAME
    if not mpath.is_file():
        raise PreparedDatasetError(
            f"{root} is not a prepared dataset (no {MANIFEST_NAME}); run `panini-dr prepare`"
        )
    manifest = json.loads(mpath.read_text())
    ds = PreparedDataset(root=root, manifest=manifest)
    if verify:
        for key, entry in manifest.get("files", {}).items():
            p = root / entry["path"]
            if not p.is_file():
                raise PreparedDatasetError(f"{key} file missing: {p}")
            if _sha256(p) != entry["sha256"]:
                raise PreparedDatasetError(
                    f"{p} changed since `panini-dr prepare` (sha256 mismatch); re-run prepare"
                )
        if ds.dense_cache is not None:
            check_dense_cache(ds)
    return ds


def check_dense_cache(ds: PreparedDataset, chunk_ids: list[str] | None = None) -> None:
    """Refuse a dense index that was not encoded from this dataset's corpus.

    Compares ``dense_cache/built_from.json`` with the corpus sha256 in the
    manifest and with the index's own ``chunk_ids.pkl``; with ``chunk_ids``
    (the corpus as the run chunks it) also with those.
    """
    cache_dir = ds.dense_cache
    if cache_dir is None:
        return
    redo = f"re-run `panini-dr prepare` with --out {ds.root} (it rebuilds the dense index)"
    stamp = _read_json(cache_dir / DENSE_STAMP_NAME)
    if stamp is None:
        raise PreparedDatasetError(
            f"{cache_dir} has no {DENSE_STAMP_NAME}, so nothing shows which corpus it was "
            f"encoded from (prepared by an older panini-dr?); {redo}"
        )
    corpus_sha = ds.manifest["files"]["corpus"]["sha256"]
    if stamp.get("corpus_sha256") != corpus_sha:
        raise PreparedDatasetError(
            f"the dense index in {cache_dir} was encoded from another corpus "
            f"(sha256 {str(stamp.get('corpus_sha256'))[:12]}..., corpus.json is {corpus_sha[:12]}...); {redo}"
        )
    if stamp.get("chunking") != ds.chunking:
        raise PreparedDatasetError(
            f"the dense index in {cache_dir} was built with chunking={stamp.get('chunking')!r}, "
            f"the dataset uses {ds.chunking!r}; {redo}"
        )
    if _cached_chunk_ids_sha256(cache_dir) != stamp.get("chunk_ids_sha256"):
        raise PreparedDatasetError(
            f"{cache_dir / 'chunk_ids.pkl'} is not the chunk-id list recorded in "
            f"{DENSE_STAMP_NAME}; {redo}"
        )
    if chunk_ids is not None and chunk_ids_sha256(chunk_ids) != stamp.get("chunk_ids_sha256"):
        raise PreparedDatasetError(
            f"the corpus chunks into ids that differ from the ones the dense index in "
            f"{cache_dir} was encoded from; {redo}"
        )


__all__ = [
    "CHUNKING_MODES",
    "DENSE_STAMP_NAME",
    "PreparedDataset",
    "PreparedDatasetError",
    "build_corpus",
    "check_dense_cache",
    "chunk_ids_sha256",
    "chunking_kwargs",
    "load_dataset_yaml",
    "load_prepared",
    "prepare_dataset",
    "read_docs_jsonl",
]
