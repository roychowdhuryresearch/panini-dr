"""Prepare BrowseComp-Plus (BCP) for `panini-dr run`.

Everything here is rebuilt from the public Tevatron releases; nothing is
downloaded from us. Sub-commands (all write under ``$PANINI_DATA_DIR`` or
``--data-dir``, default ``./data``)::

    python datasets/bcp/prepare.py corpus      # HYBRID_BCP.json   ({key: text}, 100,195 docs, ~3.3 GB)
    python datasets/bcp/prepare.py questions --decrypted <browsecomp_plus_decrypted.jsonl>
    python datasets/bcp/prepare.py dense       # bcp_dense_cache/  (Tevatron's Qwen3-Embedding-8B vectors)
    python datasets/bcp/prepare.py bm25        # indexes/bm25/     (Tevatron's Anserini/Lucene index)
    python datasets/bcp/prepare.py all --decrypted <...>

Output layout (``<root>/browsecomp_plus/``)::

    browsecomp_plus_830.json          questions (DECRYPTED: never commit or publish)
    browsecomp_plus_830_subset.json   subset file listing all 830 ids
    HYBRID_BCP.json                   corpus
    bcp_dense_cache/                  chunk_embeddings.npy, chunk_ids.pkl, faiss.index
    indexes/bm25/                     Lucene index (needs pyserini 1.2.0 + JDK 21 at run time)

The paper's run then is::

    panini-dr run --strict-rerank --subset configs/splits/cf_bcp150.json \
        --articles-cache <root>/browsecomp_plus/HYBRID_BCP.json \
        --dense-cache   <root>/browsecomp_plus/bcp_dense_cache \
        --lucene-index  <root>/browsecomp_plus/indexes/bm25

QUESTIONS ARE ENCRYPTED UPSTREAM. BrowseComp-Plus distributes queries and answers
obfuscated, with the canary "BENCHMARK DATA SHOULD NEVER APPEAR AS PLAIN TEXT
ONLINE". Decrypt them with BrowseComp-Plus's own script
(``scripts_build_index/decrypt_dataset.py`` in a checkout of
texttron/BrowseComp-Plus; the paper used commit 046949032b03), which writes
``data/browsecomp_plus_decrypted.jsonl`` inside that checkout, then pass that file
with ``--decrypted`` (or pass ``--bcp-repo <checkout>`` and this script runs the
decrypt script for you, with ``--output data/browsecomp_plus_decrypted.jsonl`` as in
the paper's run). The decrypt script needs the Hugging Face ``datasets`` package,
which the ``datasets`` extra installs. The decryption is not reimplemented here.

Keys and chunk ids are byte-identical to the paper's: document keys come from
``panini_dr.eval.browsecomp_keys.cache_key`` (``"<front-matter title> [<docid>]"``
or the bare docid), the same function the question builder uses for
``gold_pages``; the dense rows are relabelled with ``"<key>__0"`` because BCP is
run with one chunk per document.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _prep_common as common  # noqa: E402

from panini_dr.eval.bcp_dense import align_chunk_ids, load_shards  # noqa: E402
from panini_dr.eval.browsecomp_keys import cache_key  # noqa: E402

QUERIES_REPO = "Tevatron/browsecomp-plus"
CORPUS_REPO = "Tevatron/browsecomp-plus-corpus"
INDEXES_REPO = "Tevatron/browsecomp-plus-indexes"
DENSE_VARIANT = "qwen3-embedding-8b"
N_DOCS = 100_195
N_QUERIES = 830

PLAINTEXT_WARNING = (
    "WARNING: {path} holds DECRYPTED BrowseComp-Plus questions and answers. "
    "Never commit, upload or publish it (BrowseComp-Plus canary: benchmark data "
    "should never appear as plain text online)."
)


# ---------------------------------------------------------------------------
# Pure conversion functions (unit-tested on synthetic rows)
# ---------------------------------------------------------------------------


def build_corpus(docids: Iterable[Any], texts: Iterable[str | None], *,
                 min_chars: int = 1) -> tuple[dict[str, str], dict[str, int]]:
    """``{cache_key: text}`` for the corpus, plus counts.

    Port of scripts/data/build_bcp_cache.py. A duplicate key means a duplicate
    docid and is fatal rather than silently overwriting a document.
    """
    cache: dict[str, str] = {}
    skipped = untitled = 0
    for docid, text in zip(docids, texts):
        text = text or ""
        if len(text) < min_chars:
            skipped += 1
            continue
        key = cache_key(text, docid)
        if key == str(docid):
            untitled += 1
        if key in cache:
            raise common.PrepareError(f"duplicate cache key {key!r} (duplicate docid?)")
        cache[key] = text
    titles = collections.Counter(k.rsplit(" [", 1)[0] for k in cache if k.endswith("]"))
    shared = sum(v for v in titles.values() if v > 1)
    return cache, {"documents": len(cache), "skipped_short": skipped,
                   "untitled": untitled, "shared_title_documents": shared}


def docid_to_key_map(docids: Iterable[Any], texts: Iterable[str | None]) -> dict[str, str]:
    return {str(d): cache_key(t or "", d) for d, t in zip(docids, texts)}


def build_questions(decrypted_rows: Iterable[dict[str, Any]],
                    docid_to_key: dict[str, str]) -> tuple[list[dict[str, Any]], int]:
    """Decrypted BCP rows -> the slim question rows the loader reads.

    Port of scripts/data/build_bcp_questions.py: ``gold_pages`` and
    ``evidence_pages`` are corpus CACHE KEYS (not docids). Returns the rows and
    the number of judged docids with no document in the corpus.
    """
    rows: list[dict[str, Any]] = []
    missing = 0
    for r in decrypted_rows:
        def keys(field: str) -> list[str]:
            nonlocal missing
            out = []
            for d in (r.get(field) or []):
                k = docid_to_key.get(str(d.get("docid")))
                if k is None:
                    missing += 1
                else:
                    out.append(k)
            return out

        rows.append({
            "query_id": str(r.get("query_id")),
            "query": r.get("query") or "",
            "answer": r.get("answer") or "",
            "gold_pages": keys("gold_docs"),
            "evidence_pages": keys("evidence_docs"),
        })
    return rows, missing


def subset_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "subset_id": "browsecomp_plus_830",
        "benchmark": "browsecomp_plus",
        "split": "full",
        "question_ids": [r["query_id"] for r in rows],
        "stratification": {},
        "seed": 0,
        "description": "All 830 BrowseComp-Plus queries (Tevatron).",
    }


def write_dense_cache(reps: Any, lookup: list[Any], docid_to_key: dict[str, str],
                      cache_dir: Path, *, skip_faiss: bool = False) -> list[str]:
    """Relabel Tevatron's document vectors with our chunk ids and write the cache.

    Port of scripts/data/build_bcp_dense_cache.py: rows are never reordered,
    only labelled (DenseRetriever pairs embeddings with ids by position).
    """
    import numpy as np

    ids = align_chunk_ids(lookup, docid_to_key)
    if len(ids) != int(reps.shape[0]):
        raise common.PrepareError("id/row count mismatch")
    if len(set(ids)) != len(ids):
        raise common.PrepareError("duplicate chunk ids")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "chunk_embeddings.npy", np.asarray(reps, dtype="float32"))
    with (cache_dir / "chunk_ids.pkl").open("wb") as fh:
        pickle.dump(ids, fh)
    if not skip_faiss:
        common.ensure_faiss_index(cache_dir)
    return ids


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _bcp_dir(root: Path) -> Path:
    return root / "browsecomp_plus"


def _parquet_files(arg: str | None) -> list[Path]:
    """The corpus parquet file(s): ``--parquet`` (file or dir) or a HF download."""
    if arg:
        p = Path(arg)
        files = sorted(p.rglob("*.parquet")) if p.is_dir() else [p]
    else:
        print(f"downloading {CORPUS_REPO} (parquet, ~1.7 GB) ...")
        snap = common.hf_snapshot(CORPUS_REPO, allow_patterns=["*.parquet"])
        files = sorted(snap.rglob("*.parquet"))
    if not files:
        raise common.PrepareError("no corpus parquet files found")
    return files


def _read_corpus_columns(files: list[Path]) -> tuple[list[Any], list[str | None]]:
    import pyarrow.parquet as pq

    docids: list[Any] = []
    texts: list[str | None] = []
    for f in files:
        table = pq.read_table(f, columns=["docid", "text"])
        docids.extend(table.column("docid").to_pylist())
        texts.extend(table.column("text").to_pylist())
    return docids, texts


def cmd_corpus(args: argparse.Namespace) -> int:
    out = _bcp_dir(common.out_root(args.data_dir)) / "HYBRID_BCP.json"
    docids, texts = _read_corpus_columns(_parquet_files(args.parquet))
    cache, stats = build_corpus(docids, texts, min_chars=args.min_chars)
    if len(cache) != N_DOCS:
        print(f"WARNING: {len(cache):,} documents, the paper corpus has {N_DOCS:,}")
    common.write_json(out, cache)
    print(f"wrote {len(cache):,} documents -> {out}  {stats}")
    return 0


#: Where the paper's run had decrypt_dataset.py write (relative to the checkout), as in
#: BrowseComp-Plus's README: ``--output data/browsecomp_plus_decrypted.jsonl``. The
#: script's own default is ``./decrypted.jsonl``, so the path is always passed.
DECRYPTED_REL = Path("data") / "browsecomp_plus_decrypted.jsonl"


def decrypt_args(extra_args: list[str] | None) -> tuple[list[str], Path]:
    """Arguments for decrypt_dataset.py and the output path they name (relative to the checkout).

    ``--output`` defaults to :data:`DECRYPTED_REL`; a user-supplied ``--output`` wins.
    """
    args = list(extra_args or [])
    out: str | None = None
    for i, a in enumerate(args):
        if a == "--output" and i + 1 < len(args):
            out = args[i + 1]
        elif a.startswith("--output="):
            out = a.split("=", 1)[1]
    if out is None:
        out = str(DECRYPTED_REL)
        args = ["--output", out, *args]
    return args, Path(out)


def _require_hf_datasets() -> None:
    """decrypt_dataset.py imports the Hugging Face ``datasets`` package (the datasets extra)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        version("datasets")  # metadata, not an import: ./datasets/ would shadow the package
    except PackageNotFoundError as exc:
        raise common.PrepareError(
            "BrowseComp-Plus's decrypt_dataset.py needs the Hugging Face `datasets` package, "
            "which is not installed here: install the datasets extra (`uv sync --locked --inexact --extra datasets` "
            "or `pip install 'panini-dr[datasets]'`), or run the script in another environment "
            "and pass its output with --decrypted") from exc


def _decrypt_with_repo(bcp_repo: Path, extra_args: list[str] | None) -> Path:
    script = bcp_repo / "scripts_build_index" / "decrypt_dataset.py"
    if not script.is_file():
        raise common.PrepareError(f"{script} not found: is {bcp_repo} a BrowseComp-Plus checkout?")
    _require_hf_datasets()
    args, rel = decrypt_args(extra_args)
    out = rel if rel.is_absolute() else bcp_repo / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"running {script.name} in {bcp_repo} (writes {out})")
    try:
        subprocess.run([sys.executable, str(script), *args], cwd=bcp_repo, check=True)
    except subprocess.CalledProcessError as exc:
        raise common.PrepareError(
            f"{script.name} failed (exit status {exc.returncode}; its output is above)") from exc
    if not out.is_file():
        raise common.PrepareError(
            f"expected {out} after decryption; pass the decrypted file with --decrypted")
    return out


def cmd_questions(args: argparse.Namespace) -> int:
    if args.decrypted:
        decrypted = Path(args.decrypted)
    elif args.bcp_repo:
        decrypted = _decrypt_with_repo(Path(args.bcp_repo), args.decrypt_args or [])
    else:
        raise common.PrepareError(
            "pass --decrypted <browsecomp_plus_decrypted.jsonl> (made by BrowseComp-Plus's "
            "scripts_build_index/decrypt_dataset.py) or --bcp-repo <checkout>")
    root = _bcp_dir(common.out_root(args.data_dir))
    docids, texts = _read_corpus_columns(_parquet_files(args.parquet))
    mapping = docid_to_key_map(docids, texts)
    with decrypted.open() as fh:
        rows, missing = build_questions((json.loads(line) for line in fh if line.strip()), mapping)
    out_q = common.write_json(root / "browsecomp_plus_830.json", rows)
    common.write_json(root / "browsecomp_plus_830_subset.json", subset_for(rows), indent=2)
    common.require_split_ids([r["query_id"] for r in rows],
                             common.REPO_ROOT / "configs" / "splits" / "cf_bcp150.json")
    print(f"wrote {len(rows)} questions -> {out_q}")
    if len(rows) != N_QUERIES:
        print(f"WARNING: {len(rows)} queries, BrowseComp-Plus has {N_QUERIES}")
    if missing:
        print(f"WARNING: {missing} judged docids have no document in the corpus")
    print(PLAINTEXT_WARNING.format(path=out_q))
    return 0


def cmd_dense(args: argparse.Namespace) -> int:
    root = _bcp_dir(common.out_root(args.data_dir))
    if args.index_dir:
        index_dir = Path(args.index_dir)
    else:
        print(f"downloading {INDEXES_REPO}/{DENSE_VARIANT} ...")
        snap = common.hf_snapshot(INDEXES_REPO, allow_patterns=[f"{DENSE_VARIANT}/*"],
                                  local_dir=root / "indexes")
        index_dir = snap / DENSE_VARIANT
    reps, lookup = load_shards(index_dir)
    docids, texts = _read_corpus_columns(_parquet_files(args.parquet))
    ids = write_dense_cache(reps, lookup, docid_to_key_map(docids, texts),
                            root / "bcp_dense_cache", skip_faiss=args.skip_faiss)
    print(f"wrote {len(ids):,} vectors -> {root / 'bcp_dense_cache'}")
    return 0


def cmd_bm25(args: argparse.Namespace) -> int:
    root = _bcp_dir(common.out_root(args.data_dir))
    print(f"downloading {INDEXES_REPO}/bm25 ...")
    snap = common.hf_snapshot(INDEXES_REPO, allow_patterns=["bm25/*"], local_dir=root / "indexes")
    print(f"Lucene index -> {snap / 'bm25'} (query it with pyserini 1.2.0 and a JDK 21)")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    for fn in (cmd_corpus, cmd_questions, cmd_dense, cmd_bm25):
        fn(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=None, help="data root (default $PANINI_DATA_DIR or ./data)")
    ap.add_argument("--parquet", default=None,
                    help="local corpus parquet file or directory (default: download "
                         f"{CORPUS_REPO})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    min_chars_help = "drop documents shorter than this many characters (default 1, as for the paper)"
    h = f"build browsecomp_plus/HYBRID_BCP.json ({{key: text}}, 100,195 documents) from {CORPUS_REPO}"
    p = sub.add_parser("corpus", help=h, description=h[0].upper() + h[1:] + ".")
    p.add_argument("--min-chars", type=int, default=1, help=min_chars_help)
    p.set_defaults(fn=cmd_corpus)
    helps = {
        "questions": ("build browsecomp_plus/browsecomp_plus_830.json (and the subset file) from "
                      "the queries decrypted by BrowseComp-Plus's decrypt_dataset.py; never "
                      "commit or publish the result"),
        "all": "run corpus, questions, dense and bm25, in this order",
    }
    for name, fn in (("questions", cmd_questions), ("all", cmd_all)):
        p = sub.add_parser(name, help=helps[name], description=helps[name][0].upper() + helps[name][1:] + ".")
        p.add_argument("--decrypted", default=None,
                       help="the decrypted queries (browsecomp_plus_decrypted.jsonl from decrypt_dataset.py)")
        p.add_argument("--bcp-repo", default=None,
                       help="texttron/BrowseComp-Plus checkout: run its decrypt_dataset.py instead of --decrypted")
        p.add_argument("--decrypt-args", nargs="*", default=None,
                       help="extra arguments forwarded to decrypt_dataset.py")
        if name == "all":
            p.add_argument("--min-chars", type=int, default=1, help=min_chars_help)
            p.add_argument("--index-dir", default=None,
                           help=f"local {DENSE_VARIANT} index dir with corpus.shard*.pkl (default: download)")
            p.add_argument("--skip-faiss", action="store_true",
                           help="write embeddings + ids only (faiss.index is needed by `panini-dr run`)")
        p.set_defaults(fn=fn)
    h = (f"build browsecomp_plus/bcp_dense_cache/ from Tevatron's Qwen3-Embedding-8B document "
         f"vectors ({INDEXES_REPO}); no GPU needed")
    p = sub.add_parser("dense", help=h, description=h[0].upper() + h[1:] + ".")
    p.add_argument("--index-dir", default=None,
                   help=f"local {DENSE_VARIANT} index dir with corpus.shard*.pkl (default: download)")
    p.add_argument("--skip-faiss", action="store_true",
                   help="write embeddings + ids only (faiss.index is needed by `panini-dr run`)")
    p.set_defaults(fn=cmd_dense)
    h = (f"download Tevatron's Lucene BM25 index ({INDEXES_REPO}) to browsecomp_plus/indexes/bm25/ "
         "(querying it needs pyserini and a JDK 21)")
    p = sub.add_parser("bm25", help=h, description=h[0].upper() + h[1:] + ".")
    p.set_defaults(fn=cmd_bm25)
    return ap


def main(argv: list[str] | None = None) -> int:
    common.load_env_file()  # ./.env, as `panini-dr` reads it
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except common.PrepareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
