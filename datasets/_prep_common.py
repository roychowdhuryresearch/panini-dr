"""Helpers shared by the benchmark preparation scripts under ``datasets/``.

Deliberately not a package: ``datasets`` would shadow the Hugging Face library
of the same name. Each ``datasets/<benchmark>/prepare.py`` puts this directory
on ``sys.path`` and imports ``_prep_common``.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import os
import pickle
import shlex
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, Callable, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]

# Make `panini_dr` importable from a plain source checkout (no install needed).
try:  # pragma: no cover - trivial
    import panini_dr  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(REPO_ROOT / "src"))

from panini_dr.envfile import load_env_file  # noqa: E402,F401  (each script's main() calls it)
from panini_dr.paths import data_dir  # noqa: E402

#: Model slug used in dense-cache directory names (retrieval/dense.py).
DENSE_MODEL = "Qwen/Qwen3-Embedding-8B"

#: The Hugging Face dataset repos that hold the paper's frozen FRAMES-824 and
#: MoNaCo-1315 corpora and their dense caches. This is the only place they are
#: named; the dataset cards are in ``datasets/hf_cards/``.
HF_CORPUS_REPOS: dict[str, str] = {
    "frames": "roychowdhuryresearch/panini-dr-frames-corpus",  # public
    "monaco": "roychowdhuryresearch/panini-dr-monaco-corpus",  # gated: MoNaCo's terms
}


class PrepareError(RuntimeError):
    """A preparation step cannot proceed; the message says what to do."""


def out_root(arg: str | os.PathLike | None) -> Path:
    """The data root: an explicit ``--data-dir`` or ``$PANINI_DATA_DIR`` (default ./data)."""
    return Path(arg).expanduser() if arg else data_dir()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hf_snapshot(repo_id: str, *, allow_patterns: list[str], local_dir: Path | None = None,
                repo_type: str = "dataset") -> Path:
    """Download (part of) a Hugging Face repo and return the local directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise PrepareError(
            "huggingface_hub is required: pip install 'panini-dr[datasets]'"
        ) from exc
    kwargs: dict[str, Any] = {"repo_id": repo_id, "repo_type": repo_type,
                              "allow_patterns": allow_patterns}
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
    return Path(snapshot_download(**kwargs))


def hf_file(repo_id: str, filename: str, *, repo_type: str = "dataset") -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise PrepareError(
            "huggingface_hub is required: pip install 'panini-dr[datasets]'"
        ) from exc
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type))


def dense_fingerprint(articles: dict[str, str], *, model_name: str = DENSE_MODEL,
                      list_aware: bool = False) -> str:
    """The directory name DenseRetriever looks for under $PANINI_DATA_DIR/retrieval_cache.

    Computed by the retriever's own ``_fingerprint`` so the two can never drift.
    """
    from panini_dr.retrieval.dense import DenseRetriever

    fake_self = SimpleNamespace(
        corpus=SimpleNamespace(_by_title=articles, list_aware=list_aware),
        model_name=model_name,
    )
    return DenseRetriever._fingerprint(fake_self)  # type: ignore[arg-type]


def ensure_faiss_index(cache_dir: Path, *, rebuild: bool = False, batch_rows: int = 65_536) -> Path:
    """Build ``faiss.index`` from ``chunk_embeddings.npy`` when it was not downloaded.

    The published caches ship the embedding matrix and the ids only: the FAISS
    ``IndexFlatIP`` is a byte-for-byte duplicate of the matrix, so it is rebuilt
    locally (CPU, minutes) instead of doubling the download. Rows are added in
    batches from a memory map, so the peak memory is one copy of the matrix; the
    result is identical to a single ``add`` (a flat index only appends rows).
    The index is written to a temporary name and renamed, so an interrupted
    build never leaves a truncated ``faiss.index`` that a run would load.
    """
    cache_dir = Path(cache_dir)
    index_path = cache_dir / "faiss.index"
    if index_path.is_file() and not rebuild:
        return index_path
    emb = cache_dir / "chunk_embeddings.npy"
    if not emb.is_file():
        raise PrepareError(f"{emb} is missing; cannot build the FAISS index")
    try:
        import faiss  # needs the [dense] extra
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise PrepareError("building faiss.index needs faiss: pip install 'panini-dr[dense]'") from exc
    import numpy as np

    reps = np.load(emb, mmap_mode="r")
    index = faiss.IndexFlatIP(int(reps.shape[1]))
    for start in range(0, int(reps.shape[0]), batch_rows):
        index.add(np.ascontiguousarray(reps[start:start + batch_rows], dtype="float32"))
    tmp = index_path.with_name(index_path.name + ".tmp")
    faiss.write_index(index, str(tmp))
    tmp.replace(index_path)
    return index_path


def require_split_ids(question_ids: Iterable[str], split_file: Path) -> None:
    """Fail loudly unless every id of a paper split is present."""
    split = json.loads(Path(split_file).read_text())
    have = set(question_ids)
    missing = [qid for qid in split.get("question_ids", []) if qid not in have]
    if missing:
        raise PrepareError(
            f"{len(missing)} of {len(split.get('question_ids', []))} ids of "
            f"{Path(split_file).name} are not in the built question file "
            f"(first missing: {missing[:5]}). The output does not match the paper split."
        )


def write_json(path: Path, obj: Any, *, indent: int | None = None, ensure_ascii: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=ensure_ascii)
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# The published FRAMES-824 / MoNaCo-1315 corpora (Hugging Face dataset repos)
# ---------------------------------------------------------------------------
#
# Repo layout (identical for both; paths are relative to the repo root and,
# after installation, to the data root):
#
#   <benchmark>/<corpus>.json.zst                       the {title: text} corpus (zstd)
#   retrieval_cache/<fingerprint>/chunk_embeddings.npy.zst  Qwen3-Embedding-8B vectors (zstd)
#   retrieval_cache/<fingerprint>/chunk_ids.pkl          chunk id of every row
#   provenance/articles.jsonl, manifest.json, SHA256SUMS, README.md   (not installed)
#
# Installation decompresses the two .zst files, checks every byte against the
# sha256 values pinned in datasets/<benchmark>/prepare.py (not against anything
# downloaded), and rebuilds faiss.index from the embeddings, so that
# `panini-dr run` finds $PANINI_DATA_DIR/retrieval_cache/<fingerprint>/ from the
# corpus alone.

#: Read size for streaming copies, decompression and hashing.
_BLOCK = 4 << 20


@dataclass(frozen=True)
class ShippedFile:
    """One file of a published corpus repo, as shipped and as installed."""

    path: str                      # path inside the dataset repo
    sha256: str                    # of the shipped bytes
    size: int
    install_to: str                # path under the data root
    raw_sha256: str | None = None  # after decompression (None: shipped as is)
    raw_size: int | None = None

    @property
    def installed_sha256(self) -> str:
        return self.raw_sha256 or self.sha256

    @property
    def installed_size(self) -> int:
        return self.raw_size if self.raw_size is not None else self.size


@dataclass(frozen=True)
class CorpusArtefact:
    """A published corpus + dense cache, pinned to the bytes the paper's runs used."""

    benchmark: str          # "frames" | "monaco"
    label: str              # for messages, e.g. "FRAMES-824"
    repo_id: str
    gated: bool
    fingerprint: str        # dense-cache directory name (retrieval/dense.py)
    n_articles: int
    n_chunks: int
    dim: int
    corpus: ShippedFile
    embeddings: ShippedFile
    chunk_ids: ShippedFile
    faiss_sha256: str       # of faiss.index as the paper's runs had it (rebuilt locally)
    split_file: str         # configs/splits/<...>.json, for the printed run command

    @property
    def files(self) -> tuple[ShippedFile, ...]:
        return (self.corpus, self.embeddings, self.chunk_ids)

    @property
    def repo_name(self) -> str:
        return self.repo_id.rsplit("/", 1)[-1]

    def allow_patterns(self) -> list[str]:
        return [f.path for f in self.files]


def gated_help(repo_id: str) -> str:
    return (
        f"{repo_id} is a gated dataset. MoNaCo asks that its data is not redistributed, and "
        "this corpus is the set of Wikipedia pages selected by MoNaCo's gold annotations, so it "
        "is shared only with people who accept MoNaCo's terms. To get it: (1) open "
        f"https://huggingface.co/datasets/{repo_id} while logged in and accept the terms; "
        "(2) authenticate on this machine with `hf auth login` (or `huggingface-cli login`), "
        "or export HF_TOKEN=<a read token>; (3) run this command again. "
        "To install from a copy you already have, pass --from-local DIR."
    )


def hf_download_artefact(art: CorpusArtefact, *, download_dir: Path,
                         revision: str | None = None) -> Path:
    """Download the files of ``art`` from the Hub into ``download_dir``; explain every failure."""
    try:
        from huggingface_hub import get_token, snapshot_download
        from huggingface_hub.errors import (
            GatedRepoError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
        )
    except ImportError as exc:
        raise PrepareError(
            "huggingface_hub is required to download the corpus: pip install 'panini-dr[datasets]' "
            "(or install from a local copy with --from-local DIR)"
        ) from exc
    if art.gated and not get_token():
        raise PrepareError("no Hugging Face token found. " + gated_help(art.repo_id))
    try:
        path = snapshot_download(
            repo_id=art.repo_id, repo_type="dataset", revision=revision,
            allow_patterns=art.allow_patterns(), local_dir=str(download_dir),
        )
    except GatedRepoError as exc:  # before RepositoryNotFoundError: it is a subclass
        raise PrepareError(
            "the Hub refused the download: your account has not been granted access yet "
            "(or the token is not the one that accepted the terms). " + gated_help(art.repo_id)
        ) from exc
    except RepositoryNotFoundError as exc:
        hint = gated_help(art.repo_id) if art.gated else (
            "Check the repo id (--repo) and your network, or install from a local copy with "
            "--from-local DIR.")
        raise PrepareError(f"{art.repo_id} was not found on the Hugging Face Hub. {hint}") from exc
    except LocalEntryNotFoundError as exc:
        raise PrepareError(
            f"could not download {art.repo_id}: the Hub is unreachable and the files are not "
            "cached. Retry with network access, or install from a local copy with --from-local DIR."
        ) from exc
    return Path(path)


def resolve_local_source(src: Path, art: CorpusArtefact) -> Path:
    """The repo root inside ``src``: ``src`` itself or ``src/<repo name>`` (a staging dir)."""
    src = Path(src).expanduser()
    for root in (src, src / art.repo_name):
        if (root / art.corpus.path).is_file():
            return root
    raise PrepareError(
        f"{src} does not hold the {art.label} corpus: expected {art.corpus.path} in it "
        f"(or in {art.repo_name}/ under it), laid out like the Hugging Face repo {art.repo_id}"
    )


def check_shipped_file(path: Path, f: ShippedFile) -> None:
    """Size and sha256 of a shipped file against the values pinned in this repository."""
    if not path.is_file():
        raise PrepareError(f"{f.path} is missing (looked for {path})")
    size = path.stat().st_size
    if size != f.size:
        raise PrepareError(
            f"{f.path}: {size} bytes, expected {f.size} (a truncated or different download; "
            "delete it and download again)")
    digest = sha256_file(path)
    if digest != f.sha256:
        raise PrepareError(
            f"sha256 mismatch for {f.path}: got {digest}, expected {f.sha256}. This is not the "
            "file the paper's runs used; delete it and download again.")


@contextmanager
def _open_decompressed(path: Path) -> Iterator[BinaryIO]:
    """A binary stream of the decompressed contents of ``path`` (.zst, .gz or plain)."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as fh:
            yield fh
        return
    if path.suffix != ".zst":
        with path.open("rb") as fh:
            yield fh
        return
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError:
        zstandard = None
    if zstandard is not None:
        with path.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(
                raw, read_across_frames=True) as fh:
            yield fh
        return
    exe = shutil.which("zstd")
    if exe is None:
        raise PrepareError(
            f"{path.name} is zstd-compressed: install the zstd command-line tool "
            "(e.g. `apt install zstd`, `brew install zstd`) or the Python package "
            "(`pip install zstandard`), then run this again")
    proc = subprocess.Popen([exe, "-d", "-c", "-q", str(path)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    try:
        yield proc.stdout  # type: ignore[misc]
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    proc.stdout.close()
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise PrepareError(f"zstd failed on {path}: {err.strip()}")


def install_file(src: Path, f: ShippedFile, dst: Path) -> None:
    """Decompress or copy ``src`` to ``dst`` atomically, verifying the installed bytes."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    h = hashlib.sha256()
    n = 0
    try:
        with _open_decompressed(src) as fin, tmp.open("wb") as fout:
            for block in iter(lambda: fin.read(_BLOCK), b""):
                h.update(block)
                n += len(block)
                fout.write(block)
        if n != f.installed_size or h.hexdigest() != f.installed_sha256:
            raise PrepareError(
                f"{f.path} decompressed to {n} bytes with sha256 {h.hexdigest()}, expected "
                f"{f.installed_size} bytes with sha256 {f.installed_sha256}")
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def is_installed(dst: Path, f: ShippedFile) -> bool:
    return (dst.is_file() and dst.stat().st_size == f.installed_size
            and sha256_file(dst) == f.installed_sha256)


class _PlainDataUnpickler(pickle.Unpickler):
    """Refuses every global: ``chunk_ids.pkl`` must be a plain list of strings."""

    def find_class(self, module: str, name: str) -> Any:  # noqa: D401
        raise PrepareError(
            f"chunk_ids.pkl references {module}.{name}; expected a plain list of strings")


def load_chunk_ids(path: Path) -> list[str]:
    """Read ``chunk_ids.pkl`` without executing anything (no globals allowed)."""
    with Path(path).open("rb") as fh:
        ids = _PlainDataUnpickler(fh).load()
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise PrepareError(f"{path} is not a list of chunk-id strings")
    return ids


def install_corpus_artefact(art: CorpusArtefact, src_root: Path, data_root: Path, *,
                            verify_chunks: bool = False,
                            log: Callable[[str], None] = print) -> dict[str, Path]:
    """Install a published corpus from ``src_root`` (repo layout) into ``data_root``.

    Every shipped file is checked against the sha256 pinned in ``art`` before
    anything is written; every installed file is checked again after
    decompression. Already-installed files with the right bytes are kept.
    """
    src_root, data_root = Path(src_root), Path(data_root)
    for f in art.files:
        dst = data_root / f.install_to
        if is_installed(dst, f):
            log(f"ok (already installed)  {dst}")
            continue
        check_shipped_file(src_root / f.path, f)
        log(f"installing {f.path} -> {dst}")
        install_file(src_root / f.path, f, dst)

    corpus_path = data_root / art.corpus.install_to
    dense_dir = data_root / Path(art.embeddings.install_to).parent
    if dense_dir.name != art.fingerprint:
        raise PrepareError(f"internal: dense cache path {dense_dir} does not end in {art.fingerprint}")

    with corpus_path.open(encoding="utf-8") as fh:
        articles: dict[str, str] = json.load(fh)
    fp = dense_fingerprint(articles)
    if fp != art.fingerprint:
        raise PrepareError(
            f"panini_dr computes the dense-cache name {fp} for this corpus, but the cache is "
            f"{art.fingerprint}: `panini-dr run` would not find it (has the fingerprint code changed?)")
    if len(articles) != art.n_articles:
        raise PrepareError(f"{len(articles)} articles, expected {art.n_articles}")

    import numpy as np

    emb = np.load(data_root / art.embeddings.install_to, mmap_mode="r")
    ids = load_chunk_ids(data_root / art.chunk_ids.install_to)
    if tuple(emb.shape) != (art.n_chunks, art.dim) or len(ids) != art.n_chunks:
        raise PrepareError(
            f"dense cache shape {tuple(emb.shape)} with {len(ids)} ids; expected "
            f"({art.n_chunks}, {art.dim}) with {art.n_chunks} ids")

    index_path = dense_dir / "faiss.index"
    if index_path.is_file() and index_path.stat().st_size and sha256_file(index_path) == art.faiss_sha256:
        log(f"ok (already installed)  {index_path}")
    else:
        log(f"building {index_path} from the embeddings (CPU) ...")
        ensure_faiss_index(dense_dir, rebuild=True)
        if sha256_file(index_path) != art.faiss_sha256:
            log("WARNING: the rebuilt faiss.index differs from the paper's bytes (another faiss "
                "version may serialise differently). It holds the verified embeddings, so search "
                "results should not change.")
        else:
            log(f"ok  {index_path} (identical to the paper's)")

    if verify_chunks:
        from panini_dr.retrieval.corpus import load_frames_corpus

        chunks = load_frames_corpus(cache_path=corpus_path).chunks
        if [c.chunk_id for c in chunks] != ids:
            raise PrepareError(
                f"the corpus chunks into {len(chunks)} chunks whose ids differ from chunk_ids.pkl")
        log(f"ok  {len(chunks)} chunks; ids match the dense cache row for row")
    return {"corpus": corpus_path, "dense_cache": dense_dir}


def add_corpus_arguments(p: argparse.ArgumentParser, art: CorpusArtefact) -> None:
    p.add_argument("--from-local", "--from-dir", dest="from_local", default=None, metavar="DIR",
                   help=f"install from a local copy laid out like {art.repo_id} "
                        "(DIR or DIR/" + art.repo_name + "/); no network")
    p.add_argument("--repo", default=art.repo_id, help=f"Hugging Face dataset repo (default {art.repo_id})")
    p.add_argument("--revision", default=None, help="repo revision (branch, tag or commit) to download")
    p.add_argument("--download-dir", default=None,
                   help="where the compressed files are downloaded (default <data root>/_downloads/"
                        f"{art.repo_name}); deleted after a successful install unless --keep-download")
    p.add_argument("--keep-download", action="store_true", help="keep the downloaded compressed files")
    p.add_argument("--verify-chunks", action="store_true",
                   help=f"also chunk the corpus (2000/200) and check its {art.n_chunks} chunk ids "
                        "against the dense cache")


def corpus_command(args: argparse.Namespace, art: CorpusArtefact) -> int:
    """``prepare.py corpus``: download (or take a local copy of) the corpus and install it."""
    root = out_root(args.data_dir)
    if args.repo != art.repo_id:
        art = dataclasses.replace(art, repo_id=args.repo)
    downloaded: Path | None = None
    if args.from_local:
        src = resolve_local_source(Path(args.from_local), art)
        print(f"installing {art.label} from {src}")
    else:
        downloaded = Path(args.download_dir) if args.download_dir else root / "_downloads" / art.repo_name
        print(f"downloading {art.repo_id} -> {downloaded} ...")
        src = hf_download_artefact(art, download_dir=downloaded, revision=args.revision)
    placed = install_corpus_artefact(art, src, root, verify_chunks=args.verify_chunks)
    if downloaded is not None and not args.keep_download:
        shutil.rmtree(downloaded, ignore_errors=True)
        if not args.download_dir:
            try:
                downloaded.parent.rmdir()  # <data root>/_downloads, if now empty
            except OSError:
                pass
        print(f"removed the download directory {downloaded} (--keep-download keeps it)")
    print(f"corpus:      {placed['corpus']} ({art.n_articles:,} articles, {art.n_chunks:,} chunks)")
    print(f"dense cache: {placed['dense_cache']}")
    env_root = data_dir()
    if env_root.resolve() != root.resolve():
        print(f"NOTE: run with PANINI_DATA_DIR={root} (or --dense-cache {placed['dense_cache']}): "
              "`panini-dr run` looks for the dense cache under $PANINI_DATA_DIR/retrieval_cache")
    # The command REPRODUCE.md gives (--strict-rerank: a failed rerank aborts
    # instead of silently falling back to fusion order).
    print("run it with:\n"
          f"  PANINI_DATA_DIR={shlex.quote(str(root))} panini-dr run --strict-rerank "
          f"--subset {art.split_file} --articles-cache {shlex.quote(str(placed['corpus']))} "
          f"--output-dir runs/{art.benchmark}")
    return 0
