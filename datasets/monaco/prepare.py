"""Prepare MoNaCo-1315 for `panini-dr run`.

Sub-commands (write under ``$PANINI_DATA_DIR`` or ``--data-dir``, default ``./data``)::

    python datasets/monaco/prepare.py questions   # monaco/monaco_full1315.json
    python datasets/monaco/prepare.py corpus      # monaco/HYBRID_MONACO_full_v2.json + dense cache

``questions`` builds the question file from the MoNaCo release
(``allenai/MoNaCo_Benchmark`` on Hugging Face: gated, ODC-BY, and its card asks
that the unzipped data files are NOT redistributed online, so the output must
never be committed or published). Pass ``--release <monaco_version_1_release.json>``
to use a local copy, otherwise the script downloads the dataset repo (accept the
terms on Hugging Face and log in first) and looks for the release file, unzipping
an archive if needed (``--zip-password`` if the archive is protected).

The script that produced the paper's question file did not survive; this one
was rebuilt from the documented formats and then CHECKED against the paper's
file: from the MoNaCo release plus its ``execution_traces/`` it reproduces all
1315 rows exactly (same ids and order; identical ``question``, ``answer``, ``hop``
and ``gold_pages``). Rows of the release are ``{canary, ex_num, question,
decomposition, validated_answer}``; the loader (``panini_dr.eval.monaco_dataset``)
needs ``{id, question, answer, hop, gold_pages}``. We emit ``id =
"monaco_<ex_num>"``, ``answer = validated_answer`` unchanged, ``hop = "<number
of decomposition steps>step"`` (only its digits are read, for logging/failure
classification), and ``gold_pages`` = the English-Wikipedia titles named in the
``source_url`` fields of the example's execution trace (see
:func:`wikipedia_titles`; empty without traces). Nothing on the run path reads
``gold_pages``; the Obs. scorer uses them for its retrieval diagnostics only. It
checks that every id of ``configs/splits/monaco_full1315.json`` is present and
fails loudly otherwise.

``corpus`` installs the paper's frozen corpus (36,508 Wikipedia articles fetched
2026-07-22/23 and 2026-08-03; 616,096 chunks at 2000/200; 1.0 GB) and its dense
cache (10.1 GB of vectors, 5.1 GB to download; ``faiss.index`` is rebuilt
locally on CPU) from the Hugging Face dataset
``roychowdhuryresearch/panini-dr-monaco-corpus`` (named once, in
``_prep_common.HF_CORPUS_REPOS``). **That repo is gated like MoNaCo itself**:
accept the terms on its Hugging Face page, log in (``hf auth login``) or export
``HF_TOKEN``, then run this step. Every file is checked against the sha256
pinned below. ``--from-local DIR`` installs from a copy laid out like the repo,
without network. Disk: ~27 GB during installation (the download is deleted
afterwards unless ``--keep-download``), 21 GB after.

The paper's run then is::

    PANINI_DATA_DIR=<root> panini-dr run --strict-rerank --subset configs/splits/monaco_full1315.json \
        --articles-cache <root>/monaco/HYBRID_MONACO_full_v2.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _prep_common as common  # noqa: E402

MONACO_REPO = "allenai/MoNaCo_Benchmark"
RELEASE_NAME = "monaco_version_1_release.json"
CORPUS_NAME = "HYBRID_MONACO_full_v2.json"
PAPER_DENSE_FINGERPRINT = "Qwen__Qwen3-Embedding-8B__7ea9a4d678f8__de3db946f2e8"
PAPER_N_CHUNKS = 616_096
N_QUESTIONS = 1315

_CACHE = f"retrieval_cache/{PAPER_DENSE_FINGERPRINT}"
#: The published corpus, pinned to the bytes of the paper's runs. Installation
#: trusts these values, never a checksum file that comes with the download.
ARTEFACT = common.CorpusArtefact(
    benchmark="monaco",
    label="MoNaCo-1315",
    repo_id=common.HF_CORPUS_REPOS["monaco"],
    gated=True,
    fingerprint=PAPER_DENSE_FINGERPRINT,
    n_articles=36_508,
    n_chunks=PAPER_N_CHUNKS,
    dim=4096,
    corpus=common.ShippedFile(
        path=f"monaco/{CORPUS_NAME}.zst", install_to=f"monaco/{CORPUS_NAME}",
        sha256="1b475039bf7bd6f268eeab7d2f74b48858e25871d5378a4874b73b2724b60aa5", size=248_439_082,
        raw_sha256="15b7d233da85abaf8667711a9c0467b2e28d2e5d8018a7ab5fa6aa9b668774f8",
        raw_size=1_035_008_062),
    embeddings=common.ShippedFile(
        path=f"{_CACHE}/chunk_embeddings.npy.zst", install_to=f"{_CACHE}/chunk_embeddings.npy",
        sha256="335b8d7cc29e0fdc8a33d97c3e1cef5f03516907b21136cd21c256f3cf3fb36b", size=5_074_910_341,
        raw_sha256="80dcc74d99ca6f9e8d5b2c900d23b60c5422bff384b73f46874fd797bc6c59ee",
        raw_size=10_094_116_992),
    chunk_ids=common.ShippedFile(
        path=f"{_CACHE}/chunk_ids.pkl", install_to=f"{_CACHE}/chunk_ids.pkl",
        sha256="449aa7420597ddfdc328a63bb329038b3b82e5df66e9e579035c1fc10bfdcd20", size=15_424_520),
    faiss_sha256="a38fe300bcdb56b849b45d300e7406dd0b6c30974499deb8e586d39e24509126",
    split_file="configs/splits/monaco_full1315.json",
)
#: sha256 of the question file the paper's runs used. The builder reproduces it
#: byte for byte from the release and its execution traces (written with
#: ensure_ascii=False, no indentation).
PAPER_QUESTIONS_SHA256 = "7ba2b0abac1f564237892112fd1eaa19b06d7ee76a6cf6134218985be6192e15"

PLAINTEXT_WARNING = (
    "NOTE: {path} is derived from the MoNaCo release, whose terms ask that the "
    "unzipped data is not redistributed online. Do not commit or publish it."
)

#: The only URL form the paper's gold_pages were taken from.
WIKI_PREFIX = "https://en.wikipedia.org/wiki/"

# ---------------------------------------------------------------------------
# Conversion (unit-tested on synthetic rows)
# ---------------------------------------------------------------------------


def iter_release_rows(release: Any) -> Iterator[dict[str, Any]]:
    """The release is a dict keyed by question text (or, defensively, a list)."""
    if isinstance(release, dict):
        yield from (row for row in release.values() if isinstance(row, dict))
    elif isinstance(release, list):
        yield from (row for row in release if isinstance(row, dict))


def _source_url_strings(node: Any, key: str | None = None) -> Iterator[str]:
    """Every string stored under a ``source_url`` key (directly or in a list) in ``node``."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _source_url_strings(v, k)
    elif isinstance(node, (list, tuple)):
        for v in node:
            yield from _source_url_strings(v, key)
    elif isinstance(node, str) and key == "source_url":
        yield node


def wikipedia_titles(obj: Any) -> list[str]:
    """The English-Wikipedia titles named in the ``source_url`` fields of a trace, sorted.

    The rule that reproduces the paper's file exactly: a ``source_url`` value
    may hold several URLs (comma-, newline- or space-separated), so it is split
    on ``https://en.wikipedia.org/wiki/``; each piece loses its ``#fragment``,
    surrounding whitespace and trailing commas, is percent-decoded and has ``_``
    replaced by a space. A query string (``?wprov=...``) is kept, and so is a
    title containing a comma or an apostrophe. Mobile (``en.m.``), other
    language editions, scheme-less URLs and URLs outside ``source_url`` are not
    gold pages.
    """
    titles: set[str] = set()
    for s in _source_url_strings(obj):
        for piece in s.split(WIKI_PREFIX)[1:]:
            piece = piece.split("#")[0].strip().rstrip(",").strip()
            title = unquote(piece).replace("_", " ").strip()
            if title:
                titles.add(title)
    return sorted(titles)


def build_question(row: dict[str, Any], *, trace: Any = None) -> dict[str, Any]:
    steps = row.get("decomposition") or []
    return {
        "id": f"monaco_{row.get('ex_num')}",
        "question": row.get("question") or "",
        "answer": row.get("validated_answer"),
        "hop": f"{len(steps)}step",
        "gold_pages": wikipedia_titles(trace) if trace is not None else [],
    }


def build_questions(rows: Iterable[dict[str, Any]],
                    traces: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Release rows -> loader rows. ``traces`` maps ``str(ex_num)`` to that example's trace."""
    traces = traces or {}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if row.get("ex_num") is None:
            raise common.PrepareError("release row without ex_num")
        q = build_question(row, trace=traces.get(str(row.get("ex_num"))))
        if q["id"] in seen:
            raise common.PrepareError(f"duplicate id {q['id']}")
        seen.add(q["id"])
        out.append(q)
    return out


_TRACE_NUM_RE = re.compile(r"dataset_ex_(\d+)\.json$")


def load_traces(directory: Path | None) -> dict[str, Any]:
    """``execution_traces/dataset_ex_<n>.json`` files keyed by ``<n>`` (if present)."""
    if directory is None or not Path(directory).is_dir():
        return {}
    out: dict[str, Any] = {}
    for f in sorted(Path(directory).rglob("dataset_ex_*.json")):
        m = _TRACE_NUM_RE.search(f.name)
        if m:
            try:
                out[m.group(1)] = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
    return out


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _find_release(snapshot: Path, *, unzip_to: Path, password: str | None) -> tuple[Path, Path | None]:
    def search(root: Path) -> tuple[Path | None, Path | None]:
        rel = next(iter(sorted(root.rglob(RELEASE_NAME))), None)
        traces = next(iter(sorted(p for p in root.rglob("execution_traces") if p.is_dir())), None)
        return rel, traces

    rel, traces = search(snapshot)
    if rel is None:
        for z in sorted(snapshot.rglob("*.zip")):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(unzip_to, pwd=password.encode() if password else None)
        rel, traces = search(unzip_to)
    if rel is None:
        raise common.PrepareError(
            f"{RELEASE_NAME} not found in the {MONACO_REPO} download; pass --release")
    return rel, traces


def cmd_questions(args: argparse.Namespace) -> int:
    out_dir = common.out_root(args.data_dir) / "monaco"
    if args.release:
        release_path = Path(args.release)
        traces_dir = Path(args.traces) if args.traces else release_path.parent / "execution_traces"
    else:
        snap = common.hf_snapshot(MONACO_REPO, allow_patterns=["*"])
        release_path, traces_dir = _find_release(
            snap, unzip_to=out_dir / "_release", password=args.zip_password)
    rows = list(iter_release_rows(json.loads(Path(release_path).read_text())))
    questions = build_questions(rows, load_traces(traces_dir))
    out = common.write_json(out_dir / "monaco_full1315.json", questions, ensure_ascii=False)
    common.require_split_ids([q["id"] for q in questions],
                             common.REPO_ROOT / "configs" / "splits" / "monaco_full1315.json")
    n_gold = sum(1 for q in questions if q["gold_pages"])
    print(f"wrote {len(questions)} questions -> {out} ({n_gold} with gold_pages)")
    if len(questions) != N_QUESTIONS:
        print(f"WARNING: {len(questions)} questions, MoNaCo has {N_QUESTIONS}")
    digest = common.sha256_file(out)
    if digest == PAPER_QUESTIONS_SHA256:
        print("identical to the paper's question file (sha256 matches)")
    else:
        print(f"WARNING: differs from the paper's question file (sha256 {digest[:16]}..., "
              f"paper {PAPER_QUESTIONS_SHA256[:16]}...); check the release version and that "
              "--traces points at the release's execution_traces/")
    print(PLAINTEXT_WARNING.format(path=out))
    return 0


def cmd_corpus(args: argparse.Namespace) -> int:
    return common.corpus_command(args, ARTEFACT)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=None, help="data root (default $PANINI_DATA_DIR or ./data)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = (f"build monaco/monaco_full1315.json from the official {MONACO_REPO} release (gated) "
         "and check it against the paper's file")
    p = sub.add_parser("questions", help=h, description=h[0].upper() + h[1:] + ".")
    p.add_argument("--release", default=None, help=f"local {RELEASE_NAME}")
    p.add_argument("--traces", default=None,
                   help="local execution_traces/ directory (for gold_pages; optional)")
    p.add_argument("--zip-password", default=None,
                   help="password of the release archive, if it is password-protected")
    p.set_defaults(fn=cmd_questions)
    h = (f"install the paper's frozen corpus and its dense index from {ARTEFACT.repo_id} (gated), "
         "checked against pinned sha256s")
    p = sub.add_parser("corpus", help=h, description=h[0].upper() + h[1:] + ".")
    common.add_corpus_arguments(p, ARTEFACT)
    p.set_defaults(fn=cmd_corpus)
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
