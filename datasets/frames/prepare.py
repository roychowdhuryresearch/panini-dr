"""Prepare FRAMES-824 for `panini-dr run`.

Sub-commands (write under ``$PANINI_DATA_DIR`` or ``--data-dir``, default ``./data``)::

    python datasets/frames/prepare.py questions   # frames/frames_split.json (+ article_index.json)
    python datasets/frames/prepare.py corpus      # frames/HYBRID_FRAMES824.json + its dense cache

``questions`` downloads ``test.tsv`` from the Hugging Face dataset
``google/frames-benchmark`` (Apache-2.0) and converts it exactly as the paper's
``prep_frames.py`` did: ids ``"0".."823"`` (the TSV's unnamed index column),
``original_prompt`` (what the loader reads), a context/test-question split,
``answer``, the gold Wikipedia ``articles`` ({title, url}), ``reasoning_types``
and ``num_hops``. Pass ``--tsv`` to convert a local copy instead.

``corpus`` installs the paper's frozen corpus (2,513 Wikipedia articles, live
revisions fetched in 2026; 69,608 chunks at 2000/200) and its Qwen3-Embedding-8B
dense cache from the public Hugging Face dataset
``roychowdhuryresearch/panini-dr-frames-corpus`` (named once, in
``_prep_common.HF_CORPUS_REPOS``). Every file is checked against the sha256
pinned below, the two ``.zst`` files are decompressed, and ``faiss.index`` is
rebuilt from the embeddings on CPU. ``--from-local DIR`` installs from a copy
laid out like that repo, without network. The corpus cannot be rebuilt
byte-identically from live Wikipedia. The repo's ``verify/`` directory holds the
``panini-dr verify`` bundle: recorded runs of the paper configuration on 8 FRAMES
questions, which ``panini-dr verify`` replays offline against this corpus and
dense cache to check an installation and the code end to end. This step does
not download it (see datasets/README.md).

The paper's run then is::

    PANINI_DATA_DIR=<root> panini-dr run --strict-rerank --subset configs/splits/frames_full824_norm.json \
        --articles-cache <root>/frames/HYBRID_FRAMES824.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _prep_common as common  # noqa: E402

FRAMES_REPO = "google/frames-benchmark"
FRAMES_TSV = "test.tsv"
CORPUS_NAME = "HYBRID_FRAMES824.json"
#: Dense-cache directory name of the paper's corpus (retrieval/dense.py fingerprint).
PAPER_DENSE_FINGERPRINT = "Qwen__Qwen3-Embedding-8B__99c7a0f7ddf9__efb84805d53b"
PAPER_N_CHUNKS = 69_608
N_QUESTIONS = 824

_CACHE = f"retrieval_cache/{PAPER_DENSE_FINGERPRINT}"
#: The published corpus, pinned to the bytes of the paper's runs. Installation
#: trusts these values, never a checksum file that comes with the download.
ARTEFACT = common.CorpusArtefact(
    benchmark="frames",
    label="FRAMES-824",
    repo_id=common.HF_CORPUS_REPOS["frames"],
    gated=False,
    fingerprint=PAPER_DENSE_FINGERPRINT,
    n_articles=2_513,
    n_chunks=PAPER_N_CHUNKS,
    dim=4096,
    corpus=common.ShippedFile(
        path=f"frames/{CORPUS_NAME}.zst", install_to=f"frames/{CORPUS_NAME}",
        sha256="05e66bce1cfcc056894421288ec7719b9f0073bec261ecfd93d36825b890602a", size=29_668_076,
        raw_sha256="fc3015d31cb98eba5d367165160ba5c227d8801600b7ee7f2b97be227caf7e20",
        raw_size=118_318_457),
    embeddings=common.ShippedFile(
        path=f"{_CACHE}/chunk_embeddings.npy.zst", install_to=f"{_CACHE}/chunk_embeddings.npy",
        sha256="dbae811cf70db595f77dbbfd8827fb7764a29d16013b15be776ec23436ad42ec", size=575_439_762,
        raw_sha256="0a90ad9879b4686368d8cd161cfd7cab5574155cd679f9b84420c3cb6505a47a",
        raw_size=1_140_457_600),
    chunk_ids=common.ShippedFile(
        path=f"{_CACHE}/chunk_ids.pkl", install_to=f"{_CACHE}/chunk_ids.pkl",
        sha256="b488cd4e54c9061a926ef0b1fb4ef1d0fff318c1ebd65926966635832d28b877", size=1_698_904),
    faiss_sha256="d7ab3596f41ec9ce9d72db982149050c5a3e50836097ba291192fc4edd0a3eec",
    split_file="configs/splits/frames_full824_norm.json",
)

# ---------------------------------------------------------------------------
# Conversion (verbatim logic of the paper's prep_frames.py)
# ---------------------------------------------------------------------------

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WIKI_TITLE_RE = re.compile(r"wikipedia\.org/wiki/(.+?)(?:#.*)?$")


def split_question(prompt: str) -> Optional[tuple[str, str]]:
    """Split a FRAMES prompt into (context, test_question), or None."""
    sentences = SENTENCE_SPLIT_RE.split(prompt.strip())
    if len(sentences) < 2:
        return None
    if not sentences[-1].strip().endswith("?"):
        return None
    declarative = [s for s in sentences[:-1] if not s.strip().endswith("?")]
    if not declarative:
        return None
    context = " ".join(sentences[:-1]).strip()
    test_question = sentences[-1].strip()
    return context, test_question


def url_to_title(url: str) -> Optional[str]:
    match = WIKI_TITLE_RE.search(url)
    if not match:
        return None
    return unquote(match.group(1)).replace("_", " ")


def url_to_canonical(url: str) -> str:
    return re.sub(r"#.*$", "", url).replace("en.m.wikipedia", "en.wikipedia")


def build_article_list(wiki_links: list[str]) -> list[dict[str, str]]:
    """Wiki URLs -> [{title, url}] with dedup (entries may pack several URLs)."""
    expanded: list[str] = []
    for entry in wiki_links:
        parts = re.split(r",\s*(?=https?://)", entry.strip())
        expanded.extend(p.strip() for p in parts if p.strip())
    seen: set[str] = set()
    articles: list[dict[str, str]] = []
    for url in expanded:
        title = url_to_title(url)
        if title and title not in seen:
            seen.add(title)
            articles.append({"title": title, "url": url_to_canonical(url)})
    return articles


def split_frames_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """TSV rows -> question dicts (unsplittable prompts keep the full prompt)."""
    results = []
    for row in rows:
        prompt = row["Prompt"].strip()
        split = split_question(prompt)
        if split is not None:
            context, test_question = split
        else:
            context, test_question = None, prompt
        wiki_links = ast.literal_eval(row["wiki_links"])
        reasoning_types = [t.strip() for t in row.get("reasoning_types", "").split("|") if t.strip()]
        articles = build_article_list(wiki_links)
        results.append({
            "id": int(row[""]),
            "original_prompt": prompt,
            "context": context,
            "test_question": test_question,
            "answer": row["Answer"].strip(),
            "articles": articles,
            "reasoning_types": reasoning_types,
            "num_hops": len(articles),
        })
    return results


def build_article_index(questions: list[dict[str, Any]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for q in questions:
        for article in q["articles"]:
            index.setdefault(article["title"], article["url"])
    return index


def sample_dev_subset(questions: list[dict[str, Any]], n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Stratified sqrt-proportional sample by hop bucket (2/3/4/5+); for --split dev."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        buckets[str(q["num_hops"]) if q["num_hops"] <= 4 else "5+"].append(q)
    sqrt_sizes = {k: math.sqrt(len(v)) for k, v in buckets.items()}
    total = sum(sqrt_sizes.values())
    alloc = {k: round(s / total * n) for k, s in sqrt_sizes.items()}
    largest = max(alloc, key=lambda k: len(buckets[k]))
    alloc[largest] += n - sum(alloc.values())
    sampled: list[dict[str, Any]] = []
    for key in sorted(buckets):
        sampled.extend(rng.sample(buckets[key], min(alloc[key], len(buckets[key]))))
    return sampled


def read_tsv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_questions(args: argparse.Namespace) -> int:
    tsv = Path(args.tsv) if args.tsv else common.hf_file(FRAMES_REPO, FRAMES_TSV)
    questions = split_frames_questions(read_tsv(tsv))
    out = common.out_root(args.data_dir) / "frames"
    # indent=2 as the original; the loader does str(id).
    common.write_json(out / "frames_split.json", questions, indent=2)
    common.write_json(out / "article_index.json", build_article_index(questions), indent=2)
    common.require_split_ids([str(q["id"]) for q in questions],
                             common.REPO_ROOT / "configs" / "splits" / "frames_full824_norm.json")
    print(f"wrote {len(questions)} questions -> {out / 'frames_split.json'}")
    if len(questions) != N_QUESTIONS:
        print(f"WARNING: {len(questions)} questions, FRAMES has {N_QUESTIONS}")
    if args.dev_sample > 0:
        dev = sample_dev_subset(questions, args.dev_sample, seed=args.seed)
        common.write_json(out / "frames_dev.json", dev, indent=2)
        print(f"wrote {len(dev)} dev questions -> {out / 'frames_dev.json'}")
    return 0


def cmd_corpus(args: argparse.Namespace) -> int:
    return common.corpus_command(args, ARTEFACT)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=None, help="data root (default $PANINI_DATA_DIR or ./data)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = (f"build frames/frames_split.json (824 questions, ids 0-823) and frames/article_index.json "
         f"from the official {FRAMES_REPO} test.tsv")
    p = sub.add_parser("questions", help=h, description=h[0].upper() + h[1:] + ".")
    p.add_argument("--tsv", default=None, help=f"local test.tsv (default: download {FRAMES_REPO})")
    p.add_argument("--dev-sample", type=int, default=0,
                   help="also write frames_dev.json with N stratified questions")
    p.add_argument("--seed", type=int, default=42, help="seed of the --dev-sample draw (default 42)")
    p.set_defaults(fn=cmd_questions)
    h = (f"install the paper's frozen corpus and its dense index from {ARTEFACT.repo_id}, "
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
