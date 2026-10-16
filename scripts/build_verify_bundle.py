#!/usr/bin/env python
"""Maintainers: package recorded runs as a ``panini-dr verify`` bundle.

    python scripts/build_verify_bundle.py --out BUNDLE_DIR --data-dir DATA_DIR --name NAME \\
        --case NAME=CASSETTE_DIR,RUN_DIR,SUBSET_FILE,CORPUS_REL [--case ...] \\
        [--reader-tokenizer tiktoken:o200k_base] [--source-root PATH ...] \\
        [--deny LITERAL ...] [--deny-regex REGEX ...] [--description TEXT] [--recorded-with TEXT]

Each ``--case`` is one recording:

  CASSETTE_DIR  written by ``panini-dr verify record --cassette CASSETTE_DIR -- ...`` (or by the
                research harness under the same launcher)
  RUN_DIR       that recording's ``--output-dir`` (its outputs become the reference)
  SUBSET_FILE   the subset file the recording ran
  CORPUS_REL    the corpus path relative to DATA_DIR, e.g. ``frames/HYBRID_FRAMES824.json``
                (``panini-dr verify`` looks for it under the user's data dir)

The bundle gets the scrubbed cassettes, the NORMALISED reference outputs, the subset files and
a manifest with the sha256 of every file and of the data it needs (corpus, dense index chunk
ids, each question's text and gold answer). Nothing machine- or lab-specific may remain: the
finished bundle is scanned for absolute paths, IP:port pairs, the input paths, this machine's
host and user name, every service host the recordings talked to and the ``--deny`` literals /
regexes given here; any hit aborts the build (exit 1) and lists the offending strings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from panini_dr.replay.bundle import MANIFEST, BundleError, CaseSource, build_bundle


def _case(spec: str) -> CaseSource:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--case NAME=CASSETTE_DIR,RUN_DIR,SUBSET_FILE,CORPUS_REL")
    name, rest = spec.split("=", 1)
    parts = rest.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"--case {spec!r}: expected 4 comma-separated fields after NAME=")
    cassette, run_dir, subset, corpus_rel = parts
    return CaseSource(name=name, cassette=Path(cassette), run_dir=Path(run_dir), subset=Path(subset),
                      articles_cache=corpus_rel)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path, help="new bundle directory")
    ap.add_argument("--data-dir", required=True, type=Path, help="data root the recordings read (for the data pins)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--recorded-with", default="", help="one line on the code / setup that made the recordings")
    ap.add_argument("--case", action="append", type=_case, required=True, dest="cases")
    ap.add_argument("--split", default="full", help="question split of every case (FRAMES: full|dev)")
    ap.add_argument("--reader-tokenizer", default=None, choices=["tiktoken:o200k_base", "heuristic"],
                    help="what the recordings' article reader counted tokens with, when their metadata does not say")
    ap.add_argument("--source-root", action="append", default=[], type=Path,
                    help="source tree the recordings ran from (masked as <SRC> in the reference outputs)")
    ap.add_argument("--deny", action="append", default=[], help="extra literal the bundle must not contain")
    ap.add_argument("--deny-regex", action="append", default=[], help="extra regex the bundle must not match")
    a = ap.parse_args(argv)
    for c in a.cases:
        c.split = a.split
    try:
        m = build_bundle(a.out, a.cases, data_dir=a.data_dir, name=a.name, description=a.description,
                         reader_tokenizer=a.reader_tokenizer, source_roots=a.source_root,
                         deny_literals=a.deny, deny_regexes=a.deny_regex, recorded_with=a.recorded_with)
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    size = sum(f["bytes"] for f in m["files"].values())
    print(json.dumps({"bundle": str(a.out), "cases": [c["name"] for c in m["cases"]],
                      "questions": sum(len(c["questions"]) for c in m["cases"]),
                      "files": len(m["files"]) + 1, "bytes": size + (a.out / MANIFEST).stat().st_size,
                      "data": list(m["data"])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
