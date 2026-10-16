#!/usr/bin/env python3
"""Fetch the benchmark authors' grading code that `panini-dr score` imports.

Neither is redistributed in this repository (see src/panini_dr/eval/official.py):

* BrowseComp-Plus (MIT): GRADER_TEMPLATE / parse_judge_response and the qrels,
  checked out at the commit the paper's grading used.
* MoNaCo: the official V2 LLM-judge prompts and scorer. The upstream repository
  states no code licence, so it is fetched here rather than vendored. Its
  scorer and prompts are fingerprinted at load time against the version the
  paper used; the commit is not pinned (it was not recorded), pass --monaco-ref
  to choose one.

    python scripts/fetch_grader_sources.py --dest third_party
    export BCP_REPO=$PWD/third_party/BrowseComp-Plus      # the script prints these
    export MONACO_JUDGE_DIR=$PWD/third_party/monaco       # lines with absolute paths

Needs git and network access. Writes only under --dest (gitignored as third_party/).
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

BCP_URL = "https://github.com/texttron/BrowseComp-Plus"
BCP_COMMIT = "046949032b03"
MONACO_URL = "https://github.com/tomerwolgithub/monaco"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def fetch(url: str, dest: Path, ref: str | None) -> None:
    if not (dest / ".git").is_dir():
        _run(["git", "clone", "--quiet", url, str(dest)])
    if ref:
        _run(["git", "-C", str(dest), "fetch", "--quiet", "origin"])
        _run(["git", "-C", str(dest), "checkout", "--quiet", ref])
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"{dest}: {head}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="third_party", help="directory to clone into")
    ap.add_argument("--skip-bcp", action="store_true")
    ap.add_argument("--skip-monaco", action="store_true")
    ap.add_argument("--bcp-ref", default=BCP_COMMIT)
    ap.add_argument("--monaco-ref", default=None, help="commit/branch of the MoNaCo repository (default: its default branch)")
    a = ap.parse_args()
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not a.skip_bcp:
        fetch(BCP_URL, dest / "BrowseComp-Plus", a.bcp_ref)
    if not a.skip_monaco:
        fetch(MONACO_URL, dest / "monaco", a.monaco_ref)
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            from panini_dr.eval.official import load_monaco_judge

            j = load_monaco_judge(dest / "monaco", require_fingerprint=False)
            print("MoNaCo judge fingerprint matches the paper's:", j.fingerprint_ok)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not load the MoNaCo judge from {dest / 'monaco'}: {exc}")
    # Absolute paths: the exports must still work after a `cd`.
    exports = []
    if not a.skip_bcp:
        exports.append(f"export BCP_REPO={shlex.quote(str((dest / 'BrowseComp-Plus').resolve()))}")
    if not a.skip_monaco:
        exports.append(f"export MONACO_JUDGE_DIR={shlex.quote(str((dest / 'monaco').resolve()))}")
    if exports:
        print("\n" + "\n".join(exports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
