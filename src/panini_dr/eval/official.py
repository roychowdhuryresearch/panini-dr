"""Load the benchmark authors' official grading code from user checkouts.

Neither piece is redistributed in this repository:

* **BrowseComp-Plus** (texttron/BrowseComp-Plus, MIT): ``GRADER_TEMPLATE`` and
  ``parse_judge_response`` are imported from ``scripts_evaluation/evaluate_run.py``
  of a checkout at :data:`BCP_PINNED_COMMIT` (the commit the paper's grading
  imported from). Point ``BCP_REPO`` (or ``--bcp-repo``) at it;
  ``scripts/fetch_grader_sources.py`` clones it.
* **MoNaCo** (tomerwolgithub/monaco): the official V2 LLM-judge prompts and
  ``compute_llm_judge_score_V2``. The upstream repository states no licence for
  its code, so it is not vendored. Point ``MONACO_JUDGE_DIR`` (or
  ``--monaco-judge-dir``) at a checkout. The function and the two prompt
  strings are extracted by parsing (nothing else in the checkout is executed)
  and fingerprinted against the copy the paper's numbers were produced with.

Graders accept these pieces as parameters, so unit tests inject stand-ins.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import os
import sys
import types
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: texttron/BrowseComp-Plus commit the paper's BCP/FRAMES grading imported from.
BCP_PINNED_COMMIT = "046949032b03"

# Fingerprints of the MoNaCo V2 judge as used for the paper (sha256 of the
# function's source segment / of its AST dump on Python 3.12, and of each
# prompt string). Hashes only -- the text itself is not redistributed.
MONACO_FN_SOURCE_SHA256 = "39ee5009f55ce89b5aa55ec8c5485371e2a816289cf144c52e4b40f1f05abc8f"
MONACO_FN_AST_SHA256 = "3392112ec445fcd2bed3fb88e0e2fe26fd06b31f5996c17fa384e91fa31be1a9"
MONACO_SINGLE_PROMPT_SHA256 = "d7ca4a2d15562ff9b790f7ce9563f411dd3051a1ea13de2d78293b9e4848a167"
MONACO_MULTI_PROMPT_SHA256 = "ed3febe00ee79ec4d1ee7d9ae0eee6d077d2a91e95e0ff014a9dfa36f6ba5949"


class OfficialSourceError(RuntimeError):
    pass


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# BrowseComp-Plus
# ---------------------------------------------------------------------------


@dataclass
class BCPGrader:
    template: str
    parse: Callable[[str], dict[str, Any]]
    source: str = ""


def load_bcp_grader(bcp_repo: str | Path | None = None) -> BCPGrader:
    """Import GRADER_TEMPLATE / parse_judge_response from a BrowseComp-Plus checkout."""
    repo = bcp_repo or os.environ.get("BCP_REPO")
    if not repo:
        raise OfficialSourceError(
            "the BrowseComp-Plus grader template is imported from the authors' code: "
            "set BCP_REPO (or --bcp-repo) to a texttron/BrowseComp-Plus checkout at "
            f"commit {BCP_PINNED_COMMIT} (scripts/fetch_grader_sources.py clones it)"
        )
    repo = str(Path(repo).expanduser().resolve())
    if not (Path(repo) / "scripts_evaluation" / "evaluate_run.py").is_file():
        raise OfficialSourceError(f"{repo} has no scripts_evaluation/evaluate_run.py")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    # evaluate_run.py imports vllm at module top for its local-judge path, which
    # is not used here; stub it so the import succeeds. Everything used below
    # is the authors' code, loaded verbatim.
    sys.modules.setdefault("vllm", types.SimpleNamespace(LLM=None, SamplingParams=None))
    # Compiling the upstream file (on the first import, before Python caches its
    # bytecode) warns about an invalid escape sequence in one of its strings. That
    # is the authors' code, not something a user can act on; hide that warning,
    # for this import only.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"invalid escape sequence", category=SyntaxWarning)
        mod = importlib.import_module("scripts_evaluation.evaluate_run")
    return BCPGrader(template=mod.GRADER_TEMPLATE, parse=mod.parse_judge_response, source=repo)


# ---------------------------------------------------------------------------
# MoNaCo
# ---------------------------------------------------------------------------


@dataclass
class MonacoJudge:
    single_prompt: str
    multi_prompt: str
    score: Callable[[str, int], dict[str, Any] | None]
    fingerprint_ok: bool = True
    source: str = ""


def _scan(root: Path) -> tuple[str | None, dict[str, str], str]:
    """Find the V2 scoring function source and the two prompt strings under ``root``."""
    fn_src: str | None = None
    fn_file = ""
    prompts: dict[str, str] = {}
    for py in sorted(root.rglob("*.py")):
        try:
            text = py.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "compute_llm_judge_score_V2" and fn_src is None:
                fn_src = ast.get_source_segment(text, node)
                fn_file = str(py)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in ("single_answer_llm_judge_prompt", "multi_answer_llm_judge_prompt") and name not in prompts:
                    try:
                        val = ast.literal_eval(node.value)
                    except ValueError:
                        continue
                    if isinstance(val, str):
                        prompts[name] = val
    return fn_src, prompts, fn_file


def load_monaco_judge(judge_dir: str | Path | None = None, *, require_fingerprint: bool = True) -> MonacoJudge:
    """Extract the official MoNaCo V2 judge from a checkout of the authors' repository."""
    d = judge_dir or os.environ.get("MONACO_JUDGE_DIR")
    if not d:
        raise OfficialSourceError(
            "the MoNaCo V2 judge is the authors' code and is not redistributed: set "
            "MONACO_JUDGE_DIR (or --monaco-judge-dir) to a checkout of the MoNaCo "
            "repository (scripts/fetch_grader_sources.py clones it)"
        )
    root = Path(d).expanduser()
    fn_src, prompts, fn_file = _scan(root)
    if fn_src is None or len(prompts) != 2:
        raise OfficialSourceError(
            f"could not find compute_llm_judge_score_V2 and both judge prompts under {root}"
        )
    fn_tree = ast.parse(fn_src)
    fn_node = fn_tree.body[0]
    ok = (
        (_sha(fn_src) == MONACO_FN_SOURCE_SHA256 or _sha(ast.dump(fn_node)) == MONACO_FN_AST_SHA256)
        and _sha(prompts["single_answer_llm_judge_prompt"]) == MONACO_SINGLE_PROMPT_SHA256
        and _sha(prompts["multi_answer_llm_judge_prompt"]) == MONACO_MULTI_PROMPT_SHA256
    )
    if not ok and require_fingerprint:
        raise OfficialSourceError(
            f"the MoNaCo judge under {root} differs from the version the paper used "
            "(fingerprint mismatch); pass --allow-judge-mismatch to grade with it anyway"
        )
    ns: dict[str, Any] = {}
    exec(compile(fn_tree, fn_file or "<monaco_judge>", "exec"), ns)  # noqa: S102 - authors' scorer
    return MonacoJudge(
        single_prompt=prompts["single_answer_llm_judge_prompt"],
        multi_prompt=prompts["multi_answer_llm_judge_prompt"],
        score=ns["compute_llm_judge_score_V2"],
        fingerprint_ok=ok,
        source=str(root),
    )


__all__ = [
    "BCPGrader",
    "BCP_PINNED_COMMIT",
    "MonacoJudge",
    "OfficialSourceError",
    "load_bcp_grader",
    "load_monaco_judge",
]
