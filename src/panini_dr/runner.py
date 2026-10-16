"""``panini-dr run``: run PANINI-DR on one benchmark split.

This is the research runner (``run_substitution.py``) cut down to the one
system the paper evaluates. The adapter receives exactly what the paper runs
gave it: :func:`panini_dr.config.resolve_config` expands the preset, the run
refuses to start if that differs from ``configs/paper.json`` (unless drift is
allowed), and the corpus / dense cache / Lucene index are added on top the way
the old runner added them (BrowseComp-Plus is chunked one chunk per document
and its prebuilt dense cache is validated).

Outputs (one directory per run, ``<output>/<system>__<model>__<UTC time>/``):

``cell_result.json``  per-question results + the recorded adapter config
``traces/q_<id>.json``  full trajectory per question (and a Markdown render)
``run.log``           the log
``run_meta.json``     provenance: argv, package version, git commit, drift
``run_health.json``   reranker calls/failures (failures fell back to fusion order)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from panini_dr.config import (
    ResolvedConfig,
    check_pinned_defaults,
    config_drift,
    load_paper_config,
    paper_config_path,
    recorded_removed_values,
    removed_env_vars_set,
    removed_features_path,
)

_log = logging.getLogger("panini_dr.runner")

SYSTEM_ID = "ours_gsw_planner_orchestrator_v1"

#: Environment variables that change the agent's behaviour. All were unset in
#: the paper runs, so setting one is configuration drift.
BEHAVIOUR_ENV_VARS = (
    "GSW_PLANNER_TARGET_HUB_MODE",
    "GSW_STRICT_TARGET_HUB_GATE",
)


BCP_BENCHMARK_NAMES = ("browsecomp_plus", "browsecomp-plus", "bcp")

#: The LLM key when neither --api-key nor PANINI_LLM_API_KEY gives one. vLLM
#: started without --api-key accepts any key; OPENAI_API_KEY is never used.
PLACEHOLDER_LLM_API_KEY = "dummy-for-vllm"


class RunConfigError(RuntimeError):
    """The run is misconfigured (missing input, or drift without permission)."""


class RunFailedError(RuntimeError):
    """The run finished but produced nothing (every question ended with zero turns)."""


@dataclass
class DatasetSpec:
    """Everything dataset-specific a run needs."""

    benchmark: str
    subset_id: str
    questions: list[Any]
    corpus_path: Path
    chunking: str = "paragraph"  # "whole_doc" for BrowseComp-Plus
    dense_cache: Path | None = None
    lucene_index: Path | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    #: The question file graders must read (bring-your-own data only; the paper
    #: benchmarks use their default question files). Not recorded anywhere.
    questions_file: Path | None = None


def env_drift() -> dict[str, str]:
    return {k: os.environ[k] for k in BEHAVIOUR_ENV_VARS if os.environ.get(k, "").strip()}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def load_benchmark_dataset(
    *,
    subset_path: str | Path,
    articles_cache: str | Path,
    split: str = "full",
    dense_cache: str | Path | None = None,
    lucene_index: str | Path | None = None,
    limit: int = 0,
) -> DatasetSpec:
    """A paper benchmark (or any subset file): questions + corpus paths."""
    from panini_dr.eval.subset import load_benchmark_questions, load_subset

    subset_path = Path(subset_path)
    if not subset_path.is_file():
        raise RunConfigError(f"subset file not found: {subset_path}")
    if not articles_cache:
        raise RunConfigError(
            "--articles-cache is required: the {title: text} corpus JSON for this "
            "benchmark (see datasets/README.md). There is no default corpus."
        )
    corpus_path = Path(articles_cache)
    if not corpus_path.is_file():
        raise RunConfigError(f"corpus file not found: {corpus_path}")
    try:
        subset = load_subset(subset_path)
    except (OSError, ValueError) as exc:  # pydantic's ValidationError is a ValueError
        raise RunConfigError(f"cannot read the subset file {subset_path}: {exc}") from exc
    try:
        questions = load_benchmark_questions(subset, split=split)
    except (OSError, ValueError, KeyError) as exc:
        raise RunConfigError(f"cannot load the questions of {subset_path}: {exc}") from exc
    if limit > 0:
        questions = questions[:limit]
    benchmark = (subset.benchmark or "frames").lower()
    return DatasetSpec(
        benchmark=subset.benchmark,
        subset_id=subset.subset_id,
        questions=questions,
        corpus_path=corpus_path,
        # BrowseComp-Plus: one chunk per document, matching Tevatron's
        # one-vector-per-document index (as the research runner did).
        chunking="whole_doc" if benchmark in BCP_BENCHMARK_NAMES else "paragraph",
        dense_cache=Path(dense_cache) if dense_cache else None,
        lucene_index=Path(lucene_index) if lucene_index else None,
        meta={"subset_file": str(subset_path), "split": split},
    )


def load_byod_dataset(root: str | Path, *, limit: int = 0) -> DatasetSpec:
    """A directory written by ``panini-dr prepare``."""
    from panini_dr.byod import PreparedDatasetError, load_prepared
    from panini_dr.eval.subset import load_benchmark_questions, load_subset

    try:
        ds = load_prepared(root)  # checks every file and the dense index's provenance
    except PreparedDatasetError as exc:
        raise RunConfigError(str(exc)) from exc
    try:
        subset = load_subset(ds.subset_path)
        subset.dataset_path = str(ds.questions_path)
        questions = load_benchmark_questions(subset)
    except (OSError, ValueError, KeyError) as exc:
        raise RunConfigError(f"cannot load the questions of {ds.root}: {exc}") from exc
    if limit > 0:
        questions = questions[:limit]
    return DatasetSpec(
        benchmark="byod",
        subset_id=subset.subset_id,
        questions=questions,
        corpus_path=ds.corpus_path,
        chunking=ds.chunking,
        dense_cache=ds.dense_cache,
        meta={"dataset_dir": str(ds.root), "answer_types": ds.manifest.get("answer_types")},
        questions_file=ds.questions_path,
    )


def load_corpus(spec: DatasetSpec):
    from panini_dr.byod import chunking_kwargs
    from panini_dr.retrieval.corpus import load_frames_corpus

    # list_aware=False: the paper's paragraph chunker (2000/200), or one chunk
    # per document for BrowseComp-Plus (BCP_SINGLE_DOCUMENT_CHARS, overlap 0).
    return load_frames_corpus(
        cache_path=spec.corpus_path, list_aware=False, **chunking_kwargs(spec.chunking),
        benchmark=spec.benchmark, dataset_name=spec.subset_id,
    )


def build_extra(resolved: ResolvedConfig, spec: DatasetSpec, corpus: Any) -> dict[str, Any]:
    """``AdapterContext.extra``: the resolved config plus the dataset keys.

    The removed keys the paper's recorded config carries are added at their
    paper values (``configs/removed_features.json``). No code reads them; they
    are there so that the config a run records in ``cell_result.json`` is the
    paper's recorded config, key for key.
    """
    extra = dict(resolved.extra)
    extra.update(recorded_removed_values())
    extra["corpus"] = corpus
    if spec.dense_cache:
        extra["dense_cache"] = str(spec.dense_cache)
        if spec.chunking == "whole_doc":
            from panini_dr.eval.bcp_dense import validate_prebuilt_dense_cache

            validate_prebuilt_dense_cache(corpus, spec.dense_cache)
        if spec.benchmark == "byod":  # the index must be the one encoded from these chunks
            from panini_dr.byod import PreparedDatasetError, check_dense_cache, load_prepared

            try:
                check_dense_cache(load_prepared(spec.meta["dataset_dir"], verify=False),
                                  [c.chunk_id for c in corpus.chunks])
            except PreparedDatasetError as exc:
                raise RunConfigError(str(exc)) from exc
    if spec.lucene_index:
        extra["lucene_index"] = str(spec.lucene_index)
    return extra


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_state() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    if not (repo / ".git").exists():
        return {}
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return {}


def _redact_argv(argv: list[str]) -> list[str]:
    out, hide = [], False
    for a in argv:
        if hide:
            out.append("<redacted>")
            hide = False
            continue
        if a.startswith("--api-key="):
            out.append("--api-key=<redacted>")
            continue
        out.append(a)
        hide = a == "--api-key"
    return out


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


#: Seconds the LLM preflight waits for ``GET {base_url}/models``.
LLM_PREFLIGHT_TIMEOUT_S = 20.0


def llm_preflight(base_url: str, api_key: str, model: str, *,
                  timeout: float = LLM_PREFLIGHT_TIMEOUT_S) -> str | None:
    """One cheap request (``GET {base_url}/models``) before any question runs.

    Returns what is wrong, or None. Without it an unreachable server turns every
    question into a zero-turn failure and the run still looks finished.
    """
    import httpx

    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - connection refused, DNS, timeout, bad URL
        return f"cannot reach the LLM server: GET {url} failed ({type(exc).__name__}: {exc})"
    if resp.status_code in (401, 403):
        return (f"the LLM server refused the API key (GET {url}: HTTP {resp.status_code}); "
                "pass --api-key or set PANINI_LLM_API_KEY")
    if resp.status_code != 200:
        return (f"GET {url} returned HTTP {resp.status_code}; is {base_url} the server's "
                "OpenAI-compatible base URL (it usually ends in /v1)?")
    try:
        served = [m.get("id") for m in resp.json().get("data") or [] if isinstance(m, dict)]
    except Exception:  # noqa: BLE001 - not an OpenAI model list; the server answered
        served = []
    if served and model not in served:
        return (f"the LLM server at {base_url} does not serve {model!r} (it serves "
                f"{', '.join(map(str, served[:5]))}); pass --model or set PANINI_LLM_MODEL "
                "to the served name")
    return None


def rerank_preflight(spec: str) -> str | None:
    """One rerank request before any question runs. Returns an error or None."""
    from panini_dr.retrieval.rerank import maybe_wrap_reranker

    try:
        wrapped = maybe_wrap_reranker(object(), spec)
        wrapped.reranker.rerank("preflight", ["first document", "second document"], top_k=2)
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


def run_experiment(
    resolved: ResolvedConfig,
    spec: DatasetSpec,
    *,
    model: str,
    base_url: str,
    api_key: str = "",
    output_dir: str | Path = "runs",
    concurrency: int = 32,
    allow_config_drift: bool = False,
    strict_rerank: bool = False,
    skip_preflight: bool = False,
    skip_rerank_preflight: bool = False,
    verbose: bool = False,
    argv: list[str] | None = None,
    echo: Callable[[str], None] = print,
) -> Path:
    """Run every question of ``spec``; returns the run directory.

    ``skip_preflight`` skips both test requests (LLM and reranker) that are sent
    before any question; ``skip_rerank_preflight`` only the reranker's. Raises
    :class:`RunFailedError` when every question ended with zero turns.
    """
    api_key = api_key or PLACEHOLDER_LLM_API_KEY
    skip_rerank_preflight = skip_rerank_preflight or skip_preflight
    import panini_dr.adapters.ours.gsw_planner_orchestrator_v1  # noqa: F401  (registers the adapter)
    from panini_dr.adapters.base import AdapterContext, get_adapter
    from panini_dr.eval.harness import persist_cell, run_cell
    from panini_dr.retrieval import rerank
    from panini_dr.utils import setup_run_logger

    paper = load_paper_config()
    gone = removed_env_vars_set()
    if gone:
        raise RunConfigError(
            "environment variables that switched removed code are set: "
            + "; ".join(f"{k} ({why})" for k, why in gone.items())
            + ". Unset them (see configs/removed_features.json)."
        )
    drift = config_drift(resolved.flat(), paper)
    envd = env_drift()
    if (drift or envd) and not allow_config_drift:
        raise RunConfigError(_drift_message(drift, envd))
    if model != paper.get("model"):
        echo(f"note: model {model!r} differs from the paper's {paper.get('model')!r}")
    if base_url and not skip_preflight:
        err = llm_preflight(base_url, api_key, model)
        if err:
            raise RunConfigError(
                f"LLM preflight failed: {err}. No question was run "
                "(--skip-preflight skips this check)."
            )

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    cell_dir = Path(output_dir) / f"{SYSTEM_ID}__{model.replace('/', '_')}__{run_tag}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    setup_run_logger(cell_dir, verbose=verbose)

    meta: dict[str, Any] = {
        "panini_dr_version": _version(),
        "git": _git_state(),
        "argv": _redact_argv(list(argv if argv is not None else sys.argv)),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "system": SYSTEM_ID,
        "model": model,
        "base_url": base_url,
        "benchmark": spec.benchmark,
        "subset_id": spec.subset_id,
        "n_questions": len(spec.questions),
        "concurrency": concurrency,
        "dataset": {
            "corpus": str(spec.corpus_path),
            "chunking": spec.chunking,
            "dense_cache": str(spec.dense_cache) if spec.dense_cache else None,
            "lucene_index": str(spec.lucene_index) if spec.lucene_index else None,
            **spec.meta,
        },
        "paper_config_sha256": _sha256_file(paper_config_path()),
        "removed_features_sha256": _sha256_file(removed_features_path()),
        "config_drift": {k: list(v) for k, v in drift.items()},
        "behaviour_env_vars": envd,
        "strict_rerank": strict_rerank,
        "skip_preflight": skip_preflight,
        "skip_rerank_preflight": skip_rerank_preflight,
    }
    meta_path = cell_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    rerank.RERANK_HEALTH.reset()
    health_path = cell_dir / "run_health.json"

    def _write_health(extra: dict[str, Any] | None = None) -> dict[str, Any]:
        h = {"rerank": rerank.RERANK_HEALTH.snapshot(), **(extra or {})}
        health_path.write_text(json.dumps(h, indent=2) + "\n")
        return h

    reranker_spec = resolved.extra.get("reranker")
    if strict_rerank:

        def _abort(exc: BaseException, snapshot: dict[str, Any]) -> None:
            _write_health({"aborted": f"strict rerank: {type(exc).__name__}: {exc}"})
            _log.error("--strict-rerank: rerank failed (%s); aborting the run", exc)
            for h in logging.getLogger("panini_dr").handlers:
                try:
                    h.flush()
                except Exception:  # noqa: BLE001
                    pass
            os._exit(3)

        rerank.set_strict(True, on_failure=_abort)
    if reranker_spec and not skip_rerank_preflight:
        err = rerank_preflight(reranker_spec)
        if err:
            if strict_rerank:
                _write_health({"aborted": f"reranker preflight failed: {err}"})
                raise RunConfigError(f"reranker preflight failed ({err}); --strict-rerank set")
            echo(
                "WARNING: reranker preflight failed "
                f"({err}). Every search will fall back to fusion order and the run "
                "will NOT reproduce the paper configuration. Use --strict-rerank to abort."
            )

    try:
        try:
            corpus = load_corpus(spec)
        except (OSError, ValueError) as exc:  # not a {title: text} JSON file
            raise RunConfigError(f"cannot load the corpus {spec.corpus_path}: {exc}") from exc
        echo(f"corpus: {spec.corpus_path} ({len(corpus.chunks)} chunks, chunking={spec.chunking})")
        ctx = AdapterContext(
            system_id=SYSTEM_ID,
            model_id=model,
            model_name=model,
            base_url=base_url,
            api_key=api_key,
            max_turns=resolved.max_turns,
            max_prompt_tokens=resolved.max_prompt_tokens,
            max_completion_tokens=resolved.max_completion_tokens,
            extra=build_extra(resolved, spec, corpus),
        )
        adapter = get_adapter(SYSTEM_ID)(ctx)
        pinned = check_pinned_defaults(adapter, paper)
        if pinned:
            meta["pinned_default_drift"] = {k: list(v) for k, v in pinned.items()}
            meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
            if not allow_config_drift:
                raise RunConfigError(
                    "adapter defaults differ from configs/paper.json pinned_adapter_defaults "
                    f"(pinned, observed): {pinned}"
                )

        cell = run_cell(
            adapter,
            spec.questions,
            subset_id=spec.subset_id,
            benchmark=spec.benchmark,
            raw_trace_dir=cell_dir / "traces",
            judge=None,
            concurrency=concurrency,
        )
        persist_cell(cell, cell_dir / "cell_result.json")
    finally:
        if strict_rerank:
            rerank.set_strict(False)

    health = _write_health()
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")

    n_answered = sum(1 for r in cell.questions if (r.predicted_answer or "").strip())
    echo(f"=== {spec.benchmark} / {spec.subset_id}: {n_answered}/{cell.n_total} answered "
         f"-> {cell_dir}")
    echo("grade with (judge model from PANINI_JUDGE_MODEL or an added --model; judge endpoint "
         "and grader sources as in REPRODUCE.md section 3):\n  " + grade_command(spec, cell_dir))
    rh = health["rerank"]
    echo(f"reranker: {rh['calls']} calls, {rh['failures']} failures")
    if rh["failures"]:
        echo(
            f"WARNING: {rh['failures']} rerank call(s) failed and fell back to fusion order "
            f"(first error: {rh['first_errors'][:1]}). This run does not match the paper "
            "configuration; see run_health.json."
        )
    if cell.questions and all(r.trajectory.turns == 0 for r in cell.questions):
        first = next((str(e) for r in cell.questions
                      for e in (r.trajectory.extra.get("llm_error"),
                                r.trajectory.extra.get("adapter_error")) if e), "none recorded")
        raise RunFailedError(
            f"every question ({len(cell.questions)}) ended with zero turns: no LLM turn "
            f"completed, so {cell_dir} holds no result (first error: {first[:300]}). "
            "Check the LLM server and run.log."
        )
    return cell_dir


def grade_command(spec: DatasetSpec, cell_dir: str | Path) -> str:
    """A complete ``panini-dr score answers`` command for this run, ready to paste.

    The judge model comes from ``PANINI_JUDGE_MODEL`` (or an added ``--model``);
    caches and verdicts go to ``judge/`` as in REPRODUCE.md and docs/own-dataset.md.
    """
    bench = spec.benchmark.lower().replace("-", "_")
    if bench in BCP_BENCHMARK_NAMES or bench == "browsecomp":
        bench = "bcp"
    args = ["panini-dr", "score", "answers", "-b", bench, "--run", str(cell_dir)]
    if bench == "byod":
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.subset_id).strip("._") or "byod"
        questions = spec.questions_file or Path(spec.meta.get("dataset_dir", "<dataset dir>")) / "questions.jsonl"
        args += ["--questions", str(questions)]
    else:
        name = bench
        args += ["--subset", str(spec.meta.get("subset_file", "<split file>"))]
    args += ["--cache", f"judge/{name}_cache.json", "--out", f"judge/{name}_verdicts.json",
             "--backend", "openai"]
    return " ".join(shlex.quote(a) for a in args)


def _drift_message(drift: dict[str, tuple[Any, Any]], envd: dict[str, str]) -> str:
    lines = ["the resolved configuration differs from configs/paper.json:"]
    for k, (want, got) in drift.items():
        lines.append(f"  {k}: paper={want!r} resolved={got!r}")
    for k, v in envd.items():
        lines.append(f"  env {k}={v!r} changes agent behaviour (unset in the paper runs)")
    lines.append("pass --allow-config-drift to run it anyway (it is then not the paper configuration)")
    return "\n".join(lines)


def _version() -> str:
    from panini_dr import __version__

    return __version__


__all__ = [
    "BEHAVIOUR_ENV_VARS",
    "DatasetSpec",
    "RunConfigError",
    "RunFailedError",
    "SYSTEM_ID",
    "build_extra",
    "env_drift",
    "grade_command",
    "load_benchmark_dataset",
    "load_byod_dataset",
    "llm_preflight",
    "load_corpus",
    "run_experiment",
]
