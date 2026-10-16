"""``panini-dr`` command line: ``run``, ``prepare``, ``score``, ``verify``.

Environment (see ``.env.example``). Every command first reads ``.env`` from the
working directory (or the nearest parent that has one); variables already set in
the shell win.

``PANINI_LLM_BASE_URL``  OpenAI-compatible endpoint of the served LLM (``--base-url``)
``PANINI_LLM_MODEL``     served model name (``--model``; default: the paper's)
``PANINI_LLM_API_KEY``   key for that endpoint, if it needs one (``--api-key``). Never
                         taken from ``OPENAI_API_KEY`` (that is the judge's key)
``JINA_RERANK_URL``      jina-reranker-v3.5 endpoint (``.../v1/rerank``)
``PANINI_DATA_DIR``      prepared data root (default ``./data``)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer

from panini_dr.eval.score_cli import score_app
from panini_dr.replay.cli import verify_app

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    # A traceback must never print local variables (they can hold API keys).
    pretty_exceptions_show_locals=False,
    # Plain help text: Rich markup would swallow every [bracketed] part of it
    # ("[env PANINI_LLM_MODEL; default: ...]", "{bucket: [qid]}").
    rich_markup_mode=None,
    help="PANINI-DR: controlled deep research through Generative Semantic Workspaces.",
)
app.add_typer(score_app, name="score")
app.add_typer(verify_app, name="verify")


@app.callback()
def _read_env_file() -> None:
    # Runs before every sub-command parses its options (so .env values also
    # reach options that read the environment).
    from panini_dr.envfile import load_env_file

    load_env_file()


def parse_set_options(items: list[str]) -> dict[str, object]:
    """``key=value`` pairs; values are JSON when they parse (true, 5, "x"), else strings."""
    out: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        out[key] = value
    return out


def _pinned_defaults_about(pinned: dict, flat: dict) -> str:
    """One paragraph for --print-config: which limit is which."""
    return (
        "Values that govern the run but are not part of `config`: the adapter's built-in "
        "defaults, as pinned in configs/paper.json. `panini-dr run` checks them against the "
        "constructed adapter and refuses to start if one differs. Turn limits: orchestrator_max_turns = orchestrator turns per question "
        f"({pinned.get('orchestrator_max_turns')}); level_max_turns = turns per researcher "
        f"({pinned.get('level_max_turns')}); reader_max_turns = turns per article-reader call "
        f"({pinned.get('reader_max_turns')}); MAX_PARALLEL_RESEARCHERS = researchers dispatched at "
        f"once ({pinned.get('MAX_PARALLEL_RESEARCHERS')}). config.max_turns "
        f"({flat.get('max_turns')}) is none of these: it limits only the single-agent fallback a "
        "question falls back to when its plan cannot be built or ordered."
    )


@app.command()
def run(
    subset: Optional[Path] = typer.Option(
        None, "--subset", help="Split file (configs/splits/*.json). Not with --dataset."
    ),
    articles_cache: str = typer.Option(
        "", "--articles-cache", help="Corpus {title: text} JSON for the subset's benchmark."
    ),
    dense_cache: str = typer.Option(
        "", "--dense-cache",
        help="Prebuilt dense index dir. Without it the fingerprinted dir under "
        "$PANINI_DATA_DIR/retrieval_cache is used; a missing index is an error.",
    ),
    lucene_index: str = typer.Option(
        "", "--lucene-index", help="BrowseComp-Plus: Tevatron's Lucene BM25 index dir."
    ),
    dataset: Optional[Path] = typer.Option(
        None, "--dataset", help="A directory written by `panini-dr prepare` (bring your own data)."
    ),
    split: str = typer.Option("full", "--split", help="FRAMES question file: full|dev."),
    model: str = typer.Option(
        "", "--model", "-m", help="Served model name [env PANINI_LLM_MODEL; default: the paper's]."
    ),
    base_url: str = typer.Option(
        "", "--base-url", help="OpenAI-compatible base URL of the LLM [env PANINI_LLM_BASE_URL]."
    ),
    api_key: str = typer.Option(
        "", "--api-key",
        help="API key for the LLM endpoint [env PANINI_LLM_API_KEY; default: a placeholder, "
        "for servers started without a key]. OPENAI_API_KEY is never sent to it.",
    ),
    output_dir: Path = typer.Option(Path("runs"), "--output-dir", help="Where run directories go."),
    concurrency: int = typer.Option(
        0, "--concurrency", help="Questions in parallel (0 = the paper's 32)."
    ),
    limit: int = typer.Option(0, "--limit", help="Run only the first N questions (smoke)."),
    set_: list[str] = typer.Option(
        [], "--set",
        help="Override a config key, e.g. --set gsw_edge_grounded_cap=1. Treated like a flag "
        "typed on the research runner's command line. Needs --allow-config-drift. Keys whose "
        "code was removed (configs/removed_features.json) are rejected.",
    ),
    allow_config_drift: bool = typer.Option(
        False, "--allow-config-drift",
        help="Run even if the resolved config differs from configs/paper.json.",
    ),
    strict_rerank: bool = typer.Option(
        False, "--strict-rerank",
        help="Abort on the first reranker failure instead of falling back to fusion order.",
    ),
    skip_preflight: bool = typer.Option(
        False, "--skip-preflight", "--skip-rerank-preflight",
        help="Do not send the test requests to the LLM (GET /models) and the reranker before "
        "the first question (for offline or replayed runs). --skip-rerank-preflight is an alias.",
    ),
    print_config: bool = typer.Option(
        False, "--print-config",
        help="Print the resolved adapter config, its drift from paper.json and the pinned adapter "
        "defaults (orchestrator, researcher and reader turn limits, ...), then exit.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show DEBUG lines in the terminal too: per-turn tool calls and the streamed LLM "
        "text. run.log always has them.",
    ),
) -> None:
    """Run PANINI-DR (the paper configuration by default) on one split."""
    from panini_dr.config import (
        UnknownConfigKeyError,
        config_drift,
        load_paper_config,
        recorded_removed_values,
        removed_env_vars_set,
        resolve_config,
    )
    from panini_dr.runner import (
        RunConfigError,
        RunFailedError,
        env_drift,
        load_benchmark_dataset,
        load_byod_dataset,
        run_experiment,
    )

    paper = load_paper_config()
    try:
        resolved = resolve_config(parse_set_options(set_))
    except UnknownConfigKeyError as exc:
        raise typer.BadParameter(str(exc.args[0])) from exc

    if print_config:
        flat = resolved.flat()
        if dense_cache:
            flat["dense_cache"] = dense_cache
        if lucene_index:
            flat["lucene_index"] = lucene_index
        pinned = dict(paper.get("pinned_adapter_defaults", {}))
        typer.echo(json.dumps(
            {
                "config": flat,
                "removed_features": recorded_removed_values(),
                "drift_from_paper": {k: list(v) for k, v in config_drift(resolved.flat(), paper).items()},
                "behaviour_env_vars": env_drift(),
                "removed_env_vars_set": removed_env_vars_set(),
                "paper_config": "configs/paper.json",
                "pinned_adapter_defaults": pinned,
                "pinned_adapter_defaults_about": _pinned_defaults_about(pinned, flat),
            },
            indent=2, sort_keys=True,
        ))
        return

    model = model or os.environ.get("PANINI_LLM_MODEL", "") or paper["model"]
    base_url = base_url or os.environ.get("PANINI_LLM_BASE_URL", "")
    if not base_url:
        raise typer.BadParameter(
            "no LLM endpoint: set PANINI_LLM_BASE_URL or pass --base-url "
            "(the OpenAI-compatible /v1 URL of the vLLM server from serve/gpt_oss_120b.sh)"
        )
    # Only --api-key / PANINI_LLM_API_KEY: OPENAI_API_KEY (the judge's key) must
    # never reach the LLM server. run_experiment substitutes a placeholder.
    api_key = api_key or os.environ.get("PANINI_LLM_API_KEY", "")

    try:
        if dataset is not None:
            if subset is not None or articles_cache:
                raise RunConfigError("--dataset cannot be combined with --subset/--articles-cache")
            spec = load_byod_dataset(dataset, limit=limit)
            if spec.dense_cache is None and resolved.extra.get("retriever_type") != "bm25":
                raise RunConfigError(
                    f"{dataset} was prepared with --no-dense; it can only run BM25-only: "
                    "--set retriever_type=bm25 --allow-config-drift"
                )
        else:
            if subset is None:
                raise RunConfigError("pass --subset (a configs/splits/*.json file) or --dataset")
            spec = load_benchmark_dataset(
                subset_path=subset,
                articles_cache=articles_cache,
                split=split,
                dense_cache=dense_cache or None,
                lucene_index=lucene_index or None,
                limit=limit,
            )
        typer.echo(f"model={model} base_url={base_url} benchmark={spec.benchmark} "
                   f"subset={spec.subset_id} n={len(spec.questions)}")
        run_experiment(
            resolved,
            spec,
            model=model,
            base_url=base_url,
            api_key=api_key,
            output_dir=output_dir,
            concurrency=concurrency or int(paper.get("concurrency", 32)),
            allow_config_drift=allow_config_drift,
            strict_rerank=strict_rerank,
            skip_preflight=skip_preflight,
            verbose=verbose,
            argv=sys.argv,
            echo=typer.echo,
        )
    except RunConfigError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    except RunFailedError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def prepare(
    questions: Optional[Path] = typer.Option(None, "--questions", help="questions.jsonl"),
    docs: Optional[Path] = typer.Option(None, "--docs", help="docs.jsonl"),
    out: Path = typer.Option(..., "--out", help="Output directory for the prepared dataset."),
    config: Optional[Path] = typer.Option(None, "--config", help="Optional dataset.yaml."),
    name: str = typer.Option("", "--name", help="Dataset name (default: from dataset.yaml or --out)."),
    chunking: str = typer.Option(
        "", "--chunking",
        help="paragraph (2000/200 chars, as FRAMES/MoNaCo; default) or whole_doc (one chunk per doc).",
    ),
    no_dense: bool = typer.Option(
        False, "--no-dense",
        help="Skip the Qwen3-Embedding-8B index (GPU). The dataset can then only run BM25-only.",
    ),
    device: str = typer.Option("", "--device", help="Encoder device, e.g. cuda:0 (default: freest GPU)."),
    batch_size: int = typer.Option(16, "--batch-size", help="Encoder batch size."),
) -> None:
    """Build a dataset directory (corpus, questions, dense index) from your own JSONL files."""
    from panini_dr.byod import load_dataset_yaml, prepare_dataset
    from panini_dr.eval.byod_dataset import ByodSchemaError

    try:
        cfg = load_dataset_yaml(config) if config else {}
    except Exception as exc:  # noqa: BLE001 - missing file, YAML syntax, unknown keys
        typer.echo(f"error: cannot read {config}: {exc}", err=True)
        raise typer.Exit(2) from exc
    questions = questions or cfg.get("questions")
    docs = docs or cfg.get("docs")
    if not questions or not docs:
        raise typer.BadParameter("need --questions and --docs (or a dataset.yaml naming them)")
    try:
        manifest = prepare_dataset(
            questions=questions,
            docs=docs,
            out=out,
            name=name or cfg.get("name") or None,
            chunking=chunking or cfg.get("chunking") or "paragraph",
            dense=not no_dense,
            device=device or None,
            batch_size=batch_size,
            log=typer.echo,
        )
    except (ByodSchemaError, OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps({k: manifest[k] for k in ("name", "n_questions", "n_documents",
                                                     "n_chunks", "chunking", "dense")}, indent=2))


def main() -> None:  # console-script entry point
    app()


if __name__ == "__main__":
    main()
