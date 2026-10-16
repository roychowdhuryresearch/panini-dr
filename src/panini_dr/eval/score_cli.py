"""`panini-dr score`: grade run outputs with the graders of record.

    panini-dr score answers --benchmark bcp --run runs/<cell dir> [--run <second leg>] \\
        --subset configs/splits/cf_bcp150.json --cache judge/bcp_cache.json \\
        --backend bedrock --model my-judge-model --out judge/bcp_verdicts.json

    panini-dr score obs --benchmark bcp --run runs/<cell dir> --verdicts judge/bcp_verdicts.json \\
        --subset configs/splits/cf_bcp150.json --corpus <HYBRID_BCP.json> --bcp-repo <checkout>

    panini-dr score answers ... --offline        # cache only: a missing verdict is an error

    panini-dr score import-legacy-cache --benchmark bcp --legacy-cache <old cache.json> \\
        --legacy-name <old run name> --run <cell dir> [--run ...] --subset <split> --cache <new cache>

Grader per benchmark (paper behaviour): ``bcp`` and ``frames`` -> BrowseComp-Plus
GRADER_TEMPLATE, majority of 3; ``monaco`` -> MoNaCo V2 judge, max of 3 over all
questions. The judge model is yours to choose and has no default: pass
``--model`` or set ``PANINI_JUDGE_MODEL`` (the name is part of every cache key). Bring-your-own (``byod``) questions are routed by ``answer_type``
(``short`` / ``set``). Accuracy and F1 always use every question of the split as
the denominator.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import typer

from panini_dr.eval.grade_set import SetItem, grade_set, summarize_set
from panini_dr.eval.grade_short import BLANK_POLICY, ShortItem, grade_short, summarize_short
from panini_dr.eval.judge_cache import CacheMissError, JudgeCache
from panini_dr.eval.judges import JudgeModelMissingError, make_backend, resolve_judge_model
from panini_dr.eval.predictions import (
    default_questions_path,
    load_gold,
    load_predictions,
    load_subset_ids,
    normalize_benchmark,
)

score_app = typer.Typer(add_completion=False, help="Grade run outputs with the graders of record.")

# Injection point for tests (and for callers embedding the grader): when set,
# called as factory(kind, model) instead of make_backend(...).
_BACKEND_FACTORY = None


@contextmanager
def _user_errors(what: str) -> Iterator[None]:
    """Report a bad input (missing or unreadable file, wrong value, missing grader
    checkout) as ``error: <what>: ...`` with exit status 2 instead of a traceback."""
    from panini_dr.eval.judges import JudgeBackendError
    from panini_dr.eval.official import OfficialSourceError

    try:
        yield
    except (typer.Exit, typer.BadParameter):
        raise
    except KeyError as exc:
        typer.echo(f"error: {what}: missing field {exc}", err=True)
        raise typer.Exit(2) from exc
    except (OSError, ValueError, OfficialSourceError, JudgeBackendError) as exc:
        typer.echo(f"error: {what}: {exc}", err=True)
        raise typer.Exit(2) from exc


def _backend(kind, model, base_url, api, region):
    if _BACKEND_FACTORY is not None:
        return _BACKEND_FACTORY(kind, model)
    with _user_errors("cannot set up the judge"):
        return make_backend(kind, model, base_url=base_url or None, api=api, region=region or None)


def _benchmark(name: str) -> str:
    try:
        return normalize_benchmark(name)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--benchmark") from exc


_MODEL_HELP = "Judge model name (no default; else PANINI_JUDGE_MODEL). Part of every cache key."


def _judge_model(model: Optional[str]) -> str:
    """``--model`` or ``$PANINI_JUDGE_MODEL``; a clear error (exit 2) when neither is set."""
    try:
        return resolve_judge_model(model)
    except JudgeModelMissingError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc


def _buckets_for(subset_id: str, buckets: Optional[Path]) -> dict | None:
    if buckets:
        return json.loads(Path(buckets).read_text())
    if subset_id == "cf_bcp150":
        from panini_dr.config import configs_dir

        p = configs_dir() / "splits" / "cf_bcp150_buckets.json"
        if p.is_file():
            return json.loads(p.read_text())
    return None


def _short_gold(answer) -> str:
    return str(answer) if not isinstance(answer, list) else "; ".join(map(str, answer))


class _OfflineBackend:
    """Stands in for a judge in offline mode; a call is a bug (misses raise first)."""

    name = "offline"

    def __init__(self, model: str) -> None:
        self.model = model

    def complete(self, prompt: str) -> str:  # pragma: no cover - guarded by CacheMissError
        raise RuntimeError("offline mode: a judge call was attempted")


def _load_inputs(bench, questions, subset, run):
    """(gold, subset id, qids, predictions) for one benchmark, checked."""
    if questions is None and bench == "byod":
        raise typer.BadParameter(
            "your own data has no default question file: pass the prepared questions.jsonl "
            "(the one in the `panini-dr prepare` output directory)", param_hint="--questions")
    if subset is None and bench != "byod":
        raise typer.BadParameter("--subset is required (e.g. configs/splits/<split>.json)")
    with _user_errors("cannot read the question file"):
        qpath = questions or default_questions_path(bench)
        gold = load_gold(bench, qpath)
    if subset is not None:
        with _user_errors("cannot read the subset file"):
            subset_id, qids = load_subset_ids(subset)
    else:
        subset_id, qids = "all", list(gold)
    missing_gold = [q for q in qids if q not in gold]
    if missing_gold:
        raise typer.BadParameter(
            f"{len(missing_gold)} subset ids are not in {qpath} (e.g. {missing_gold[:5]})")
    with _user_errors("cannot read the run's traces"):
        preds, dups = load_predictions(list(run))
    if dups:
        typer.echo(f"WARNING: {len(dups)} question ids appear in more than one trace; "
                   f"the first is kept (e.g. {dups[:5]})")
    return gold, subset_id, qids, preds


@score_app.command("answers")
def score_answers(
    benchmark: str = typer.Option(..., "--benchmark", "-b", help="bcp | frames | monaco | byod"),
    run: list[Path] = typer.Option(..., "--run", help="Run (cell) dir with traces/; repeat for multi-leg runs."),
    cache: Path = typer.Option(..., "--cache", help="Judge cache JSON (created if missing; written only here)."),
    questions: Optional[Path] = typer.Option(None, "--questions", help="Question file (default: the prepared file for the benchmark)."),
    subset: Optional[Path] = typer.Option(None, "--subset", help="Split JSON with question_ids (the denominator). Required except for byod."),
    buckets: Optional[Path] = typer.Option(None, "--buckets", help="{bucket: [qid]} for per-bucket counts (automatic for cf_bcp150)."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write per-question verdicts + summary here (input of `score obs`)."),
    backend: str = typer.Option("bedrock", "--backend", help="openai | bedrock"),
    model: Optional[str] = typer.Option(None, "--model", help=_MODEL_HELP),
    judge_base_url: str = typer.Option("", "--judge-base-url", help="OpenAI-compatible base URL (openai backend)."),
    judge_api: str = typer.Option("chat", "--judge-api", help="openai backend: chat | responses"),
    bedrock_region: str = typer.Option("", "--bedrock-region", help="Bedrock region (else PANINI_BEDROCK_REGION / AWS_REGION)."),
    passes: int = typer.Option(3, "--passes", help="Judge passes per question (paper: 3)."),
    workers: int = typer.Option(16, "--workers", help="Judge requests in parallel."),
    bcp_repo: Optional[Path] = typer.Option(None, "--bcp-repo", help="BrowseComp-Plus checkout (else BCP_REPO)."),
    monaco_judge_dir: Optional[Path] = typer.Option(None, "--monaco-judge-dir", help="MoNaCo repository checkout (else MONACO_JUDGE_DIR)."),
    allow_judge_mismatch: bool = typer.Option(False, "--allow-judge-mismatch", help="Grade with a MoNaCo judge whose fingerprint differs from the paper's."),
    offline: bool = typer.Option(False, "--offline", help="Cache only: never build a judge backend or call a judge; any verdict missing from the cache is an error (exit 2)."),
) -> None:
    """Grade a run's final answers; print accuracy / F1 over the whole split."""
    model = _judge_model(model)
    bench = _benchmark(benchmark)
    gold, subset_id, qids, preds = _load_inputs(bench, questions, subset, run)

    be = _OfflineBackend(model) if offline else _backend(backend, model, judge_base_url, judge_api, bedrock_region)
    with _user_errors(f"cannot read the judge cache {cache}"):
        jc = JudgeCache(cache)
    short_ids = [q for q in qids if gold[q].answer_type == "short"]
    set_ids = [q for q in qids if gold[q].answer_type == "set"]
    records: dict[str, dict] = {}
    summary: dict = dict(benchmark=bench, subset_id=subset_id, n=len(qids), judge_model=model,
                         backend="offline" if offline else backend, passes=passes,
                         n_traces=sum(1 for q in qids if q in preds))

    if short_ids:
        from panini_dr.eval.official import load_bcp_grader

        with _user_errors("short answers are graded with BrowseComp-Plus's grader"):
            g = load_bcp_grader(bcp_repo)
        items = [ShortItem(q, gold[q].question, _short_gold(gold[q].answer), preds[q])
                 for q in short_ids if q in preds]
        try:
            recs = grade_short(items, complete=be.complete, template=g.template, parse=g.parse,
                               model=model, cache=jc, passes=passes,
                               blank_policy=BLANK_POLICY[bench], workers=workers, offline=offline)
        except CacheMissError as exc:
            typer.echo(f"ERROR: {exc}", err=True)
            raise typer.Exit(2) from exc
        records.update(recs)
        summary["short"] = summarize_short(recs, short_ids, buckets=_buckets_for(subset_id, buckets))
        s = summary["short"]
        typer.echo(f"[{bench}] short-answer accuracy: {s['correct']}/{s['n']} = {100 * s['accuracy']:.1f}%"
                   f"  (blank {s['n_blank']}, no trace {s['n_missing']}, errors {s['n_errors']})")
        for b, c in (s.get("buckets") or {}).items():
            typer.echo(f"    {b}: {c['correct']}/{c['n']}")
    if set_ids:
        from panini_dr.eval.official import load_monaco_judge

        with _user_errors("set answers are graded with MoNaCo's judge"):
            mj = load_monaco_judge(monaco_judge_dir, require_fingerprint=not allow_judge_mismatch)
        items = [SetItem(q, gold[q].question, gold[q].answer, preds[q]) for q in set_ids if q in preds]
        try:
            recs = grade_set(items, complete=be.complete, single_prompt=mj.single_prompt,
                             multi_prompt=mj.multi_prompt, score=mj.score, model=model, cache=jc,
                             passes=passes, workers=workers, offline=offline)
        except CacheMissError as exc:
            typer.echo(f"ERROR: {exc}", err=True)
            raise typer.Exit(2) from exc
        records.update(recs)
        summary["set"] = summarize_set(recs, set_ids)
        s = summary["set"]
        typer.echo(f"[{bench}] set-answer mean F1 (max of {passes} passes, over all {s['n']}): {s['f1']:.4f}"
                   f"  (mean over passes {s['f1_mean_over_passes']:.4f}; judged {s['n_judged']}, "
                   f"blank {s['n_blank']}, no trace {s['n_missing']}, errors {s['n_errors']})")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(summary=summary, records=records), indent=1))
        typer.echo(f"verdicts written to {out}")


@score_app.command("import-legacy-cache")
def import_legacy_cache(
    benchmark: str = typer.Option(..., "--benchmark", "-b", help="bcp | frames | monaco"),
    legacy_cache: Path = typer.Option(..., "--legacy-cache", help="Old cache JSON keyed '<name>:<qid>' (read only)."),
    legacy_name: str = typer.Option(..., "--legacy-name", help="The old key prefix, e.g. 'cf_bcp_capped_cap5' or 'ours_cap5#items'."),
    run: list[Path] = typer.Option(..., "--run", help="The run those verdicts graded (traces/); repeat for multi-leg runs."),
    cache: Path = typer.Option(..., "--cache", help="New-format judge cache to add the verdicts to (created if missing)."),
    subset: Path = typer.Option(..., "--subset", help="Split JSON with question_ids."),
    questions: Optional[Path] = typer.Option(None, "--questions", help="Question file (default: the prepared file)."),
    model: Optional[str] = typer.Option(None, "--model", help="The judge model the old verdicts came from (no default; else PANINI_JUDGE_MODEL)."),
    passes: int = typer.Option(3, "--passes", help="Judge passes per question in the old verdicts."),
    bcp_repo: Optional[Path] = typer.Option(None, "--bcp-repo", help="BrowseComp-Plus checkout (else BCP_REPO)."),
    monaco_judge_dir: Optional[Path] = typer.Option(None, "--monaco-judge-dir", help="MoNaCo repository checkout (else MONACO_JUDGE_DIR)."),
    allow_judge_mismatch: bool = typer.Option(False, "--allow-judge-mismatch", help="Import verdicts for a MoNaCo judge whose fingerprint differs from the paper's."),
    report: Optional[Path] = typer.Option(None, "--report", help="Write the per-question import report (JSON) here."),
) -> None:
    """Re-key verdicts from an old name:qid cache onto this repo's cache keys.

    Each verdict is tied to the run's own answer and the rebuilt prompt; a record
    whose stored answer, vote count or gold length disagrees is not imported and
    makes the command exit 1. No judge is called.
    """
    from panini_dr.eval.legacy_cache import import_set, import_short

    model = _judge_model(model)
    bench = _benchmark(benchmark)
    if bench == "byod":
        raise typer.BadParameter("legacy caches exist for bcp, frames and monaco only")
    gold, _subset_id, qids, preds = _load_inputs(bench, questions, subset, run)
    with _user_errors(f"cannot read the legacy cache {legacy_cache}"):
        legacy = json.loads(Path(legacy_cache).read_text())
    with _user_errors(f"cannot read the judge cache {cache}"):
        jc = JudgeCache(cache)
    present = [q for q in qids if q in preds]
    if bench == "monaco":
        from panini_dr.eval.official import load_monaco_judge

        with _user_errors("set answers are graded with MoNaCo's judge"):
            mj = load_monaco_judge(monaco_judge_dir, require_fingerprint=not allow_judge_mismatch)
        items = [SetItem(q, gold[q].question, gold[q].answer, preds[q]) for q in present]
        rep = import_set(legacy, legacy_name, items, single_prompt=mj.single_prompt,
                         multi_prompt=mj.multi_prompt, model=model, passes=passes, put=jc.put)
    else:
        from panini_dr.eval.official import load_bcp_grader

        with _user_errors("short answers are graded with BrowseComp-Plus's grader"):
            g = load_bcp_grader(bcp_repo)
        items = [ShortItem(q, gold[q].question, _short_gold(gold[q].answer), preds[q]) for q in present]
        rep = import_short(legacy, legacy_name, items, template=g.template, model=model,
                           passes=passes, blank_policy=BLANK_POLICY[bench], put=jc.put)
    rep.no_trace = [q for q in qids if q not in preds]
    wrote = jc.save()
    d = rep.as_dict()
    typer.echo(f"[{bench}] {legacy_name}: imported {d['imported']} of {len(qids)} "
               f"(answer checked {d['verified']}, not checkable {d['unverified']}); "
               f"blank {d['blank']}, no legacy verdict {d['missing']}, no trace {d['no_trace']}, "
               f"conflicts {len(d['conflicts'])}; cache {'written' if wrote else 'unchanged'}: {cache}")
    for q, why in list(rep.conflicts.items())[:20]:
        typer.echo(f"  conflict {q}: {why}")
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(dict(benchmark=bench, legacy_name=legacy_name, model=model,
                                          passes=passes, n=len(qids), **d), indent=1))
    if rep.conflicts:
        raise typer.Exit(1)


@score_app.command("obs")
def score_obs(
    benchmark: str = typer.Option(..., "--benchmark", "-b", help="bcp | frames | monaco"),
    run: list[Path] = typer.Option(..., "--run", help="Run (cell) dir with traces/; repeat for multi-leg runs."),
    verdicts: Path = typer.Option(..., "--verdicts", help="The --out file of `score answers` for the same run."),
    subset: Path = typer.Option(..., "--subset", help="Split JSON (configs/splits/...)."),
    corpus: Path = typer.Option(..., "--corpus", help="The {title: text} corpus the run used."),
    questions: Optional[Path] = typer.Option(None, "--questions", help="Question file (default: the prepared file)."),
    buckets: Optional[Path] = typer.Option(None, "--buckets", help="bcp: {difficulty: [qid]} (default configs/splits/cf_bcp150_buckets.json)."),
    qrels_dir: Optional[Path] = typer.Option(None, "--qrels-dir", help="bcp: topics-qrels/ (default <BCP_REPO>/topics-qrels)."),
    bcp_repo: Optional[Path] = typer.Option(None, "--bcp-repo", help="bcp: BrowseComp-Plus checkout (else BCP_REPO)."),
    doctext_cache: Optional[Path] = typer.Option(None, "--doctext-cache", help="bcp: pickle cache of the corpus subset."),
    packets: str = typer.Option("", "--packets", help="monaco: glob of existing packet JSONs (default: build them from the traces)."),
    out: Optional[Path] = typer.Option(None, "--out", help="Directory for per-question rows + summary."),
) -> None:
    """Observed-support (Obs.) score of a run, from its traces and its verdicts."""
    import os

    from panini_dr.eval import obs as O

    bench = _benchmark(benchmark)
    if bench == "byod":
        raise typer.BadParameter("Obs. is defined for bcp, frames and monaco")
    qpath = questions or default_questions_path(bench)
    with _user_errors(f"cannot read the verdicts {verdicts}"):
        v = json.loads(Path(verdicts).read_text())
        records = v.get("records", v)
    if bench == "bcp":
        from panini_dr.config import configs_dir

        repo = bcp_repo or os.environ.get("BCP_REPO")
        qd = qrels_dir or (Path(repo) / "topics-qrels" if repo else None)
        if qd is None:
            raise typer.BadParameter("bcp Obs. needs --qrels-dir or --bcp-repo / BCP_REPO")
        kw = dict(buckets=str(buckets or configs_dir() / "splits" / "cf_bcp150_buckets.json"),
                  qrels_dir=str(qd), doctext_cache=str(doctext_cache) if doctext_cache else None)
    else:
        kw = dict(split=str(subset))
    with _user_errors("cannot load the Obs. inputs"):
        ds = O.Dataset(bench, questions=str(qpath), corpus=str(corpus), **kw)
    if bench == "bcp":
        with _user_errors("cannot read the subset file"):
            _, ids = load_subset_ids(subset)
        if set(ids) != set(ds.qids):
            raise typer.BadParameter("bcp Obs. is defined on the CF-150 split (the buckets file); --subset differs")
    runs = [str(r) for r in run]
    if bench == "monaco":
        s = O.score_monaco_obs(ds, records, run_dirs=runs, packets_glob=packets or None,
                               corpus=str(corpus), out_dir=str(out) if out else None)
        typer.echo(f"[monaco] Obs. {s['obs']:.4f}  official {s['official']:.4f}  (n={s['n']}, "
                   f"with answer {s['with_pred']}, credited items {s['credited_items']})")
    else:
        s = O.score_short_obs(ds, records, run_dirs=runs, out_dir=str(out) if out else None)
        a = s["aggregates"]["all"]
        typer.echo(f"[{bench}] Obs. {100 * (s['obs'] or 0):.1f}  official {100 * (s['official'] or 0):.1f}  "
                   f"(n={s['n']}, blank {a.get('n_blank')}, no trace {a.get('trace_missing')})")


__all__ = ["score_app"]
