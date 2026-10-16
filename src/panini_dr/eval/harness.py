"""Eval harness — runs one adapter against one subset, produces a CellResult.

Usage
-----
.. code-block:: python

    from panini_dr.adapters import AdapterContext, get_adapter
    from panini_dr.eval import load_subset
    from panini_dr.eval.subset import load_benchmark_questions
    from panini_dr.eval.harness import run_cell

    subset = load_subset("configs/splits/frames_full824_norm.json")
    questions = load_benchmark_questions(subset, split="full")
    ctx = AdapterContext(system_id="ours_gsw_planner_orchestrator_v1", model_id=..., ...)
    adapter = get_adapter(ctx.system_id)(ctx)
    cell = run_cell(adapter, questions, subset_id=subset.subset_id, benchmark="frames")
    cell.summary_row()

The harness does not spawn processes or manage vLLM servers — that's
the CLI driver's job. It assumes the adapter's model endpoint is up.
"""

from __future__ import annotations

import statistics
import time
import traceback
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from panini_dr.adapters.base import Adapter
from panini_dr.eval.failure_classifier import classify, histogram
from panini_dr.eval.frames_dataset import FramesQuestion
# The in-run LLM judge of the research repo is not shipped: grading is done
# offline by ``panini-dr score``. ``judge`` stays a duck-typed hook
# (``.judge(question=, gold=, predicted=)``) and defaults to None.
LLMJudge = Any
from panini_dr.eval.scoring import any_alias_match, best_f1, exact_match, token_f1
from panini_dr.utils import get_logger, render_question_markdown

_log = get_logger("harness")
from panini_dr.models.failure_modes import FailureMode
from panini_dr.models.trace import CellResult, QuestionResult, Trajectory


def _config_snapshot_value(value: Any, *, key: str = "") -> Any:
    """Convert runtime adapter config into a secret-safe JSON snapshot."""

    key_lower = key.lower()
    if "api_key" in key_lower or key_lower.endswith("_secret"):
        return "<redacted>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(child_key): _config_snapshot_value(
                child_value,
                key=str(child_key),
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _config_snapshot_value(item, key=key)
            for item in value
        ]
    if key_lower == "corpus":
        chunks = getattr(value, "chunks", None)
        return {
            "type": type(value).__name__,
            "source": str(getattr(value, "source", "")),
            "n_chunks": len(chunks) if chunks is not None else None,
        }
    return {"type": type(value).__name__}


def _config_snapshot(extra: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _config_snapshot_value(value, key=str(key))
        for key, value in extra.items()
    }


def _answer_to_str(value: Any) -> str:
    """Render a gold/predicted answer as a string for the FRAMES-style
    exact-match/token-F1 metrics and the string-typed ``QuestionResult`` fields.

    MoNaCo gold answers (and collection predictions) are *lists* — often a
    list-of-rows. FRAMES ``exact_match``/``token_f1`` call ``.lower()`` on their
    arguments and ``QuestionResult.gold_answer`` is typed ``str``, so a raw list
    would crash scoring. We render rows as ``a | b`` and join items with
    newlines. This string is only used for the throwaway FRAMES metrics and for
    display; the authoritative MoNaCo metric (``MonacoVerdict.f1``) is computed
    separately from the *structured* gold below.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        rows: list[str] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                rows.append(" | ".join("" if c is None else str(c) for c in item))
            else:
                rows.append(str(item))
        return "\n".join(rows)
    return str(value)


def _score_one(
    q: FramesQuestion,
    trajectory: Trajectory,
    predicted: str,
    *,
    judge: LLMJudge | None = None,
) -> QuestionResult:
    """Score a single question using exact-match + token F1 + (optional) LLM judge."""
    # MoNaCo gold (and collection predictions) may be lists; the FRAMES string
    # metrics and the str-typed result fields need a stringified form. The judge
    # still receives the *structured* gold so its set-overlap F1 is exact.
    pred_str = _answer_to_str(predicted)
    gold_str = _answer_to_str(q.answer)
    em = exact_match(pred_str, gold_str)
    f1 = token_f1(pred_str, gold_str)

    result = QuestionResult(
        question_id=q.id,
        question=q.question,
        gold_answer=gold_str,
        trajectory=trajectory,
        predicted_answer=pred_str,
        exact_match=em,
        f1=f1,
    )

    # LLM-as-judge grading. Runs even when exact_match is already true — we
    # want to confirm or catch disagreements (and record the reason for audit).
    if judge is not None:
        verdict = judge.judge(
            question=q.question, gold=q.answer, predicted=pred_str
        )
        result.judge_correct = verdict.correct
        # MoNaCo: the set-overlap verdict carries the official partial-credit
        # F1 (MonacoVerdict.f1); use it so mean_f1 reflects the real set metric
        # rather than token overlap. FRAMES verdicts have no f1 -> unchanged.
        _verdict_f1 = getattr(verdict, "f1", None)
        if _verdict_f1 is not None:
            result.f1 = float(_verdict_f1)
        result.notes = (
            f"judge={verdict.correct}; reason={verdict.reason}"
            if verdict.reason else f"judge={verdict.correct}"
        )
        if verdict.error:
            result.trajectory.extra.setdefault("judge_errors", []).append(verdict.error)

    result.failure_mode = classify(result, num_hops=q.num_hops)
    return result


def _empty_trajectory(system_id: str, model_id: str, question_id: str, error: str) -> Trajectory:
    return Trajectory(
        system_id=system_id,
        model_id=model_id,
        question_id=question_id,
        extra={"adapter_error": error},
    )


def run_cell(
    adapter: Adapter,
    questions: Iterable[FramesQuestion],
    *,
    subset_id: str,
    benchmark: str,
    on_question_done=None,
    raw_trace_dir: Path | str | None = None,
    judge: LLMJudge | None = None,
    concurrency: int = 1,
) -> CellResult:
    """Run ``adapter`` over the given questions and return an aggregated CellResult.

    If ``judge`` is provided, each question result also receives a
    ``judge_correct`` verdict. The headline ``accuracy`` stays on
    exact_match — judge verdicts are reported alongside via
    ``n_judge_correct`` / ``judge_accuracy`` so both signals are visible.

    ``concurrency``: when > 1, run that many ``adapter.run_question``
    calls in parallel via a ThreadPoolExecutor. Each Q is an
    independent network-bound LLM session; the adapter's shared state
    (``llm``, ``retriever``, ``corpus``) is read-only at query time so
    concurrent calls are safe. Per-Q logging is interleaved but
    timestamped so traces remain readable. Default 1 = sequential
    (no behavior change). Tune to provider rate limits (~4–8 is safe
    for Bedrock gpt-oss-120b at our TPM).
    """

    results: list[QuestionResult] = []
    cell = CellResult(
        system_id=adapter.ctx.system_id,
        model_id=adapter.ctx.model_id,
        benchmark=benchmark,
        subset_id=subset_id,
        config={
            "max_turns": adapter.ctx.max_turns,
            "max_prompt_tokens": adapter.ctx.max_prompt_tokens,
            "max_completion_tokens": adapter.ctx.max_completion_tokens,
            **_config_snapshot(adapter.ctx.extra),
        },
    )
    trace_dir = Path(raw_trace_dir) if raw_trace_dir else None
    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)

    # Fail-fast: if the first N questions all fail with the same
    # configuration-level error (e.g. 401 auth, connection refused to vLLM,
    # model-not-found), stop the cell so the user can fix it without burning
    # through every question. Per-question hiccups don't trip this guard.
    _FAIL_FAST_AFTER = 3
    _fail_errors: list[str] = []

    questions_list = list(questions)
    n_total_plan = len(questions_list)
    _log.info(
        "cell start | system=%s model=%s subset=%s n=%s concurrency=%d",
        adapter.ctx.system_id, adapter.ctx.model_id, subset_id,
        n_total_plan, concurrency,
    )

    def _process_one(i: int, q: FramesQuestion) -> tuple[QuestionResult, str | None]:
        """Run a single question end-to-end. Returns (result, fail_error_or_None)."""
        # A hop count of 0 means the dataset has none (FRAMES-style files without
        # num_hops, your own data): leave the field out of the log line.
        _log.info(
            "q%s start (%s/%s)%s | %s",
            q.id, i, n_total_plan, f" hops={q.num_hops}" if q.num_hops else "",
            q.question[:140],
        )
        t0 = time.time()
        try:
            trajectory = adapter.run_question(
                q.question,
                question_id=q.id,
                articles=q.articles,
            )
            predicted = trajectory.final_answer
        except Exception as exc:  # noqa: BLE001
            trajectory = _empty_trajectory(
                adapter.ctx.system_id, adapter.ctx.model_id, q.id, str(exc)
            )
            trajectory.extra["traceback"] = traceback.format_exc()
            predicted = ""
        trajectory.wall_time_s = trajectory.wall_time_s or (time.time() - t0)

        _llm_err = trajectory.extra.get("llm_error") or trajectory.extra.get("adapter_error")
        _stopped_reason = trajectory.extra.get("stopped_reason", "")
        fail_err: str | None = None
        if trajectory.turns == 0 and _llm_err:
            fail_err = str(_llm_err)
            _log.warning("q%s zero-turn failure: %s", q.id, _llm_err)
        if _llm_err and _stopped_reason == "llm_error":
            _log.warning(
                "q%s mid-question LLM error after turn %d (pred empty): %s",
                q.id, trajectory.turns, _llm_err,
            )

        for tc in trajectory.tool_calls:
            _log.debug(
                "q%s turn %d | %s(%s) -> %s%s",
                q.id, tc.turn, tc.name,
                ", ".join(f"{k}={v!r}" for k, v in list(tc.args.items())[:3]),
                tc.result_preview[:200].replace("\n", " "),
                f" [error: {tc.error}]" if tc.error else "",
            )

        result = _score_one(q, trajectory, predicted, judge=judge)

        if trace_dir:
            (trace_dir / f"q_{q.id}.json").write_text(
                trajectory.model_dump_json(indent=2)
            )
            try:
                (trace_dir / f"q_{q.id}.md").write_text(
                    render_question_markdown(result)
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning("failed to render q%s markdown: %s", q.id, exc)

        mark = "✓" if result.exact_match else ("~" if result.judge_correct else "✗")
        judge_tag = (
            "" if result.judge_correct is None
            else f" judge={'✓' if result.judge_correct else '✗'}"
        )
        _log.info(
            "q%s done  %s | turns=%d tools=%d f1=%.2f mode=%s wall=%.1fs tokens_out=%d%s | pred=%r gold=%r",
            q.id, mark, trajectory.turns, len(trajectory.tool_calls),
            result.f1, result.failure_mode.value, trajectory.wall_time_s,
            trajectory.completion_tokens, judge_tag,
            (predicted or "")[:80], (q.answer or "")[:80],
        )
        return result, fail_err

    if concurrency <= 1:
        # Sequential — preserves Phase-1/-2 fail-fast semantics exactly.
        for i, q in enumerate(questions_list, start=1):
            result, fail_err = _process_one(i, q)
            if fail_err:
                _fail_errors.append(fail_err)
            elif result.trajectory.turns > 0 or result.predicted_answer:
                _fail_errors.clear()
            results.append(result)
            if on_question_done:
                on_question_done(result)
            if len(_fail_errors) >= _FAIL_FAST_AFTER and len(set(_fail_errors)) == 1:
                _log.error(
                    "cell fail-fast: first %d Qs all failed with the same error %r — "
                    "aborting before more LLM calls",
                    _FAIL_FAST_AFTER, _fail_errors[0],
                )
                break
    else:
        # Parallel — drop fail-fast (each worker is independent and we'd
        # rather see all results than abort mid-flight).
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results_by_id: dict[str, QuestionResult] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {
                ex.submit(_process_one, i, q): q.id
                for i, q in enumerate(questions_list, start=1)
            }
            for fut in as_completed(futures):
                qid = futures[fut]
                try:
                    result, _ = fut.result()
                except Exception as exc:  # noqa: BLE001
                    _log.error("q%s worker raised: %s", qid, exc)
                    continue
                results_by_id[qid] = result
                if on_question_done:
                    on_question_done(result)
        # Preserve original question order in CellResult.
        for q in questions_list:
            if q.id in results_by_id:
                results.append(results_by_id[q.id])

    if len(_fail_errors) >= _FAIL_FAST_AFTER and len(set(_fail_errors)) == 1:
        # Sequential path may set this; surface for downstream debug.
        cell.config["aborted_reason"] = _fail_errors[-1]

    cell.questions = results
    cell.n_total = len(results)
    cell.n_correct = sum(1 for r in results if r.exact_match)
    cell.accuracy = cell.n_correct / cell.n_total if cell.n_total else 0.0
    cell.mean_f1 = statistics.fmean(r.f1 for r in results) if results else 0.0

    # LLM-judge aggregates — only count questions where the judge actually ran.
    judged = [r for r in results if r.judge_correct is not None]
    cell.n_judge_seen = len(judged)
    cell.n_judge_correct = sum(1 for r in judged if r.judge_correct)
    cell.judge_accuracy = (
        cell.n_judge_correct / cell.n_judge_seen if cell.n_judge_seen else 0.0
    )
    cell.mean_turns = statistics.fmean(r.trajectory.turns for r in results) if results else 0.0
    cell.mean_tool_calls = (
        statistics.fmean(len(r.trajectory.tool_calls) for r in results) if results else 0.0
    )
    cell.mean_prompt_tokens = (
        statistics.fmean(r.trajectory.prompt_tokens for r in results) if results else 0.0
    )
    cell.mean_completion_tokens = (
        statistics.fmean(r.trajectory.completion_tokens for r in results) if results else 0.0
    )
    cell.mean_wall_time_s = (
        statistics.fmean(r.trajectory.wall_time_s for r in results) if results else 0.0
    )
    cell.failure_histogram = histogram(results)
    cell.completed_at = datetime.now(timezone.utc)
    return cell


def persist_cell(cell: CellResult, path: str | Path) -> None:
    """Write a CellResult to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cell.model_dump_json(indent=2))


def load_cell(path: str | Path) -> CellResult:
    """Load a persisted CellResult."""
    return CellResult.model_validate_json(Path(path).read_text())


__all__ = [
    "run_cell",
    "persist_cell",
    "load_cell",
    "_score_one",
    "any_alias_match",
    "best_f1",
]
