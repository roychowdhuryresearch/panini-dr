"""Read what a run predicted, and the gold the graders compare it with.

Predictions come from a run's traces (``<run>/traces/q_<id>.json`` ->
``final_answer``), as in the research graders; several run dirs can be given
(the BrowseComp-Plus run of record ran as two legs).

Gold rows are read RAW from the prepared question files so the prompt text is
exactly what the paper's graders sent:

* ``bcp``    browsecomp_plus_830.json rows ``{query_id, query, answer}``
* ``frames`` frames_split.json rows; the grader used ``test_question`` and ``answer``
* ``monaco`` monaco_full1315.json rows ``{id, question, answer}``
* ``byod``   questions.jsonl rows ``{id, question, answer, answer_type?}``
  (``answer_type`` defaults to ``short`` for a string answer and ``set`` for a list)
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BENCHMARKS = ("bcp", "frames", "monaco", "byod")


def normalize_benchmark(name: str) -> str:
    n = (name or "").strip().lower().replace("-", "_")
    if n in ("browsecomp_plus", "browsecomp", "bcp"):
        return "bcp"
    if n not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}; expected one of {BENCHMARKS}")
    return n


def trace_files(run_dir: str | Path) -> list[str]:
    d = str(run_dir)
    return sorted(glob.glob(os.path.join(d, "traces", "q_*.json"))) or sorted(
        glob.glob(os.path.join(d, "*", "traces", "q_*.json"))
    )


def load_predictions(run_dirs: list[str | Path]) -> tuple[dict[str, str], list[str]]:
    """``{qid: final_answer}`` over all run dirs, and the qids seen more than once
    (the first occurrence wins)."""
    preds: dict[str, str] = {}
    dups: list[str] = []
    for d in run_dirs:
        files = trace_files(d)
        if not files:
            raise FileNotFoundError(f"no traces/q_*.json under {d}")
        for p in files:
            t = json.loads(Path(p).read_text())
            qid = str(t.get("question_id"))
            if qid in preds:
                dups.append(qid)
                continue
            preds[qid] = str(t.get("final_answer") or "")
    return preds, dups


@dataclass
class GoldRow:
    qid: str
    question: str
    answer: Any
    answer_type: str  # short | set
    raw: dict[str, Any]


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    rows = json.loads(p.read_text())
    if isinstance(rows, dict):
        rows = rows.get("questions") or []
    return rows


def default_answer_type(answer: Any) -> str:
    return "set" if isinstance(answer, list) else "short"


def load_gold(benchmark: str, path: str | Path) -> dict[str, GoldRow]:
    bench = normalize_benchmark(benchmark)
    out: dict[str, GoldRow] = {}
    for r in _read_rows(path):
        if bench == "bcp":
            qid = str(r.get("query_id", r.get("id", "")))
            out[qid] = GoldRow(qid, str(r.get("query", r.get("question", ""))), r.get("answer"), "short", r)
        elif bench == "frames":
            qid = str(r["id"])
            question = r.get("test_question") or r.get("original_prompt") or r.get("question") or ""
            out[qid] = GoldRow(qid, question, r.get("answer"), "short", r)
        elif bench == "monaco":
            qid = str(r["id"])
            out[qid] = GoldRow(qid, str(r.get("question", "")), r.get("answer"), "set", r)
        else:
            qid = str(r["id"])
            at = str(r.get("answer_type") or default_answer_type(r.get("answer"))).lower()
            if at not in ("short", "set"):
                raise ValueError(f"question {qid}: answer_type must be 'short' or 'set', got {at!r}")
            out[qid] = GoldRow(qid, str(r.get("question", "")), r.get("answer"), at, r)
    return out


def default_questions_path(benchmark: str) -> Path:
    bench = normalize_benchmark(benchmark)
    if bench == "bcp":
        from panini_dr.eval.browsecomp_dataset import default_questions_path as f

        return f()
    if bench == "frames":
        from panini_dr.eval.frames_dataset import default_questions_path as f

        return f("full")
    if bench == "monaco":
        from panini_dr.eval.monaco_dataset import default_questions_path as f

        return f()
    raise ValueError("byod needs --questions <questions.jsonl>")


def load_subset_ids(path: str | Path) -> tuple[str, list[str]]:
    d = json.loads(Path(path).read_text())
    return str(d.get("subset_id", Path(path).stem)), [str(q) for q in d["question_ids"]]


__all__ = [
    "BENCHMARKS",
    "GoldRow",
    "default_answer_type",
    "default_questions_path",
    "load_gold",
    "load_predictions",
    "load_subset_ids",
    "normalize_benchmark",
    "trace_files",
]
