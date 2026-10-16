"""Set-answer grader of record: MoNaCo's official V2 LLM judge, 3 passes, max over passes.

Used for MoNaCo and for bring-your-own datasets whose questions have
``answer_type: set``. This is the instrument behind the paper's MoNaCo 0.549:
the item-level re-judge (``judge_items_pipeline.py`` in the research repo),
which keeps every raw verdict so the Obs. scorer can see which predicted items
the judge credited.

Faithful to that pipeline:

* gold: ``answer`` (a list; a scalar becomes a one-element list), each element
  rendered with ``str()`` -- a multi-column row becomes its Python list repr --
  ``glen = len(answer)`` and ``correct_answer = "; ".join(...)``;
* prompt: the authors' single-answer prompt when ``glen == 1``, else the
  multi-answer prompt, with the STRIPPED prediction as the response;
* each pass: the reply's numeric fields are de-markdowned (``**13**`` -> ``13``,
  which the authors' parser cannot read), then scored with the authors'
  ``compute_llm_judge_score_V2`` (``None`` -> all zeros);
* per question: F1 = max over passes (the mean over passes is reported too);
* blank predictions are not judged and score 0; a question whose judging
  raised scores 0 and is reported (re-running retries it);
* the mean is over EVERY question of the split (blank, missing and failed = 0).
"""

from __future__ import annotations

import re
import statistics as st
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from panini_dr.eval.judge_cache import CacheMissError, JudgeCache, cache_key

GRADER_NAME = "set_monaco_v2_items"


@dataclass
class SetItem:
    qid: str
    question: str
    answer: Any          # list (rows or items) or scalar
    prediction: str


def gold_list(answer: Any) -> list[str]:
    a = answer if isinstance(answer, list) else [answer]
    return [str(x) for x in a]


def demarkdown(raw: str) -> str:
    """Strip markdown emphasis around the judge reply's NUMERIC fields only."""
    for field in ("final answer length:", "final precision:", "precision:"):
        raw = re.sub(
            r"(?i)(" + re.escape(field) + r"\s*)\*+\s*([0-9][0-9.]*)\s*\*+",
            lambda m: m.group(1) + m.group(2),
            raw,
        )
    return raw


def parse_credited(raw: str, glen: int, pred: str, score: dict[str, Any]) -> tuple[list[str], int | None, bool]:
    """The judge's own list of credited predicted items, plus its stated predicted length."""
    if glen == 1:
        return ([pred] if float(score.get("judge_score", 0) or 0) > 0 else []), 1, True
    kw = "\noverlapping answers:"
    if kw not in raw:
        return [], None, False
    chunk = raw.split(kw)[1].strip()
    chunk = chunk.split("\nfinal")[0]
    parts = [p.strip() for p in chunk.replace("###NULL", "###").split("###")]
    items = [p for p in parts if p and p.upper() != "NULL"]
    plen = None
    m = re.search(r"final answer length:\s*\**\s*(\d+)", raw)
    if m:
        plen = int(m.group(1))
    return items, plen, True


def build_prompt(item: SetItem, single_prompt: str, multi_prompt: str) -> tuple[str, int]:
    gl = gold_list(item.answer)
    glen = len(gl)
    tmpl = single_prompt if glen == 1 else multi_prompt
    prompt = tmpl.format(question=item.question, response=str(item.prediction or "").strip(),
                         correct_answer="; ".join(gl))
    return prompt, glen


def set_cache_key(item: SetItem, *, single_prompt: str, multi_prompt: str, model: str,
                  passes: int) -> str:
    """The cache key :func:`grade_set` looks the verdict of ``item`` up under."""
    prompt, _ = build_prompt(item, single_prompt, multi_prompt)
    return cache_key(grader=GRADER_NAME, model=model, passes=passes, qid=item.qid,
                     prediction=str(item.prediction or "").strip(), prompt=prompt)


def judge_one(
    item: SetItem,
    *,
    complete: Callable[[str], str],
    single_prompt: str,
    multi_prompt: str,
    score: Callable[[str, int], dict[str, Any] | None],
    passes: int = 3,
) -> dict[str, Any]:
    pred = str(item.prediction or "").strip()
    prompt, glen = build_prompt(item, single_prompt, multi_prompt)
    ps = []
    for _ in range(max(1, passes)):
        raw = demarkdown(complete(prompt))
        s = score(raw, glen) or {}
        cred, plen, ok = parse_credited(raw, glen, pred, s)
        ps.append(dict(
            f1=float(s.get("judge_score", 0.0) or 0.0),
            precision=float(s.get("precision", 0.0) or 0.0),
            recall=float(s.get("recall", s.get("precision", 0.0)) or 0.0),
            credited=cred, judge_pred_len=plen, parsed=ok, raw=raw[-3000:],
        ))
    return dict(glen=glen, passes=ps)


def question_scores(rec: dict[str, Any] | None) -> dict[str, float]:
    if not rec or not rec.get("passes"):
        return dict(f1_max=0.0, f1_mean=0.0, precision=0.0, recall=0.0)
    ps = rec["passes"]
    best = max(ps, key=lambda p: p["f1"])
    return dict(
        f1_max=float(best["f1"]),
        f1_mean=float(sum(p["f1"] for p in ps) / len(ps)),
        precision=float(best["precision"]),
        recall=float(best["recall"]),
    )


def grade_set(
    items: list[SetItem],
    *,
    complete: Callable[[str], str],
    single_prompt: str,
    multi_prompt: str,
    score: Callable[[str, int], dict[str, Any] | None],
    model: str,
    cache: JudgeCache | None = None,
    passes: int = 3,
    workers: int = 16,
    offline: bool = False,
) -> dict[str, dict[str, Any]]:
    """Judge every non-blank prediction (cached). Returns ``{qid: record}``.

    ``offline=True`` never calls the judge: a non-blank prediction whose verdict
    is not in the cache raises :class:`CacheMissError` (listing every such
    question) before anything is judged or written.
    """
    cache = JudgeCache(None) if cache is None else cache
    out: dict[str, dict[str, Any]] = {}
    todo: list[tuple[str, SetItem]] = []
    for it in items:
        pred = str(it.prediction or "").strip()
        if not pred:
            out[it.qid] = dict(blank=True, glen=len(gold_list(it.answer)), passes=[])
            continue
        key = set_cache_key(it, single_prompt=single_prompt, multi_prompt=multi_prompt,
                            model=model, passes=passes)
        hit = cache.get(key)
        if hit is not None:
            out[it.qid] = dict(hit, cached=True)
        else:
            todo.append((key, it))
    if offline and todo:
        raise CacheMissError([it.qid for _, it in todo])

    def work(pair: tuple[str, SetItem]) -> dict[str, Any]:
        _, it = pair
        return judge_one(it, complete=complete, single_prompt=single_prompt,
                         multi_prompt=multi_prompt, score=score, passes=passes)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(work, p): p for p in todo}
        for fut in as_completed(futs):
            key, it = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001 -- scored 0, reported, retried next run
                out[it.qid] = dict(error=f"{type(exc).__name__}: {exc}", passes=[],
                                   glen=len(gold_list(it.answer)))
                continue
            cache.put(key, rec)
            out[it.qid] = rec
    cache.save()
    return out


def summarize_set(records: dict[str, dict[str, Any]], qids: list[str]) -> dict[str, Any]:
    """Mean F1 over ALL ``qids``: max over passes (of record) and mean over passes."""
    n = len(qids)
    rows = {q: question_scores(records.get(q)) for q in qids}
    judged = [q for q in qids if (records.get(q) or {}).get("passes")]
    multi = [q for q in qids if int((records.get(q) or {}).get("glen") or 0) > 1]
    single = [q for q in qids if int((records.get(q) or {}).get("glen") or 0) == 1]

    def m(ids: list[str], k: str) -> float:
        return st.mean(rows[q][k] for q in ids) if ids else 0.0

    n_pass = sum(len((records.get(q) or {}).get("passes") or []) for q in qids)
    n_parsed = sum(1 for q in qids for p in (records.get(q) or {}).get("passes") or [] if p.get("parsed"))
    return dict(
        n=n,
        f1=m(qids, "f1_max"),
        f1_mean_over_passes=m(qids, "f1_mean"),
        precision=m(qids, "precision"),
        recall=m(qids, "recall"),
        f1_multi=m(multi, "f1_max"), n_multi=len(multi),
        f1_single=m(single, "f1_max"), n_single=len(single),
        n_judged=len(judged),
        n_blank=sum(1 for q in qids if (records.get(q) or {}).get("blank")),
        n_missing=sum(1 for q in qids if q not in records),
        n_errors=sum(1 for q in qids if "error" in (records.get(q) or {})),
        parse_rate=(n_parsed / n_pass) if n_pass else None,
    )


__all__ = [
    "GRADER_NAME",
    "SetItem",
    "build_prompt",
    "demarkdown",
    "gold_list",
    "grade_set",
    "judge_one",
    "parse_credited",
    "question_scores",
    "set_cache_key",
    "summarize_set",
]
