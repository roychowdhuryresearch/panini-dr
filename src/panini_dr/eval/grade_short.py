"""Short-answer grader of record: BrowseComp-Plus GRADER_TEMPLATE, 3 passes, majority vote.

Used for BrowseComp-Plus and FRAMES (the paper re-graded every FRAMES row with
the same instrument as the BCP column) and for bring-your-own datasets whose
questions have ``answer_type: short``.

Faithful to the research graders (``bcp_official_judge.py`` /
``frames_bcp_grader.py``):

* prompt = ``GRADER_TEMPLATE.format(question=..., response=..., correct_answer=...)``
  with the authors' template, each verdict parsed with their
  ``parse_judge_response``;
* a pass votes "correct" when ``parse(...)["correct"]`` is True (or the string
  yes/true); the question is correct when yes-votes > passes / 2;
* blank predictions, per benchmark:
  - ``bcp``: graded, with the response rendered as ``"(no answer)"``
    (``bcp_official_judge.py``);
  - ``frames`` and bring-your-own: scored incorrect without a judge call
    (``frames_bcp_grader.py``; a whitespace-only answer is blank).

Accuracy is reported over every question of the split: a question with no
trace or no verdict counts as incorrect and stays in the denominator. (The BCP
script divided by the number of traces found; for a complete run, as in the
paper, the two are identical.)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from panini_dr.eval.judge_cache import CacheMissError, JudgeCache, cache_key

GRADER_NAME = "short_bcp_template"

#: How a blank prediction is handled, per benchmark (see module docstring).
BLANK_POLICY = {
    "bcp": "judge_as_no_answer",
    "browsecomp_plus": "judge_as_no_answer",
    "frames": "zero_without_call",
    "byod": "zero_without_call",
}


@dataclass
class ShortItem:
    qid: str
    question: str
    gold: str
    prediction: str


def count_yes(verdicts: Iterable[dict[str, Any]]) -> int:
    """Yes-votes, counted exactly as the research graders did.

    ``parse_judge_response`` returns ``correct`` as a boolean; a raw "yes"/"true"
    string is accepted too.
    """
    return sum(
        1
        for v in verdicts
        if v.get("correct") is True or str(v.get("correct")).lower() in ("yes", "true")
    )


def majority(yes: int, n: int) -> bool:
    return yes > n / 2


def is_blank_without_call(prediction: str, blank_policy: str) -> bool:
    """True when the prediction is scored 0 without a judge call (blank, zero policy)."""
    return blank_policy == "zero_without_call" and not str(prediction or "").strip()


def build_short_prompt(item: ShortItem, template: str, blank_policy: str) -> str:
    """The exact grader prompt for ``item`` ("" when no call is made)."""
    pred = str(item.prediction or "")
    if is_blank_without_call(pred, blank_policy):
        return ""
    response = pred if blank_policy == "zero_without_call" else (pred or "(no answer)")
    return template.format(question=item.question, response=response, correct_answer=item.gold)


def short_cache_key(item: ShortItem, *, template: str, model: str, passes: int,
                    blank_policy: str) -> str:
    """The cache key :func:`grade_short` looks the verdict of ``item`` up under."""
    return cache_key(grader=GRADER_NAME, model=model, passes=passes, qid=item.qid,
                     prediction=str(item.prediction or ""),
                     prompt=build_short_prompt(item, template, blank_policy))


def grade_one(
    item: ShortItem,
    *,
    complete: Callable[[str], str],
    template: str,
    parse: Callable[[str], dict[str, Any]],
    passes: int = 3,
    blank_policy: str = "zero_without_call",
) -> tuple[str, dict[str, Any]]:
    """Grade one prediction. Returns (prompt, record); prompt is "" when no call is made."""
    pred = str(item.prediction or "")
    if is_blank_without_call(pred, blank_policy):
        return "", dict(correct=False, votes="0/0", extracted=None, pred="", blank=True)
    prompt = build_short_prompt(item, template, blank_policy)
    verdicts = [parse(complete(prompt)) for _ in range(max(1, passes))]
    yes = count_yes(verdicts)
    return prompt, dict(
        correct=majority(yes, len(verdicts)),
        votes=f"{yes}/{len(verdicts)}",
        extracted=verdicts[0].get("extracted_final_answer"),
        pred=pred[:200],
        blank=not pred.strip(),
    )


def grade_short(
    items: list[ShortItem],
    *,
    complete: Callable[[str], str],
    template: str,
    parse: Callable[[str], dict[str, Any]],
    model: str,
    cache: JudgeCache | None = None,
    passes: int = 3,
    blank_policy: str = "zero_without_call",
    workers: int = 16,
    offline: bool = False,
) -> dict[str, dict[str, Any]]:
    """Grade every item (cached). Returns ``{qid: record}``; failures carry ``error``.

    ``offline=True`` never calls the judge: a verdict that is not in the cache
    raises :class:`CacheMissError` (listing every such question) before anything
    is graded or written. Blank predictions scored 0 without a call need no cache.
    """
    cache = JudgeCache(None) if cache is None else cache
    out: dict[str, dict[str, Any]] = {}
    todo: list[tuple[str, ShortItem]] = []
    for it in items:
        key = short_cache_key(it, template=template, model=model, passes=passes,
                              blank_policy=blank_policy)
        hit = cache.get(key)
        if hit is not None:
            out[it.qid] = dict(hit, cached=True)
        else:
            todo.append((key, it))
    if offline:
        misses = [it.qid for _, it in todo if not is_blank_without_call(it.prediction, blank_policy)]
        if misses:
            raise CacheMissError(misses)

    def work(pair: tuple[str, ShortItem]) -> tuple[str, str, dict[str, Any]]:
        key, it = pair
        _prompt, rec = grade_one(it, complete=complete, template=template, parse=parse,
                                 passes=passes, blank_policy=blank_policy)
        return key, it.qid, rec

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(work, p): p for p in todo}
        for fut in as_completed(futs):
            key, it = futs[fut]
            try:
                _, qid, rec = fut.result()
            except Exception as exc:  # noqa: BLE001 -- reported, scored 0, retried next run
                out[it.qid] = dict(correct=False, error=f"{type(exc).__name__}: {exc}",
                                   pred=str(it.prediction or "")[:200])
                continue
            cache.put(key, rec)
            out[qid] = rec
    cache.save()
    return out


def summarize_short(
    records: dict[str, dict[str, Any]],
    qids: list[str],
    *,
    buckets: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Accuracy over ALL ``qids`` (missing = incorrect), plus per-bucket counts."""
    n = len(qids)
    correct = [q for q in qids if (records.get(q) or {}).get("correct") is True]
    out: dict[str, Any] = dict(
        n=n,
        correct=len(correct),
        accuracy=(len(correct) / n) if n else 0.0,
        n_graded=sum(1 for q in qids if q in records and "error" not in records[q]),
        n_blank=sum(1 for q in qids if (records.get(q) or {}).get("blank")),
        n_missing=sum(1 for q in qids if q not in records),
        n_errors=sum(1 for q in qids if "error" in (records.get(q) or {})),
    )
    if buckets:
        cs = set(correct)
        out["buckets"] = {
            b: dict(n=len(ids), correct=sum(1 for q in ids if q in cs)) for b, ids in buckets.items()
        }
    return out


__all__ = [
    "BLANK_POLICY",
    "GRADER_NAME",
    "ShortItem",
    "build_short_prompt",
    "count_yes",
    "grade_one",
    "grade_short",
    "is_blank_without_call",
    "majority",
    "short_cache_key",
    "summarize_short",
]
