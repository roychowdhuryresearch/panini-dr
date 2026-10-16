"""Import judge verdicts from the research repo's ``<name>:<qid>``-keyed caches.

The research graders keyed verdicts on a run name and a question id only
(``bcp_official_judge.py``, ``frames_bcp_grader.py``: ``"<name>:<qid>"``;
the MoNaCo item re-judge, ``judge_items_pipeline.py``: ``"<name>#items:<qid>"``).
:mod:`panini_dr.eval.judge_cache` keys a verdict on everything that determines
it, so an old verdict can only be reused once it is tied to the answer it
graded. This module does that tie, and refuses whenever it cannot be checked:

* the run's own answer (``final_answer`` of its traces) is the answer the new
  key hashes; the prompt is rebuilt from the question file and the official
  template exactly as the grader would build it;
* **short answers** (BCP / FRAMES): the old record's ``pred`` must equal the
  trace answer (BCP stored the first 200 characters, FRAMES the whole answer)
  and its vote string must agree with its verdict and the number of passes;
  otherwise the record is reported as a conflict and not imported;
* **set answers** (MoNaCo): the old record must have one entry per pass and
  the gold length of the question; old item records store no prediction, so
  the answer can be checked only where the judge credited a single-answer
  prediction (it then echoes the prediction). Such records are counted as
  ``verified``, the others as ``unverified`` (imported, and reported);
* a blank answer: FRAMES-style blanks are scored without a call (nothing to
  import is needed); a MoNaCo blank is never judged.

Nothing here calls a judge.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from panini_dr.eval.grade_set import SetItem, gold_list, set_cache_key
from panini_dr.eval.grade_short import (
    ShortItem,
    is_blank_without_call,
    majority,
    short_cache_key,
)

#: How many leading characters of the answer the old BCP cache kept in ``pred``.
LEGACY_PRED_CHARS = 200

_VOTES = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


@dataclass
class ImportReport:
    """What happened to each question of the split."""

    imported: list[str] = field(default_factory=list)
    verified: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    blank: list[str] = field(default_factory=list)
    no_trace: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    conflicts: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(
            imported=len(self.imported), verified=len(self.verified),
            unverified=len(self.unverified), blank=len(self.blank),
            no_trace=len(self.no_trace), missing=len(self.missing),
            conflicts=dict(self.conflicts),
            missing_ids=list(self.missing), unverified_ids=list(self.unverified),
        )


def legacy_key(name: str, qid: str) -> str:
    return f"{name}:{qid}"


def _pred_matches(legacy_pred: Any, pred: str) -> bool:
    if legacy_pred is None:
        return False
    lp = str(legacy_pred)
    return lp == pred or lp == pred[:LEGACY_PRED_CHARS]


def import_short(
    legacy: dict[str, Any],
    name: str,
    items: list[ShortItem],
    *,
    template: str,
    model: str,
    passes: int,
    blank_policy: str,
    put: Callable[[str, dict[str, Any]], None],
) -> ImportReport:
    """Re-key BCP-template verdicts of run ``name`` for ``items`` (answers from the traces)."""
    rep = ImportReport()
    for it in items:
        pred = str(it.prediction or "")
        old = legacy.get(legacy_key(name, it.qid))
        if old is None:
            if is_blank_without_call(pred, blank_policy):
                rep.blank.append(it.qid)
            else:
                rep.missing.append(it.qid)
            continue
        blank = is_blank_without_call(pred, blank_policy)
        # frames_bcp_grader.py stored pred="" for any whitespace-only answer
        same = (str(old.get("pred") or "").strip() == "") if blank else _pred_matches(old.get("pred"), pred)
        if not same:
            rep.conflicts[it.qid] = "the cached answer differs from the run's answer"
            continue
        m = _VOTES.match(str(old.get("votes") or ""))
        if not m:
            rep.conflicts[it.qid] = f"unreadable vote string {old.get('votes')!r}"
            continue
        yes, n = int(m.group(1)), int(m.group(2))
        if blank:
            if n != 0 or old.get("correct") is not False:
                rep.conflicts[it.qid] = "a blank answer has a judged verdict"
                continue
        elif n != passes:
            rep.conflicts[it.qid] = f"{n} judge passes cached, {passes} requested"
            continue
        if not blank and bool(old.get("correct")) != majority(yes, n):
            rep.conflicts[it.qid] = "the verdict disagrees with its own votes"
            continue
        rec = dict(correct=bool(old.get("correct")), votes=f"{yes}/{n}",
                   extracted=old.get("extracted"), pred=pred[:LEGACY_PRED_CHARS],
                   blank=not pred.strip(), imported_from=legacy_key(name, it.qid))
        put(short_cache_key(it, template=template, model=model, passes=passes,
                            blank_policy=blank_policy), rec)
        rep.imported.append(it.qid)
        (rep.blank if blank else rep.verified).append(it.qid)
    return rep


def import_set(
    legacy: dict[str, Any],
    name: str,
    items: list[SetItem],
    *,
    single_prompt: str,
    multi_prompt: str,
    model: str,
    passes: int,
    put: Callable[[str, dict[str, Any]], None],
) -> ImportReport:
    """Re-key MoNaCo item verdicts (``{glen, passes: [...]}``) of run ``name``."""
    rep = ImportReport()
    for it in items:
        pred = str(it.prediction or "").strip()
        if not pred:
            rep.blank.append(it.qid)
            continue
        old = legacy.get(legacy_key(name, it.qid))
        if old is None:
            rep.missing.append(it.qid)
            continue
        ps = old.get("passes") or []
        glen = len(gold_list(it.answer))
        if len(ps) != passes:
            rep.conflicts[it.qid] = f"{len(ps)} judge passes cached, {passes} requested"
            continue
        if int(old.get("glen") or 0) != glen:
            rep.conflicts[it.qid] = f"cached gold length {old.get('glen')} != {glen}"
            continue
        echoed = [p.get("credited") for p in ps if glen == 1 and p.get("credited")]
        if echoed and any(list(c) != [pred] for c in echoed):
            rep.conflicts[it.qid] = "the cached credited answer differs from the run's answer"
            continue
        rec = dict(glen=glen, passes=ps, imported_from=legacy_key(name, it.qid))
        put(set_cache_key(it, single_prompt=single_prompt, multi_prompt=multi_prompt,
                          model=model, passes=passes), rec)
        rep.imported.append(it.qid)
        (rep.verified if echoed else rep.unverified).append(it.qid)
    return rep


__all__ = ["ImportReport", "LEGACY_PRED_CHARS", "import_set", "import_short", "legacy_key"]
