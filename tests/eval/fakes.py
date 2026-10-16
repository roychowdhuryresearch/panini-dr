"""Stand-ins for the authors' grading code and for judge transports.

Nothing here is the official BrowseComp-Plus or MoNaCo code: the template,
parser and V2 scorer below only mimic the shapes of their inputs/outputs so the
graders' own logic (votes, passes, aggregation, denominators, caching) can be
tested offline. All questions are made up.
"""
from __future__ import annotations

import re
import threading

FAKE_TEMPLATE = "Q: {question}\nR: {response}\nGOLD: {correct_answer}\n"


def fake_parse(reply: str) -> dict:
    m = re.search(r"correct:\s*(yes|no)", reply)
    ex = re.search(r"extracted_final_answer:\s*(.*)", reply)
    return {"correct": (m.group(1) == "yes") if m else None,
            "extracted_final_answer": ex.group(1).strip() if ex else None}


SINGLE_PROMPT = "SINGLE\n[question]: {question}\n[response]: '{response}'\n[correct_answer]: {correct_answer}\n"
MULTI_PROMPT = "MULTI\n[question]: {question}\n[response]: '{response}'\n[correct_answer]: {correct_answer}\n"


def fake_v2_score(judgement: str, gold_len: int):
    """Mimics the V2 scorer's output keys (not its code)."""
    if gold_len == 1:
        m = re.search(r"final precision:\s*([0-9.]+)", judgement)
        p = float(m.group(1)) if m else 0.0
        return {"judge_score": p, "precision": p}
    m = re.search(r"\nfinal answer length:\s*(\d+)", judgement)
    if not m or "\noverlapping answers:" not in judgement:
        return None
    plen = int(m.group(1))
    chunk = judgement.split("\noverlapping answers:")[1].strip()
    answers = [a for a in chunk.split("###") if a.strip() and a.strip() != "NULL"]
    nc = len(answers)
    if plen == 0 or nc == 0:
        return {"judge_score": 0.0, "precision": 0.0, "recall": 0.0}
    r = min(nc, gold_len) / gold_len
    p = nc / max(plen, nc)
    return {"judge_score": 2 * p * r / (p + r), "precision": p, "recall": r}


class ScriptedJudge:
    """A judge transport replying from a function of the prompt; counts calls."""

    name = "fake"

    def __init__(self, reply, model="fake-judge"):
        self.reply = reply
        self.model = model
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def complete(self, prompt: str) -> str:
        with self._lock:
            self.calls.append(prompt)
            i = len(self.calls)
        return self.reply(prompt, i) if callable(self.reply) else self.reply
