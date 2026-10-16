"""Bring-your-own-dataset questions (``questions.jsonl``).

One JSON object per line::

    {"id": "q1", "question": "...", "answer": "Paris"}
    {"id": "q2", "question": "...", "answer": ["a", "b"], "answer_type": "set"}

``answer`` is a string or a list (a list of rows is allowed for multi-column
set answers). ``answer_type`` is ``short`` (one short answer, graded with the
short-answer grader) or ``set`` (a collection, graded with the set-F1 judge).
When it is missing it defaults to ``short`` for a string answer and ``set``
for a list. Other keys are kept in ``raw``.

The objects are shaped like ``FramesQuestion`` so the harness can run them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

ANSWER_TYPES = ("short", "set")


class ByodSchemaError(ValueError):
    """A questions.jsonl line does not follow the schema above."""


class ByodQuestion(BaseModel):
    id: str
    question: str
    answer: Any = ""
    answer_type: str = "short"
    num_hops: int = 0
    articles: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any], *, where: str = "") -> "ByodQuestion":
        qid = raw.get("id")
        question = raw.get("question")
        if qid is None or str(qid).strip() == "":
            raise ByodSchemaError(f"{where}: missing 'id'")
        if not isinstance(question, str) or not question.strip():
            raise ByodSchemaError(f"{where}: missing or empty 'question'")
        if "answer" not in raw:
            raise ByodSchemaError(f"{where}: missing 'answer'")
        answer = raw["answer"]
        if not isinstance(answer, (str, list)):
            raise ByodSchemaError(f"{where}: 'answer' must be a string or a list")
        answer_type = raw.get("answer_type") or ("set" if isinstance(answer, list) else "short")
        if answer_type not in ANSWER_TYPES:
            raise ByodSchemaError(
                f"{where}: answer_type must be one of {ANSWER_TYPES}, got {answer_type!r}"
            )
        return cls(
            id=str(qid),
            question=question,
            answer=answer,
            answer_type=answer_type,
            num_hops=int(raw.get("num_hops") or 0),
            raw=raw,
        )


def read_questions_jsonl(path: str | Path) -> list[ByodQuestion]:
    """Parse and validate a questions.jsonl file (ids must be unique)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"questions file not found: {path}")
    out: list[ByodQuestion] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path.name}:{lineno}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ByodSchemaError(f"{where}: not valid JSON ({exc})") from exc
            if not isinstance(raw, dict):
                raise ByodSchemaError(f"{where}: each line must be a JSON object")
            q = ByodQuestion.from_raw(raw, where=where)
            if q.id in seen:
                raise ByodSchemaError(f"{where}: duplicate id {q.id!r}")
            seen.add(q.id)
            out.append(q)
    if not out:
        raise ByodSchemaError(f"{path}: no questions")
    return out


def load_byod(*, path: str | Path | None) -> list[ByodQuestion]:
    if not path:
        raise FileNotFoundError(
            "a byod subset needs dataset_path (the prepared questions.jsonl)"
        )
    return read_questions_jsonl(path)


__all__ = ["ANSWER_TYPES", "ByodQuestion", "ByodSchemaError", "load_byod", "read_questions_jsonl"]
