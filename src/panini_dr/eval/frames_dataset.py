"""FRAMES dataset loader.

Reads the FRAMES question file written by ``datasets/frames/prepare.py``
(``$PANINI_DATA_DIR/frames/frames_split.json`` for ``--split full``), or any
file passed explicitly (a subset's ``dataset_path``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

def default_questions_path(split: str = "full") -> Path:
    """``$PANINI_DATA_DIR/frames/frames_{dev,split}.json``, resolved at call time."""
    from panini_dr.paths import data_dir

    name = "frames_dev.json" if split == "dev" else "frames_split.json"
    return data_dir() / "frames" / name


# Kept for importers; resolved once at import. Prefer default_questions_path().
FRAMES_DEFAULT_DEV_PATH = default_questions_path("dev")
FRAMES_DEFAULT_FULL_PATH = default_questions_path("full")


class FramesQuestion(BaseModel):
    """One FRAMES question."""

    id: str
    question: str
    answer: str
    articles: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_types: list[str] = Field(default_factory=list)
    num_hops: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "FramesQuestion":
        # The FRAMES dev split stores two variants:
        #   - ``original_prompt`` — the well-formed full question,
        #     e.g. "Joel Oshiro Dyck played ice hockey for three teams.
        #     Which of those teams were dissolved after 2018-19?"
        #   - ``test_question``   — just the tail, e.g. "Which of those
        #     teams were dissolved after 2018-19?"  — contains unbound
        #     pronouns ("those teams", "that film") that make the question
        #     ambiguous without the preceding context sentence.
        # We want the grounded variant, so ``original_prompt`` wins.
        # A plain ``question`` key (the bring-your-own-dataset schema) is the
        # last resort; FRAMES files always carry ``original_prompt``.
        question = (
            raw.get("original_prompt")
            or raw.get("test_question")
            or raw.get("question")
            or ""
        )
        return cls(
            id=str(raw.get("id", "")),
            question=question,
            answer=raw.get("answer", ""),
            articles=list(raw.get("articles") or []),
            reasoning_types=list(raw.get("reasoning_types") or []),
            num_hops=int(raw.get("num_hops") or 0),
            raw=raw,
        )


def load_frames(
    *,
    split: str = "dev",
    path: Path | str | None = None,
) -> list[FramesQuestion]:
    """Load the dev (100 Qs) or full (824 Qs) FRAMES split.

    Parameters
    ----------
    split:
        ``"dev"`` (default) or ``"full"``. Ignored if ``path`` is given.
    path:
        Explicit path override.
    """
    if path is None:
        path = default_questions_path(split)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"FRAMES question file not found at {path!s}. Build it with "
            "`python datasets/frames/prepare.py` (writes $PANINI_DATA_DIR/frames/) "
            "or set the subset's dataset_path."
        )
    with path.open() as f:
        raw = json.load(f)
    return [FramesQuestion.from_raw(item) for item in raw]
