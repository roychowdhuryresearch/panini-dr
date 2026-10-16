"""MoNaCo dataset loader (eval side).

MoNaCo (Multi-hop Natural COmplex questions) answers are *multi-value*:
the gold ``validated_answer`` is a list of rows, where each row is a
tuple/list of atomic cells (multi-column answers), or — for
single-column answers — a flat list of items. We keep the structured
gold intact so the MoNaCo judge can compute set-overlap F1 over the
flattened fact set, while also exposing FRAMES-shaped attributes
(``id``, ``question``, ``num_hops``, ``articles``) so the existing
harness can consume MoNaCo questions the same way it consumes FRAMES
questions.

Source: the question file written by ``datasets/monaco/prepare.py``
(``$PANINI_DATA_DIR/monaco/monaco_full1315.json``, or ``$MONACO_QUESTIONS``,
or a subset's ``dataset_path``), each ``{id, question, answer, hop,
gold_pages}``. The article text is loaded separately via ``--articles-cache``
(there are no per-question ``articles`` payloads — retrieval runs against the
shared corpus).

There is no fallback to a smaller question file: a missing file is an error,
and so is a subset id that the file does not contain (see
``eval.subset.load_benchmark_questions``).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_HOP_RE = re.compile(r"(\d+)")


def default_questions_path() -> Path:
    """``$MONACO_QUESTIONS`` or ``$PANINI_DATA_DIR/monaco/monaco_full1315.json``.

    Resolved at call time so the override works in any import order.
    """
    from panini_dr.paths import data_dir

    return Path(
        os.environ.get("MONACO_QUESTIONS")
        or data_dir() / "monaco" / "monaco_full1315.json"
    )


def hops_from_hop(hop: Any) -> int:
    """Parse ``"3hop"`` / ``"7hop"`` (or a bare int) to an int number of hops."""
    if isinstance(hop, (int, float)):
        return int(hop)
    match = _HOP_RE.search(str(hop or ""))
    return int(match.group(1)) if match else 0


def flatten_answer(answer: Any) -> list[str]:
    """Recursively flatten a MoNaCo gold answer into atomic string items.

    Gold answers come in three shapes:
      - flat list:        ``["Poland", "Lithuania", ...]``
      - list of rows:     ``[["As the World Turns", "1956 - 2010"], ...]``
      - nested/mixed:      ``[["33.1%"], [["Congo", "25.3%"], ...]]``

    Every atomic cell (str / int / float / bool) becomes one item; empty
    strings are dropped. Order is preserved but duplicates are kept here
    (the judge de-duplicates via set semantics).
    """
    items: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            for child in node:
                _walk(child)
        elif isinstance(node, bool):
            items.append(str(node))
        elif isinstance(node, (str, int, float)):
            text = str(node).strip()
            if text:
                items.append(text)
        elif node is None:
            return
        else:  # pragma: no cover - defensive
            text = str(node).strip()
            if text:
                items.append(text)

    _walk(answer)
    return items


class MonacoQuestion(BaseModel):
    """One MoNaCo question, shaped like ``FramesQuestion`` for the harness."""

    id: str
    question: str
    # The raw structured gold: list-of-lists (rows of cells) or a flat list.
    answer: list[Any] = Field(default_factory=list)
    gold_pages: list[str] = Field(default_factory=list)
    hop: str = ""
    num_hops: int = 0
    # MoNaCo has no per-question article payloads — retrieval uses the shared
    # HYBRID corpus loaded via --articles-cache — so this stays empty. It
    # exists only so the harness can pass ``articles=q.articles`` uniformly.
    articles: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "MonacoQuestion":
        answer = raw.get("answer")
        if not isinstance(answer, list):
            answer = [] if answer is None else [answer]
        return cls(
            id=str(raw.get("id", "")),
            question=str(raw.get("question", "") or ""),
            answer=answer,
            gold_pages=list(raw.get("gold_pages") or []),
            hop=str(raw.get("hop", "") or ""),
            num_hops=hops_from_hop(raw.get("hop")),
            articles=list(raw.get("articles") or []),
            raw=raw,
        )

    def gold_items(self) -> list[str]:
        """Flattened set of atomic gold fact strings."""
        return flatten_answer(self.answer)

    def gold_text(self) -> str:
        """Human-readable ``; ``-joined rendering of the gold facts.

        Used for the ``QuestionResult.gold_answer`` string field and for
        logging. Multi-column rows are rendered as ``a | b``.
        """
        parts: list[str] = []
        for row in self.answer:
            if isinstance(row, (list, tuple)):
                cells = flatten_answer(row)
                if cells:
                    parts.append(" | ".join(cells))
            else:
                cell = flatten_answer(row)
                if cell:
                    parts.append(cell[0])
        return "; ".join(parts)


def load_monaco_eval(*, path: Path | str | None = None) -> list[MonacoQuestion]:
    """Load MoNaCo questions.

    Parameters
    ----------
    path:
        Explicit path override. Defaults to :func:`default_questions_path`.
    """
    if path is None:
        path = default_questions_path()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"MoNaCo question file not found at {path!s}. Build it with "
            "`python datasets/monaco/prepare.py` (writes "
            "$PANINI_DATA_DIR/monaco/monaco_full1315.json), or set MONACO_QUESTIONS "
            "or the subset's dataset_path. There is no fallback to another file."
        )
    with path.open() as f:
        raw = json.load(f)
    return [MonacoQuestion.from_raw(item) for item in raw]
