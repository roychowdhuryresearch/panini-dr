"""BrowseComp-Plus questions, shaped like ``FramesQuestion`` for the harness.

BrowseComp-Plus (Tevatron, ACL 2026) pairs 830 multi-constraint queries with a
fixed, human-verified corpus of 100,195 documents. Unlike FRAMES and MoNaCo it
ships per-query relevance judgments -- gold documents (which contain the answer)
and evidence documents (needed to reach it) -- so it is the only one of the three
that can measure **retrieval coverage per question**. That instrument is the
reason this benchmark is worth wiring in: its absence on MoNaCo is what let a
wrong-corpus run masquerade as an architecture failure for half a day on
2026-08-12.

Answers are short factoid strings (median 2 words), so this routes to the FRAMES
all-or-nothing judge, NOT MoNaCo's set-F1 -- which ``subset.py`` already does by
falling through to the default branch. Note that our judge is not their official
Qwen3-32B, so numbers from here are internally comparable but NOT
leaderboard-comparable; say so on any figure.

Retrieval uses the shared corpus via ``--articles-cache HYBRID_BCP.json``, so
``articles`` stays empty exactly as it does for MoNaCo.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field



def default_questions_path() -> Path:
    """Where the questions live, resolved AT CALL TIME.

    Deliberately a function, not a module constant. Resolving here means the
    override works from a shell export, a test monkeypatch, or a launcher
    script, in every ordering. MoNaCo follows the same rule.
    """
    from panini_dr.paths import data_dir

    return Path(
        os.environ.get("BROWSECOMP_QUESTIONS")
        or data_dir() / "browsecomp_plus" / "browsecomp_plus_830.json"
    )


class BrowseCompQuestion(BaseModel):
    """One BrowseComp-Plus query, shaped like ``FramesQuestion``."""

    id: str
    question: str
    answer: list[Any] = Field(default_factory=list)
    gold_pages: list[str] = Field(default_factory=list)
    hop: str = ""
    num_hops: int = 0
    # No per-question article payloads -- retrieval uses the shared corpus
    # loaded via --articles-cache. Exists so the harness can pass
    # ``articles=q.articles`` uniformly across benchmarks.
    articles: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "BrowseCompQuestion":
        answer = raw.get("answer")
        if not isinstance(answer, list):
            answer = [] if answer is None else [answer]
        return cls(
            id=str(raw.get("query_id", raw.get("id", ""))),
            question=str(raw.get("query", raw.get("question", "")) or ""),
            answer=answer,
            gold_pages=list(raw.get("gold_pages") or []),
            articles=list(raw.get("articles") or []),
            raw=raw,
        )

    def gold_items(self) -> list[str]:
        return [str(a) for a in self.answer if str(a).strip()]

    def gold_text(self) -> str:
        return "; ".join(self.gold_items())


def load_browsecomp_plus(
    *, path: Path | str | None = None
) -> list[BrowseCompQuestion]:
    """Load the BrowseComp-Plus questions file."""
    path = Path(path) if path is not None else default_questions_path()
    if not path.exists():
        raise FileNotFoundError(
            f"BrowseComp-Plus questions not found at {path!s}. Build it with "
            "`python datasets/bcp/prepare.py questions`, or set BROWSECOMP_QUESTIONS "
            "to point at another file."
        )
    with path.open() as f:
        rows = json.load(f)
    if isinstance(rows, dict):
        rows = rows.get("questions") or []
    return [BrowseCompQuestion.from_raw(r) for r in rows]
