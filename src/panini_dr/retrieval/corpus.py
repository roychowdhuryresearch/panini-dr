"""Article corpus loader + chunker.

A corpus is one JSON object ``{title: full text}`` (the ``--articles-cache``
file that ``panini-dr prepare`` or the scripts under ``datasets/`` write). We
chunk each article into ~512-token paragraphs and expose a simple store.

There is deliberately no default corpus: a run without one fails with
:class:`CorpusNotConfiguredError` instead of silently loading some other
benchmark's articles.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from panini_dr.retrieval.manifest import USER_DATASET, CorpusManifest, build_corpus_manifest


# Conservative token estimate: 1 token ≈ 4 chars. ~512 tokens ≈ 2000 chars.
DEFAULT_CHUNK_CHARS = 2000
DEFAULT_CHUNK_OVERLAP = 200
_log = logging.getLogger(__name__)


@dataclass(slots=True)
class Chunk:
    """One searchable chunk."""

    chunk_id: str
    title: str
    text: str
    char_start: int
    char_end: int


# A row is a bullet, a numbered entry, or a pipe-delimited table line; three in
# a row is a list. Kept in step with the reader's own detector.
_ROW_RE = re.compile(r"^\s*(?:[-*•]\s+\S|\d+[.)]\s+\S|[^|\n]{1,120}\|)")
_MIN_LIST_ROWS = 8
# A table chunk may run 6x a prose chunk. Measured on MoNaCo: 27% of list runs
# corpus-wide are cut by a boundary (36% on the articles behind our
# under-emission losses), and a half-table both scores worse against a query and
# hands the reader a fragment. Beyond this it still splits — on row boundaries.
MAX_TABLE_CHARS = 12_000


def _list_runs(text: str) -> list[tuple[int, int]]:
    """[(start, end)] for every stretch of >= _MIN_LIST_ROWS consecutive rows."""
    out: list[tuple[int, int]] = []
    pos, run_start, n = 0, None, 0
    for line in text.splitlines(keepends=True):
        if line.strip() and _ROW_RE.match(line):
            if run_start is None:
                run_start, n = pos, 0
            n += 1
        else:
            if run_start is not None and n >= _MIN_LIST_ROWS:
                out.append((run_start, pos))
            run_start, n = None, 0
        pos += len(line)
    if run_start is not None and n >= _MIN_LIST_ROWS:
        out.append((run_start, pos))
    return out


def _split_table(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split one list run into <= MAX_TABLE_CHARS pieces, never mid-row."""
    if end - start <= MAX_TABLE_CHARS:
        return [(start, end)]
    spans: list[tuple[int, int]] = []
    s = start
    cur = start
    for m in re.finditer(r"[^\n]*\n|[^\n]+$", text[start:end]):
        le = start + m.end()
        if le - s > MAX_TABLE_CHARS and cur > s:
            spans.append((s, cur))
            s = cur
        cur = le
    if cur > s:
        spans.append((s, cur))
    return spans


def _split_spans(
    text: str, max_chars: int, overlap: int, *, list_aware: bool = False
) -> list[tuple[int, int]]:
    """Chunk boundaries for one article.

    ``list_aware=False`` is the shipped behaviour, byte-identical. With it on,
    a list/table run becomes its own chunk (up to MAX_TABLE_CHARS) instead of
    being sliced at the 2,000-char prose cap, and the prose around it is
    chunked exactly as before. Off by default because the corpus fingerprint
    keys the dense index — flipping it re-embeds every chunk.
    """
    if not list_aware:
        return _split_on_paragraphs(text, max_chars, overlap)
    runs = _list_runs(text)
    if not runs:
        return _split_on_paragraphs(text, max_chars, overlap)
    spans: list[tuple[int, int]] = []
    pos = 0
    for rs, re_ in runs:
        if rs > pos:
            spans.extend((pos + s, pos + e)
                         for s, e in _split_on_paragraphs(
                             text[pos:rs], max_chars, overlap))
        spans.extend(_split_table(text, rs, re_))
        pos = re_
    if pos < len(text):
        spans.extend((pos + s, pos + e)
                     for s, e in _split_on_paragraphs(
                         text[pos:], max_chars, overlap))
    return [(s, e) for s, e in spans if e > s]


def _split_on_paragraphs(text: str, max_chars: int, overlap: int) -> list[tuple[int, int]]:
    """Greedy paragraph-aware chunking with hard-cap fallback.

    A "paragraph" is a run of non-empty lines (separated by blank
    lines). When a paragraph itself exceeds ``max_chars`` (e.g., a
    Wikipedia table flattened to one-row-per-line with no blank
    separators), we force-split it on line boundaries. As a final
    safety net, any single line longer than ``max_chars`` is
    character-sliced.
    """
    paragraphs = [m for m in re.finditer(r"[^\n]+(?:\n[^\n]+)*", text)]
    if not paragraphs:
        return [(0, len(text))]

    # First split each over-long paragraph into line-bounded
    # sub-spans of <= max_chars each. Paragraphs under the cap pass
    # through unchanged.
    sub_spans: list[tuple[int, int]] = []
    for p in paragraphs:
        if p.end() - p.start() <= max_chars:
            sub_spans.append((p.start(), p.end()))
            continue
        # Walk lines inside the over-long paragraph, accumulating up
        # to max_chars per sub-span.
        body_start = p.start()
        body_end = p.end()
        body = text[body_start:body_end]
        line_offsets: list[tuple[int, int]] = []  # (start, end) within body
        cur = 0
        for line_match in re.finditer(r"[^\n]*(?:\n|$)", body):
            ls, le = line_match.start(), line_match.end()
            if le == ls:
                continue
            line_offsets.append((ls, le))
        if not line_offsets:
            line_offsets = [(0, len(body))]
        sub_start = line_offsets[0][0]
        sub_end = sub_start
        for ls, le in line_offsets:
            if le - sub_start > max_chars:
                if sub_end > sub_start:
                    sub_spans.append((body_start + sub_start, body_start + sub_end))
                sub_start = ls
                sub_end = le
                # Final safety net: a single line longer than max_chars
                # gets character-sliced into max_chars pieces.
                if le - ls > max_chars:
                    pos = ls
                    while pos < le:
                        chunk_end = min(pos + max_chars, le)
                        sub_spans.append((body_start + pos, body_start + chunk_end))
                        pos = chunk_end
                    sub_start = le
                    sub_end = le
            else:
                sub_end = le
        if sub_end > sub_start:
            sub_spans.append((body_start + sub_start, body_start + sub_end))

    # Now greedily re-merge adjacent sub-spans up to max_chars (this
    # gives us paragraph-aware chunks where possible while never
    # exceeding the cap on a single chunk).
    spans: list[tuple[int, int]] = []
    if not sub_spans:
        return [(0, len(text))]
    start = sub_spans[0][0]
    cur_end = sub_spans[0][1]
    for s, e in sub_spans[1:]:
        if e - start > max_chars and cur_end > start:
            spans.append((start, cur_end))
            start = max(cur_end - overlap, s)
            cur_end = e
        else:
            cur_end = e
    if start < cur_end:
        spans.append((start, cur_end))
    return spans


class CorpusNotConfiguredError(RuntimeError):
    """Raised when a run has no corpus (no silent fallback to a default one)."""


def require_corpus(extra: dict) -> "ArticleCorpus":
    """Return ``extra["corpus"]`` or fail loudly.

    The adapters used to fall back to an old FRAMES article cache when no corpus
    was passed, which made a misconfigured run look like a real (bad) result.
    """
    corpus = extra.get("corpus")
    if not corpus:
        raise CorpusNotConfiguredError(
            "no corpus configured: pass --articles-cache <corpus.json> (or "
            "--dataset <prepared dir>) to `panini-dr run`, or put an ArticleCorpus "
            "in AdapterContext.extra['corpus']."
        )
    return corpus


class ArticleCorpus:
    """In-memory corpus of FRAMES articles, chunked for retrieval."""

    def __init__(
        self,
        chunks: list[Chunk],
        by_title: dict[str, str],
        manifest: CorpusManifest | None = None,
    ):
        self.chunks = chunks
        self._by_title = by_title
        # Set by load_frames_corpus; the dense cache key reads it so a
        # re-chunked corpus never loads the previous chunking's index.
        self.list_aware = False
        self._by_id = {c.chunk_id: c for c in chunks}
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.chunks)

    def article_text(self, title: str) -> str:
        return self._by_title.get(title, "")

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def titles(self) -> list[str]:
        return sorted(self._by_title.keys())


def load_frames_corpus(
    *,
    cache_path: Path | str | None = None,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    list_aware: bool = False,
    benchmark: str = "frames",
    dataset_name: str | None = None,
) -> ArticleCorpus:
    """Load the pre-cached FRAMES article dict and produce an ArticleCorpus.

    ``list_aware`` keeps list/table runs whole instead of slicing them at the
    prose cap. It changes chunk_ids, so it changes the corpus fingerprint and
    forces the dense index to be rebuilt.

    ``benchmark`` / ``dataset_name`` only label the corpus in its manifest and
    in the log line below: a paper benchmark (``frames``, ``monaco``,
    ``browsecomp_plus``) or ``byod`` with the name of a dataset written by
    ``panini-dr prepare``. They do not change the chunks.
    """
    if not cache_path:
        raise CorpusNotConfiguredError(
            "load_frames_corpus() needs an explicit cache_path: the {title: text} "
            "corpus JSON written by `panini-dr prepare` or datasets/<benchmark>/prepare.py."
        )
    path = Path(cache_path)
    if not path.is_file():
        raise CorpusNotConfiguredError(f"corpus file not found: {path}")
    with path.open() as f:
        data: dict[str, str] = json.load(f)

    chunks: list[Chunk] = []
    for title, text in data.items():
        if not text:
            continue
        spans = _split_spans(text, chunk_chars, chunk_overlap,
                             list_aware=list_aware)
        for i, (s, e) in enumerate(spans):
            chunks.append(
                Chunk(
                    chunk_id=f"{title}__{i}",
                    title=title,
                    text=text[s:e],
                    char_start=s,
                    char_end=e,
                )
            )
    manifest = build_corpus_manifest(
        path=path,
        articles=data,
        structured_unit_count=len(chunks),
        benchmark=benchmark,
        dataset_name=dataset_name,
    )
    if (benchmark or "").strip().lower() == USER_DATASET:
        _log.info(
            "Corpus %s: your own dataset (%d documents, %d chunks)",
            manifest.corpus_id,
            len(data),
            len(chunks),
        )
    elif not manifest.eligible_for_official_evaluation:
        _log.warning(
            "Corpus %s has scope=%s and official_evaluation_eligible=false; "
            "benchmark scores with this corpus are control runs only.",
            manifest.corpus_id,
            manifest.scope.value,
        )

    corpus = ArticleCorpus(chunks=chunks, by_title=data, manifest=manifest)
    corpus.list_aware = list_aware
    return corpus
