"""Agentic read: a sub-agent answers one question about one article.

`read` used to return a 12,000-char window and leave extraction to the parent.
Measured consequence: we hold 70.1% of gold items in retrieved evidence and emit
37.1% of them — readout efficiency 0.578. This module moves extraction to a
sub-agent that sees the whole article, and makes its claim checkable: every cited
span is verified against the text, and an answer that loses all its citations is
demoted to not-found rather than trusted.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from panini_dr.adapters.ours._planner_exec import _extract_json_object

# A model's character offsets drift; requiring exactness would drop good
# citations. When an offset IS supplied, proximity is the preferred match: it
# picks the occurrence the model pointed at rather than the first one in the
# article. It is NOT a veto, though — a verbatim span whose offset drifted (or
# was never given) is drift, not confabulation, and dropping it demotes a
# correct answer to a CACHED not_found. So a windowed miss falls back to the
# whole text; the guard that matters is "does this span occur at all".
CITATION_SLACK_CHARS = 200


@dataclass
class ReadResult:
    answer: str | None = None
    derivation: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    inputs_grounded: bool = False
    not_found_reason: str | None = None
    sections_checked: list[str] = field(default_factory=list)
    sections_skipped: list[str] = field(default_factory=list)
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # How many candidate rows the question's qualifier was applied to. None when
    # the reader did not report one (the normal case for a point question) — see
    # to_dict for why that is deliberately not the same as zero.
    candidates_considered: int | None = None
    # The article's head fact-table, returned verbatim to the caller. This is the
    # channel the reader used to destroy: with raw text replaced by
    # {answer, citations}, the parent saw an infobox on 1 of 2,477 FRAMES reads
    # (0%), against 74% for a plain offset-0 read — and scored 648 vs 667.
    infobox: str = ""
    # Loop instrumentation. The briefed reader is an agent, not one call, so
    # cost direction is uncertain (6x less context per call, but potentially
    # 3 calls instead of 1). Recorded so it is measured, not assumed.
    turns: int = 0
    tool_calls: int = 0
    # Set when the reply was cut off and salvaged: the citation list is PARTIAL.
    # Surfaced to the parent for the same reason as `coverage` — it cannot see the
    # article, so it cannot tell a short evidence list from a truncated one.
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.error is not None:
            out["error"] = self.error
            return out
        if self.answer is not None:
            out["answer"] = self.answer
            if self.derivation:
                out["derivation"] = self.derivation
                out["inputs_grounded"] = self.inputs_grounded
        else:
            out["not_found"] = True
            out["reason"] = self.not_found_reason or "not_stated_on_page"
        out["citations"] = self.citations
        out["sections_checked"] = self.sections_checked
        if self.sections_skipped:
            out["sections_skipped"] = self.sections_skipped
        if self.truncated:
            out["truncated"] = True
        if self.infobox:
            out["infobox"] = self.infobox
        # Coverage: how much of the candidate set the qualifier let through.
        #
        # This exists because the reader's worst failure looks like a success. On
        # monaco_88 the parent asked for "collaborations (multiple primary
        # artists)"; the reader returned 15 items at precision 1.00 and outcome
        # "answer", so nothing signalled a problem — but gold also counted "X
        # featuring Y", 30 more items, and the parent could not tell because it no
        # longer receives the article text. "matched 15, considered 100" makes the
        # narrowing visible and re-askable at the cost of one line.
        #
        # Emitted only when the reader reported a coherent count: a considered
        # count below the number of citations is impossible, and a nonsense ratio
        # would be acted on as if it were real, which is worse than no signal.
        matched = len(self.citations)
        if self.candidates_considered is not None and self.candidates_considered >= matched:
            out["coverage"] = {"matched": matched, "considered": self.candidates_considered}
        # NB: there is deliberately no `cached` flag here. A ReadResult does not
        # know whether it is being served from the ledger — `_tool_read` stamps
        # `payload["cached"] = True` on the returned dict, which is the single
        # mechanism for it.
        return out


def verify_citations(
    full_text: str, citations: Any, slack: int = CITATION_SLACK_CHARS
) -> tuple[list[dict[str, Any]], int]:
    """Keep citations whose span occurs in the article; drop the rest.

    Returns ``(kept, dropped_count)``. Kept citations carry the span's *real*
    offset, so downstream consumers get a usable pointer rather than the model's
    estimate.

    Match order, and why:

    1. When an ``offset`` key is **present**, search ``[offset-slack,
       offset+len(span)+slack]`` first. This is the preferred path: it resolves a
       span that occurs many times to the occurrence the model pointed at.
    2. When the key is **absent**, or present but the window missed, search the
       WHOLE text. An absent offset is not "offset 0" — treating it as such
       searched only the first ~200 chars and dropped a verbatim span sitting at
       char 35,000, demoting a correct answer to a not_found that ``_tool_read``
       then CACHED for the rest of the question.

    ``citations`` is unvetted model JSON, not a trusted shape: a non-list value
    (str, dict, int, ...) yields no citations and nothing to count as dropped;
    a list entry that is not itself a dict is skipped and counted as dropped,
    exactly like an empty span already is.
    """
    kept: list[dict[str, Any]] = []
    dropped = 0
    if not isinstance(citations, list):
        return kept, dropped
    low = full_text.lower()
    for cit in citations:
        if not isinstance(cit, dict):
            dropped += 1
            continue
        span = str(cit.get("span") or "")
        if not span.strip():
            dropped += 1
            continue
        needle = span.lower()
        idx = -1
        if "offset" in cit:
            try:
                claimed = int(cit["offset"])
            except (TypeError, ValueError):
                claimed = 0
            start = max(0, claimed - slack)
            idx = low.find(needle, start, claimed + len(span) + slack)
        if idx < 0:                      # no offset given, or the window missed
            idx = low.find(needle)
        if idx < 0:
            dropped += 1
            continue
        kept.append({"span": span, "offset": idx})
    return kept, dropped


def _candidate_count(value: Any) -> int | None:
    """Parse the reader's ``candidates_considered`` or return None.

    Unvetted model JSON: anything that is not a non-negative plain int is dropped
    rather than coerced. ``bool`` is excluded explicitly — it passes
    ``isinstance(x, int)`` in Python, and ``True`` would silently become a
    considered-count of 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


# The reader is an agent, not one call: it may search and page before answering.
# Measured at 4: every read that hit the cap had already spent two turns on find
# with real hits back — the budget ran out mid-investigation, not on a runaway.
MAX_READER_TURNS = 8


BRIEFED_READER_SYSTEM = (
    "You answer ONE question about ONE article, and you cite the exact text you "
    "used.\n"
    "\n"
    "You are given, up front: the article's INFOBOX (its head fact table), a "
    "SECTION MAP of where things are, and the RETRIEVED PASSAGE — the part of "
    "the article that actually matched the search that led here. Most questions "
    "are answerable from the infobox or that passage alone. Check them first.\n"
    "\n"
    "If they are not enough you have two tools over the FULL article:\n"
    "- `find(pattern)` — regex search the whole article; returns matches with "
    "their character offsets and surrounding text. Use this when you know what "
    "string you are looking for. It is exact and cheap.\n"
    "- `read_full(offset)` — a raw window of the article starting at `offset`. "
    "Use the section map to choose the offset rather than paging blindly.\n"
    "\n"
    f"You have at most {MAX_READER_TURNS} turns. When you have the answer, or "
    "are confident the article does not contain it, stop and reply.\n"
    "\n"
    "Answer the property that was ASKED FOR, not the nearest one on the page. "
    "If the passage states a related but different property — the spike's "
    "height when the flower's was asked for, the career total when a single "
    "season was asked for — `find` the exact one before answering. A verbatim "
    "citation of the wrong property is still a wrong answer.\n"
    "Never conclude that something is absent from the article on the strength "
    "of the passage alone: `find` it across the full article first.\n"
    "\n"
    "You MAY compute an answer the article does not state literally (arithmetic, "
    "date differences, counts) — put the calculation in `derivation` and cite the "
    "spans you computed FROM.\n"
    "Every claim needs a citation: the exact substring from the article and its "
    "approximate character offset. Do not invent spans; a citation that does not "
    "occur in the article is discarded and your answer is thrown away with it.\n"
    "Keep each span SHORT — the shortest exact substring that pins the claim.\n"
    "If the article does not support an answer, say so — a scoped negative is a "
    "useful result, not a failure.\n"
    "When the question asks you to FILTER a list, also report "
    "`candidates_considered`: how many entries you applied the filter to.\n"
    "Reply with ONE JSON object and nothing else. It must be exactly one of "
    "these four shapes:\n"
    '1. an answer:      {"answer": "...", "derivation": "... or null", '
    '"citations": [{"span": "exact text", "offset": 1234}], '
    '"candidates_considered": 100}\n'
    '2. a scoped miss:  {"not_found": true, "reason": "why the article does not '
    'answer it"}\n'
    '3. search it:      {"tool": "find", "pattern": "regex to search for"}\n'
    '4. read a window:  {"tool": "read_full", "offset": 38078}\n'
    "\n"
    "Shapes 3 and 4 are how you use the tools — the result comes back and you "
    "reply again. Do not describe a tool call in prose; emit the JSON."
)

# How much of the retrieved chunk to hand over. The chunk is the passage search
# matched, so it is the highest-value text we have; a cap only guards against a
# pathologically large one.
MAX_BRIEFING_CHUNK_CHARS = 6_000
MAX_BRIEFING_MAP_ENTRIES = 60


# A "row" is a bullet, a numbered entry, or a pipe-delimited table line. Three
# in a row is a list; two is a coincidence.
_ROW_RE = re.compile(r"^\s*(?:[-*•]\s+\S|\d+[.)]\s+\S|[^|\n]{1,120}\|)")
_MIN_RUN_ROWS = 3
# A run has to be a LIST before completing it is worth anything. Measured on the
# first attempt with the gate at 3: the completion fired on 60% of reads, median
# +7 rows, because `key | value` infobox lines and reference blocks all count as
# rows. Point-fact reads got padded with fact-table fragments and single-answer
# F1 fell to 0.846 against champion's 0.892, while multi-value rose to 0.468 vs
# 0.434 — the mechanism worked and the trigger paid for it on every other read.
_MIN_COMPLETABLE_ROWS = 8
# ...and the part hanging outside the chunk has to be worth a round trip.
_MIN_ROWS_OUTSIDE = 4
# The completion has to stay far below a whole-article dump — the briefing exists
# because the 30k-char dump lost. 12k chars is ~3x a normal chunk and ~6x smaller
# than the article mean.
MAX_RUN_COMPLETION_CHARS = 12_000


def _row_runs(text: str) -> list[tuple[int, int, int]]:
    """[(start, end, n_rows)] for every stretch of >= _MIN_RUN_ROWS rows."""
    out: list[tuple[int, int, int]] = []
    pos = 0
    run_start: int | None = None
    n = 0
    for line in text.splitlines(keepends=True):
        if line.strip() and _ROW_RE.match(line):
            if run_start is None:
                run_start, n = pos, 0
            n += 1
        else:
            if run_start is not None and n >= _MIN_RUN_ROWS:
                out.append((run_start, pos, n))
            run_start, n = None, 0
        pos += len(line)
    if run_start is not None and n >= _MIN_RUN_ROWS:
        out.append((run_start, pos, n))
    return out


def _rows_between(text: str, lo: int, hi: int) -> int:
    """How many row-lines fall in [lo, hi) — the size of what a widening buys."""
    if hi <= lo:
        return 0
    return sum(1 for line in text[lo:hi].splitlines()
               if line.strip() and _ROW_RE.match(line))


def complete_cut_run(
    full_text: str, chunk_start: int, chunk_end: int,
    max_chars: int = MAX_RUN_COMPLETION_CHARS,
) -> tuple[int, int, int, bool]:
    """Widen [chunk_start, chunk_end) to cover any list/table run it cuts.

    Returns ``(start, end, rows_added, truncated)``. When the chunk cuts no run
    the span is returned unchanged and ``rows_added`` is 0 — the ordinary
    point-fact read pays nothing.

    Why this exists, measured on MoNaCo-200: 27% of list/table runs corpus-wide
    are cut by a chunk boundary and 36% on the articles behind our
    under-emission losses. Of every gold item the reader missed, 37% sat in a
    retrieved article OUTSIDE the chunk it was briefed on — reachable via
    read_full, but only by paginating against an 8-turn cap. Fragmentation does
    not block access; it makes complete access unaffordable.
    """
    if not full_text or chunk_end <= chunk_start:
        return chunk_start, chunk_end, 0, False
    start, end = chunk_start, chunk_end
    for rs, re_, nrows in _row_runs(full_text):
        if re_ <= chunk_start or rs >= chunk_end:
            continue                      # run does not touch the chunk
        if nrows < _MIN_COMPLETABLE_ROWS:
            continue                      # an infobox / short block, not a list
        outside = (_rows_between(full_text, rs, chunk_start)
                   + _rows_between(full_text, chunk_end, re_))
        if outside < _MIN_ROWS_OUTSIDE:
            continue                      # nothing material hanging off the edge
        start = min(start, rs)
        end = max(end, re_)
    if (start, end) == (chunk_start, chunk_end):
        return chunk_start, chunk_end, 0, False
    truncated = False
    if end - start > max_chars:
        end = start + max_chars
        truncated = True
    added = (_rows_between(full_text, start, chunk_start)
             + _rows_between(full_text, chunk_end, end))
    return start, end, added, truncated


def build_reader_briefing(
    *,
    title: str,
    question: str,
    full_text: str,
    chunk_text: str,
    chunk_start: int,
    outline: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The reader's opening context: infobox + section map + retrieved passage.

    Deliberately NOT the whole article. The single-shot reader received ~30k
    chars and no indication of which chunk matched; 24% of its reads concerned a
    passage >12k chars in, and it went looking for something the retriever had
    already located. Here we hand over what is already known and let the reader
    ask for the rest via `find` / `read_full`.
    """
    ib = extract_infobox(full_text)
    parts = [f"ARTICLE: {title}   ({len(full_text):,} chars)",
             f"QUESTION: {question}"]
    parts.append("INFOBOX\n" + ib if ib
                 else "INFOBOX\n(this article has no infobox)")
    if outline:
        rows = []
        for e in outline[:MAX_BRIEFING_MAP_ENTRIES]:
            st = _safe_int(e.get("start"), 0)
            name = str(e.get("section") or "")
            if not name:
                continue
            # Mark the section the retrieved passage falls in, so the reader can
            # see where it already is rather than inferring it.
            here = " <- the retrieved passage is in this section" if (
                st <= chunk_start and chunk_start < st + 100_000
                and all(_safe_int(o.get("start"), 0) <= chunk_start
                        or _safe_int(o.get("start"), 0) > chunk_start
                        for o in outline)
                and st == max((_safe_int(o.get("start"), 0) for o in outline
                               if _safe_int(o.get("start"), 0) <= chunk_start),
                              default=-1)
            ) else ""
            rows.append(f"  {name} ... {st:,}{here}")
        if rows:
            parts.append("SECTION MAP\n" + "\n".join(rows))
    body = (chunk_text or "")[:MAX_BRIEFING_CHUNK_CHARS]
    # If the chunk boundary cut a list or table, hand over the WHOLE run. The
    # reader could always have paged to it, but paging a 334-row table split
    # across 16 chunks does not fit an 8-turn budget.
    r_start, r_end, added, cut = complete_cut_run(
        full_text, chunk_start, chunk_start + len(chunk_text or ""))
    if added:
        body = full_text[r_start:r_end]
        note = (f"  [+{added} row(s) restored: the chunk boundary cut a "
                f"list/table]")
        if cut:
            note += (f"\n  [TRUNCATED at char {r_end:,} — continue with "
                     f"read_full(offset={r_end})]")
        parts.append(
            f"RETRIEVED PASSAGE  (chars {r_start:,}-{r_end:,}) "
            f"— this is what the search matched\n{note}\n{body}"
        )
    else:
        parts.append(
            f"RETRIEVED PASSAGE  (chars {chunk_start:,}-{chunk_start + len(body):,}) "
            f"— this is what the search matched\n{body}"
        )
    system = BRIEFED_READER_SYSTEM
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


# The head fact-table. Wikipedia renders it as an "[INFOBOX]" marker followed by
# "key | value" lines, ending at the first blank line that precedes prose. Capped
# because a few pages carry enormous infoboxes and this rides on every ask().
INFOBOX_MARKER = "[INFOBOX]"
# 2,000 truncated 7.2% of FRAMES infoboxes and 16.9% of MoNaCo ones (median
# 634 chars lost, max 2,932). FRAMES p99 is 3,556; champion's 12k raw window
# never cut one.
MAX_INFOBOX_CHARS = 4_000


def extract_infobox(full_text: str) -> str:
    """Return the article's infobox verbatim, or "" when it has none.

    Deliberately literal: the marker is emitted by our own corpus renderer, so a
    marker-based cut is exact where a heuristic would guess. Everything after the
    table's first blank-line break is prose and is excluded.
    """
    i = (full_text or "").find(INFOBOX_MARKER)
    if i < 0:
        return ""
    tail = full_text[i + len(INFOBOX_MARKER):]
    end = tail.find("\n\n")
    body = (tail[:end] if end > 0 else tail).strip()
    return body[:MAX_INFOBOX_CHARS]


def _salvage_truncated_json(text: str) -> dict[str, Any] | None:
    """Recover a partial object from a reply that was cut off mid-emission.

    Returns None when ``text`` is not truncated JSON — either it parses already,
    or it is not JSON at all. Only truncation is recoverable; genuine nonsense
    must stay an error.

    Why this is worth doing: the schema emits ``answer`` before ``citations``, so
    a reply that runs out of room has usually finished the answer and is partway
    through the evidence list. Discarding it throws away a complete answer over a
    missing bracket. Every salvaged citation is still verified against the article
    afterwards, so this loosens no grounding guarantee — it only declines to
    treat "the list was cut short" as "there is no evidence".
    """
    start = text.find("{")
    if start < 0:
        return None
    stack: list[str] = []
    in_str = esc = False
    # Last index at which a value had just finished, with the containers open at
    # that moment. Cutting anywhere else leaves a half-written value behind.
    safe: tuple[int, tuple[str, ...]] | None = None
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            safe = (i + 1, tuple(stack))
        elif ch == ",":
            safe = (i, tuple(stack))          # cut BEFORE the comma
    if not stack:
        return None                            # complete: nothing to salvage
    if safe is None:
        return None                            # nothing finished before the cut
    cut, open_at_cut = safe
    try:
        return json.loads(text[start:cut] + "".join(reversed(open_at_cut)))
    except Exception:  # noqa: BLE001 - a failed salvage is just "not salvageable"
        return None


def parse_read_response(
    text: str,
    full_text: str,
    sections_checked: list[str],
    sections_skipped: list[str],
    infobox: str = "",
) -> ReadResult:
    base = ReadResult(
        sections_checked=list(sections_checked),
        sections_skipped=list(sections_skipped),
        infobox=infobox,
    )
    try:
        obj = _extract_json_object(text)
    except Exception as exc:  # noqa: BLE001 - any parse failure is an error, not an answer
        obj = _salvage_truncated_json(text)
        if obj is None:
            base.error = f"unparseable reader response: {exc}"
            return base
        base.truncated = True

    # Read before the not_found branch: "considered 100, matched 0" is exactly the
    # case where a scoped negative is informative rather than merely empty.
    base.candidates_considered = _candidate_count(obj.get("candidates_considered"))

    if obj.get("not_found"):
        base.not_found_reason = str(obj.get("reason") or "not_stated_on_page")
        return base

    answer = obj.get("answer")
    if answer is None or not str(answer).strip():
        base.not_found_reason = "reader returned no answer"
        return base

    kept, dropped = verify_citations(full_text, obj.get("citations") or [])
    if not kept:
        # Confabulation guard: no verifiable evidence means no answer.
        base.not_found_reason = "citations_unverifiable"
        return base

    base.answer = str(answer).strip()
    deriv = obj.get("derivation")
    base.derivation = str(deriv).strip() if deriv else None
    base.citations = kept
    # Grounded means every citation the model offered verified — not merely
    # that at least one did (a partial drop is a partial confabulation).
    # A salvaged reply lost citations to the cut, not to confabulation — but it
    # still lost them, so claiming full grounding would be false either way.
    base.inputs_grounded = dropped == 0 and not base.truncated
    return base


MAX_READ_CHARS = 150_000

# The budget that actually binds is TOKENS, not characters. monaco_613 sent a
# 121,028-char article — comfortably under MAX_READ_CHARS — and the server
# rejected it: "Input length (71074) exceeds model's maximum context length
# (65536)". Polling tables run ~1.7 chars/token where prose runs ~4, so a single
# character budget either wastes most of the context on prose or overflows on
# tables. It overflowed, and a 400 loses the whole read on exactly the
# table-heavy list pages the reader exists for.
#
# Input and output share one context, so they are derived from it rather than
# picked independently — two free-standing magic numbers drift apart, and the
# first re-run showed exactly that: the input budget was fixed while the output
# budget stayed at 4,000 and became the new failure.
MODEL_CONTEXT_TOKENS = 65_536

# 12,000, not 4,000. A 100-row list answer cites ~100 spans, and on this model
# the reasoning trace is charged against the same completion budget, so 4,000 ran
# out mid-JSON: every truncation observed hit exactly 4000 completion tokens and
# cost the whole read. This buys roughly 3x the room at the price of ~4,000
# tokens of input budget, which the token-density fix above already made
# affordable.
READ_COMPLETION_TOKENS = 12_000

# System prompt, question, skipped-section note and chat-template scaffolding.
_PROMPT_OVERHEAD_TOKENS = 1_500

# What is left for article text. Generous margin retained deliberately:
# `estimate_tokens` is an estimate, and a 400 costs the entire read.
MAX_READ_TOKENS = MODEL_CONTEXT_TOKENS - READ_COMPLETION_TOKENS - _PROMPT_OVERHEAD_TOKENS

# o200k_base is not gpt-oss's exact vocabulary, but on the article that failed it
# counted 70,766 against the server's reported 71,074 — within 0.5%, in 0.03s.
# Absent tiktoken the heuristic below takes over; nothing here is a hard dependency.
try:  # pragma: no cover - import-time capability probe
    import tiktoken as _tiktoken

    _ENCODER: Any = _tiktoken.get_encoding("o200k_base")
except Exception:  # noqa: BLE001 - any failure just means "estimate instead"
    _ENCODER = None

# Word-ish and punctuation runs. Counts each number and pipe in a table row
# separately, which is why it tracks density rather than assuming a flat ratio.
_TOKENISH_RE = re.compile(r"\w+|[^\w\s]")

# Measured real/tokenish on the failing article was 1.299. 1.4 keeps the fallback
# on the conservative side of that, since under-estimating is what causes a 400.
_TOKENISH_TO_TOKENS = 1.4

# Applied to the derived character limit. Density is measured over the WHOLE
# article but spent on a PREFIX, and the prefix can be denser than the mean.
_BUDGET_SAFETY = 0.95


# BPE merging is quadratic in the length of a repeated-character run: measured on
# o200k_base, encoding a run of spaces cost 0.08s at 10k chars, 1.71s at 50k,
# 4.70s at 100k and 81s at 400k. The reader estimates on EVERY read, so one
# article padded with a long run would stall a question for over a minute.
# Collapsing runs to 100 makes it instant (0.000s) and leaves real prose untouched
# (225k chars in 0.022s), because no meaningful text repeats one character 100
# times.
_RUN_RE = re.compile(r"(.)\1{99,}", re.S)
_RUN_KEEP = 100

# Tokens charged for each character dropped by the collapse. Runs pack far denser
# than 4 chars/token, so charging them at the PROSE rate deliberately
# over-estimates: over-estimating shrinks the budget, under-estimating causes the
# 400 this whole change exists to prevent.
_COLLAPSED_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate the token cost of ``text``.

    Exact-ish when tiktoken is importable, heuristic otherwise. Never raises: a
    tokeniser failure must degrade to an estimate, not break the read.
    """
    if not text:
        return 0
    collapsed = _RUN_RE.sub(lambda m: m.group(1) * _RUN_KEEP, text)
    # Charged separately rather than ignored: the collapse is a speed device, and
    # the dropped characters are still sent to the model.
    dropped = (len(text) - len(collapsed)) // _COLLAPSED_CHARS_PER_TOKEN
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(collapsed)) + dropped
        except Exception:  # noqa: BLE001 - fall through to the heuristic
            pass
    return int(len(_TOKENISH_RE.findall(collapsed)) * _TOKENISH_TO_TOKENS) + dropped


def effective_char_limit(
    full_text: str,
    char_limit: int = MAX_READ_CHARS,
    token_limit: int = MAX_READ_TOKENS,
) -> int:
    """Characters we can afford to send, given ``full_text``'s own token density.

    Prose is left alone — it hits ``char_limit`` long before ``token_limit`` — so
    this only bites on the dense articles that were overflowing. The limit is
    scaled by the article's measured chars-per-token rather than a fixed ratio,
    because the ratio is the thing that varies (1.7 for polling tables, ~4 for
    prose); assuming either one is what produced the bug.
    """
    tokens = estimate_tokens(full_text)
    if tokens <= token_limit:
        return char_limit
    density = len(full_text) / tokens          # chars per token, this article
    return max(1, min(char_limit, int(token_limit * density * _BUDGET_SAFETY)))

def _safe_int(value: Any, default: int) -> int:
    """Coerce ``value`` to ``int``, degrading to ``default`` instead of raising.

    An outline entry's ``start`` is unvetted, like ``verify_citations``'s
    ``offset`` — missing, ``None``, or non-numeric must not crash selection.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---- briefed reader: tools + loop -----------------------------------------
_FIND_CONTEXT_CHARS = 220
_MAX_FIND_MATCHES = 8
# How many matched spans an exhausted read hands back as partial evidence.
_MAX_PARTIAL_SPANS = 6

# Measured over 146 reads: 84% of not_found reads searched the article first, but
# only 33% of ANSWERS did. The reader reaches for find when it is about to give
# up and skips it when it believes it already has the answer — which is backwards
# for the two losses it caused. q185 answered "None of the actors became
# President" off a cast list (Trump has a cameo); q318 answered the SPIKE height
# when asked for the FLOWER height, citing the spike verbatim. Both cited real
# text; both never searched. Verbatim is not the same as relevant.
_ABSENCE_RE = re.compile(
    r"\b(none|no one|nobody|not any|there (?:are|were|is|was) no|"
    r"does not|did not|do not|never|neither)\b", re.I)


def answer_needs_a_search(*, answer: str) -> bool:
    """True when a passage-only answer has not earned the right to be final.

    One trigger, and only one: the answer asserts absence. You cannot establish
    that an article says nothing by reading part of it.

    A second trigger was built and REFUTED before it shipped: "no citation
    contains an attribute term from the question", aimed at q318 (answered the
    SPIKE height when asked the FLOWER height). Replayed over the run's reads it
    fired on 62% of answers and all 13 inspected fires were false positives —
    citing a bare value ("1929", "5:31", "November 12, 1988") is the CORRECT
    citation for a point fact; the attribute word lives in the question, not in
    the span. It would have bought one extra LLM call on two reads in three. The
    wrong-attribute case is handled in the prompt instead, where a false positive
    costs nothing.
    """
    return bool(_ABSENCE_RE.search(answer or ""))


def reader_find(full_text: str, pattern: str) -> dict[str, Any]:
    """Regex over the WHOLE article, with offsets and surrounding text.

    Exact, where "read 30k chars and hope" is not. A bad regex is the model's
    error, not a crash: report zero matches and say why.
    """
    try:
        rx = re.compile(pattern, re.I)
    except re.error as exc:
        return {"match_count": 0, "matches": [], "error": f"bad regex: {exc}"}
    out = []
    for m in rx.finditer(full_text or ""):
        lo = max(0, m.start() - _FIND_CONTEXT_CHARS)
        hi = min(len(full_text), m.end() + _FIND_CONTEXT_CHARS)
        out.append({"offset": m.start(), "text": full_text[lo:hi]})
        if len(out) >= _MAX_FIND_MATCHES:
            break
    # The COUNT is the truth; only the LIST is capped. Reporting len(out) meant a
    # saturated find said "8 matches" — measured on MoNaCo, 27.1% of find calls
    # returned exactly 8 — so a reader enumerating a 40-row table was told the
    # table had 8 rows, and 33% of list-intent asks came back with one item.
    total = sum(1 for _ in rx.finditer(full_text or ""))
    return {"match_count": total, "matches": out,
            "truncated": total > len(out)}


MAX_TOOL_RESULT_CHARS = 20_000


def render_tool_result(res: dict[str, Any]) -> str:
    """Serialise a tool result for the reader, trimming the PAYLOAD not the JSON.

    This was ``json.dumps(res)[:20000]`` — a cut mid-string with no closing
    brace. Measured: 41% of FRAMES and 50% of MoNaCo ``read_full`` windows
    exceeded the cap, so the reader's own escalation-to-full-article path handed
    back invalid JSON half the time, with no signal that anything was missing.
    """
    body = json.dumps(res)
    if len(body) <= MAX_TOOL_RESULT_CHARS:
        return body
    trimmed = dict(res)
    text = str(trimmed.get("text") or "")
    if text:
        keep = max(0, MAX_TOOL_RESULT_CHARS - (len(body) - len(text)) - 200)
        trimmed["text"] = text[:keep]
        trimmed["truncated_for_context"] = True
        out = json.dumps(trimmed)
        if len(out) <= MAX_TOOL_RESULT_CHARS:
            return out
    # No single oversized field to trim (e.g. a huge match list): drop matches.
    trimmed["matches"] = (trimmed.get("matches") or [])[:2]
    trimmed["truncated_for_context"] = True
    return json.dumps(trimmed)[:MAX_TOOL_RESULT_CHARS]


def reader_read_full(full_text: str, offset: int = 0) -> dict[str, Any]:
    """A raw window of the article, token-budgeted and pageable.

    Reuses the density-aware budget, so a 300k-char article is not dumped, and
    the caller is told where the window ends so it can page on.
    """
    total = len(full_text or "")
    start = max(0, min(int(offset or 0), total))
    limit = min(effective_char_limit(full_text or ""), MAX_READ_CHARS)
    end = min(start + limit, total)
    return {
        "window": {"start": start, "end": end},
        "article_chars": total,
        "text": (full_text or "")[start:end],
        "truncated": end < total,
    }


def read_article_briefed(
    *,
    title: str,
    full_text: str,
    question: str,
    chunk_text: str,
    chunk_start: int,
    outline: list[dict[str, Any]],
    llm: Any,
    seed: int | None = None,
    max_tokens: int = READ_COMPLETION_TOKENS,
) -> ReadResult:
    """Briefed reader: opens on infobox + map + retrieved passage, escalates via
    find/read_full, capped at MAX_READER_TURNS.

    Never raises — any failure returns a ReadResult with ``error`` set, matching
    the contract callers already rely on.
    """
    ib = extract_infobox(full_text)
    # What the reader ACTUALLY examined, so a not_found stays scoped. Under the
    # single-shot reader this was the list of sections it was handed; the briefed
    # reader is not handed sections, so reporting them would overclaim. It reads
    # the infobox and the retrieved passage, then whatever its tools return —
    # that is what gets recorded, and each tool call appends to it below.
    secs = (["infobox"] if ib else []) + ["retrieved passage"]
    msgs = build_reader_briefing(
        title=title, question=question, full_text=full_text,
        chunk_text=chunk_text, chunk_start=chunk_start, outline=outline,
    )
    last = ReadResult(sections_checked=secs, infobox=ib)
    tool_calls = 0
    pushed_back = False
    total_prompt_tokens = 0
    total_completion_tokens = 0
    # Every span `find` actually matched, kept so an exhausted read can hand it
    # back. Measured in the 2026-08-01 postmortem: 86-89% of budget-exhausted
    # reads HAD a find that returned matches and still returned citations: [],
    # and the not_found slice owned 100% of arm A's FRAMES deficit (21 vs 27,
    # with both other slices exactly tied).
    matched_spans: list[dict[str, Any]] = []
    for turn in range(1, MAX_READER_TURNS + 1):
        try:
            resp = llm.chat(msgs, temperature=0.0, max_tokens=max_tokens, seed=seed)
        except Exception as exc:  # noqa: BLE001 - never raise into the tool loop
            last.error = str(exc)[:300]
            last.prompt_tokens = total_prompt_tokens
            last.completion_tokens = total_completion_tokens
            last.turns, last.tool_calls = turn, tool_calls
            return last
        total_prompt_tokens += int(getattr(resp, "prompt_tokens", 0) or 0)
        total_completion_tokens += int(getattr(resp, "completion_tokens", 0) or 0)
        raw = resp.text or resp.reasoning_content or ""
        try:
            obj = _extract_json_object(raw)
        except Exception:  # noqa: BLE001
            obj = _salvage_truncated_json(raw)
        if isinstance(obj, dict) and obj.get("tool") in ("find", "read_full"):
            if turn == MAX_READER_TURNS:
                break                    # no budget left to use the result
            tool_calls += 1
            if obj["tool"] == "find":
                pat = str(obj.get("pattern") or "")
                res = reader_find(full_text, pat)
                secs.append(f"find({pat[:60]}) -> {res['match_count']} match(es)")
                for m in (res.get("matches") or [])[:_MAX_PARTIAL_SPANS]:
                    txt = str(m.get("text") or "").strip()
                    if txt:
                        matched_spans.append({"span": txt[:400],
                                              "offset": _safe_int(m.get("offset"), 0)})
            else:
                off = _safe_int(obj.get("offset"), 0)
                res = reader_read_full(full_text, off)
                secs.append(f"read_full(offset={off}) -> chars "
                            f"{res['window']['start']}-{res['window']['end']}")
            msgs = msgs + [
                # The REQUEST only. Measured: 167 replies in one run and 58 in
                # another carried the tool request AND an invented result after
                # it — usually {"match_count": 0, "matches": []}, once "Tool
                # execution failed: read_full not supported in this environment"
                # (a string our code never emits). Echoing the whole reply put
                # the model's imaginary "0 matches" in the history immediately
                # before the real result, leaving it to arbitrate between them —
                # and an invented failure is exactly what stops it searching
                # again. It only ever sees what it ASKED for.
                {"role": "assistant", "content": json.dumps(obj)},
                {"role": "user", "content": "TOOL RESULT\n" + render_tool_result(res)},
            ]
            continue
        result = parse_read_response(raw, full_text, secs, [], infobox=ib)
        result.prompt_tokens = total_prompt_tokens
        result.completion_tokens = total_completion_tokens
        result.turns, result.tool_calls = turn, tool_calls
        if (result.error is None and result.answer is not None
                and tool_calls == 0 and not pushed_back
                and turn < MAX_READER_TURNS
                and answer_needs_a_search(answer=result.answer)):
            # Once only: the budget is for investigating, not for arguing.
            pushed_back = True
            msgs = msgs + [
                {"role": "assistant", "content": raw[:2000]},
                {"role": "user", "content": (
                    "Your answer says the article does not contain something, "
                    "but you only saw the retrieved passage — you cannot "
                    "establish absence from part of an article. Run "
                    "find(pattern) over the FULL article first, then answer. "
                    "If the search confirms it, repeat your answer unchanged."
                )},
            ]
            continue
        if result.error is None:
            return result
        last = result
        # The reply did not parse. Re-sending the SAME messages just gets the same
        # reply — measured: 13 of 46 reads burned all four turns this way, every
        # one on "no opening brace in response", 52 wasted calls. Tell the model
        # what was wrong so the retry can differ.
        msgs = msgs + [
            {"role": "assistant", "content": raw[:2000]},
            {"role": "user", "content": (
                "That reply could not be parsed: " + str(result.error) + ". "
                "Reply with ONE JSON object and nothing else — either "
                '{"answer": ..., "citations": [...]}, or '
                '{"not_found": true, "reason": ...}, or a tool request '
                '{"tool": "find", "pattern": ...} / '
                '{"tool": "read_full", "offset": ...}.'
            )},
        ]
    if last.answer is None and last.not_found_reason is None:
        # A stale parse error from an earlier turn used to suppress this, so a read
        # that escalated and then ran out of budget surfaced as outcome=error —
        # and the error path returns no article text at all. Exhaustion is a
        # not_found, and it carries the infobox back like every other not_found.
        last.error = None
        last.not_found_reason = (
            f"reader reached its {MAX_READER_TURNS}-turn budget without an answer")
        # Hand back what find DID match rather than an empty citation list. These
        # spans came out of the article, so they verify by construction; this is
        # strictly more information than the parent had before, and it is still a
        # not_found — no answer is being asserted.
        if not last.citations and matched_spans:
            kept, _ = verify_citations(full_text, matched_spans[:_MAX_PARTIAL_SPANS])
            last.citations = kept
            last.not_found_reason += (
                f" — returning {len(kept)} span(s) that find did match")
    last.turns, last.tool_calls, last.infobox = MAX_READER_TURNS, tool_calls, ib
    last.prompt_tokens = total_prompt_tokens
    last.completion_tokens = total_completion_tokens
    return last
