"""Phase-1 context compaction for the GSW researcher ReAct loop.

When --gsw-context-compaction is on, the researcher archives full tool
results and replaces STALE search tool messages in the live LLM transcript
with compact pointer placeholders (chunk_id + title), keeping the most
recent search results full. Read outputs stay full for now because they are
high-value evidence. The off-path is never invoked, so default-off behavior
is byte-identical.

Thread-safety: each researcher owns its own ToolResultArchive. Pool workers
never write a shared archive (mirrors _refutation_log / working memory).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

# --- tunable Phase-1 policy constants (kept here, easy to sweep) ---
SEARCH_FULL_WINDOW = 2           # keep the last N search results full
MAX_PLACEHOLDER_CHUNKS = 5       # cap chunk pointers listed in a placeholder
COMPACT_ONLY_SUCCESSFUL_SEARCHES = True

_COMPACTABLE_TOOLS = ("search",)


@dataclass
class _ArchiveEntry:
    archive_id: str
    tool: str
    turn: int
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass
class CompactionMetrics:
    search_messages_compacted: int = 0
    read_messages_compacted: int = 0
    placeholder_chars: int = 0
    raw_chars_archived: int = 0
    estimated_prompt_chars_saved: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "search_messages_compacted": self.search_messages_compacted,
            "read_messages_compacted": self.read_messages_compacted,
            "placeholder_chars": self.placeholder_chars,
            "raw_chars_archived": self.raw_chars_archived,
            "estimated_prompt_chars_saved": self.estimated_prompt_chars_saved,
        }


@dataclass
class ToolResultArchive:
    """Per-researcher store of full tool results, keyed by tool_call_id."""

    by_call_id: dict[str, _ArchiveEntry] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)
    _metrics: CompactionMetrics = field(default_factory=CompactionMetrics)

    def store(
        self,
        call_id: str,
        tool: str,
        turn: int,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> str:
        n = self._counts.get(tool, 0) + 1
        self._counts[tool] = n
        archive_id = f"{tool}_{n:04d}"
        raw_json = json.dumps(result, default=str)
        self._metrics.raw_chars_archived += len(raw_json)
        self.by_call_id[call_id] = _ArchiveEntry(
            archive_id=archive_id,
            tool=tool,
            turn=turn,
            args=dict(args or {}),
            result=result if isinstance(result, dict) else {},
        )
        return archive_id

    def get(self, call_id: str) -> Optional[_ArchiveEntry]:
        return self.by_call_id.get(call_id)

    def record_compaction(
        self, *, tool: str, raw_chars: int, placeholder_chars: int
    ) -> None:
        if tool == "search":
            self._metrics.search_messages_compacted += 1
        elif tool == "read":
            self._metrics.read_messages_compacted += 1
        self._metrics.placeholder_chars += placeholder_chars
        self._metrics.estimated_prompt_chars_saved += max(
            raw_chars - placeholder_chars, 0
        )

    def metrics(self) -> dict[str, int]:
        return self._metrics.to_dict()


def build_search_placeholder(entry: _ArchiveEntry) -> dict[str, Any]:
    results = (entry.result or {}).get("results", []) or []
    top: list[dict[str, str]] = []
    for r in results[:MAX_PLACEHOLDER_CHUNKS]:
        chunk = {"chunk_id": str(r.get("chunk_id", "")), "title": str(r.get("title", ""))}
        top.append(chunk)
    note = "Full search result elided to save context. Use read(chunk_id) if needed."
    out = {
        "compacted_tool_result": True,
        "archive_id": entry.archive_id,
        "tool": "search",
        "query": str(entry.args.get("query", "")),
        "top_chunks": top,
        "note": note,
    }
    return out


_BUILDERS = {"search": build_search_placeholder}
_WINDOWS = {"search": SEARCH_FULL_WINDOW}


def _is_compacted(content: str) -> bool:
    return '"compacted_tool_result": true' in content or '"compacted_tool_result":true' in content


def _is_successful_search(entry: _ArchiveEntry) -> bool:
    return bool((entry.result or {}).get("results"))


def compact_messages(
    messages: list[dict[str, Any]],
    archive: ToolResultArchive,
) -> int:
    """Replace stale search tool messages with compact placeholders.

    Walks `messages` in order, mapping each tool-role message to its archive
    entry via tool_call_id. Keeps the last SEARCH_FULL_WINDOW search results
    full; compacts older ones. Skips error results, non-search tools, and
    already-compacted messages (idempotent). Mutates `messages` in place —
    only the `content` string of stale tool messages. Returns the count newly
    compacted.
    """
    per_tool: dict[str, list[int]] = {tool: [] for tool in _COMPACTABLE_TOOLS}
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        entry = archive.get(msg.get("tool_call_id", ""))
        if entry is None or entry.tool not in per_tool:
            continue
        per_tool[entry.tool].append(i)

    n_compacted = 0
    for tool, idxs in per_tool.items():
        window = _WINDOWS[tool]
        stale = idxs[:-window] if window > 0 else list(idxs)
        for i in stale:
            msg = messages[i]
            content = msg.get("content", "")
            if _is_compacted(content):
                continue
            entry = archive.get(msg.get("tool_call_id", ""))
            if entry is None:
                continue
            if (entry.result or {}).get("error"):
                continue
            if (
                tool == "search"
                and COMPACT_ONLY_SUCCESSFUL_SEARCHES
                and not _is_successful_search(entry)
            ):
                continue
            raw_chars = len(content)
            placeholder = json.dumps(_BUILDERS[tool](entry))
            msg["content"] = placeholder
            archive.record_compaction(
                tool=tool,
                raw_chars=raw_chars,
                placeholder_chars=len(placeholder),
            )
            n_compacted += 1
    return n_compacted
