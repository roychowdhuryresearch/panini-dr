"""Normaliser for run directories (``<output-dir>/<system>__<model>__<run_tag>/``).

Every rule below exists because the field is NOT a decision of the harness: it is a wall-clock
measurement, an absolute path, or an order that the harness itself does not define. Each rule is
listed in RULES with its justification; every comparison report prints the rule list.

What is compared after normalisation
  * cell_result.json (minus the per-question trajectories, which are compared via traces/)
  * traces/q_<id>.json (the full Trajectory: messages, tool calls with full results, extra/*)
  * traces/q_<id>.md is a rendering of the same QuestionResult; compared after the same rules.

The normaliser is idempotent (normalising a normalised document changes nothing), which is what
lets a verify bundle ship its reference outputs already normalised.
"""

from __future__ import annotations

import json
import re
from typing import Any

# key -> justification.  Keys are dropped wherever they appear (any depth).
DROP_KEYS: dict[str, str] = {
    "wall_time_s": "wall-clock duration of a question (time.time() deltas); never read back by the harness",
    "mean_wall_time_s": "mean of wall_time_s over the cell",
    "duration_s": "per-tool-call wall-clock duration (time.time() deltas in the researcher/orchestrator loops)",
    "started_at": "CellResult creation timestamp (datetime.utcnow)",
    "completed_at": "CellResult completion timestamp (datetime.now)",
    "build_timestamp": "corpus-manifest load time (retrieval/manifest.py, datetime.now)",
}

# list-valued keys whose element ORDER is set by thread scheduling, not by the harness logic.
UNORDERED_LISTS: dict[str, str] = {
    "gsw_checker_calls": (
        "appended by parallel sibling researchers into ONE shared sink (the orchestrator hoists "
        "self._checker_sink before pool.map so every sibling appends to the same list): element order = "
        "completion order of threads, a logging artefact; compared as a multiset"),
}

RULES = (
    [f"drop key '{k}': {v}" for k, v in DROP_KEYS.items()]
    + [f"multiset list '{k}': {v}" for k, v in UNORDERED_LISTS.items()]
    + [
        "path strings: the run directory, the source tree root(s) and the cassette root are replaced by "
        "<RUN_DIR>/<SRC>/<CASSETTE> placeholders (they differ between record and replay output dirs and "
        "between the original export and the extracted repo)",
        "run_tag: the cell directory name <system>__<model>__<YYYYmmdd_HHMMSS> is compared without the "
        "timestamp (datetime.now in the runner)",
        "markdown traces: the rendered '| Wall time | Ns |' and '_(took Ns)_' numbers are masked (same quantities as wall_time_s / duration_s)",
    ]
)

_RUN_TAG = re.compile(r"__\d{8}_\d{6}")


def rules(multiset: bool) -> list[str]:
    """RULES as they apply: with ``multiset=False`` the multiset rule is disabled (strict)."""
    return [("(DISABLED: strict order) " + r) if (not multiset and r.startswith("multiset")) else r
            for r in RULES]


class Normaliser:
    def __init__(self, replacements: list[tuple[str, str]] | None = None,
                 extra_drop: list[str] | None = None, string_maps: list[tuple[str, str]] | None = None,
                 multiset: bool = True):
        # longest first so nested paths are replaced before their parents
        self.replacements = sorted(replacements or [], key=lambda kv: -len(kv[0]))
        self.drop = dict(DROP_KEYS)
        for k in extra_drop or []:
            self.drop[k] = "user --ignore-key"
        self.string_maps = string_maps or []
        self.multiset = multiset

    def s(self, v: str) -> str:
        for old, new in self.replacements:
            if old:
                v = v.replace(old, new)
        for old, new in self.string_maps:
            v = v.replace(old, new)
        return _RUN_TAG.sub("__<RUN_TAG>", v)

    def __call__(self, obj: Any, key: str | None = None) -> Any:
        if isinstance(obj, dict):
            return {self.s(k): self(v, k) for k, v in obj.items() if k not in self.drop}
        if isinstance(obj, list):
            items = [self(v) for v in obj]
            if self.multiset and key in UNORDERED_LISTS:
                items = sorted(items, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
            return items
        if isinstance(obj, str):
            # some fields hold JSON serialised to a string (ToolCall.result_full); normalise inside too
            if obj[:1] in "{[" and len(obj) > 1:
                try:
                    inner = json.loads(obj)
                except Exception:  # noqa: BLE001
                    return self.s(obj)
                return {"__json__": self(inner, key)}
            return self.s(obj)
        return obj

    # rendered durations in traces/q_<id>.md (utils/trace_markdown.py): "| Wall time | 136.30s |" and
    # "### Tool call N: `search`  _(took 1.61s)_".  Only the number is masked; the line stays compared.
    _MD_WALL = re.compile(r"(\|\s*Wall time\s*\|\s*)\d+(?:\.\d+)?s")
    _MD_TOOK = re.compile(r"\(took \d+(?:\.\d+)?s\)")

    def md(self, text: str) -> list[str]:
        t = self._MD_WALL.sub(r"\1<WALL>", self.s(text))
        t = self._MD_TOOK.sub("(took <DURATION>)", t)
        return t.splitlines()


def diff_paths(a: Any, b: Any, path: str = "", out: list[str] | None = None, limit: int = 25) -> list[str]:
    """First `limit` JSON paths where a and b differ (structural, ordered)."""
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if type(a) is not type(b):
        out.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
    elif isinstance(a, dict):
        for k in list(dict.fromkeys([*a.keys(), *b.keys()])):
            if k not in a:
                out.append(f"{path}.{k}: missing on left")
            elif k not in b:
                out.append(f"{path}.{k}: missing on right")
            else:
                diff_paths(a[k], b[k], f"{path}.{k}", out, limit)
            if len(out) >= limit:
                break
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: list length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff_paths(x, y, f"{path}[{i}]", out, limit)
            if len(out) >= limit:
                break
    elif a != b:
        ra, rb = repr(a), repr(b)
        out.append(f"{path}: {ra[:160]} != {rb[:160]}")
    return out


__all__ = ["DROP_KEYS", "RULES", "UNORDERED_LISTS", "Normaliser", "diff_paths", "rules"]
