"""A judge-verdict cache that cannot hand back a stale verdict.

The research graders keyed their caches on ``<run name>:<qid>``, so re-using a
name (or changing the judge) silently returned old verdicts. Here a key carries
everything that determines a verdict:

    <grader>|<judge model>|x<passes>|<qid>|a:<sha256(prediction)[:16]>|p:<sha256(prompt)[:16]>

so a changed answer, a changed judge model, a different number of passes or a
changed prompt (question, gold, template) is a cache miss.

The cache lives only where the user points it (no default location), is
written atomically (temp file + rename) and only when something changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


class CacheMissError(RuntimeError):
    """Raised in offline mode when a verdict would need a judge call.

    ``qids`` lists every question whose verdict is not in the cache."""

    def __init__(self, qids: list[str]) -> None:
        self.qids = list(qids)
        super().__init__(
            f"offline mode: {len(self.qids)} verdict(s) not in the judge cache "
            f"(e.g. {self.qids[:5]}); no judge call was made"
        )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def cache_key(
    *, grader: str, model: str, passes: int, qid: str, prediction: str, prompt: str = ""
) -> str:
    return (
        f"{grader}|{model}|x{int(passes)}|{qid}"
        f"|a:{sha256_hex(prediction)[:16]}|p:{sha256_hex(prompt)[:16]}"
    )


class JudgeCache:
    """JSON-file cache of judge records, safe for concurrent writers in one process."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._data: dict[str, Any] = {}
        self._dirty = False
        self._lock = threading.Lock()
        if self.path is not None and self.path.is_file():
            self._data = json.loads(self.path.read_text())

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def get(self, key: str) -> Any:
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, record: Any) -> None:
        with self._lock:
            if self._data.get(key) != record:
                self._data[key] = record
                self._dirty = True

    @property
    def dirty(self) -> bool:
        return self._dirty

    def save(self) -> bool:
        """Write atomically if anything changed. Returns True when a write happened."""
        with self._lock:
            if self.path is None or not self._dirty:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w") as fh:
                    json.dump(self._data, fh)
                os.replace(tmp, self.path)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
            self._dirty = False
            return True


__all__ = ["CacheMissError", "JudgeCache", "cache_key", "sha256_hex"]
