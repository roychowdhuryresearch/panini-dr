"""Per-question event-order gate for replay (``order="recorded"``).

A replay serves every response from the cassette, so without a gate the
harness's threads would race through a question in whatever order the scheduler
picks. Within one question several threads (parallel researcher siblings) share
state, e.g. the per-question read ledger, so the ORDER of their events decides
what each one sees. The gate reproduces the recorded order of hook events within
each question:

The response for entry ``e`` (recorded completion time ``t1``) is released only
when, in this replay,

  (a) every response of the same question recorded as completing before ``e.t1``
      has been released, and
  (b) every request of the same question recorded as starting before ``e.t1``
      has arrived at a hook.

(b) is what orders thread-local CPU work: a request arriving proves its thread
finished the work between its previous response and this request (e.g. writing
the shared read ledger). Every dependency points strictly backwards in recorded
time, so a faithful replay cannot deadlock. A question that misses, or waits
longer than ``timeout_s``, is marked diverged and ungated (its outputs will not
match anyway, and the run must still finish).

Questions are the roots of the lineage strings (``q:<id>`` or ``main``); calls
made outside any question (lineage root ``main``) are never gated.
"""

from __future__ import annotations

import collections
import heapq
import threading
import time
from typing import Any, Callable


def root_of(lineage: str | None) -> str:
    """The question a lineage belongs to (``q:<id>``), or ``main``."""
    return (lineage or "main").split("/")[0]


class EventOrderGate:
    """Release recorded responses in recorded event order, per question.

    ``lock`` must be the lock the cassette holds while it hands out entries
    (:meth:`note_arrival` is called with it held), so arrivals and releases are
    ordered against each other.
    """

    def __init__(self, entries: list[dict[str, Any]], lock: threading.RLock, *, order: str = "recorded",
                 timeout_s: float = 60.0, log: Callable[..., None] | None = None) -> None:
        if order not in ("recorded", "free"):
            raise ValueError(f"order must be 'recorded' or 'free', not {order!r}")
        self.order = order
        self.timeout_s = timeout_s
        self._log = log or (lambda *a: None)
        self._cv = threading.Condition(lock)
        self.released: set[int] = set()
        self.arrived: set[int] = set()
        self.diverged: set[str] = set()
        self.stats: collections.Counter = collections.Counter()
        # Per question, two min-heaps over the recorded timeline: responses not yet
        # RELEASED keyed by t1, requests not yet ARRIVED keyed by t0.
        self._unreleased: dict[str, list] = collections.defaultdict(list)
        self._unarrived: dict[str, list] = collections.defaultdict(list)
        for e in entries:
            q = root_of(e.get("lineage"))
            self._unreleased[q].append((e["t1"], e["i"]))
            self._unarrived[q].append((e["t0"], e["i"]))
        for h in [*self._unreleased.values(), *self._unarrived.values()]:
            heapq.heapify(h)

    def _min_open(self, heap: list, done: set[int], skip: int) -> float:
        while heap and heap[0][1] in done:
            heapq.heappop(heap)
        if heap and heap[0][1] == skip:
            # e itself: look past it without removing it
            return min((t for t, i in heap if i != skip and i not in done), default=float("inf"))
        return heap[0][0] if heap else float("inf")

    def blocked(self, q: str, e: dict[str, Any]) -> str | None:
        """What ``e`` still waits for: ``"response"``, ``"request"`` or None."""
        if self._min_open(self._unreleased.get(q, []), self.released, e["i"]) < e["t1"]:
            return "response"
        if self._min_open(self._unarrived.get(q, []), self.arrived, e["i"]) < e["t1"]:
            return "request"
        return None

    def note_arrival(self, i: int) -> None:
        """Entry ``i``'s request reached a hook (caller holds the cassette lock)."""
        self.arrived.add(i)
        self._cv.notify_all()

    def mark_diverged(self, lineage: str | None) -> None:
        with self._cv:
            self.diverged.add(root_of(lineage))
            self._cv.notify_all()

    def release(self, e: dict[str, Any]) -> None:
        """Block until ``e``'s response may be returned, then mark it released."""
        q = root_of(e.get("lineage"))
        with self._cv:
            if self.order == "recorded" and q != "main":
                deadline = time.monotonic() + self.timeout_s
                waited = None
                while q not in self.diverged:
                    why = self.blocked(q, e)
                    if why is None:
                        break
                    waited = why
                    left = deadline - time.monotonic()
                    if left <= 0:
                        self.stats["timeouts"] += 1
                        self.diverged.add(q)
                        self._log(f"[cassette] gate timeout in {q} (waiting on a {why}); question ungated")
                        break
                    self._cv.wait(timeout=min(left, 1.0))
                self.stats[f"waited_{waited}" if waited else "immediate"] += 1
            self.released.add(e["i"])
            self._cv.notify_all()


__all__ = ["EventOrderGate", "root_of"]
