"""Cassette store: record / replay of external responses, keyed by request content.

Layout of a cassette directory (record mode writes it; replay mode only reads it):

    meta.json        run metadata (source commit, resolved config, service versions, ...)
    calls.jsonl      one line per external call, in completion order:
                       {"i", "kind", "key", "lineage", "lseq", "site", "role", "thread",
                        "t0", "t1", "req", "resp" | "exc"}
                     "req" is the request with every message / tool schema replaced by a
                     blob reference {"$blob": sha256} (content-addressed, see blobs.jsonl).
    blobs.jsonl      {"h": sha256, "v": <json value>}  -- each distinct message / tool list once
    leaks.jsonl      network connections made OUTSIDE any hook during record (should be empty)

Keys are sha256 over the canonical JSON of the FULL request (not call order; calls run
concurrently).  Canonical JSON = json.dumps(..., ensure_ascii=False, separators=(",", ":"))
WITHOUT sort_keys: dict insertion order is preserved on purpose, because the wire order of
tool-schema properties reaches the model's prompt.  The only value normalisation applied is
integral floats -> int on the top-level sampling scalars (temperature, top_p, max_tokens, seed),
where 0 and 0.0 are the same request to the server.

Repeated identical requests are stored as a list per key and consumed in recorded order,
with lineage affinity: among the remaining entries for a key, the one recorded by the same
logical thread lineage at the same per-lineage call index is preferred, then the same lineage,
then the oldest.  (Lineage = question id / pool task path; see :mod:`panini_dr.replay.patches`.)
"""

from __future__ import annotations

import base64
import collections
import difflib
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from panini_dr.replay.gate import EventOrderGate, root_of

CALLS = "calls.jsonl"
BLOBS = "blobs.jsonl"
LEAKS = "leaks.jsonl"
META = "meta.json"

# ---------------------------------------------------------------------------
# canonical JSON + hashing
# ---------------------------------------------------------------------------

_NUMERIC_SCALARS = ("temperature", "top_p", "max_tokens", "max_completion_tokens", "seed")


def _fallback(o: Any) -> Any:
    # numpy scalars / arrays, pydantic models, dataclasses: make them JSON but flag them.
    try:
        import numpy as np  # noqa: WPS433

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return {"__ndarray__": encode_array(o)}
    except Exception:  # noqa: BLE001
        pass
    if hasattr(o, "model_dump"):
        return {"__model__": type(o).__name__, "v": o.model_dump()}
    if isinstance(o, (set, frozenset)):
        return {"__set__": sorted(repr(x) for x in o)}
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, bytes):
        return {"__bytes__": base64.b64encode(o).decode()}
    return {"__nonjson__": f"{type(o).__module__}.{type(o).__qualname__}", "repr": repr(o)}


def canon(obj: Any) -> str:
    """Canonical JSON: insertion order kept, no whitespace, non-ASCII kept."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_fallback)


def canon_sorted(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_fallback)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def normalise_scalars(req: dict[str, Any]) -> dict[str, Any]:
    out = dict(req)
    for k in _NUMERIC_SCALARS:
        v = out.get(k)
        if isinstance(v, float) and v.is_integer():
            out[k] = int(v)
    return out


def request_key(req: dict[str, Any]) -> str:
    """sha256 of the canonical JSON of the full request."""
    return sha(canon(normalise_scalars(req)))


# ---------------------------------------------------------------------------
# numpy array codec (query embeddings)
# ---------------------------------------------------------------------------


def encode_array(a: Any) -> dict[str, Any]:
    import numpy as np

    a = np.ascontiguousarray(a)
    return {"dtype": str(a.dtype), "shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}


def decode_array(d: dict[str, Any]) -> Any:
    import numpy as np

    buf = base64.b64decode(d["b64"])
    return np.frombuffer(buf, dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()


# ---------------------------------------------------------------------------
# blob compaction of requests (messages / tools are stored once)
# ---------------------------------------------------------------------------

_BLOB_FIELDS = ("messages", "tools", "sentences", "documents")


def compact_request(req: dict[str, Any], put_blob) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in req.items():
        if k == "messages" and isinstance(v, list):
            out[k] = [{"$blob": put_blob(m)} for m in v]
        elif k in _BLOB_FIELDS and v is not None:
            out[k] = {"$blob": put_blob(v)}
        else:
            out[k] = v
    return out


def expand_request(creq: dict[str, Any], blobs: dict[str, Any]) -> dict[str, Any]:
    def _x(v):
        if isinstance(v, dict) and set(v) == {"$blob"}:
            return blobs[v["$blob"]]
        return v

    out: dict[str, Any] = {}
    for k, v in creq.items():
        if k == "messages" and isinstance(v, list):
            out[k] = [_x(m) for m in v]
        else:
            out[k] = _x(v)
    return out


# ---------------------------------------------------------------------------
# exceptions: record + rebuild
# ---------------------------------------------------------------------------


def serialise_exception(exc: BaseException) -> dict[str, Any]:
    d: dict[str, Any] = {
        "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "str": str(exc),
        "args": [a if isinstance(a, (str, int, float, bool, type(None))) else repr(a) for a in exc.args],
        "attrs": {},
    }
    for attr in ("message", "status_code", "body", "code", "param", "type", "request_id"):
        if hasattr(exc, attr):
            try:
                v = getattr(exc, attr)
                json.dumps(v)
                d["attrs"][attr] = v
            except Exception:  # noqa: BLE001
                d["attrs"][attr] = repr(getattr(exc, attr, None))
    resp = getattr(exc, "response", None)
    if resp is not None and hasattr(resp, "status_code"):
        try:
            d["response"] = {"status_code": resp.status_code, "text": resp.text[:20000]}
        except Exception:  # noqa: BLE001
            d["response"] = {"status_code": getattr(resp, "status_code", None), "text": ""}
    req = getattr(exc, "request", None)
    if req is not None and hasattr(req, "url"):
        try:
            d["request_url_path"] = str(req.url.path)
        except Exception:  # noqa: BLE001
            pass
    return d


def _import_attr(path: str):
    mod, _, name = path.rpartition(".")
    parts = name.split(".")
    import importlib

    while True:
        try:
            m = importlib.import_module(mod)
            break
        except ImportError:
            mod, _, head = mod.rpartition(".")
            parts.insert(0, head)
            if not mod:
                raise
    obj = m
    for p in parts:
        obj = getattr(obj, p)
    return obj


def rebuild_exception(d: dict[str, Any]) -> BaseException:
    """Recreate the recorded exception: same class, same str(), same key attributes."""
    try:
        cls = _import_attr(d["type"])
        if not (isinstance(cls, type) and issubclass(cls, BaseException)):
            raise TypeError
    except Exception:  # noqa: BLE001
        cls = RuntimeError
    exc = cls.__new__(cls)
    try:
        BaseException.__init__(exc, *d.get("args", [d.get("str", "")]))
    except Exception:  # noqa: BLE001
        pass
    for k, v in (d.get("attrs") or {}).items():
        try:
            setattr(exc, k, v)
        except Exception:  # noqa: BLE001
            pass
    try:
        import httpx

        url = "http://cassette.invalid" + d.get("request_url_path", "/")
        request = httpx.Request("POST", url)
        try:
            exc.request = request  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        if d.get("response"):
            r = d["response"]
            try:
                exc.response = httpx.Response(r["status_code"] or 500, text=r.get("text", ""), request=request)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    if str(exc) != d.get("str", str(exc)):
        # Last resort: make str() match exactly (the harness stores str(exc) in traces).
        try:
            exc.args = (d["str"],)
        except Exception:  # noqa: BLE001
            pass
    return exc


# ---------------------------------------------------------------------------
# errors raised by the tool
# ---------------------------------------------------------------------------


class CassetteMiss(RuntimeError):
    """Replay found no recorded response for this request."""


class CassetteNetworkLeak(RuntimeError):
    """Replay attempted a network connection outside every hooked call."""


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------


class Cassette:
    """One cassette directory, opened for ``record`` or ``replay``.

    In replay mode ``order="recorded"`` turns on the per-question event-order gate
    (:class:`panini_dr.replay.gate.EventOrderGate`); ``"free"`` releases every
    response as soon as it is found.
    """

    def __init__(self, root: Path, mode: str, report_dir: Path | None = None, latency_scale: float = 0.0,
                 order: str = "free", gate_timeout_s: float = 60.0):
        self.root = Path(root)
        self.mode = mode
        self.lock = threading.RLock()
        self.t_origin = time.monotonic()
        self.latency_scale = latency_scale
        self.counts: collections.Counter = collections.Counter()
        self.lineage_counters: collections.Counter = collections.Counter()
        self.misses: list[dict[str, Any]] = []
        self.leaks: list[dict[str, Any]] = []
        self.pick_stats: collections.Counter = collections.Counter()
        self.order = order
        self.gate_timeout_s = gate_timeout_s
        self.gate: EventOrderGate | None = None
        if mode == "record":
            self.root.mkdir(parents=True, exist_ok=True)
            for f in (CALLS, BLOBS, LEAKS):
                if (self.root / f).exists():
                    raise SystemExit(f"cassette: {self.root / f} exists; refusing to overwrite a cassette")
            self._calls_f = open(self.root / CALLS, "a", encoding="utf-8")
            self._blobs_f = open(self.root / BLOBS, "a", encoding="utf-8")
            self._leaks_f = open(self.root / LEAKS, "a", encoding="utf-8")
            self._seen_blobs: set[str] = set()
            self._i = 0
        elif mode == "replay":
            self.report_dir = Path(report_dir) if report_dir else None
            self._load()
        else:
            raise ValueError(f"cassette mode must be 'record' or 'replay', not {mode!r}")

    # ---------------- record ----------------
    def put_blob(self, v: Any) -> str:
        s = canon(v)
        h = sha(s)
        with self.lock:
            if h not in self._seen_blobs:
                self._seen_blobs.add(h)
                self._blobs_f.write('{"h":"%s","v":%s}\n' % (h, s))
        return h

    def record(self, *, kind: str, req: dict[str, Any], key: str, ctx: dict[str, Any], t0: float, t1: float,
               resp: Any = None, exc: dict[str, Any] | None = None) -> None:
        creq = compact_request(req, self.put_blob)
        with self.lock:
            entry = {"i": self._i, "kind": kind, "key": key, **ctx,
                     "t0": round(t0 - self.t_origin, 4), "t1": round(t1 - self.t_origin, 4), "req": creq}
            if exc is not None:
                entry["exc"] = exc
            else:
                entry["resp"] = resp
            self._i += 1
            self._calls_f.write(canon(entry) + "\n")
            self._calls_f.flush()
            self.counts[kind] += 1

    def record_leak(self, info: dict[str, Any]) -> None:
        with self.lock:
            self.leaks.append(info)
            if self.mode == "record":
                self._leaks_f.write(canon(info) + "\n")
                self._leaks_f.flush()

    def close(self) -> None:
        if self.mode == "record":
            with self.lock:
                for f in (self._calls_f, self._blobs_f, self._leaks_f):
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                        f.close()
                    except Exception:  # noqa: BLE001
                        pass

    # ---------------- replay ----------------
    def _load(self) -> None:
        self.blobs: dict[str, Any] = {}
        with open(self.root / BLOBS, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    self.blobs[d["h"]] = d["v"]
        self.by_key: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        self.all_entries: list[dict[str, Any]] = []
        with open(self.root / CALLS, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    self.all_entries.append(e)
        # consume in RECORDED REQUEST order (t0), not completion order
        for e in sorted(self.all_entries, key=lambda e: (e["t0"], e["i"])):
            self.by_key[e["key"]].append(e)
        self.n_recorded = len(self.all_entries)
        self.gate = EventOrderGate(self.all_entries, self.lock, order=self.order,
                                   timeout_s=self.gate_timeout_s, log=eprint)

    def mark_diverged(self, lineage: str | None) -> None:
        self.gate.mark_diverged(lineage)

    def take(self, key: str, lineage: str, lseq: int) -> dict[str, Any] | None:
        with self.lock:
            lst = self.by_key.get(key)
            if not lst:
                return None
            pick = None
            for j, e in enumerate(lst):
                if e.get("lineage") == lineage and e.get("lseq") == lseq:
                    pick = j
                    self.pick_stats["exact_lineage_seq"] += 1
                    break
            if pick is None:
                for j, e in enumerate(lst):
                    if e.get("lineage") == lineage:
                        pick = j
                        self.pick_stats["same_lineage"] += 1
                        break
            if pick is None:
                pick = 0
                self.pick_stats["fifo_fallback"] += 1
            e = lst.pop(pick)
            if not lst:
                del self.by_key[key]
            self.counts[e["kind"]] += 1
            self.gate.note_arrival(e["i"])
            return e

    def next_lseq(self, lineage: str) -> int:
        with self.lock:
            n = self.lineage_counters[lineage]
            self.lineage_counters[lineage] += 1
            return n

    def leftovers(self) -> list[dict[str, Any]]:
        with self.lock:
            return [e for lst in self.by_key.values() for e in lst]

    # ---------------- miss reporting ----------------
    def report_miss(self, *, kind: str, req: dict[str, Any], key: str, ctx: dict[str, Any]) -> Path | None:
        """Log a miss; with a report dir, write the request, its nearest recorded
        neighbour and a unified diff of the two (``misses/miss_NNNN.{json,diff}``)."""
        with self.lock:
            n = len(self.misses)
            info = {"n": n, "kind": kind, "key": key, **ctx}
            self.misses.append(info)
        if not self.report_dir:
            return None
        d = self.report_dir / "misses"
        d.mkdir(parents=True, exist_ok=True)
        nearest, why = self.nearest(kind, req, ctx)
        out = {"miss": info, "request": req, "nearest_selection": why,
               "nearest": None if nearest is None else {k: v for k, v in nearest.items() if k != "req"},
               "nearest_request": None if nearest is None else expand_request(nearest["req"], self.blobs)}
        (d / f"miss_{n:04d}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_fallback))
        diff = ""
        if nearest is not None:
            a = pretty_request(expand_request(nearest["req"], self.blobs)).splitlines(keepends=True)
            b = pretty_request(req).splitlines(keepends=True)
            diff = "".join(difflib.unified_diff(a, b, fromfile=f"recorded i={nearest['i']} lineage={nearest.get('lineage')} lseq={nearest.get('lseq')}",
                                                tofile=f"replay lineage={ctx.get('lineage')} lseq={ctx.get('lseq')}", n=3))
            if not diff:
                diff = "(requests render identically; difference is in key-order or number formatting only)\n"
                if canon_sorted(expand_request(nearest["req"], self.blobs)) == canon_sorted(req):
                    diff += "(equal modulo dict key order)\n"
        (d / f"miss_{n:04d}.diff").write_text(diff)
        return d / f"miss_{n:04d}.diff"

    def nearest(self, kind: str, req: dict[str, Any], ctx: dict[str, Any]):
        q = (ctx.get("lineage") or "").split("/")[0]
        msgs = req.get("messages") or []
        mh = [sha(canon(m)) for m in msgs] if isinstance(msgs, list) else []
        best, best_score = None, None
        for e in self.all_entries:
            if e["kind"] != kind:
                continue
            score = 0
            if q and (e.get("lineage") or "").split("/")[0] == q:
                score += 10 ** 7
            if e.get("lineage") == ctx.get("lineage"):
                score += 10 ** 6
            if e.get("role") and e.get("role") == ctx.get("role"):
                score += 10 ** 5
            em = e["req"].get("messages")
            if isinstance(em, list) and mh:
                lcp = 0
                for a, b in zip(em, mh):
                    if isinstance(a, dict) and a.get("$blob") == b:
                        lcp += 1
                    else:
                        break
                score += lcp * 100 - abs(len(em) - len(mh))
            elif kind != "llm":
                # non-chat requests: shared prefix of the canonical body
                a = canon(expand_request(e["req"], self.blobs))
                b = canon(req)
                score += len(os.path.commonprefix([a, b])) // 50
            if best_score is None or score > best_score:
                best, best_score = e, score
        why = "same question > same lineage > same role > longest common message prefix"
        return best, why

    @property
    def diverged(self) -> set[str]:
        return set(self.gate.diverged) if self.gate is not None else set()

    @property
    def gate_stats(self) -> collections.Counter:
        return self.gate.stats if self.gate is not None else collections.Counter()


def pretty_request(req: dict[str, Any]) -> str:
    """Line-oriented rendering of a request for unified diffs: one block per message."""
    lines: list[str] = []
    for k, v in req.items():
        if k == "messages" and isinstance(v, list):
            for i, m in enumerate(v):
                lines.append(f"=== message[{i}] role={m.get('role') if isinstance(m, dict) else '?'}")
                if isinstance(m, dict):
                    for mk, mv in m.items():
                        if isinstance(mv, str):
                            lines.append(f"  {mk}:")
                            lines.extend("    " + ln for ln in mv.split("\n"))
                        else:
                            lines.append(f"  {mk}: {json.dumps(mv, ensure_ascii=False, default=_fallback)}")
                else:
                    lines.append("  " + json.dumps(m, ensure_ascii=False, default=_fallback))
        elif isinstance(v, (list, dict)):
            lines.append(f"=== {k}")
            lines.extend("  " + ln for ln in json.dumps(v, ensure_ascii=False, indent=1, default=_fallback).split("\n"))
        else:
            lines.append(f"=== {k}: {json.dumps(v, ensure_ascii=False, default=_fallback)}")
    return "\n".join(lines) + "\n"


def eprint(*a: Any) -> None:
    print(*a, file=sys.stderr, flush=True)


__all__ = [
    "BLOBS",
    "CALLS",
    "LEAKS",
    "META",
    "Cassette",
    "CassetteMiss",
    "CassetteNetworkLeak",
    "canon",
    "canon_sorted",
    "compact_request",
    "decode_array",
    "encode_array",
    "eprint",
    "expand_request",
    "normalise_scalars",
    "pretty_request",
    "rebuild_exception",
    "request_key",
    "root_of",
    "serialise_exception",
    "sha",
]
