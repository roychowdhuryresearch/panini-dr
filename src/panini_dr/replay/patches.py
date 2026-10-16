"""Runtime monkeypatches that route every external, nondeterministic dependency through the cassette.

Installed by :mod:`panini_dr.replay.launcher` BEFORE the harness runs; no harness source is edited.

  kind "llm"        <pkg>.models.llm_client.LLMClient.chat  (the single LLM entry point; all roles)
  kind "embed"      sentence_transformers.SentenceTransformer.encode   (Qwen3 query encoder)
  kind "http_post"  httpx.post                         (jina reranker; any other module-level httpx.post)
  guard             socket.socket.connect / connect_ex  (any connection made outside a hook:
                                                         record -> logged to leaks.jsonl,
                                                         replay -> CassetteNetworkLeak)
  lineage           concurrent.futures.ThreadPoolExecutor.submit  (propagates a logical thread
                                                         lineage "q:<qid>/<task>:<arg>#n" into pool
                                                         workers; used for duplicate-key affinity and
                                                         for attributing calls to questions)
"""

from __future__ import annotations

import collections
import dataclasses
import importlib
import inspect
import json
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from panini_dr.replay.store import (
    Cassette,
    CassetteMiss,
    CassetteNetworkLeak,
    canon,
    decode_array,
    encode_array,
    eprint,
    rebuild_exception,
    request_key,
    serialise_exception,
)

# Frames inside this package are the tool itself, not the call site.
_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_tls = threading.local()
_submit_lock = threading.Lock()
_submit_counts: collections.Counter = collections.Counter()

CAS: Cassette | None = None
MODE = "record"

DEFAULT_LLM_HOOK = "{pkg}.models.llm_client.LLMClient.chat"
DEFAULT_LLM_RESPONSE = "{pkg}.models.llm_client.LLMResponse"


# ---------------------------------------------------------------------------
# hook-depth flag (socket guard) and lineage
# ---------------------------------------------------------------------------


class _InHook:
    def __enter__(self):
        _tls.depth = getattr(_tls, "depth", 0) + 1

    def __exit__(self, *a):
        _tls.depth -= 1


def in_hook() -> bool:
    return getattr(_tls, "depth", 0) > 0


def current_lineage() -> str:
    return getattr(_tls, "lineage", None) or "main"


def _task_label(fn: Any, args: tuple) -> str:
    for a in args:
        if hasattr(a, "id") and (hasattr(a, "question") or hasattr(a, "answer")):
            return f"q:{getattr(a, 'id')}"
    name = getattr(fn, "__name__", None) or type(fn).__name__
    if args and isinstance(args[0], str) and len(args[0]) <= 120:
        return f"{name}:{args[0]}"
    return name


def _patched_submit_factory(orig_submit):
    def submit(self, fn, /, *args, **kwargs):
        parent = current_lineage()
        label = _task_label(fn, args)
        with _submit_lock:
            n = _submit_counts[(parent, label)]
            _submit_counts[(parent, label)] += 1
        if label.startswith("q:"):
            child = label + (f"#{n}" if n else "")
        else:
            child = f"{parent}/{label}" + (f"#{n}" if n else "")

        def run(*a, **k):
            prev = getattr(_tls, "lineage", None)
            _tls.lineage = child
            try:
                return fn(*a, **k)
            finally:
                _tls.lineage = prev

        return orig_submit(self, run, *args, **kwargs)

    submit.__cassette_patched__ = True
    return submit


def install_lineage() -> None:
    if not getattr(ThreadPoolExecutor.submit, "__cassette_patched__", False):
        ThreadPoolExecutor.submit = _patched_submit_factory(ThreadPoolExecutor.submit)


# ---------------------------------------------------------------------------
# call context (non-key metadata stored with every entry)
# ---------------------------------------------------------------------------


def _call_context(skip_files: tuple[str, ...]) -> dict[str, Any]:
    lineage = current_lineage()
    ctx: dict[str, Any] = {"lineage": lineage, "lseq": CAS.next_lseq(lineage) if CAS else 0,
                           "thread": threading.current_thread().name}
    site: list[str] = []
    role = None
    f = sys._getframe(2)
    depth = 0
    while f is not None and depth < 40:
        fn = f.f_code.co_filename
        if not fn.startswith(_TOOL_DIR) and not any(fn.endswith(s) for s in skip_files):
            if len(site) < 4:
                site.append(f.f_code.co_name)
            if role is None:
                slf = f.f_locals.get("self")
                r = getattr(slf, "role", None) if slf is not None else None
                if isinstance(r, str) and hasattr(slf, "chat"):
                    role = r
        if len(site) >= 4 and role is not None:
            break
        f = f.f_back
        depth += 1
    ctx["site"] = "<".join(site)
    ctx["role"] = role
    return ctx


def _maybe_sleep(entry: dict[str, Any]) -> None:
    if CAS is not None:
        CAS.gate.release(entry)
    if CAS is not None and CAS.latency_scale > 0:
        dt = max(0.0, float(entry.get("t1", 0)) - float(entry.get("t0", 0)))
        time.sleep(dt * CAS.latency_scale)


def _miss(kind: str, req: dict[str, Any], key: str, ctx: dict[str, Any]):
    CAS.mark_diverged(ctx.get("lineage"))
    path = CAS.report_miss(kind=kind, req=req, key=key, ctx=ctx)
    eprint(f"[cassette] MISS kind={kind} key={key[:16]} lineage={ctx.get('lineage')} lseq={ctx.get('lseq')} "
           f"role={ctx.get('role')} site={ctx.get('site')} -> {path}")
    return CassetteMiss(f"cassette miss: kind={kind} key={key[:16]} lineage={ctx.get('lineage')} (diff: {path})")


# ---------------------------------------------------------------------------
# LLM hook
# ---------------------------------------------------------------------------

_LLM_KEY_ARGS = ("messages", "tools", "tool_choice", "temperature", "max_tokens", "seed", "response_format", "stop")
_LLM_NONKEY_ARGS = ("stream",)


def _llm_request(client: Any, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    temperature = params.get("temperature")
    if temperature is None:
        temperature = getattr(client, "default_temperature", None)
    max_tokens = params.get("max_tokens")
    if max_tokens is None:
        max_tokens = getattr(client, "default_max_tokens", None)
    req: dict[str, Any] = {
        "kind": "llm",
        "model": getattr(client, "model", None),
        "messages": params.get("messages"),
        "tools": params.get("tools"),
        "tool_choice": params.get("tool_choice"),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": params.get("seed"),
        "response_format": params.get("response_format"),
        "stop": params.get("stop"),
    }
    extra = {k: v for k, v in params.items()
             if k not in _LLM_KEY_ARGS and k not in _LLM_NONKEY_ARGS and v is not None}
    if extra:
        req["extra"] = extra
    nonkey = {k: params.get(k) for k in _LLM_NONKEY_ARGS}
    nonkey["client"] = {"provider": getattr(client, "provider", None),
                        "base_url": getattr(client, "_base_url", None),
                        "reasoning_effort": getattr(client, "reasoning_effort", None),
                        "class": f"{type(client).__module__}.{type(client).__qualname__}"}
    # snapshot (callers mutate message lists after the call returns)
    return json.loads(canon(req)), nonkey


def _serialise_response(resp: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(resp):
        d = {f.name: getattr(resp, f.name) for f in dataclasses.fields(resp)}
    elif hasattr(resp, "model_dump"):
        d = resp.model_dump()
    else:
        d = dict(vars(resp))
    return {"class": f"{type(resp).__module__}.{type(resp).__qualname__}", "fields": json.loads(canon(d))}


def install_llm_hook(module_path: str, cls_name: str, method: str, response_cls_path: str) -> dict[str, Any]:
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    orig = getattr(cls, method)
    if getattr(orig, "__cassette_patched__", False):
        return {}
    sig = inspect.signature(orig)
    rmod, _, rname = response_cls_path.rpartition(".")
    resp_cls = getattr(importlib.import_module(rmod), rname)
    skip = (os.path.basename(inspect.getsourcefile(orig) or "llm_client.py"),)

    def chat(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        params = dict(bound.arguments)
        params.pop(next(iter(sig.parameters)))  # self
        req, nonkey = _llm_request(self, params)
        key = request_key(req)
        ctx = _call_context(skip)
        ctx.update(nonkey)
        if MODE == "record":
            t0 = time.monotonic()
            try:
                with _InHook():
                    resp = orig(self, *args, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                CAS.record(kind="llm", req=req, key=key, ctx=ctx, t0=t0, t1=time.monotonic(),
                           exc=serialise_exception(exc))
                raise
            CAS.record(kind="llm", req=req, key=key, ctx=ctx, t0=t0, t1=time.monotonic(),
                       resp=_serialise_response(resp))
            return resp
        entry = CAS.take(key, ctx["lineage"], ctx["lseq"])
        if entry is None:
            raise _miss("llm", req, key, ctx)
        _maybe_sleep(entry)
        if "exc" in entry:
            raise rebuild_exception(entry["exc"])
        fields = json.loads(canon(entry["resp"]["fields"]))
        names = {f.name for f in dataclasses.fields(resp_cls)} if dataclasses.is_dataclass(resp_cls) else None
        if names is not None:
            unknown = set(fields) - names
            if unknown:
                raise CassetteMiss(f"cassette: response class {response_cls_path} lacks recorded fields {sorted(unknown)}")
        return resp_cls(**fields)

    chat.__cassette_patched__ = True
    chat.__wrapped__ = orig
    setattr(cls, method, chat)
    return {"llm_hook": f"{module_path}.{cls_name}.{method}", "llm_response_class": response_cls_path,
            "llm_signature": str(sig)}


# ---------------------------------------------------------------------------
# query-embedding hook (SentenceTransformer)
# ---------------------------------------------------------------------------

MAX_RECORDED_ENCODE = 1000  # a corpus-scale encode under the cassette means a dense cache is missing


def install_embed_hook() -> dict[str, Any]:
    import sentence_transformers as st

    real = st.SentenceTransformer
    if getattr(real, "__cassette_patched__", False):
        return {}

    class CassetteSentenceTransformer:
        __cassette_patched__ = True

        def __init__(self, model_name_or_path=None, *args, **kwargs):
            self._cassette_model = str(model_name_or_path)
            self._cassette_ctor = {"args": [repr(a) for a in args],
                                   "kwargs": {k: repr(v) for k, v in kwargs.items()}}
            self._real = None
            if MODE == "record":
                with _InHook():
                    self._real = real(model_name_or_path, *args, **kwargs)

        def encode(self, sentences, *args, **kwargs):
            n = 1 if isinstance(sentences, str) else len(sentences)
            if n > MAX_RECORDED_ENCODE:
                raise RuntimeError(f"cassette: refusing a corpus-scale encode of {n} texts "
                                   f"(dense cache missing or stale?)")
            kw = {k: v for k, v in kwargs.items() if k not in ("show_progress_bar",)}
            req = json.loads(canon({"kind": "embed", "model": self._cassette_model,
                                    "sentences": sentences, "args": list(args), "kwargs": kw}))
            key = request_key(req)
            ctx = _call_context(("sentence_transformers",))
            ctx["ctor"] = self._cassette_ctor
            if MODE == "record":
                t0 = time.monotonic()
                try:
                    with _InHook():
                        out = self._real.encode(sentences, *args, **kwargs)
                except BaseException as exc:  # noqa: BLE001
                    CAS.record(kind="embed", req=req, key=key, ctx=ctx, t0=t0, t1=time.monotonic(),
                               exc=serialise_exception(exc))
                    raise
                import numpy as np

                if not isinstance(out, np.ndarray):
                    raise RuntimeError(f"cassette: encode returned {type(out)}; only numpy is recorded")
                CAS.record(kind="embed", req=req, key=key, ctx=ctx, t0=t0, t1=time.monotonic(),
                           resp={"ndarray": encode_array(out)})
                return out
            entry = CAS.take(key, ctx["lineage"], ctx["lseq"])
            if entry is None:
                raise _miss("embed", req, key, ctx)
            _maybe_sleep(entry)
            if "exc" in entry:
                raise rebuild_exception(entry["exc"])
            return decode_array(entry["resp"]["ndarray"])

        def __getattr__(self, name):
            if name.startswith("_cassette") or name == "_real":
                raise AttributeError(name)
            if self._real is None:
                raise AttributeError(f"cassette replay: SentenceTransformer.{name} is not available "
                                     f"(only .encode is replayed)")
            return getattr(self._real, name)

    st.SentenceTransformer = CassetteSentenceTransformer
    return {"embed_hook": "sentence_transformers.SentenceTransformer -> CassetteSentenceTransformer.encode"}


# ---------------------------------------------------------------------------
# httpx.post hook (jina reranker)
# ---------------------------------------------------------------------------


def install_http_post_hook() -> dict[str, Any]:
    import httpx

    orig = httpx.post
    if getattr(orig, "__cassette_patched__", False):
        return {}

    def post(url, *args, **kwargs):
        u = httpx.URL(str(url))
        req: dict[str, Any] = {"kind": "http_post", "path": u.path}
        for k in ("json", "data", "content", "params"):
            if kwargs.get(k) is not None:
                v = kwargs[k]
                if isinstance(v, bytes):
                    v = v.decode("utf-8", "replace")
                req[k] = v
        req = json.loads(canon(req))
        key = request_key(req)
        ctx = _call_context(("httpx/_api.py",))
        ctx["url_host"] = f"{u.scheme}://{u.host}:{u.port}" if u.port else f"{u.scheme}://{u.host}"
        ctx["timeout"] = repr(kwargs.get("timeout"))
        ctx["header_names"] = sorted((kwargs.get("headers") or {}).keys())
        if MODE == "record":
            t0 = time.monotonic()
            try:
                with _InHook():
                    resp = orig(url, *args, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                CAS.record(kind="http_post", req=req, key=key, ctx=ctx, t0=t0, t1=time.monotonic(),
                           exc=serialise_exception(exc))
                raise
            CAS.record(kind="http_post", req=req, key=key, ctx=ctx, t0=t0, t1=time.monotonic(),
                       resp={"status_code": resp.status_code,
                             "content_type": resp.headers.get("content-type", ""),
                             "text": resp.text})
            return resp
        entry = CAS.take(key, ctx["lineage"], ctx["lseq"])
        if entry is None:
            raise _miss("http_post", req, key, ctx)
        _maybe_sleep(entry)
        if "exc" in entry:
            raise rebuild_exception(entry["exc"])
        r = entry["resp"]
        headers = {"content-type": r["content_type"]} if r.get("content_type") else {}
        return httpx.Response(r["status_code"], content=r["text"].encode("utf-8"), headers=headers,
                              request=httpx.Request("POST", str(url)))

    post.__cassette_patched__ = True
    httpx.post = post
    return {"http_post_hook": "httpx.post"}


# ---------------------------------------------------------------------------
# socket guard
# ---------------------------------------------------------------------------


def install_socket_guard() -> dict[str, Any]:
    if getattr(socket.socket.connect, "__cassette_patched__", False):
        return {}
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex

    def _check(sock, address):
        if in_hook():
            return
        stack = []
        f = sys._getframe(2)
        while f is not None and len(stack) < 12:
            stack.append(f"{os.path.basename(f.f_code.co_filename)}:{f.f_code.co_name}:{f.f_lineno}")
            f = f.f_back
        info = {"address": repr(address), "family": str(getattr(sock, "family", "")),
                "lineage": current_lineage(), "stack": stack, "mode": MODE}
        if CAS is not None:
            CAS.record_leak(info)
        eprint(f"[cassette] NETWORK LEAK ({MODE}) to {address!r} from {stack[:4]}")
        if MODE == "replay":
            raise CassetteNetworkLeak(f"cassette replay: network connection to {address!r} outside any hook")

    def connect(self, address):
        _check(self, address)
        return orig_connect(self, address)

    def connect_ex(self, address):
        _check(self, address)
        return orig_connect_ex(self, address)

    connect.__cassette_patched__ = True
    connect_ex.__cassette_patched__ = True
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    return {"socket_guard": "socket.socket.connect/connect_ex"}


def install_all(cas: Cassette, mode: str, pkg: str = "panini_dr", llm_hook: str = DEFAULT_LLM_HOOK,
                llm_response: str = DEFAULT_LLM_RESPONSE) -> dict[str, Any]:
    """Install every hook (idempotent) and point them at ``cas``. Returns what was patched."""
    global CAS, MODE
    CAS, MODE = cas, mode
    installed: dict[str, Any] = {}
    installed.update(install_socket_guard())
    install_lineage()
    installed["lineage"] = "concurrent.futures.ThreadPoolExecutor.submit"
    installed.update(install_http_post_hook())
    installed.update(install_embed_hook())
    mod_cls, _, meth = llm_hook.rpartition(".")
    mod, _, cls = mod_cls.rpartition(".")
    installed.update(install_llm_hook(mod.format(pkg=pkg), cls, meth, llm_response.format(pkg=pkg)))
    return installed


__all__ = [
    "DEFAULT_LLM_HOOK",
    "DEFAULT_LLM_RESPONSE",
    "current_lineage",
    "in_hook",
    "install_all",
    "install_embed_hook",
    "install_http_post_hook",
    "install_lineage",
    "install_llm_hook",
    "install_socket_guard",
]
