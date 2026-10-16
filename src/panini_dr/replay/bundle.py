"""Verify bundles: what ``panini-dr verify`` replays, and how maintainers build one.

A bundle directory holds, for one or more *cases* (one recorded run each)::

    manifest.json                    format, cases, data requirements, sha256 of every file
    subsets/<subset>.json            the questions of each case (ids only)
    cassettes/<case>/calls.jsonl     every external response of the recorded run
    cassettes/<case>/blobs.jsonl     the request bodies they answer (content-addressed)
    cassettes/<case>/leaks.jsonl     empty: the recording made no unhooked connection
    cassettes/<case>/meta.json       scrubbed recording metadata (versions, counts, config)
    reference/<case>/cell_result.json        the recorded run's outputs, NORMALISED
    reference/<case>/traces/q_<id>.{json,md}  (see panini_dr.replay.normalise)

It carries no corpus: the corpus, its dense index and the question file come from the
user's data directory, and the manifest pins them (sha256 of the corpus, of the dense
index's chunk-id list, and of each subset question's text and gold answer).

Building a bundle SCRUBS everything machine- or lab-specific: the LLM base URL and the
reranker host in every recorded call are replaced by placeholders, the recording
metadata is rebuilt from a whitelist, and the reference outputs are normalised (run
directory, data directory and source roots become placeholders). The finished bundle is
then scanned for absolute paths, IP:port pairs and every identifier derived from the
build inputs and the build machine (paths, host name, user name, service hosts) plus any
``deny`` patterns the maintainer adds; a single hit aborts the build.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import shutil
import socket
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from panini_dr.replay.compare import cell_dir
from panini_dr.replay.normalise import Normaliser
from panini_dr.replay.store import BLOBS, CALLS, LEAKS, META, canon, expand_request, request_key

FORMAT = "panini-dr-verify-bundle/1"
MANIFEST = "manifest.json"
LLM_BASE_URL_PLACEHOLDER = "<LLM_BASE_URL>"
SERVICE_HOST_PLACEHOLDER = "<SERVICE_HOST>"

#: Patterns no bundle file may contain (in addition to the identifiers derived at build time).
DEFAULT_DENY_REGEXES: tuple[str, ...] = (
    # an absolute POSIX path (not the path part of a URL)
    r"(?<![\w.:/~-])/(?:home|mnt|media|tmp|Users|root|scratch|var|opt|srv|nfs|net|data\d*|workspace)/",
    r"\b[A-Za-z]:(?:\\){1,4}(?:Users|home)\b",           # a Windows user path (raw or JSON-escaped)
    r"\b(?:localhost|127\.0\.0\.1)\b",
    r"\b(?:\d{1,3}\.){3}\d{1,3}:\d{2,5}\b",               # IP:port
)


class BundleError(RuntimeError):
    """A bundle cannot be built or used as is."""


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# scrubbing
# ---------------------------------------------------------------------------


def scrub_call(entry: dict[str, Any]) -> dict[str, Any]:
    """One calls.jsonl entry with its non-key service locations replaced.

    Only metadata changes: ``key``, ``req`` and ``resp`` are left byte-identical (the
    request is what replay matches on; the response is what it serves).
    """
    e = dict(entry)
    client = e.get("client")
    if isinstance(client, dict) and client.get("base_url"):
        e["client"] = {**client, "base_url": LLM_BASE_URL_PLACEHOLDER}
    if e.get("url_host"):
        e["url_host"] = SERVICE_HOST_PLACEHOLDER
    return e


def scrub_args(args: list[str]) -> list[str]:
    """Runner arguments without secrets, URLs or local paths (kept as basenames)."""
    out: list[str] = []
    hide = False
    for i, a in enumerate(args):
        if hide:
            out.append("<redacted>")
            hide = False
            continue
        if a in ("--api-key",):
            out.append(a)
            hide = True
            continue
        if "://" in a:
            out.append("<URL>")
        elif a.startswith(("/", "~")) or (i == 0 and os.sep in a):
            out.append("<PATH>/" + Path(a).name)
        else:
            out.append(a)
    return out


def _service_summary(models: Any, version: Any) -> dict[str, Any] | None:
    if not isinstance(models, dict) or not isinstance(models.get("data"), list) or not models["data"]:
        return None
    m = models["data"][0]
    out = {k: m.get(k) for k in ("id", "max_model_len", "owned_by") if m.get(k) is not None}
    root = m.get("root")
    if isinstance(root, str) and not root.startswith(("/", "~", ".")) and "://" not in root:
        out["model"] = root  # a hub id such as an organisation/model name; local paths are dropped
    if isinstance(version, dict) and version.get("version"):
        out["server_version"] = version["version"]
    return out


_META_KEEP = ("tool", "mode", "pkg", "started_utc", "finished_utc", "wall_s", "runner_exit", "source_commit",
              "subset_id", "benchmark", "system_id", "model_id", "n_total", "answers", "config",
              "counts_by_kind", "calls_by_question", "n_exceptions_recorded", "n_leaks", "patches",
              "reader_tokenizer")
_ENV_KEEP = ("PYTHONHASHSEED", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def scrub_meta(meta: dict[str, Any], *, reader_tokenizer: str | None = None) -> dict[str, Any]:
    """Recording metadata rebuilt from a whitelist (no paths, hosts, argv or cwd)."""
    out = {k: meta[k] for k in _META_KEEP if k in meta}
    if reader_tokenizer and not out.get("reader_tokenizer"):
        out["reader_tokenizer"] = reader_tokenizer
    out["env"] = {k: v for k, v in (meta.get("env") or {}).items() if k in _ENV_KEEP}
    out["versions"] = {k: (v.split()[0] if k == "python" and isinstance(v, str) else v)
                       for k, v in (meta.get("versions") or {}).items() if k not in ("executable", "platform")}
    services = meta.get("services") or {}
    out["services"] = {k: v for k, v in (
        ("llm", _service_summary(services.get("llm_models"), services.get("llm_version"))),
        ("reranker", _service_summary(services.get("jina_models"), services.get("jina_version"))),
    ) if v}
    if meta.get("runner"):
        out["runner"] = scrub_args(list(meta["runner"]))
    if meta.get("runner_script"):
        out["runner_script"] = Path(meta["runner_script"]).name
    return out


# ---------------------------------------------------------------------------
# scanning
# ---------------------------------------------------------------------------


def _url_hosts(values: Iterable[Any]) -> set[str]:
    """Host names of the URLs in ``values`` (strings, or lists / dicts of them)."""
    hosts: set[str] = set()

    def visit(v: Any) -> None:
        if isinstance(v, str) and "://" in v:
            for m in re.finditer(r"[a-zA-Z][\w+.-]*://[^\s\"'<>]+", v):
                h = urlsplit(m.group(0)).hostname
                if h:
                    hosts.add(h)
        elif isinstance(v, dict):
            for x in v.values():
                visit(x)
        elif isinstance(v, list):
            for x in v:
                visit(x)

    for v in values:
        visit(v)
    return hosts


def derived_deny_literals(*, inputs: Iterable[Path], metas: Iterable[dict[str, Any]],
                          service_urls: Iterable[str] = (), extra: Iterable[str] = (),
                          machine: bool = True) -> list[str]:
    """Identifiers that must not reach a bundle.

    The build inputs' paths, the recording's source repo / cwd / runner paths, the hosts
    of every service URL the recording talked to (from its argv, environment and calls)
    and, with ``machine``, this machine's host name, user name and home directory.
    """
    lits: set[str] = set()
    for p in inputs:
        for q in {str(Path(p)), str(Path(p).resolve())}:
            lits.add(q)
    if machine:
        for get in (socket.gethostname, getpass.getuser, lambda: str(Path.home())):
            try:
                lits.add(get())
            except Exception:  # noqa: BLE001
                pass
    hosts = _url_hosts(service_urls)
    for m in metas:
        for k in ("source_repo", "cwd", "runner_script"):
            if isinstance(m.get(k), str):
                lits.add(m[k])
        for a in [*(m.get("runner") or []), *(m.get("argv") or [])]:
            if isinstance(a, str) and a.startswith("/"):
                lits.add(a)
        hosts |= _url_hosts([m.get("runner"), m.get("argv"), m.get("env")])
    lits.update(h for h in hosts if h != "localhost" and not h.endswith(".invalid"))
    lits.update(x for x in extra if x)
    # very short literals would match ordinary text; the regexes cover generic paths
    return sorted(x for x in lits if len(x) >= 4 and x not in ("/", "/tmp", "/home"))


def _literal_re(lit: str) -> re.Pattern:
    """``lit`` as a whole token: not glued to letters / digits on either side."""
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(lit) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def scan_bundle(root: Path, *, literals: Iterable[str] = (), regexes: Iterable[str] = DEFAULT_DENY_REGEXES,
                skip: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Every hit of a deny literal / regex in any bundle file: [{file, pattern, count, sample}]."""
    root = Path(root)
    lits = [x for x in literals if x]
    pats = [re.compile(r) for r in regexes]
    lit_re = [_literal_re(x) for x in lits]
    hits: list[dict[str, Any]] = []
    skip = set(skip)
    for p in sorted(root.rglob("*")):
        if not p.is_file() or str(p.relative_to(root)) in skip:
            continue
        text = p.read_bytes().decode("utf-8", "replace")
        for name, rx in [*zip(lits, lit_re), *((r.pattern, r) for r in pats)]:
            found = list(rx.finditer(text))
            if found:
                m = found[0]
                hits.append({"file": str(p.relative_to(root)), "pattern": name, "count": len(found),
                             "sample": text[max(0, m.start() - 40): m.end() + 40]})
    return hits


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------


@dataclass
class CaseSource:
    """One recorded run to package as a verify case."""

    name: str
    cassette: Path
    run_dir: Path                 # the record run's --output-dir (or its cell dir)
    subset: Path                  # the subset file the run used
    articles_cache: str           # corpus path RELATIVE to the data dir, e.g. frames/HYBRID_FRAMES824.json
    split: str = "full"
    concurrency: int | None = None
    model: str | None = None
    dense_cache: str | None = None  # dense index dir relative to the data dir (default: fingerprint dir)
    extra_run_args: list[str] = field(default_factory=list)


def _runner_value(runner: list[str], name: str) -> str | None:
    for i, a in enumerate(runner):
        if a == name and i + 1 < len(runner):
            return runner[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def _questions_file(benchmark: str, split: str) -> str:
    b = (benchmark or "frames").lower()
    if b == "monaco":
        return "monaco/monaco_full1315.json"
    if b in ("browsecomp_plus", "browsecomp-plus", "bcp"):
        return "browsecomp_plus/browsecomp_plus_830.json"
    if b == "byod":
        raise BundleError("bring-your-own datasets are not supported in verify bundles")
    return "frames/frames_dev.json" if split == "dev" else "frames/frames_split.json"


def dense_fingerprint_dir(corpus_path: Path, benchmark: str, model_name: str) -> str:
    """``retrieval_cache/<fingerprint>``: where ``panini-dr run`` looks for the dense index."""
    from panini_dr.retrieval.dense import DenseRetriever
    from panini_dr.runner import BCP_BENCHMARK_NAMES, DatasetSpec, load_corpus

    chunking = "whole_doc" if (benchmark or "").lower() in BCP_BENCHMARK_NAMES else "paragraph"
    corpus = load_corpus(DatasetSpec(benchmark=benchmark, subset_id="", questions=[], corpus_path=corpus_path,
                                     chunking=chunking))
    fp = DenseRetriever._fingerprint(types.SimpleNamespace(corpus=corpus, model_name=model_name))
    return f"retrieval_cache/{fp}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _copy_cassette(src: Path, dst: Path, *, reader_tokenizer: str | None) -> dict[str, Any]:
    """Scrubbed copy of a cassette; checks every key still matches its request."""
    dst.mkdir(parents=True, exist_ok=True)
    leaks = (src / LEAKS).read_text() if (src / LEAKS).exists() else ""
    if leaks.strip():
        raise BundleError(f"{src}: the recording made connections outside the hooks (leaks.jsonl); "
                          "it is not a closed recording")
    blobs = {d["h"]: d["v"] for d in _read_jsonl(src / BLOBS)}
    entries = _read_jsonl(src / CALLS)
    bad = [e["i"] for e in entries if request_key(expand_request(e["req"], blobs)) != e["key"]]
    if bad:
        raise BundleError(f"{src}: {len(bad)} recorded calls do not hash to their key (e.g. i={bad[:5]})")
    with open(dst / CALLS, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(canon(scrub_call(e)) + "\n")
    shutil.copyfile(src / BLOBS, dst / BLOBS)
    (dst / LEAKS).write_text("")
    meta = json.loads((src / META).read_text()) if (src / META).exists() else {}
    _write_json(dst / META, scrub_meta(meta, reader_tokenizer=reader_tokenizer))
    kinds: dict[str, int] = {}
    paths: set[str] = set()
    embed_models: set[str] = set()
    urls: set[str] = set()
    for e in entries:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        if isinstance(e.get("client"), dict) and e["client"].get("base_url"):
            urls.add(e["client"]["base_url"])
        if e.get("url_host"):
            urls.add(e["url_host"])
        if e["kind"] == "http_post":
            paths.add(e["req"].get("path") or "")
        if e["kind"] == "embed":
            embed_models.add(e["req"].get("model") or "")
    return {"meta": meta, "n_calls": len(entries), "counts_by_kind": kinds, "http_paths": sorted(paths),
            "embed_models": sorted(embed_models), "service_urls": sorted(urls)}


def _normalise_reference(cell: Path, dst: Path, replacements: list[tuple[str, str]]) -> dict[str, Any]:
    """Write the normalised reference outputs of one run; returns its cell_result."""
    n = Normaliser(replacements, multiset=False)
    cr = json.loads((cell / "cell_result.json").read_text())
    _write_json(dst / "cell_result.json", n(cr))
    for q in cr.get("questions", []):
        qid = q["question_id"]
        tj, tm = cell / "traces" / f"q_{qid}.json", cell / "traces" / f"q_{qid}.md"
        if not tj.exists():
            raise BundleError(f"{cell}: trace {tj.name} is missing")
        _write_json(dst / "traces" / tj.name, n(json.loads(tj.read_text())))
        if tm.exists():
            (dst / "traces" / tm.name).write_text("\n".join(n.md(tm.read_text())) + "\n", encoding="utf-8")
    return cr


def build_bundle(out: Path, cases: list[CaseSource], *, data_dir: Path, name: str, description: str = "",
                 reader_tokenizer: str | None = None, source_roots: Iterable[Path] = (),
                 deny_literals: Iterable[str] = (), deny_regexes: Iterable[str] = (),
                 recorded_with: str = "", scan_machine_identifiers: bool = True) -> dict[str, Any]:
    """Build a verify bundle at ``out`` (must not exist). Returns the manifest.

    ``reader_tokenizer`` states what the recording's article reader counted tokens with
    (``tiktoken:o200k_base`` or ``heuristic``) when the cassette's metadata does not say.
    ``scan_machine_identifiers`` adds this machine's host name, user name and home
    directory to the scan (on by default; tests on shared CI runners turn it off).
    """
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise BundleError(f"{out} exists and is not empty")
    data_dir = Path(data_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "format": FORMAT, "name": name, "description": description,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recorded_with": recorded_with, "cases": [], "data": {}, "files": {},
    }
    metas: list[dict[str, Any]] = []
    inputs: list[Path] = [data_dir, *[Path(r) for r in source_roots]]
    tokenizers: set[str] = set()
    hashseeds: set[str] = set()
    service_urls: set[str] = set()
    for c in cases:
        cell = cell_dir(c.run_dir).resolve()
        inputs += [Path(c.cassette), Path(c.run_dir), cell, Path(c.subset)]
        cas_info = _copy_cassette(Path(c.cassette), out / "cassettes" / c.name, reader_tokenizer=reader_tokenizer)
        meta = cas_info["meta"]
        metas.append(meta)
        service_urls.update(cas_info["service_urls"])
        tok = meta.get("reader_tokenizer") or reader_tokenizer
        if tok:
            tokenizers.add(tok)
        hashseeds.add(str((meta.get("env") or {}).get("PYTHONHASHSEED", "0")))

        subset = json.loads(Path(c.subset).read_text())
        subset.pop("dataset_path", None)
        subset_rel = f"subsets/{Path(c.subset).name}"
        _write_json(out / subset_rel, subset)

        reps = [(str(cell), "<RUN_DIR>"), (str(cell.parent), "<OUT_DIR>"), (str(data_dir), "<DATA_DIR>")]
        reps += [(str(Path(r).resolve()), "<SRC>") for r in source_roots]
        cr = _normalise_reference(cell, out / "reference" / c.name, reps)
        qids = [q["question_id"] for q in cr.get("questions", [])]
        if list(subset.get("question_ids") or []) != qids:
            raise BundleError(f"case {c.name}: the run's questions {qids} are not the subset's "
                              f"{subset.get('question_ids')}")

        runner = list(meta.get("runner") or [])
        model = c.model or cr.get("model_id") or _runner_value(runner, "--model")
        conc = c.concurrency or int(_runner_value(runner, "--concurrency") or 32)
        benchmark = subset.get("benchmark") or cr.get("benchmark") or "frames"
        split = c.split or subset.get("split") or "full"

        corpus_path = data_dir / c.articles_cache
        if not corpus_path.is_file():
            raise BundleError(f"case {c.name}: corpus {corpus_path} not found")
        if c.articles_cache not in manifest["data"]:
            manifest["data"][c.articles_cache] = {"role": "corpus", "sha256": sha256_file(corpus_path),
                                                  "bytes": corpus_path.stat().st_size}
        if "embed" in cas_info["counts_by_kind"]:
            model_name = (cas_info["embed_models"] or [""])[0]
            dense_rel = c.dense_cache or dense_fingerprint_dir(corpus_path, benchmark, model_name)
            ddir = data_dir / dense_rel
            files = {}
            for fn in ("chunk_ids.pkl", "chunk_embeddings.npy", "faiss.index"):
                p = ddir / fn
                if not p.is_file():
                    raise BundleError(f"case {c.name}: dense index file {p} not found")
                files[fn] = {"bytes": p.stat().st_size}
            files["chunk_ids.pkl"]["sha256"] = sha256_file(ddir / "chunk_ids.pkl")
            manifest["data"].setdefault(dense_rel, {"role": "dense index", "encoder": model_name,
                                                    "for_corpus": c.articles_cache, "files": files})
        qfile = _questions_file(benchmark, split)
        manifest["data"].setdefault(qfile, {"role": "questions",
                                            "check": "each case's subset questions must match (text and gold answer)"})

        run_args = ["--subset", "{bundle}/" + subset_rel, "--articles-cache", "{data}/" + c.articles_cache,
                    "--split", split, "--model", model, "--concurrency", str(conc)]
        if c.dense_cache:
            run_args += ["--dense-cache", "{data}/" + c.dense_cache]
        run_args += list(c.extra_run_args)
        manifest["cases"].append({
            "name": c.name, "benchmark": benchmark, "subset_id": subset.get("subset_id"),
            "subset": subset_rel, "cassette": f"cassettes/{c.name}", "reference": f"reference/{c.name}",
            "questions_file": qfile,
            "questions": {q["question_id"]: {"question_sha256": sha256_text(q.get("question") or ""),
                                             "answer_sha256": sha256_text(str(q.get("gold_answer") or ""))}
                          for q in cr.get("questions", [])},
            "answers": {q["question_id"]: q.get("predicted_answer") for q in cr.get("questions", [])},
            "run_args": run_args,
            "rerank_url_path": (cas_info["http_paths"] or ["/v1/rerank"])[0],
            "n_recorded_calls": cas_info["n_calls"], "counts_by_kind": cas_info["counts_by_kind"],
        })
        manifest.setdefault("llm_model", model)
        manifest.setdefault("source_commit", meta.get("source_commit"))
        manifest.setdefault("versions", scrub_meta(meta).get("versions"))
        manifest.setdefault("services", scrub_meta(meta).get("services"))
    if len(tokenizers) > 1:
        raise BundleError(f"cases disagree on the reader tokenizer: {sorted(tokenizers)}")
    if len(hashseeds) != 1:
        raise BundleError(f"cases disagree on PYTHONHASHSEED: {sorted(hashseeds)}")
    manifest["reader_tokenizer"] = next(iter(tokenizers), None)
    manifest["env"] = {"PYTHONHASHSEED": next(iter(hashseeds))}

    literals = derived_deny_literals(inputs=inputs, metas=metas, service_urls=service_urls,
                                     extra=deny_literals, machine=scan_machine_identifiers)
    hits = scan_bundle(out, literals=literals, regexes=[*DEFAULT_DENY_REGEXES, *deny_regexes])
    if hits:
        raise BundleError("bundle contains machine- or lab-specific strings:\n" + "\n".join(
            f"  {h['file']}: {h['pattern']!r} x{h['count']} ...{h['sample']!r}..." for h in hits[:20]))
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != MANIFEST:
            manifest["files"][str(p.relative_to(out))] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    _write_json(out / MANIFEST, manifest)
    hits = scan_bundle(out, literals=literals, regexes=[*DEFAULT_DENY_REGEXES, *deny_regexes])
    if hits:
        raise BundleError(f"manifest contains machine- or lab-specific strings: {hits[:5]}")
    return manifest


# ---------------------------------------------------------------------------
# using
# ---------------------------------------------------------------------------


def load_manifest(bundle: Path) -> dict[str, Any]:
    p = Path(bundle) / MANIFEST
    if not p.is_file():
        raise BundleError(f"{bundle} is not a verify bundle (no {MANIFEST})")
    m = json.loads(p.read_text())
    if m.get("format") != FORMAT:
        raise BundleError(f"{p}: unsupported bundle format {m.get('format')!r} (expected {FORMAT})")
    return m


def check_integrity(bundle: Path, manifest: dict[str, Any]) -> list[str]:
    """Files that are missing or differ from the manifest's sha256."""
    problems = []
    for rel, info in sorted(manifest.get("files", {}).items()):
        p = Path(bundle) / rel
        if not p.is_file():
            problems.append(f"{rel}: missing")
        elif sha256_file(p) != info.get("sha256"):
            problems.append(f"{rel}: modified (sha256 differs from the manifest)")
    return problems


def expand_run_args(args: list[str], *, bundle: Path, data_dir: Path) -> list[str]:
    return [a.replace("{bundle}", str(bundle)).replace("{data}", str(data_dir)) for a in args]


__all__ = [
    "DEFAULT_DENY_REGEXES",
    "FORMAT",
    "MANIFEST",
    "BundleError",
    "CaseSource",
    "build_bundle",
    "check_integrity",
    "dense_fingerprint_dir",
    "derived_deny_literals",
    "expand_run_args",
    "load_manifest",
    "scan_bundle",
    "scrub_args",
    "scrub_call",
    "scrub_meta",
    "sha256_file",
    "sha256_text",
]
