"""``panini-dr verify``: replay a bundle's recorded runs and compare them strictly.

For every case of the bundle this

1. checks the bundle (sha256 of every file) and the data it needs (corpus sha256,
   dense index, the subset questions' text and gold answer, the reader's tokenizer),
2. runs ``panini-dr run`` on the case's questions in a fresh process with every LLM,
   query-embedding and reranker response served from the cassette
   (``python -m panini_dr.replay --mode replay``; PYTHONHASHSEED from the recording,
   CUDA_VISIBLE_DEVICES="", HF offline, a socket guard that turns any connection into
   a failure, and the per-question event-order gate; ``--skip-preflight``, since the
   run's LLM and reranker test requests are not part of a recording),
3. compares the run's outputs with the bundle's reference outputs: the final answer
   raw, the normalised trace, the embedded trajectory, the Markdown rendering and
   the cell's config + aggregates, in strict order (see :mod:`panini_dr.replay.normalise`
   for the only fields a comparison ignores), and requires that every recorded call
   was served exactly once (no miss, nothing left over, no network leak).

Exit status: 0 PASS, 1 FAIL, 2 the bundle or the data cannot be used.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from panini_dr.replay.bundle import (
    BundleError,
    check_integrity,
    expand_run_args,
    load_manifest,
    sha256_file,
    sha256_text,
)
from panini_dr.replay.compare import cell_dir, compare_cells, replay_gate
from panini_dr.replay.launcher import frozen_env
from panini_dr.replay.normalise import DROP_KEYS, Normaliser

REPLAY_LLM_BASE_URL = "http://llm.replay.invalid/v1"
REPLAY_RERANK_HOST = "http://rerank.replay.invalid"

#: Variables that change what the harness does; the replay runs with them EMPTY so a
#: shell export or a ``.env`` file (read by every ``panini-dr`` command, but never over a
#: variable that is already set, even to empty) cannot change the run.
NEUTRALISED_ENV = (
    "GSW_READ_TITLE_FALLBACK", "GSW_PLANNER_TARGET_HUB_MODE", "GSW_STRICT_TARGET_HUB_GATE",
    "GSW_DENSE_GPU", "GSW_FAISS_THREADS", "PANINI_STRICT_RERANK", "MONACO_QUESTIONS",
    "BROWSECOMP_QUESTIONS", "PANINI_SIDECAR_BASE_URL", "PANINI_LLM_MODEL", "PANINI_LLM_API_KEY",
)

MAX_DIFF_LINES = 40
MAX_PATH_DIFFS = 8


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def _tiktoken_offline() -> bool:
    """Whether tiktoken's o200k_base encoding loads WITHOUT network access."""
    try:
        import tiktoken
    except Exception:  # noqa: BLE001
        return False
    orig, orig_ex = socket.socket.connect, socket.socket.connect_ex

    def refuse(*a, **k):
        raise OSError("network disabled by panini-dr verify preflight")

    socket.socket.connect, socket.socket.connect_ex = refuse, refuse
    try:
        tiktoken.get_encoding("o200k_base")
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        socket.socket.connect, socket.socket.connect_ex = orig, orig_ex


def preflight(bundle: Path, manifest: dict[str, Any], data_dir: Path, cases: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """(errors, notes): errors make verify exit 2 before replaying anything."""
    errors: list[str] = []
    notes: list[str] = []
    if not data_dir.is_dir():
        return [f"data dir {data_dir} does not exist (pass --data-dir or set PANINI_DATA_DIR)"], notes
    missing_mods = [m for m in ("numpy", "faiss", "torch", "sentence_transformers")
                    if importlib.util.find_spec(m) is None]
    if missing_mods:
        errors.append(f"python packages missing: {', '.join(missing_mods)}. verify replays the hybrid "
                      "retriever, which needs the 'dense' extra (CPU only is fine): "
                      "uv sync --locked --inexact --extra dense")
    for rel, info in manifest.get("data", {}).items():
        p = data_dir / rel
        role = info.get("role")
        if role == "corpus":
            if not p.is_file():
                errors.append(f"corpus {p} not found (the bundle's runs read {rel} under the data dir)")
            elif p.stat().st_size != info.get("bytes") or sha256_file(p) != info.get("sha256"):
                errors.append(f"corpus {p} is not the recorded corpus (sha256 {info.get('sha256')})")
        elif role == "dense index":
            for fn, f in (info.get("files") or {}).items():
                fp = p / fn
                if not fp.is_file():
                    errors.append(f"dense index file {fp} not found (the dense index of {info.get('for_corpus')})")
                elif fp.stat().st_size != f.get("bytes") or ("sha256" in f and sha256_file(fp) != f["sha256"]):
                    errors.append(f"dense index file {fp} differs from the recorded index")
        elif role == "questions" and not p.is_file():
            errors.append(f"question file {p} not found")
    if errors:
        return errors, notes
    errors += _check_questions(bundle, data_dir, cases)
    tok = manifest.get("reader_tokenizer")
    if tok:
        have = "tiktoken:o200k_base" if _tiktoken_offline() else "heuristic"
        if have != tok:
            if tok.startswith("tiktoken"):
                errors.append("the article reader of the recorded runs counted tokens with tiktoken's o200k_base "
                              "encoding, which is not available offline here. Fetch it once with network access: "
                              "python -c \"import tiktoken; tiktoken.get_encoding('o200k_base')\"")
            else:
                errors.append("the recorded runs counted reader tokens with the heuristic (no tiktoken), but "
                              "tiktoken's o200k_base encoding is available here, so the reader would read "
                              "different amounts of text")
    have_v = _local_versions()
    for k, v in (manifest.get("versions") or {}).items():
        if k in have_v and isinstance(v, str) and not v.startswith("unavailable") and have_v[k] != v:
            notes.append(f"{k} {have_v[k]} here, {v} in the recording")
    return errors, notes


def _local_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for mod, dist in (("numpy", "numpy"), ("faiss", "faiss-cpu"), ("httpx", "httpx"), ("openai", "openai")):
        try:
            out[mod] = version(dist)
        except PackageNotFoundError:
            pass
    return out


def _check_questions(bundle: Path, data_dir: Path, cases: list[dict[str, Any]]) -> list[str]:
    """The data dir's question file must hold each case's questions unchanged."""
    from panini_dr.eval.subset import load_benchmark_questions, load_subset

    errors = []
    saved = {k: os.environ.get(k) for k in ("PANINI_DATA_DIR", "MONACO_QUESTIONS", "BROWSECOMP_QUESTIONS")}
    os.environ["PANINI_DATA_DIR"] = str(data_dir)
    os.environ.pop("MONACO_QUESTIONS", None)
    os.environ.pop("BROWSECOMP_QUESTIONS", None)
    try:
        for case in cases:
            subset = load_subset(bundle / case["subset"])
            split = case["run_args"][case["run_args"].index("--split") + 1] if "--split" in case["run_args"] else "full"
            try:
                qs = {q.id: q for q in load_benchmark_questions(subset, split=split)}
            except Exception as exc:  # noqa: BLE001
                errors.append(f"[{case['name']}] cannot load its questions from {data_dir / case['questions_file']}: {exc}")
                continue
            for qid, want in case.get("questions", {}).items():
                q = qs.get(qid)
                if q is None:
                    errors.append(f"[{case['name']}] question {qid} is not in {case['questions_file']}")
                elif sha256_text(q.question) != want["question_sha256"] or \
                        sha256_text(str(q.answer)) != want["answer_sha256"]:
                    errors.append(f"[{case['name']}] question {qid} in {case['questions_file']} differs from the "
                                  "recorded one (text or gold answer)")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return errors


# ---------------------------------------------------------------------------
# one case
# ---------------------------------------------------------------------------


def child_env(manifest: dict[str, Any], case: dict[str, Any], data_dir: Path) -> dict[str, str]:
    from panini_dr.replay.launcher import with_package_on_path

    env = frozen_env("replay", str((manifest.get("env") or {}).get("PYTHONHASHSEED", "0")))
    for k in NEUTRALISED_ENV:
        env[k] = ""
    env["PANINI_DATA_DIR"] = str(data_dir)
    env["PANINI_LLM_BASE_URL"] = REPLAY_LLM_BASE_URL
    env["JINA_RERANK_URL"] = REPLAY_RERANK_HOST + case.get("rerank_url_path", "/v1/rerank")
    return with_package_on_path(env)


def child_argv(bundle: Path, case: dict[str, Any], data_dir: Path, out: Path, report: Path,
               gate_timeout: float) -> list[str]:
    return [sys.executable, "-m", "panini_dr.replay", "--mode", "replay", "--cassette", str(bundle / case["cassette"]),
            "--report-dir", str(report), "--gate-timeout", str(gate_timeout), "--",
            "run", *expand_run_args(case["run_args"], bundle=bundle, data_dir=data_dir),
            "--base-url", REPLAY_LLM_BASE_URL, "--api-key", "replay", "--skip-preflight",
            "--output-dir", str(out)]


def clip_diff(lines: list[str], width: int = 200) -> list[str]:
    """Shorten long unified-diff lines; a changed ``-``/``+`` pair is cut around its first difference."""
    out = list(lines)
    for i in range(len(out) - 1):
        a, b = out[i], out[i + 1]
        if a.startswith("-") and b.startswith("+") and not a.startswith("---") and (len(a) > width or len(b) > width):
            k = next((j for j in range(1, min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            lo = max(1, k - width // 3)
            out[i] = a[0] + ("..." if lo > 1 else "") + a[lo:lo + width] + ("..." if lo + width < len(a) else "")
            out[i + 1] = b[0] + ("..." if lo > 1 else "") + b[lo:lo + width] + ("..." if lo + width < len(b) else "")
    return [ln if len(ln) <= width + 8 else ln[:width] + "..." for ln in out]


def _tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def run_case(bundle: Path, manifest: dict[str, Any], case: dict[str, Any], data_dir: Path, out_root: Path,
             *, gate_timeout: float, echo: Callable[[str], None]) -> dict[str, Any]:
    name = case["name"]
    out, report_dir, log = out_root / name, out_root / f"{name}_cassette", out_root / f"{name}.log"
    qids = list(case.get("questions", {}))
    echo(f"[{name}] replaying {len(qids)} questions ({case.get('n_recorded_calls')} recorded calls) ...")
    t0 = time.time()
    with open(log, "w") as lf:
        proc = subprocess.run(child_argv(bundle, case, data_dir, out, report_dir, gate_timeout),
                              env=child_env(manifest, case, data_dir), cwd=out_root,
                              stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    res: dict[str, Any] = {"name": name, "rc": proc.returncode, "wall_s": round(wall, 1), "pass": False,
                           "n_questions": len(qids), "n_pass": 0}
    summary_p = report_dir / "summary.json"
    try:
        cell = cell_dir(out)
    except FileNotFoundError:
        cell = None
    if cell is None or not summary_p.is_file():
        echo(f"  the replay did not finish (exit {proc.returncode}); last lines of {log.name}:")
        for line in _tail(log).splitlines():
            echo(f"    {line}")
        echo(f"[{name}] FAIL ({wall:.1f} s)")
        return res
    summary = json.loads(summary_p.read_text())
    reps = [(str(cell), "<RUN_DIR>"), (str(cell.parent), "<OUT_DIR>"), (str(data_dir), "<DATA_DIR>"),
            (str(bundle), "<BUNDLE>")]
    report = compare_cells(bundle / case["reference"], cell, Normaliser([], multiset=False),
                           Normaliser(reps, multiset=False), limit=MAX_PATH_DIFFS)
    gate = replay_gate(summary)
    for qid, r in report["questions"].items():
        if "why" in r:
            echo(f"  q{qid:<6} FAIL  ({r['why']})")
            continue
        verdict = "PASS" if r["pass"] else "FAIL"
        line = f"  q{qid:<6} {verdict}  answer={r['answer_left']!r}"
        if not r["answer_equal"]:
            line += f" -> replay {r['answer_right']!r}"
        line += f"  (turns {r['turns']}, tool calls {r['tool_calls']})"
        echo(line)
        if not r["pass"]:
            parts = [f"answer {'same' if r['answer_equal'] else 'DIFFERENT'}",
                     f"trace {'same' if r['trace_equal'] else 'DIFFERENT'}",
                     f"result {'same' if r['result_equal'] else 'DIFFERENT'}",
                     f"markdown {'same' if r['md_equal'] else 'DIFFERENT'}"]
            echo(f"           {'; '.join(parts)}")
            for d in r["diffs"]:
                echo(f"           {d[:220]}")
    res["n_pass"] = sum(1 for r in report["questions"].values() if r.get("pass"))
    echo(f"  cell_result (config + aggregates): {'same' if report['cell']['equal'] else 'DIFFERENT'}")
    for d in report["cell"]["diffs"]:
        echo(f"    {d[:220]}")
    served = sum((summary.get("pick_stats") or {}).values())
    echo(f"  cassette: {served}/{summary.get('n_recorded')} recorded calls served, {gate['misses']} misses, "
         f"{gate['leftover']} left over, {gate['leaks']} network leaks")
    if summary.get("misses"):
        m0 = summary["misses"][0]
        echo(f"  first miss: kind={m0.get('kind')} question={str(m0.get('lineage') or '').split('/')[0]} "
             f"role={m0.get('role')} site={m0.get('site')}")
        diff = report_dir / "misses" / "miss_0000.diff"
        if diff.is_file():
            lines = clip_diff(diff.read_text(errors="replace").splitlines())
            echo(f"  request diff, recorded (-) vs replay (+), first {min(len(lines), MAX_DIFF_LINES)} of {len(lines)} lines:")
            for line in lines[:MAX_DIFF_LINES]:
                echo(f"    {line}")
    for leak in (summary.get("leaks") or [])[:3]:
        echo(f"  network leak to {leak.get('address')} from {leak.get('stack', [])[:4]}")
    if proc.returncode not in (0, 3, 4):
        echo(f"  the replay process exited {proc.returncode}; last lines of {log.name}:")
        for line in _tail(log, 15).splitlines():
            echo(f"    {line}")
    ok = report["verdict"] == "PASS" and gate["ok"] and proc.returncode == 0
    res.update({"pass": ok, "summary": summary, "report": report})
    echo(f"[{name}] {'PASS' if ok else 'FAIL'} ({wall:.1f} s)")
    return res


# ---------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------


def verify_bundle(bundle: Path, *, data_dir: Path | None = None, keep_output: Path | None = None,
                  only: list[str] | None = None, gate_timeout: float = 60.0,
                  echo: Callable[[str], None] = print) -> int:
    """Replay every case of ``bundle`` and compare; returns the exit status (0/1/2)."""
    bundle = Path(bundle).resolve()
    try:
        manifest = load_manifest(bundle)
    except BundleError as exc:
        echo(f"error: {exc}")
        return 2
    data_dir = Path(data_dir or os.environ.get("PANINI_DATA_DIR") or "data").expanduser().resolve()
    cases = [c for c in manifest["cases"] if not only or c["name"] in only]
    if only and len(cases) != len(set(only)):
        echo(f"error: unknown case(s) {sorted(set(only) - {c['name'] for c in manifest['cases']})}; "
             f"the bundle has {[c['name'] for c in manifest['cases']]}")
        return 2
    n_q = sum(len(c.get("questions", {})) for c in cases)
    echo(f"panini-dr verify: {manifest.get('name')} ({len(cases)} case(s), {n_q} questions)")
    echo(f"  bundle   {bundle}")
    echo(f"  data dir {data_dir}")
    if manifest.get("recorded_with"):
        echo(f"  recorded with {manifest['recorded_with']}")

    integrity = check_integrity(bundle, manifest)
    if integrity:
        echo(f"  integrity: {len(integrity)} bundle file(s) do not match the manifest -> the verdict will be FAIL")
        for p in integrity[:10]:
            echo(f"    {p}")
    else:
        echo(f"  integrity: all {len(manifest.get('files', {}))} bundle files match the manifest")
    errors, notes = preflight(bundle, manifest, data_dir, cases)
    for n in notes:
        echo(f"  note: {n}")
    if errors:
        for e in errors:
            echo(f"  error: {e}")
        echo("cannot verify: fix the data / environment above (nothing was replayed)")
        return 2
    echo("  data: corpus, dense index and questions match the recording")
    echo("  comparison: strict (answers raw; traces, trajectories, markdown and cell config/aggregates "
         f"after dropping only {', '.join(DROP_KEYS)}, run-dir paths and the run-tag timestamp)")

    if keep_output is not None:
        out_root = Path(keep_output).resolve()
        if out_root.exists() and any(out_root.iterdir()):
            echo(f"error: --keep-output {out_root} exists and is not empty")
            return 2
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        out_root = Path(tempfile.mkdtemp(prefix="panini-verify-"))
    results = []
    try:
        for case in cases:
            results.append(run_case(bundle, manifest, case, data_dir, out_root, gate_timeout=gate_timeout, echo=echo))
    finally:
        if keep_output is None:
            shutil.rmtree(out_root, ignore_errors=True)
    n_pass = sum(r["n_pass"] for r in results)
    ok = all(r["pass"] for r in results) and not integrity
    wall = sum(r["wall_s"] for r in results)
    if ok:
        echo(f"PASS: {n_pass}/{n_q} questions reproduce the reference outputs exactly ({wall:.0f} s of replay)")
    else:
        why = [f"{r['name']}: {r['n_pass']}/{r['n_questions']} questions identical" for r in results]
        if integrity:
            why.append("the bundle was modified")
        echo(f"FAIL: {'; '.join(why)}")
        if keep_output is None:
            echo("  re-run with --keep-output DIR to keep the replay outputs, logs and per-miss reports")
        else:
            echo(f"  replay outputs, logs and miss reports: {out_root}")
    return 0 if ok else 1


__all__ = ["NEUTRALISED_ENV", "child_argv", "child_env", "clip_diff", "preflight", "run_case", "verify_bundle"]
