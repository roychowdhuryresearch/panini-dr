"""Record / replay launcher: run the harness with every external call routed through a cassette.

    python -m panini_dr.replay --mode {record,replay} --cassette DIR [options] -- <runner args...>

By default the runner is the ``panini-dr`` command line itself, run in this process
(``-- run --subset ... --output-dir ...``). A ``run`` always gets ``--skip-preflight``: its
test requests to the LLM (``GET /models``) and the reranker are not part of what a cassette
records (the GET would be a network leak), so record and replay both leave them out. ``--runner-script PATH`` runs any script as
``__main__`` instead (e.g. the research harness the reference recordings came from; pass its
package with ``--pkg`` so the LLM hook patches the right ``LLMClient``).

The harness source is never edited: every patch is applied at runtime in this process
(see :mod:`panini_dr.replay.patches`), then the runner is executed.

Environment frozen by the launcher (the process re-execs itself so they apply from interpreter start):
  PYTHONHASHSEED   fixed (default 0; replay uses the value recorded in meta.json). str-hash
                   randomisation changes set-iteration order, and the harness renders set-ordered
                   lists into prompts (e.g. _plan_quality.py iterates set(...) of placeholders/years).
  HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE = 1   (no hub traffic while loading the encoder)
  CUDA_VISIBLE_DEVICES = ""  in replay unless --replay-gpu (replay needs no GPU: vectors come from tape)
Exit status: the runner's status, or 3 if replay saw a key miss / network leak, 4 if recorded calls
were left unconsumed (the replay made fewer calls than the recording).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import platform
import runpy
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_PKG = "panini_dr"
ENV_KEYS = ("JINA_RERANK_URL", "MONACO_QUESTIONS", "BROWSECOMP_QUESTIONS", "GSW_MEMORY_ROOT",
            "GSW_PLANNER_TARGET_HUB_MODE", "GSW_STRICT_TARGET_HUB_GATE", "GSW_READ_TITLE_FALLBACK",
            "GSW_FAISS_THREADS", "GSW_DENSE_GPU", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE", "PYTHONHASHSEED", "PYTHONPATH", "JAVA_HOME", "PANINI_DATA_DIR")
_REEXEC_FLAG = "_PANINI_REPLAY_REEXEC"
EXIT_MISS = 3
EXIT_LEFTOVER = 4


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def parse(argv: list[str]):
    if "--" not in argv:
        raise SystemExit("usage: python -m panini_dr.replay [options] -- <runner args...>")
    i = argv.index("--")
    own, runner = argv[:i], argv[i + 1:]
    if not runner:
        raise SystemExit("missing runner arguments after --")
    p = argparse.ArgumentParser(prog="python -m panini_dr.replay")
    p.add_argument("--mode", choices=["record", "replay"], required=True)
    p.add_argument("--cassette", required=True)
    p.add_argument("--pkg", default=DEFAULT_PKG, help="harness package name (default panini_dr)")
    p.add_argument("--runner-script", default="",
                   help="run this script as __main__ with the runner args (default: the panini-dr CLI)")
    p.add_argument("--llm-hook", default="{pkg}.models.llm_client.LLMClient.chat",
                   help="dotted <module>.<Class>.<method>; {pkg} expands")
    p.add_argument("--llm-response-class", default="{pkg}.models.llm_client.LLMResponse")
    p.add_argument("--source-repo", default="", help="git repo whose HEAD is recorded as the source commit")
    p.add_argument("--report-dir", default="", help="replay: where misses/summary go "
                   "(default <cassette>/replay_reports/<utc>_<pid>)")
    p.add_argument("--hashseed", default=None, help="PYTHONHASHSEED (record default 0; replay default = recorded)")
    p.add_argument("--replay-latency-scale", type=float, default=0.0,
                   help="replay: sleep recorded_latency*scale before returning each response (0 = instant)")
    p.add_argument("--replay-gpu", action="store_true", help="replay: keep CUDA devices visible")
    p.add_argument("--replay-order", choices=["recorded", "free"], default="recorded",
                   help="replay: 'recorded' = reproduce the recorded order of hook events per question (a response is "
                        "released only after every same-question response recorded as completing earlier has been "
                        "released AND every same-question request recorded as starting earlier has arrived); "
                        "'free' = release immediately")
    p.add_argument("--gate-timeout", type=float, default=60.0)
    p.add_argument("--no-probe", action="store_true", help="record: do not query the services' /models and /version")
    return p.parse_args(own), runner


def wanted_hashseed(args) -> str:
    want = args.hashseed
    if args.mode == "replay" and want is None:
        meta = Path(args.cassette) / "meta.json"
        if meta.exists():
            want = json.loads(meta.read_text()).get("env", {}).get("PYTHONHASHSEED")
    return str(want if want is not None else "0")


def frozen_env(mode: str, hashseed: str, *, replay_gpu: bool = False, base: dict | None = None) -> dict:
    """The environment a record/replay process must start with."""
    env = dict(os.environ if base is None else base)
    env.update({"PYTHONHASHSEED": hashseed, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    if mode == "replay" and not replay_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def with_package_on_path(env: dict) -> dict:
    """Make sure a fresh interpreter can import this package (it may only be on sys.path)."""
    src = str(Path(__file__).resolve().parents[2])
    parts = [p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p]
    if src not in parts:
        env["PYTHONPATH"] = os.pathsep.join([src, *parts])
    return env


def _reexec_if_needed(args, argv: list[str]) -> None:
    want = wanted_hashseed(args)
    env = frozen_env(args.mode, want, replay_gpu=args.replay_gpu)
    changed = any(os.environ.get(k) != env.get(k)
                  for k in ("PYTHONHASHSEED", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES"))
    if changed and os.environ.get(_REEXEC_FLAG) != "1":
        env[_REEXEC_FLAG] = "1"
        with_package_on_path(env)
        os.execve(sys.executable, [sys.executable, "-m", "panini_dr.replay", *argv], env)
    if os.environ.get("PYTHONHASHSEED") != want:
        raise SystemExit(f"cassette: PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')!r}, wanted {want!r}")


def _runner_opt(runner: list[str], name: str) -> str | None:
    for i, a in enumerate(runner):
        if a == name and i + 1 < len(runner):
            return runner[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def _probe_services(runner: list[str]) -> dict:
    import httpx

    out: dict = {}
    base = _runner_opt(runner, "--base-url") or os.environ.get("PANINI_LLM_BASE_URL", "")
    if base:
        root = base.rstrip("/")
        root = root[: -len("/v1")] if root.endswith("/v1") else root
        for name, url in (("llm_models", base.rstrip("/") + "/models"), ("llm_version", root + "/version")):
            try:
                r = httpx.get(url, timeout=10)
                out[name] = r.json() if r.status_code == 200 else {"status": r.status_code}
            except Exception as exc:  # noqa: BLE001
                out[name] = {"error": repr(exc)}
    jina = os.environ.get("JINA_RERANK_URL", "")
    if jina:
        jroot = jina.split("/v1/")[0]
        for name, url in (("jina_models", jroot + "/v1/models"), ("jina_version", jroot + "/version")):
            try:
                r = httpx.get(url, timeout=10)
                out[name] = r.json() if r.status_code == 200 else {"status": r.status_code}
            except Exception as exc:  # noqa: BLE001
                out[name] = {"error": repr(exc)}
    return out


def _versions() -> dict:
    """Installed versions of the packages that shape a run (from package metadata: no imports)."""
    from importlib.metadata import PackageNotFoundError, version

    v = {"python": sys.version, "platform": platform.platform(), "executable": sys.executable}
    for name, dists in (("openai", ("openai",)), ("httpx", ("httpx",)), ("torch", ("torch",)),
                        ("sentence_transformers", ("sentence-transformers",)), ("transformers", ("transformers",)),
                        ("faiss", ("faiss-cpu", "faiss-gpu", "faiss")), ("numpy", ("numpy",)),
                        ("pyserini", ("pyserini",)), ("tiktoken", ("tiktoken",))):
        v[name] = "unavailable"
        for d in dists:
            try:
                v[name] = version(d)
                break
            except PackageNotFoundError:
                continue
    return v


def _git_head(repo: str) -> dict:
    if not repo:
        return {}
    try:
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        return {"source_repo": repo, "source_commit": head}
    except Exception as exc:  # noqa: BLE001
        return {"source_repo": repo, "source_commit_error": repr(exc)}


def _resolved_config(runner: list[str], since: float) -> dict:
    out_dir = _runner_opt(runner, "--output-dir")
    if not out_dir:
        return {}
    cells = [c for c in glob.glob(os.path.join(out_dir, "*", "cell_result.json")) if os.path.getmtime(c) >= since - 5]
    if not cells:
        return {"cell_result": None}
    cell = max(cells, key=os.path.getmtime)
    d = json.loads(Path(cell).read_text())
    return {"cell_result": cell, "config": d.get("config"), "subset_id": d.get("subset_id"),
            "benchmark": d.get("benchmark"), "system_id": d.get("system_id"), "model_id": d.get("model_id"),
            "n_total": d.get("n_total"),
            "answers": {q["question_id"]: q.get("predicted_answer") for q in d.get("questions", [])}}


def _reader_tokenizer(pkg: str) -> str | None:
    """Which token counter the article reader used: tiktoken's o200k_base or the heuristic."""
    mod = sys.modules.get(f"{pkg}.adapters.ours._agentic_read")
    if mod is None or not hasattr(mod, "_ENCODER"):
        return None
    return "tiktoken:o200k_base" if getattr(mod, "_ENCODER") is not None else "heuristic"


SKIP_PREFLIGHT_FLAGS = ("--skip-preflight", "--skip-rerank-preflight")


def with_skip_preflight(runner: list[str], runner_script: str = "") -> list[str]:
    """``runner`` with ``--skip-preflight`` added when it is a ``panini-dr run`` without it."""
    if runner_script or not runner or runner[0] != "run" or any(a in SKIP_PREFLIGHT_FLAGS for a in runner):
        return list(runner)
    return [*runner, "--skip-preflight"]


def _run_runner(args, runner: list[str]) -> int:
    """Run the harness in this process; returns its exit status."""
    if args.runner_script:
        runner_path = os.path.abspath(args.runner_script)
        sys.argv = [runner_path, *runner]
        sys.path.insert(0, os.path.dirname(runner_path))
        runpy.run_path(runner_path, run_name="__main__")
        return 0
    from panini_dr.cli import main as cli_main

    sys.argv = ["panini-dr", *runner]
    cli_main()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args, runner = parse(argv)
    _reexec_if_needed(args, argv)
    runner = with_skip_preflight(runner, args.runner_script)

    from panini_dr.replay import patches as P
    from panini_dr.replay.store import Cassette, eprint

    cas_dir = Path(args.cassette).resolve()
    started = time.time()
    meta: dict = {"tool": "panini_dr.replay", "mode": args.mode, "pkg": args.pkg, "started_utc": _utc(),
                  "argv": [sys.executable, "-m", "panini_dr.replay", *argv], "runner": runner,
                  "runner_script": args.runner_script or None,
                  "cwd": os.getcwd(), "env": {k: os.environ.get(k) for k in ENV_KEYS if k in os.environ}}
    meta.update(_git_head(args.source_repo))
    if args.mode == "record":
        if (cas_dir / "calls.jsonl").exists():
            raise SystemExit(f"cassette {cas_dir} already holds a recording")
        meta["services"] = {} if args.no_probe else _probe_services(runner)  # before the socket guard
        cas = Cassette(cas_dir, "record")
        report_dir = cas_dir
    else:
        report_dir = Path(args.report_dir) if args.report_dir else (
            cas_dir / "replay_reports" / f"{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}")
        report_dir.mkdir(parents=True, exist_ok=True)
        cas = Cassette(cas_dir, "replay", report_dir=report_dir, latency_scale=args.replay_latency_scale,
                       order=args.replay_order, gate_timeout_s=args.gate_timeout)
        rec_meta = json.loads((cas_dir / "meta.json").read_text()) if (cas_dir / "meta.json").exists() else {}
        meta["recorded_meta"] = {k: rec_meta.get(k) for k in ("source_commit", "pkg", "started_utc", "runner")}
        meta["n_recorded"] = cas.n_recorded
        if rec_meta.get("runner") and rec_meta["runner"][1:] != runner[1:]:
            meta["runner_args_differ_from_record"] = True
            eprint("[cassette] note: runner arguments differ from the recording (paths may legitimately differ)")

    meta["versions"] = _versions()
    meta["patches"] = P.install_all(cas, args.mode, args.pkg, args.llm_hook, args.llm_response_class)
    eprint(f"[cassette] mode={args.mode} cassette={cas_dir} pkg={args.pkg} PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')} "
           f"patches={sorted(meta['patches'])}")

    rc = 0
    try:
        rc = _run_runner(args, runner)
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except BaseException as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        rc = 1
        meta["runner_exception"] = repr(exc)
    finally:
        cas.close()

    meta["finished_utc"] = _utc()
    meta["wall_s"] = round(time.time() - started, 1)
    meta["runner_exit"] = rc
    meta["reader_tokenizer"] = _reader_tokenizer(args.pkg)
    meta.update(_resolved_config(runner, started))
    meta["counts_by_kind"] = dict(cas.counts)
    meta["n_leaks"] = len(cas.leaks)
    final = rc
    if args.mode == "record":
        calls_by_lineage: dict = {}
        n_exc = 0
        with open(cas_dir / "calls.jsonl", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                q = (e.get("lineage") or "").split("/")[0]
                calls_by_lineage.setdefault(q, {}).setdefault(e["kind"], 0)
                calls_by_lineage[q][e["kind"]] += 1
                n_exc += "exc" in e
        meta["calls_by_question"] = calls_by_lineage
        meta["n_exceptions_recorded"] = n_exc
        (cas_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        eprint(f"[cassette] recorded {sum(cas.counts.values())} calls {dict(cas.counts)}; leaks={len(cas.leaks)}; "
               f"wall={meta['wall_s']}s -> {cas_dir}")
    else:
        left = cas.leftovers()
        meta["n_misses"] = len(cas.misses)
        meta["misses"] = cas.misses
        meta["n_leftover"] = len(left)
        meta["leftover"] = [{k: e.get(k) for k in ("i", "kind", "lineage", "lseq", "role", "site")} for e in left]
        meta["pick_stats"] = dict(cas.pick_stats)
        meta["replay_order"] = args.replay_order
        meta["gate_stats"] = dict(cas.gate_stats)
        meta["diverged_questions"] = sorted(cas.diverged)
        meta["leaks"] = cas.leaks
        (report_dir / "summary.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        verdict = "OK" if not cas.misses and not cas.leaks and not left else "DIVERGED"
        eprint(f"[cassette] replay {verdict}: served={sum(cas.counts.values()) or sum(cas.pick_stats.values())} "
               f"misses={len(cas.misses)} leftover={len(left)} leaks={len(cas.leaks)} picks={dict(cas.pick_stats)} "
               f"order={args.replay_order} gate={dict(cas.gate_stats)} "
               f"wall={meta['wall_s']}s -> {report_dir}/summary.json")
        if cas.misses or cas.leaks:
            final = EXIT_MISS
        elif left:
            final = EXIT_LEFTOVER
    sys.stdout.flush()
    sys.stderr.flush()
    return final if final else rc


__all__ = ["EXIT_LEFTOVER", "EXIT_MISS", "frozen_env", "main", "parse", "wanted_hashseed", "with_package_on_path",
           "with_skip_preflight"]
