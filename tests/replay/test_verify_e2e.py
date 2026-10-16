"""End to end: record a toy cassette, bundle it, ``panini-dr verify`` it, then break it.

Everything runs on CPU without network or external data. The recording runs the paper
configuration (``panini-dr run``: planner, orchestrator, resolver, verifier, researcher,
reader, evidence judge, synthesis validator; hybrid BM25 + dense retrieval with the jina
reranker client) with the scripted LLM of tests/e2e, a fake reranker server and stub
encoder / faiss / torch modules (tests/replay/_stubs). ``panini-dr verify`` then replays
it in a fresh process with none of those fakes installed: every response must come
from the cassette.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from panini_dr.replay.bundle import CaseSource, build_bundle, sha256_file
from tests.e2e.fakes import TOY_QUESTIONS, write_toy_corpus

REPO = Path(__file__).resolve().parents[2]
STUBS = Path(__file__).resolve().parent / "_stubs"


def _env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    for k in ("GSW_READ_TITLE_FALLBACK", "GSW_PLANNER_TARGET_HUB_MODE", "GSW_STRICT_TARGET_HUB_GATE",
              "GSW_DENSE_GPU", "GSW_FAISS_THREADS", "PANINI_STRICT_RERANK", "PANINI_LLM_MODEL"):
        env.pop(k, None)
    env["PYTHONPATH"] = os.pathsep.join([str(STUBS), str(REPO / "src"), str(REPO)])
    env.update(extra)
    return env


@pytest.fixture(scope="module")
def toy_bundle(tmp_path_factory):
    work = tmp_path_factory.mktemp("replay_e2e")
    data = work / "data"
    (data / "frames").mkdir(parents=True)
    write_toy_corpus(data / "frames" / "corpus.json")
    (data / "frames" / "frames_split.json").write_text(json.dumps(
        [{"id": q.id, "original_prompt": q.question, "answer": q.answer} for q in TOY_QUESTIONS]))
    (work / "subset.json").write_text(json.dumps({
        "subset_id": "toy_verify", "benchmark": "frames", "split": "full",
        "question_ids": [q.id for q in TOY_QUESTIONS]}))
    rec = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "_record_toy.py"), str(work)],
        env=_env(PYTHONHASHSEED="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PANINI_DATA_DIR=str(data),
                 JINA_RERANK_URL="http://reranker.invalid/v1/rerank"),
        cwd=work, capture_output=True, text=True, timeout=600)
    assert rec.returncode == 0, rec.stdout[-3000:] + rec.stderr[-3000:]
    meta = json.loads((work / "cassette" / "meta.json").read_text())
    assert meta["n_leaks"] == 0 and meta["reader_tokenizer"] == "heuristic"
    assert set(meta["counts_by_kind"]) == {"llm", "embed", "http_post"}        # all three hooks recorded
    assert meta["answers"] == {q.id: q.answer for q in TOY_QUESTIONS}
    manifest = build_bundle(work / "bundle", [CaseSource(
        name="toy", cassette=work / "cassette", run_dir=work / "record_run", subset=work / "subset.json",
        articles_cache="frames/corpus.json")], data_dir=data, name="toy bundle",
        scan_machine_identifiers=False)
    return work, data, manifest


def _verify(bundle: Path, data: Path, *extra: str) -> subprocess.CompletedProcess:
    # PANINI_DATA_DIR / PYTHONHASHSEED deliberately wrong here: verify must set what the replay needs
    return subprocess.run(
        [sys.executable, "-m", "panini_dr.cli", "verify", "--bundle", str(bundle), "--data-dir", str(data), *extra],
        env=_env(PANINI_DATA_DIR="/nonexistent", PYTHONHASHSEED="random", JINA_RERANK_URL="http://wrong.invalid/x"),
        cwd=bundle.parent, capture_output=True, text=True, timeout=600)


def test_bundle_is_self_contained_and_scrubbed(toy_bundle):
    work, data, manifest = toy_bundle
    b = work / "bundle"
    assert manifest["cases"][0]["counts_by_kind"]["llm"] > 0
    assert set(manifest["data"]) == {"frames/corpus.json", "frames/frames_split.json",
                                     next(k for k in manifest["data"] if k.startswith("retrieval_cache/"))}
    for rel, info in manifest["files"].items():
        assert sha256_file(b / rel) == info["sha256"]
    for p in b.rglob("*"):
        if p.is_file():
            t = p.read_text(errors="replace")
            assert str(work) not in t and "llm.invalid" not in t and "reranker.invalid" not in t, p
    # reference outputs are stored normalised: no wall-clock fields
    ref = json.loads((b / "reference" / "toy" / "cell_result.json").read_text())
    assert "mean_wall_time_s" not in ref and all("wall_time_s" not in q["trajectory"] for q in ref["questions"])


def test_verify_passes_on_the_recording(toy_bundle):
    work, data, _ = toy_bundle
    res = _verify(work / "bundle", data, "--keep-output", str(work / "pass_out"))
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    for q in TOY_QUESTIONS:
        assert f"q{q.id}" in out and f"PASS  answer={q.answer!r}" in out
    assert "0 misses, 0 left over, 0 network leaks" in out
    assert "PASS: 2/2 questions reproduce the reference outputs exactly" in out
    summary = json.loads((work / "pass_out" / "toy_cassette" / "summary.json").read_text())
    assert summary["env"]["PYTHONHASHSEED"] == "0" and summary["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert summary["n_misses"] == 0 and summary["n_leftover"] == 0 and summary["leaks"] == []


def _corrupt_one_byte(bundle: Path, old: bytes, new: bytes) -> None:
    """Change one byte inside the first recorded planner response that contains ``old``."""
    assert len(old) == len(new) and sum(a != b for a, b in zip(old, new)) == 1
    p = bundle / "cassettes" / "toy" / "calls.jsonl"
    lines = p.read_bytes().split(b"\n")
    for n, line in enumerate(lines):
        e = json.loads(line) if line.strip() else {}
        if e.get("role") == "planner" and old in line[line.index(b'"resp"'):]:
            j = line.index(old, line.index(b'"resp"'))
            lines[n] = line[:j] + new + line[j + len(old):]
            break
    else:
        raise AssertionError("no planner response to corrupt")
    p.write_bytes(b"\n".join(lines))


def test_verify_fails_on_a_one_byte_change_with_a_useful_diff(toy_bundle):
    work, data, manifest = toy_bundle
    bad = work / "bundle_corrupt"
    shutil.copytree(work / "bundle", bad)
    _corrupt_one_byte(bad, b"Zorvan Klett", b"Zorvan Klitt")
    res = _verify(bad, data)
    out = res.stdout + res.stderr
    assert res.returncode == 1, out
    assert "cassettes/toy/calls.jsonl: modified" in out                     # the integrity check
    assert "qtoy1   FAIL" in out and "PASS  answer='Lindqvar'" in out        # only the touched question
    assert "first miss: kind=llm question=q:toy1" in out
    assert "Klitt" in out.split("request diff")[1]                           # the diff shows the change
    assert out.rstrip().splitlines()[-2].startswith("FAIL:")


def test_replay_divergence_fails_even_when_the_manifest_is_updated(toy_bundle):
    """A tampered bundle whose manifest matches still FAILs: the replay itself diverges."""
    work, data, _ = toy_bundle
    bad = work / "bundle_consistent"
    shutil.copytree(work / "bundle", bad)
    _corrupt_one_byte(bad, b"Zorvan Klett", b"Zorvan Klitt")
    m = json.loads((bad / "manifest.json").read_text())
    rel = "cassettes/toy/calls.jsonl"
    m["files"][rel]["sha256"] = sha256_file(bad / rel)
    (bad / "manifest.json").write_text(json.dumps(m))
    res = _verify(bad, data)
    out = res.stdout + res.stderr
    assert res.returncode == 1, out
    assert f"all {len(m['files'])} bundle files match the manifest" in out
    assert "qtoy1   FAIL" in out and "misses" in out


def test_verify_refuses_wrong_data_before_replaying(toy_bundle, tmp_path):
    work, data, _ = toy_bundle
    other = tmp_path / "data"
    shutil.copytree(data, other)
    corpus = json.loads((other / "frames" / "corpus.json").read_text())
    corpus["Veltravia"] += " A new sentence."
    (other / "frames" / "corpus.json").write_text(json.dumps(corpus))
    res = _verify(work / "bundle", other)
    assert res.returncode == 2, res.stdout
    assert "is not the recorded corpus" in res.stdout and "replaying" not in res.stdout
