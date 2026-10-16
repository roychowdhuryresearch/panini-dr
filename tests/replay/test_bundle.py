"""Bundle scrubbing, scanning and integrity (no recording needed)."""

from __future__ import annotations

import json

from panini_dr.replay.bundle import (
    check_integrity,
    derived_deny_literals,
    scan_bundle,
    scrub_args,
    scrub_call,
    scrub_meta,
    sha256_file,
)
from panini_dr.replay.store import canon


def test_scrub_call_replaces_service_locations_only():
    e = {"i": 3, "kind": "llm", "key": "k" * 64, "lineage": "q:1", "lseq": 0,
         "client": {"provider": "vllm", "base_url": "http://gpu-box.lab.example:8001/v1", "class": "x.LLMClient"},
         "t0": 1.0, "t1": 2.0, "req": {"kind": "llm", "messages": [{"$blob": "ab"}]}, "resp": {"fields": {"text": "hi"}}}
    s = scrub_call(e)
    assert s["client"]["base_url"] == "<LLM_BASE_URL>" and s["client"]["provider"] == "vllm"
    assert {k: v for k, v in s.items() if k != "client"} == {k: v for k, v in e.items() if k != "client"}
    assert e["client"]["base_url"].startswith("http://gpu-box")           # the input is not modified
    h = scrub_call({"kind": "http_post", "url_host": "http://10.0.0.7:8011", "req": {}, "resp": {}})
    assert h["url_host"] == "<SERVICE_HOST>"
    assert list(json.loads(canon(s))) == list(e)                            # key order kept


def test_scrub_meta_keeps_a_whitelist():
    meta = {
        "tool": "panini_dr.replay", "mode": "record", "pkg": "panini_dr", "source_commit": "abc123",
        "source_repo": "/srv/lab/repo", "cwd": "/srv/lab/repo", "argv": ["/srv/lab/venv/bin/python", "-m", "x"],
        "env": {"PYTHONHASHSEED": "0", "PYTHONPATH": "/srv/lab/repo/src", "JINA_RERANK_URL": "http://h:1/v1/rerank"},
        "versions": {"python": "3.12.11 (main, ...)", "numpy": "2.2.6", "executable": "/srv/lab/venv/bin/python",
                     "platform": "Linux-lab-box"},
        "services": {"llm_models": {"data": [{"id": "my-llm", "root": "/srv/lab/models/llm", "max_model_len": 65536}]},
                     "llm_version": {"version": "0.24.0"},
                     "jina_models": {"data": [{"id": "jina-rerank", "root": "jinaai/jina-reranker-v3.5"}]}},
        "runner": ["run", "--subset", "/srv/lab/s.json", "--base-url", "http://h:1/v1", "--api-key", "sk-x"],
        "counts_by_kind": {"llm": 3}, "cell_result": "/srv/lab/out/cell_result.json",
    }
    out = scrub_meta(meta, reader_tokenizer="heuristic")
    text = json.dumps(out)
    assert "/srv/lab" not in text and "http://h" not in text and "sk-x" not in text
    assert out["env"] == {"PYTHONHASHSEED": "0"}
    assert out["versions"] == {"python": "3.12.11", "numpy": "2.2.6"}
    assert out["services"]["llm"] == {"id": "my-llm", "max_model_len": 65536, "server_version": "0.24.0"}
    assert out["services"]["reranker"]["model"] == "jinaai/jina-reranker-v3.5"
    assert out["runner"] == ["run", "--subset", "<PATH>/s.json", "--base-url", "<URL>", "--api-key", "<redacted>"]
    assert out["reader_tokenizer"] == "heuristic" and out["source_commit"] == "abc123"
    assert scrub_args(["/x/y/run.py", "--k", "v"]) == ["<PATH>/run.py", "--k", "v"]


def test_scan_finds_paths_hosts_and_derived_identifiers(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps({
        "text": "Naturalized citizens; see https://en.wikipedia.org/home/page and the tokens 8001",
        "leak": "cache at /home/someone/data",
    }) + "\n")
    (tmp_path / "b.json").write_text('{"host": "gpu-box.lab.example", "svc": "10.1.2.3:8011"}')
    lits = derived_deny_literals(inputs=[tmp_path / "in"], metas=[{"runner": ["--base-url", "http://gpu-box.lab.example:8001/v1"]}],
                                 extra=["turali"], machine=False)
    assert "gpu-box.lab.example" in lits
    hits = scan_bundle(tmp_path, literals=lits)
    found = {(h["file"], h["pattern"]) for h in hits}
    assert ("b.json", "gpu-box.lab.example") in found
    assert any(f == "a.jsonl" and "home" in p for f, p in found)          # the absolute path
    assert any(f == "b.json" and r"\d{2,5}" in p for f, p in found)       # IP:port
    assert not any(p == "turali" for _, p in found)                       # not inside "Naturalized"
    assert not any("wikipedia" in h["sample"] and "home" in h["pattern"] for h in hits)  # URL paths are fine


def test_integrity_check_names_modified_and_missing_files(tmp_path):
    (tmp_path / "x.txt").write_text("hello")
    (tmp_path / "y.txt").write_text("world")
    manifest = {"files": {n: {"sha256": sha256_file(tmp_path / n)} for n in ("x.txt", "y.txt")}}
    assert check_integrity(tmp_path, manifest) == []
    (tmp_path / "x.txt").write_text("hellp")
    (tmp_path / "y.txt").unlink()
    assert check_integrity(tmp_path, manifest) == ["x.txt: modified (sha256 differs from the manifest)",
                                                   "y.txt: missing"]
