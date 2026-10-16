"""Misusing the command line gives `error: ...` and exit status 2, never a traceback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from panini_dr.cli import app

runner = CliRunner()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    for var in ("BCP_REPO", "MONACO_JUDGE_DIR", "PANINI_JUDGE_MODEL", "BROWSECOMP_QUESTIONS", "MONACO_QUESTIONS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PANINI_DATA_DIR", str(tmp_path / "no_data"))
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run1" / "traces").mkdir(parents=True)
    (tmp_path / "run1" / "traces" / "q_a.json").write_text(json.dumps({"question_id": "a", "final_answer": "x"}))
    (tmp_path / "fq.json").write_text(json.dumps([{"id": "a", "original_prompt": "Q?", "answer": "x"}]))
    (tmp_path / "q.jsonl").write_text(json.dumps({"id": "a", "question": "Q?", "answer": "x"}) + "\n")
    (tmp_path / "corpus.json").write_text(json.dumps({"T": "hello world"}))
    (tmp_path / "bad.json").write_text("not json")
    subset = {"subset_id": "s", "benchmark": "frames", "split": "full", "question_ids": ["a"],
              "dataset_path": str(tmp_path / "fq.json")}
    (tmp_path / "s.json").write_text(json.dumps(subset))
    (tmp_path / "s_missing.json").write_text(json.dumps({**subset, "question_ids": ["zz"]}))
    return tmp_path


SCORE = ["score", "answers", "--run", "run1", "--cache", "c.json", "--model", "m"]


@pytest.mark.parametrize("args, message", [
    ([*SCORE, "-b", "byod"], "--questions"),
    ([*SCORE, "-b", "byod", "--questions", "nope.jsonl"], "cannot read the question file"),
    ([*SCORE, "-b", "frames", "--subset", "s.json", "--questions", "fq.json", "--offline"], "BCP_REPO"),
    ([*SCORE, "-b", "frames", "--subset", "s.json", "--questions", "fq.json", "--offline",
      "--bcp-repo", "nope"], "scripts_evaluation/evaluate_run.py"),
    ([*SCORE, "-b", "xyz"], "unknown benchmark"),
    ([*SCORE, "-b", "frames", "--subset", "bad.json", "--questions", "fq.json"], "cannot read the subset file"),
    ([*SCORE, "-b", "frames", "--subset", "s.json"], "cannot read the question file"),
    (["score", "answers", "--run", "nope", "--cache", "c.json", "--model", "m", "-b", "frames",
      "--subset", "s.json", "--questions", "fq.json"], "no traces"),
    ([*SCORE, "-b", "frames", "--subset", "s.json", "--questions", "fq.json", "--backend", "foo"],
     "unknown judge backend"),
    (["score", "answers", "--run", "run1", "--cache", "bad.json", "--model", "m", "-b", "frames",
      "--subset", "s.json", "--questions", "fq.json", "--offline"], "cannot read the judge cache"),
    (["score", "obs", "-b", "frames", "--run", "run1", "--verdicts", "nope.json", "--subset", "s.json",
      "--corpus", "corpus.json", "--questions", "fq.json"], "cannot read the verdicts"),
    (["score", "import-legacy-cache", "-b", "frames", "--legacy-cache", "nope.json", "--legacy-name", "x",
      "--run", "run1", "--cache", "c.json", "--subset", "s.json", "--questions", "fq.json", "--model", "m"],
     "cannot read the legacy cache"),
    (["run", "--subset", "s_missing.json", "--articles-cache", "corpus.json"], "not in the loaded question file"),
    (["run", "--subset", "bad.json", "--articles-cache", "corpus.json"], "cannot read the subset file"),
    (["run", "--subset", "s.json", "--articles-cache", "bad.json", "--skip-preflight", "--output-dir", "r",
      "--set", "retriever_type=bm25", "--allow-config-drift"], "cannot load the corpus"),
    (["run", "--dataset", "nope"], "not a prepared dataset"),
    (["prepare", "--config", "nope.yaml", "--out", "o"], "cannot read nope.yaml"),
    (["prepare", "--config", "bad.json", "--out", "o"], "expected a mapping"),
])
def test_misuse_is_a_clean_error(inputs, args, message):
    res = runner.invoke(app, args)
    assert res.exit_code == 2, res.output
    assert isinstance(res.exception, SystemExit), repr(res.exception)  # no traceback
    assert message in res.output


def test_tracebacks_never_show_local_variables():
    """Typer's pretty tracebacks print locals by default, and locals can hold API keys."""
    assert app.pretty_exceptions_show_locals is False
