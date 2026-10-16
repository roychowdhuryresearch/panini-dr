import json
import sys
import types

from typer.testing import CliRunner

from panini_dr.eval import score_cli
from tests.eval.fakes import ScriptedJudge, fake_parse
from tests.eval.test_official_sources import _fake_monaco_checkout


def _fake_bcp_checkout(root):
    se = root / "scripts_evaluation"
    se.mkdir(parents=True)
    (se / "__init__.py").write_text("")
    (se / "evaluate_run.py").write_text(
        "import re\n"
        "GRADER_TEMPLATE = 'Q: {question}\\nR: {response}\\nGOLD: {correct_answer}\\n'\n"
        "def parse_judge_response(t):\n"
        "    m = re.search(r'correct:\\s*(yes|no)', t)\n"
        "    return {'correct': (m.group(1) == 'yes') if m else None, 'extracted_final_answer': None}\n")


def _write_run(run, preds):
    (run / "traces").mkdir(parents=True)
    for qid, ans in preds.items():
        (run / "traces" / f"q_{qid}.json").write_text(json.dumps({"question_id": qid, "final_answer": ans}))


def _isolate_bcp_module(monkeypatch):
    for m in [m for m in sys.modules if m.startswith("scripts_evaluation")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=None, SamplingParams=None))


def _judge(prompt, i):
    """Short prompts: 'yes' when the response equals the gold. Set prompts: credit one item."""
    if prompt.startswith("Q: "):
        r = prompt.split("\nR: ")[1].split("\n")[0]
        g = prompt.split("\nGOLD: ")[1].split("\n")[0]
        return f"correct: {'yes' if r == g else 'no'}"
    if prompt.startswith("MULTI"):
        return "extracted_final_answers: x\nfinal answer length: 2\noverlapping answers: Red"
    return "final precision: 1.0"


def test_score_answers_bcp_two_legs(tmp_path, monkeypatch):
    _fake_bcp_checkout(tmp_path / "bcp")
    _isolate_bcp_module(monkeypatch)
    q = [{"query_id": "1", "query": "Made-up question one?", "answer": "Alpha"},
         {"query_id": "2", "query": "Made-up question two?", "answer": "Beta"},
         {"query_id": "3", "query": "Made-up question three?", "answer": "Gamma"},
         {"query_id": "4", "query": "Made-up question four?", "answer": "Delta"}]
    (tmp_path / "q.json").write_text(json.dumps(q))
    (tmp_path / "split.json").write_text(json.dumps({"subset_id": "toy", "question_ids": ["1", "2", "3", "4"]}))
    _write_run(tmp_path / "leg1", {"1": "Alpha", "2": "Wrong"})
    _write_run(tmp_path / "leg2", {"3": ""})                          # blank; q4 has no trace
    judge = ScriptedJudge(_judge)
    monkeypatch.setattr(score_cli, "_BACKEND_FACTORY", lambda kind, model: judge)
    res = CliRunner().invoke(score_cli.score_app, [
        "answers", "--model", "my-judge-model", "--benchmark", "bcp", "--run", str(tmp_path / "leg1"), "--run", str(tmp_path / "leg2"),
        "--questions", str(tmp_path / "q.json"), "--subset", str(tmp_path / "split.json"),
        "--cache", str(tmp_path / "cache.json"), "--bcp-repo", str(tmp_path / "bcp"),
        "--out", str(tmp_path / "verdicts.json"), "--workers", "1"])
    assert res.exit_code == 0, res.output
    assert "1/4 = 25.0%" in res.output
    v = json.loads((tmp_path / "verdicts.json").read_text())
    assert v["summary"]["short"]["n_missing"] == 1
    assert "R: (no answer)" in "".join(judge.calls)                   # BCP grades a blank answer
    n_calls = len(judge.calls)
    res = CliRunner().invoke(score_cli.score_app, [
        "answers", "--model", "my-judge-model", "--benchmark", "bcp", "--run", str(tmp_path / "leg1"), "--run", str(tmp_path / "leg2"),
        "--questions", str(tmp_path / "q.json"), "--subset", str(tmp_path / "split.json"),
        "--cache", str(tmp_path / "cache.json"), "--bcp-repo", str(tmp_path / "bcp")])
    assert res.exit_code == 0 and len(judge.calls) == n_calls          # all cached


def test_score_answers_byod_routes_by_answer_type(tmp_path, monkeypatch):
    _fake_bcp_checkout(tmp_path / "bcp")
    _isolate_bcp_module(monkeypatch)
    _fake_monaco_checkout(tmp_path / "monaco")
    rows = [{"id": "a", "question": "Capital of the toy land?", "answer": "Quux"},
            {"id": "b", "question": "Colours of the toy flag?", "answer": ["Red", "Blue"], "answer_type": "set"},
            {"id": "c", "question": "Toy blank?", "answer": "Zed"}]
    (tmp_path / "questions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    _write_run(tmp_path / "run", {"a": "Quux", "b": "Red, Green", "c": " "})
    judge = ScriptedJudge(_judge)
    monkeypatch.setattr(score_cli, "_BACKEND_FACTORY", lambda kind, model: judge)
    res = CliRunner().invoke(score_cli.score_app, [
        "answers", "--model", "my-judge-model", "--benchmark", "byod", "--run", str(tmp_path / "run"),
        "--questions", str(tmp_path / "questions.jsonl"), "--cache", str(tmp_path / "c.json"),
        "--bcp-repo", str(tmp_path / "bcp"), "--monaco-judge-dir", str(tmp_path / "monaco"),
        "--allow-judge-mismatch", "--workers", "1"])
    assert res.exit_code == 0, res.output
    assert "short-answer accuracy: 1/2" in res.output                 # c is blank: 0, no call
    assert "set-answer mean F1" in res.output
    assert not any("Toy blank?" in c for c in judge.calls)


def test_score_answers_requires_a_subset_for_benchmarks(tmp_path):
    (tmp_path / "q.json").write_text("[]")
    _write_run(tmp_path / "run", {"1": "x"})
    res = CliRunner().invoke(score_cli.score_app, [
        "answers", "--model", "my-judge-model", "--benchmark", "frames", "--run", str(tmp_path / "run"),
        "--questions", str(tmp_path / "q.json"), "--cache", str(tmp_path / "c.json")])
    assert res.exit_code != 0


def _bcp_toy(tmp_path):
    q = [{"query_id": "1", "query": "Made-up question one?", "answer": "Alpha"},
         {"query_id": "2", "query": "Made-up question two?", "answer": "Beta"}]
    (tmp_path / "q.json").write_text(json.dumps(q))
    (tmp_path / "split.json").write_text(json.dumps({"subset_id": "toy", "question_ids": ["1", "2"]}))
    _write_run(tmp_path / "run", {"1": "Alpha", "2": "Wrong"})
    return ["--model", "my-judge-model", "--benchmark", "bcp", "--run", str(tmp_path / "run"), "--questions", str(tmp_path / "q.json"),
            "--subset", str(tmp_path / "split.json"), "--bcp-repo", str(tmp_path / "bcp")]


def test_score_answers_offline_never_builds_a_backend(tmp_path, monkeypatch):
    _fake_bcp_checkout(tmp_path / "bcp")
    _isolate_bcp_module(monkeypatch)
    common = _bcp_toy(tmp_path)

    def boom(kind, model):
        raise AssertionError("offline mode built a judge backend")
    monkeypatch.setattr(score_cli, "_BACKEND_FACTORY", boom)
    res = CliRunner().invoke(score_cli.score_app, ["answers", *common, "--cache", str(tmp_path / "c.json"), "--offline"])
    assert res.exit_code == 2 and "not in the judge cache" in res.output
    assert not (tmp_path / "c.json").exists()


def test_import_legacy_cache_then_offline_score(tmp_path, monkeypatch):
    _fake_bcp_checkout(tmp_path / "bcp")
    _isolate_bcp_module(monkeypatch)
    common = _bcp_toy(tmp_path)
    legacy = {"old:1": {"correct": True, "votes": "3/3", "extracted": "Alpha", "pred": "Alpha"},
              "old:2": {"correct": False, "votes": "0/3", "extracted": "Wrong", "pred": "Wrong"},
              "other:1": {"correct": False, "votes": "0/3", "extracted": None, "pred": "Alpha"}}
    (tmp_path / "legacy.json").write_text(json.dumps(legacy))
    before = (tmp_path / "legacy.json").read_bytes()
    monkeypatch.setattr(score_cli, "_BACKEND_FACTORY", lambda kind, model: (_ for _ in ()).throw(AssertionError()))
    res = CliRunner().invoke(score_cli.score_app, [
        "import-legacy-cache", *common, "--legacy-cache", str(tmp_path / "legacy.json"), "--legacy-name", "old",
        "--cache", str(tmp_path / "new.json"), "--report", str(tmp_path / "report.json")])
    assert res.exit_code == 0, res.output
    assert "imported 2 of 2" in res.output and "conflicts 0" in res.output
    assert (tmp_path / "legacy.json").read_bytes() == before            # the old cache is read only
    assert json.loads((tmp_path / "report.json").read_text())["imported"] == 2
    res = CliRunner().invoke(score_cli.score_app, ["answers", *common, "--cache", str(tmp_path / "new.json"),
                                                   "--offline", "--out", str(tmp_path / "v.json")])
    assert res.exit_code == 0, res.output
    assert "1/2 = 50.0%" in res.output
    assert json.loads((tmp_path / "v.json").read_text())["summary"]["backend"] == "offline"


def test_import_legacy_cache_reports_a_stale_verdict(tmp_path, monkeypatch):
    _fake_bcp_checkout(tmp_path / "bcp")
    _isolate_bcp_module(monkeypatch)
    common = _bcp_toy(tmp_path)
    legacy = {"old:1": {"correct": True, "votes": "3/3", "pred": "Alpha"},
              "old:2": {"correct": True, "votes": "3/3", "pred": "A different answer"}}
    (tmp_path / "legacy.json").write_text(json.dumps(legacy))
    res = CliRunner().invoke(score_cli.score_app, [
        "import-legacy-cache", *common, "--legacy-cache", str(tmp_path / "legacy.json"), "--legacy-name", "old",
        "--cache", str(tmp_path / "new.json")])
    assert res.exit_code == 1 and "conflict 2" in res.output and "imported 1 of 2" in res.output


def test_score_answers_needs_a_judge_model(tmp_path, monkeypatch):
    _fake_bcp_checkout(tmp_path / "bcp")
    _isolate_bcp_module(monkeypatch)
    common = [a for a in _bcp_toy(tmp_path) if a not in ("--model", "my-judge-model")]
    monkeypatch.delenv("PANINI_JUDGE_MODEL", raising=False)
    monkeypatch.setattr(score_cli, "_BACKEND_FACTORY", lambda kind, model: (_ for _ in ()).throw(AssertionError()))
    res = CliRunner().invoke(score_cli.score_app, ["answers", *common, "--cache", str(tmp_path / "c.json")])
    assert res.exit_code == 2 and "no judge model" in res.output
    assert not (tmp_path / "c.json").exists()
    # the environment variable names it instead; the name reaches the cache keys
    judge = ScriptedJudge(_judge)
    seen = []
    monkeypatch.setattr(score_cli, "_BACKEND_FACTORY", lambda kind, model: seen.append(model) or judge)
    monkeypatch.setenv("PANINI_JUDGE_MODEL", "my-judge-model")
    res = CliRunner().invoke(score_cli.score_app, ["answers", *common, "--cache", str(tmp_path / "c.json"),
                                                   "--workers", "1"])
    assert res.exit_code == 0, res.output
    assert seen == ["my-judge-model"]
    assert all("|my-judge-model|" in k for k in json.loads((tmp_path / "c.json").read_text()))
