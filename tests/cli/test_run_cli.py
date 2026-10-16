"""`panini-dr run` / `--print-config` through the real CLI, with faked services."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from panini_dr.cli import app
from panini_dr.config import load_paper_config
from tests.e2e.fakes import TOY_QUESTIONS

runner = CliRunner()


def _frames_style_split(tmp_path: Path) -> Path:
    qfile = tmp_path / "frames_toy.json"
    qfile.write_text(json.dumps([
        {"id": q.id, "original_prompt": q.question, "answer": q.answer} for q in TOY_QUESTIONS
    ]))
    subset = tmp_path / "toy_split.json"
    subset.write_text(json.dumps({
        "subset_id": "toy_split", "benchmark": "frames", "split": "full",
        "question_ids": [q.id for q in TOY_QUESTIONS], "dataset_path": str(qfile),
    }))
    return subset


def _cell_dir(out: Path) -> Path:
    dirs = [d for d in out.iterdir() if d.is_dir()]
    assert len(dirs) == 1, dirs
    return dirs[0]


def test_print_config_is_the_paper_config():
    res = runner.invoke(app, ["run", "--print-config"])
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    # the resolved config plus the removed keys the paper recorded
    assert {**out["config"], **out["removed_features"]} == load_paper_config()["config"]
    assert not set(out["config"]) & set(out["removed_features"])
    assert out["drift_from_paper"] == {}
    assert out["removed_env_vars_set"] == {}


def test_print_config_shows_the_pinned_adapter_defaults():
    res = runner.invoke(app, ["run", "--print-config"])
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    pinned = out["pinned_adapter_defaults"]
    assert pinned == load_paper_config()["pinned_adapter_defaults"]
    # the turn limits REPRODUCE.md quotes, next to the inert fallback limit
    assert (pinned["orchestrator_max_turns"], pinned["level_max_turns"],
            pinned["reader_max_turns"], pinned["MAX_PARALLEL_RESEARCHERS"]) == (12, 20, 8, 8)
    assert out["config"]["max_turns"] == 50
    about = out["pinned_adapter_defaults_about"]
    assert "orchestrator_max_turns" in about and "level_max_turns = turns per researcher (20)" in about
    assert "config.max_turns (50)" in about and "fallback" in about


def test_setting_a_removed_key_is_an_error():
    from panini_dr.config import removed_keys

    key = next(iter(removed_keys()))
    res = runner.invoke(app, ["run", "--print-config", "--set", f"{key}=1"])
    assert res.exit_code != 0
    assert "was removed" in res.output
    assert "configs/removed_features.json" in res.output


def test_print_config_reports_drift():
    res = runner.invoke(app, ["run", "--print-config", "--set", "gsw_edge_grounded_cap=1"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["drift_from_paper"] == {"gsw_edge_grounded_cap": [5, 1]}


def test_run_paper_config_end_to_end(fake_services, toy_corpus_path, tmp_path, monkeypatch):
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    out = tmp_path / "runs"
    split = _frames_style_split(tmp_path)
    res = runner.invoke(app, [
        "run", "--subset", str(split),
        "--articles-cache", str(toy_corpus_path), "--output-dir", str(out),
        "--concurrency", "2", "--strict-rerank",
    ])
    assert res.exit_code == 0, res.output
    cell_dir = _cell_dir(out)
    cell = json.loads((cell_dir / "cell_result.json").read_text())
    preds = {q["question_id"]: q["predicted_answer"] for q in cell["questions"]}
    assert preds == {q.id: q.answer for q in TOY_QUESTIONS}
    # the recorded config is the paper's plus the dataset key `corpus`
    recorded = dict(cell["config"])
    assert recorded.pop("corpus")["n_chunks"] > 0
    assert recorded == load_paper_config()["config"]
    meta = json.loads((cell_dir / "run_meta.json").read_text())
    assert meta["config_drift"] == {} and meta["benchmark"] == "frames"
    assert meta["n_questions"] == len(TOY_QUESTIONS) and "finished_at" in meta
    health = json.loads((cell_dir / "run_health.json").read_text())
    assert health["rerank"]["calls"] > 0 and health["rerank"]["failures"] == 0
    assert (cell_dir / "run.log").is_file()
    assert (cell_dir / "traces" / "q_toy2.json").is_file()
    # the closing hint is a complete grading command for this benchmark and split
    assert (f"panini-dr score answers -b frames --run {cell_dir} --subset {split} "
            "--cache judge/frames_cache.json --out judge/frames_verdicts.json --backend openai"
            ) in res.output


def test_rerank_failures_are_reported(fake_services, toy_corpus_path, tmp_path, monkeypatch):
    llm, reranker = fake_services
    reranker.fail = True
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    out = tmp_path / "runs"
    res = runner.invoke(app, [
        "run", "--subset", str(_frames_style_split(tmp_path)), "--limit", "1",
        "--articles-cache", str(toy_corpus_path), "--output-dir", str(out),
    ])
    assert res.exit_code == 0, res.output
    assert "reranker preflight failed" in res.output
    assert "fell back to fusion order" in res.output
    health = json.loads((_cell_dir(out) / "run_health.json").read_text())
    assert health["rerank"]["failures"] > 0


def test_strict_rerank_preflight_aborts(fake_services, toy_corpus_path, tmp_path, monkeypatch):
    llm, reranker = fake_services
    reranker.fail = True
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    res = runner.invoke(app, [
        "run", "--subset", str(_frames_style_split(tmp_path)), "--strict-rerank",
        "--articles-cache", str(toy_corpus_path), "--output-dir", str(tmp_path / "runs"),
    ])
    assert res.exit_code == 2
    assert "preflight failed" in res.output
    assert llm.calls == []


@pytest.mark.parametrize(
    "args, env, message",
    [
        (["--set", "gsw_edge_grounded_cap=1"], {}, "gsw_edge_grounded_cap: paper=5 resolved=1"),
        ([], {"GSW_PLANNER_TARGET_HUB_MODE": "hybrid"}, "GSW_PLANNER_TARGET_HUB_MODE"),
    ],
)
def test_drift_is_refused(fake_services, toy_corpus_path, tmp_path, monkeypatch, args, env, message):
    llm, _ = fake_services
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    res = runner.invoke(app, [
        "run", "--subset", str(_frames_style_split(tmp_path)),
        "--articles-cache", str(toy_corpus_path), "--output-dir", str(tmp_path / "runs"), *args,
    ])
    assert res.exit_code == 2
    assert message in res.output and "--allow-config-drift" in res.output
    assert llm.calls == []


def test_a_removed_env_var_is_refused_even_with_drift_allowed(
        fake_services, toy_corpus_path, tmp_path, monkeypatch):
    """GSW_READ_TITLE_FALLBACK switched code that no longer exists: setting it is
    an error, not drift."""
    llm, _ = fake_services
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    monkeypatch.setenv("GSW_READ_TITLE_FALLBACK", "1")
    res = runner.invoke(app, [
        "run", "--subset", str(_frames_style_split(tmp_path)), "--allow-config-drift",
        "--articles-cache", str(toy_corpus_path), "--output-dir", str(tmp_path / "runs"),
    ])
    assert res.exit_code == 2
    assert "GSW_READ_TITLE_FALLBACK" in res.output
    assert "removed_features.json" in res.output
    assert llm.calls == []


def test_drift_allowed_is_recorded(fake_services, toy_corpus_path, tmp_path, monkeypatch):
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    out = tmp_path / "runs"
    res = runner.invoke(app, [
        "run", "--subset", str(_frames_style_split(tmp_path)), "--limit", "1",
        "--articles-cache", str(toy_corpus_path), "--output-dir", str(out),
        "--set", "gsw_edge_grounded_cap=1", "--allow-config-drift",
    ])
    assert res.exit_code == 0, res.output
    meta = json.loads((_cell_dir(out) / "run_meta.json").read_text())
    assert meta["config_drift"] == {"gsw_edge_grounded_cap": [5, 1]}


def test_missing_inputs_are_explicit_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("PANINI_LLM_BASE_URL", raising=False)
    res = runner.invoke(app, ["run", "--subset", str(_frames_style_split(tmp_path))])
    assert res.exit_code != 0 and "PANINI_LLM_BASE_URL" in res.output
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    res = runner.invoke(app, ["run", "--subset", str(_frames_style_split(tmp_path))])
    assert res.exit_code == 2 and "--articles-cache is required" in res.output
    res = runner.invoke(app, ["run"])
    assert res.exit_code == 2 and "--subset" in res.output


def test_missing_subset_ids_are_an_error(toy_corpus_path, tmp_path, monkeypatch):
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    subset = _frames_style_split(tmp_path)
    data = json.loads(subset.read_text())
    data["question_ids"].append("not_in_file")
    subset.write_text(json.dumps(data))
    res = runner.invoke(app, ["run", "--subset", str(subset), "--articles-cache", str(toy_corpus_path)])
    assert res.exit_code != 0
    assert "not in the loaded question file" in str(res.exception) + res.output


def test_skip_rerank_preflight_sends_no_test_request(fake_services, toy_corpus_path, tmp_path, monkeypatch):
    import panini_dr.runner as runner_mod

    def _must_not_run(spec):  # pragma: no cover - only reached on regression
        raise AssertionError("rerank_preflight ran despite --skip-rerank-preflight")

    monkeypatch.setattr(runner_mod, "rerank_preflight", _must_not_run)
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    out = tmp_path / "runs"
    res = runner.invoke(app, [
        "run", "--subset", str(_frames_style_split(tmp_path)), "--limit", "1",
        "--skip-rerank-preflight",
        "--articles-cache", str(toy_corpus_path), "--output-dir", str(out),
    ])
    assert res.exit_code == 0, res.output
    meta = json.loads((_cell_dir(out) / "run_meta.json").read_text())
    assert meta["skip_rerank_preflight"] is True


@pytest.mark.parametrize("llm_key, expected", [(None, "dummy-for-vllm"), ("llm-key", "llm-key")])
def test_the_judge_key_is_never_sent_to_the_llm(fake_services, toy_corpus_path, tmp_path, monkeypatch,
                                                 llm_key, expected):
    """OPENAI_API_KEY (the judge's key) must not become the LLM server's key."""
    from panini_dr.models import llm_client

    keys: list[str] = []

    class _RecordingOpenAI:
        def __init__(self, *, base_url=None, api_key=None, **kwargs):
            keys.append(api_key)

    monkeypatch.setattr(llm_client, "OpenAI", _RecordingOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-judge-secret")
    if llm_key is None:
        monkeypatch.delenv("PANINI_LLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("PANINI_LLM_API_KEY", llm_key)
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    res = runner.invoke(app, [
        "run", "--subset", str(_frames_style_split(tmp_path)), "--limit", "1",
        "--articles-cache", str(toy_corpus_path), "--output-dir", str(tmp_path / "runs"),
    ])
    assert res.exit_code == 0, res.output
    assert keys and set(keys) == {expected}


def test_llm_client_uses_openai_key_only_for_openai(monkeypatch):
    from panini_dr.models import llm_client

    seen: dict[str, str] = {}

    class _RecordingOpenAI:
        def __init__(self, *, base_url=None, api_key=None, **kwargs):
            seen[base_url or "openai"] = api_key

    monkeypatch.setattr(llm_client, "OpenAI", _RecordingOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-judge-secret")
    llm_client.LLMClient(model="openai/gpt-oss-120b", base_url="http://llm.invalid/v1")
    llm_client.LLMClient(model="gpt-4o-mini")
    assert seen == {"http://llm.invalid/v1": "dummy-for-vllm", "openai": "sk-judge-secret"}


# ---------------------------------------------------------------------------
# LLM preflight and the zero-turn guard
# ---------------------------------------------------------------------------


def _run_args(tmp_path: Path, toy_corpus_path: Path, *extra: str) -> list[str]:
    return ["run", "--subset", str(_frames_style_split(tmp_path)), "--limit", "1",
            "--articles-cache", str(toy_corpus_path), "--output-dir", str(tmp_path / "runs"), *extra]


def test_llm_preflight_runs_before_any_question(fake_services, toy_corpus_path, tmp_path, monkeypatch):
    llm, _ = fake_services
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    res = runner.invoke(app, _run_args(tmp_path, toy_corpus_path))
    assert res.exit_code == 0, res.output
    assert llm.preflight_calls == ["http://llm.invalid/v1/models"]


@pytest.mark.parametrize("problem, message", [
    ("down", "cannot reach the LLM server"),
    ("model", "does not serve 'openai/gpt-oss-120b'"),
])
def test_llm_preflight_failure_stops_the_run(fake_services, toy_corpus_path, tmp_path, monkeypatch,
                                             problem, message):
    llm, reranker = fake_services
    if problem == "down":
        llm.down = True
    else:
        llm.served_models = ("some/other-model",)
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    res = runner.invoke(app, _run_args(tmp_path, toy_corpus_path))
    assert res.exit_code == 2, res.output
    assert "LLM preflight failed" in res.output and message in res.output
    assert "--skip-preflight" in res.output
    assert llm.calls == [] and reranker.calls == 0 and not (tmp_path / "runs").exists()


def test_llm_preflight_reports_a_rejected_key(monkeypatch):
    import httpx

    from panini_dr.runner import llm_preflight

    seen = {}

    def get(url, headers=None, timeout=None):
        seen.update(url=url, headers=headers, timeout=timeout)
        return httpx.Response(401, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    err = llm_preflight("http://llm.invalid/v1/", "k", "m")
    assert "refused the API key" in err and "PANINI_LLM_API_KEY" in err
    assert seen == {"url": "http://llm.invalid/v1/models", "headers": {"Authorization": "Bearer k"},
                    "timeout": 20.0}


@pytest.mark.parametrize("flag", ["--skip-preflight", "--skip-rerank-preflight"])
def test_skip_preflight_skips_both_test_requests(fake_services, toy_corpus_path, tmp_path, monkeypatch, flag):
    import panini_dr.runner as runner_mod

    llm, _ = fake_services
    llm.down = True  # would fail the LLM preflight

    def _must_not_run(*a, **k):  # pragma: no cover - only reached on regression
        raise AssertionError("a preflight ran despite " + flag)

    monkeypatch.setattr(runner_mod, "rerank_preflight", _must_not_run)
    monkeypatch.setattr(runner_mod, "llm_preflight", _must_not_run)
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    res = runner.invoke(app, _run_args(tmp_path, toy_corpus_path, flag))
    assert res.exit_code == 0, res.output
    meta = json.loads((_cell_dir(tmp_path / "runs") / "run_meta.json").read_text())
    assert meta["skip_preflight"] is True and meta["skip_rerank_preflight"] is True


def test_a_run_where_every_question_has_zero_turns_fails(fake_services, toy_corpus_path, tmp_path, monkeypatch):
    from panini_dr.models.llm_client import LLMClient

    def chat(self, *a, **k):
        raise ConnectionError("LLM went away after the preflight")

    monkeypatch.setattr(LLMClient, "chat", chat)
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    res = runner.invoke(app, _run_args(tmp_path, toy_corpus_path))
    assert res.exit_code == 1, res.output
    assert "ended with zero turns" in res.output and "went away" in res.output
    cell = json.loads((_cell_dir(tmp_path / "runs") / "cell_result.json").read_text())
    assert cell["questions"] and all(q["trajectory"]["turns"] == 0 for q in cell["questions"])


def test_grade_command_names_the_grader_of_each_benchmark(tmp_path):
    from panini_dr.runner import DatasetSpec, grade_command

    bcp = DatasetSpec(benchmark="browsecomp_plus", subset_id="cf_bcp150", questions=[],
                      corpus_path=tmp_path / "c.json",
                      meta={"subset_file": "configs/splits/cf_bcp150.json"})
    assert grade_command(bcp, "runs/bcp/x y") == (
        "panini-dr score answers -b bcp --run 'runs/bcp/x y' --subset configs/splits/cf_bcp150.json "
        "--cache judge/bcp_cache.json --out judge/bcp_verdicts.json --backend openai")
    mine = DatasetSpec(benchmark="byod", subset_id="my data", questions=[],
                       corpus_path=tmp_path / "c.json", meta={"dataset_dir": "data/mydata"},
                       questions_file=Path("data/mydata/questions.jsonl"))
    assert grade_command(mine, "runs/m") == (
        "panini-dr score answers -b byod --run runs/m --questions data/mydata/questions.jsonl "
        "--cache judge/my_data_cache.json --out judge/my_data_verdicts.json --backend openai")
