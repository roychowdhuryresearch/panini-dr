"""Record and replay leave out `panini-dr run`'s LLM / reranker test requests."""

from __future__ import annotations

from panini_dr.replay.launcher import with_skip_preflight
from panini_dr.replay.verify import child_argv


def test_the_launcher_adds_skip_preflight_to_a_panini_dr_run():
    assert with_skip_preflight(["run", "--subset", "s.json"]) == ["run", "--subset", "s.json", "--skip-preflight"]
    # already there (or its alias): unchanged
    assert with_skip_preflight(["run", "--skip-preflight"]) == ["run", "--skip-preflight"]
    assert with_skip_preflight(["run", "--skip-rerank-preflight"]) == ["run", "--skip-rerank-preflight"]
    # another harness (--runner-script) or another command: unchanged
    assert with_skip_preflight(["--system", "x"], "runner.py") == ["--system", "x"]
    assert with_skip_preflight(["score", "answers"]) == ["score", "answers"]


def test_verify_replays_skip_the_preflight(tmp_path):
    case = {"cassette": "cassettes/c", "run_args": ["--subset", "{bundle}/subsets/s.json"]}
    argv = child_argv(tmp_path, case, tmp_path / "data", tmp_path / "out", tmp_path / "rep", 60.0)
    run = argv[argv.index("run"):]
    assert "--skip-preflight" in run and "--skip-rerank-preflight" not in run
