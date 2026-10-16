"""`.env` is read the same way by every entry point (panini_dr.envfile)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from panini_dr.cli import app
from panini_dr.envfile import load_env_file

REPO = Path(__file__).resolve().parents[2]
runner = CliRunner()
KEY = "PANINI_TEST_DOTENV_VALUE"


def _clean(monkeypatch, *keys: str) -> None:
    """Remove ``keys`` now and restore their absence after the test."""
    for k in keys:
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)


def _env_file(tmp_path: Path, text: str, monkeypatch) -> Path:
    (tmp_path / ".env").write_text(text)
    monkeypatch.chdir(tmp_path)
    return tmp_path / ".env"


@pytest.mark.parametrize("args", [
    ["run", "--print-config"],
    ["prepare", "--out", "prepared"],
    ["score", "answers", "-b", "frames", "--run", "nope", "--cache", "c.json"],
    ["verify", "--bundle", "no_bundle"],
])
def test_every_command_reads_the_env_file(tmp_path, monkeypatch, args):
    _clean(monkeypatch, KEY)
    _env_file(tmp_path, f"{KEY}=from-dotenv\n", monkeypatch)
    runner.invoke(app, args)
    assert os.environ.get(KEY) == "from-dotenv"


def test_a_parent_directory_env_file_is_found(tmp_path, monkeypatch):
    _clean(monkeypatch, KEY)
    (tmp_path / ".env").write_text(f"{KEY}=parent\n")
    (tmp_path / "sub").mkdir()
    monkeypatch.chdir(tmp_path / "sub")
    assert load_env_file() == tmp_path / ".env"
    assert os.environ[KEY] == "parent"


def test_the_shell_wins_over_the_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv(KEY, "shell")
    _env_file(tmp_path, f"{KEY}=from-dotenv\n", monkeypatch)
    load_env_file()
    assert os.environ[KEY] == "shell"


def test_verify_replays_keep_behaviour_variables_blank(tmp_path, monkeypatch):
    """A .env cannot switch on a behaviour variable inside a verify replay."""
    from panini_dr.replay.verify import NEUTRALISED_ENV, child_env

    _clean(monkeypatch, "GSW_READ_TITLE_FALLBACK", "GSW_DENSE_GPU")
    _env_file(tmp_path, "GSW_READ_TITLE_FALLBACK=1\nGSW_DENSE_GPU=cuda:1\n", monkeypatch)
    runner.invoke(app, ["verify", "--bundle", "no_bundle"])  # the parent reads .env
    assert os.environ["GSW_READ_TITLE_FALLBACK"] == "1"
    env = child_env({"env": {"PYTHONHASHSEED": "0"}}, {}, tmp_path)
    assert all(env[k] == "" for k in NEUTRALISED_ENV)
    # the replayed `panini-dr run` reads the same .env: the blanks must survive it
    monkeypatch.setattr(os, "environ", dict(env))
    load_env_file()
    assert os.environ["GSW_READ_TITLE_FALLBACK"] == "" and os.environ["GSW_DENSE_GPU"] == ""


def _load_script(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_dataset_scripts_read_the_env_file(tmp_path, monkeypatch):
    frames = _load_script("datasets/frames/prepare.py", "_panini_prep_frames_envfile")
    tsv = tmp_path / "test.tsv"
    links = repr(["https://en.wikipedia.org/wiki/Page_0"])
    tsv.write_text("\tPrompt\tAnswer\twiki_links\treasoning_types\n"
                   f"0\tFact 0 is stated. What is fact 0?\tv0\t{links}\t\n")
    monkeypatch.setattr(frames.common, "require_split_ids", lambda ids, split: None)
    _clean(monkeypatch, "PANINI_DATA_DIR")
    _env_file(tmp_path, f"PANINI_DATA_DIR={tmp_path / 'from_env'}\n", monkeypatch)
    assert frames.main(["questions", "--tsv", str(tsv)]) == 0
    rows = json.loads((tmp_path / "from_env" / "frames" / "frames_split.json").read_text())
    assert [str(r["id"]) for r in rows] == ["0"]


def test_empty_values_are_ignored(tmp_path, monkeypatch):
    """`HF_HOME=` copied from .env.example must not set HF_HOME to "" (cache -> ./hub)."""
    _clean(monkeypatch, "HF_HOME", KEY, "PANINI_TEST_DOTENV_BLANK")
    _env_file(tmp_path, f"HF_HOME=\nPANINI_TEST_DOTENV_BLANK=   \n{KEY}=set\n", monkeypatch)
    load_env_file()
    assert "HF_HOME" not in os.environ and "PANINI_TEST_DOTENV_BLANK" not in os.environ
    assert os.environ[KEY] == "set"


def test_an_empty_shell_value_is_not_replaced(tmp_path, monkeypatch):
    monkeypatch.setenv(KEY, "")
    _env_file(tmp_path, f"{KEY}=from-dotenv\n", monkeypatch)
    load_env_file()
    assert os.environ[KEY] == ""


def test_env_example_sets_only_the_required_variables(tmp_path, monkeypatch):
    """Every optional line of .env.example is commented out; copying it sets two URLs."""
    from dotenv import dotenv_values

    values = dotenv_values(REPO / ".env.example")
    assert set(values) == {"PANINI_LLM_BASE_URL", "JINA_RERANK_URL"}
    assert all(v for v in values.values())
