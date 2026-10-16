"""Question-file loaders: default locations and the plain `question` key."""
from __future__ import annotations

import json

import pytest

from panini_dr.eval.frames_dataset import FramesQuestion, default_questions_path, load_frames


def test_frames_prefers_original_prompt_then_test_question_then_question():
    assert FramesQuestion.from_raw(
        {"id": "1", "original_prompt": "full", "test_question": "tail", "question": "q",
         "answer": "a"}).question == "full"
    assert FramesQuestion.from_raw(
        {"id": "1", "test_question": "tail", "question": "q", "answer": "a"}).question == "tail"
    # A plain `question` key used to load as an empty question.
    assert FramesQuestion.from_raw({"id": "1", "question": "q", "answer": "a"}).question == "q"


def test_frames_default_paths_follow_panini_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PANINI_DATA_DIR", str(tmp_path))
    assert default_questions_path("full") == tmp_path / "frames" / "frames_split.json"
    assert default_questions_path("dev") == tmp_path / "frames" / "frames_dev.json"
    with pytest.raises(FileNotFoundError, match="datasets/frames/prepare.py"):
        load_frames(split="full")
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "frames_split.json").write_text(
        json.dumps([{"id": "0", "question": "Which river?", "answer": "Merrow"}]))
    assert [q.question for q in load_frames(split="full")] == ["Which river?"]


def test_browsecomp_default_path_follows_panini_data_dir(tmp_path, monkeypatch):
    from panini_dr.eval.browsecomp_dataset import default_questions_path as bcp_path

    monkeypatch.delenv("BROWSECOMP_QUESTIONS", raising=False)
    monkeypatch.setenv("PANINI_DATA_DIR", str(tmp_path))
    assert bcp_path() == tmp_path / "browsecomp_plus" / "browsecomp_plus_830.json"
