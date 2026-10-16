from __future__ import annotations

import json

from panini_dr.eval.monaco_dataset import load_monaco_eval
from panini_dr.eval.subset import PilotSubset, load_benchmark_questions


def test_monaco_questions_override_is_resolved_at_call_time(tmp_path, monkeypatch):
    path = tmp_path / "monaco.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "full-only",
                    "question": "q",
                    "answer": ["a"],
                    "hop": "3step",
                    "gold_pages": [],
                }
            ]
        )
    )

    # The module is already imported above. A call-time override must still win.
    monkeypatch.setenv("MONACO_QUESTIONS", str(path))
    questions = load_monaco_eval()

    assert [question.id for question in questions] == ["full-only"]


def test_subset_can_pin_a_non_default_monaco_question_file(tmp_path):
    path = tmp_path / "monaco.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "pinned",
                    "question": "q",
                    "answer": ["a"],
                    "hop": "2step",
                    "gold_pages": [],
                }
            ]
        )
    )
    subset = PilotSubset(
        subset_id="pinned",
        benchmark="monaco",
        split="full",
        question_ids=["pinned"],
        dataset_path=str(path),
    )

    assert [question.id for question in load_benchmark_questions(subset)] == ["pinned"]


# --- no silent fallback to a smaller question file --------------------------


def _write(path, ids):
    path.write_text(json.dumps([
        {"id": i, "question": "q", "answer": ["a"], "hop": "3step", "gold_pages": []}
        for i in ids
    ]))


def test_default_path_is_the_full_file_under_panini_data_dir(tmp_path, monkeypatch):
    from panini_dr.eval.monaco_dataset import default_questions_path

    monkeypatch.delenv("MONACO_QUESTIONS", raising=False)
    monkeypatch.setenv("PANINI_DATA_DIR", str(tmp_path))
    assert default_questions_path() == tmp_path / "monaco" / "monaco_full1315.json"


def test_missing_question_file_is_an_error_not_a_fallback(tmp_path, monkeypatch):
    import pytest

    monkeypatch.delenv("MONACO_QUESTIONS", raising=False)
    monkeypatch.setenv("PANINI_DATA_DIR", str(tmp_path))
    # a smaller file sitting where the old fallback used to look is ignored
    (tmp_path / "monaco").mkdir()
    _write(tmp_path / "monaco" / "monaco_subset200.json", ["m1"])
    with pytest.raises(FileNotFoundError, match="no fallback"):
        load_monaco_eval()


def test_subset_ids_missing_from_the_question_file_raise(tmp_path):
    import pytest

    from panini_dr.eval.subset import SubsetIdsMissingError

    path = tmp_path / "monaco.json"
    _write(path, ["m1", "m2"])
    subset = PilotSubset(subset_id="s", benchmark="monaco", split="full",
                         question_ids=["m1", "m2", "m3"], dataset_path=str(path))
    with pytest.raises(SubsetIdsMissingError, match="1 are not in the loaded question file"):
        load_benchmark_questions(subset)


def test_complete_subset_keeps_subset_order(tmp_path):
    path = tmp_path / "monaco.json"
    _write(path, ["m1", "m2", "m3"])
    subset = PilotSubset(subset_id="s", benchmark="monaco", split="full",
                         question_ids=["m3", "m1"], dataset_path=str(path))
    assert [q.id for q in load_benchmark_questions(subset)] == ["m3", "m1"]
