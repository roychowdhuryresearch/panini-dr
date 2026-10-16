"""BrowseComp-Plus question loading + benchmark routing."""
import json

import pytest

from panini_dr.eval.browsecomp_dataset import (
    BrowseCompQuestion,
    load_browsecomp_plus,
)
from panini_dr.eval.subset import PilotSubset, load_benchmark_questions


def _write(tmp_path, rows):
    p = tmp_path / "bcp.json"
    p.write_text(json.dumps(rows))
    return p


def test_maps_bcp_fields_onto_the_harness_shape(tmp_path):
    p = _write(tmp_path, [{
        "query_id": "769",
        "query": "Name the institution that ...",
        "answer": "Queen Arwa University",
        "gold_pages": ["Arwa University holds annual cultural activities [5412]"],
        "evidence_pages": ["Sanaa - Wikipedia [46697]"],
    }])
    qs = load_browsecomp_plus(path=p)
    assert len(qs) == 1
    q = qs[0]
    assert q.id == "769"
    assert q.question.startswith("Name the institution")
    # answer is a single short string but the harness expects a list
    assert q.answer == ["Queen Arwa University"]
    assert q.gold_pages == ["Arwa University holds annual cultural activities [5412]"]
    # evidence is kept for the retrieval-coverage metric MoNaCo cannot provide
    assert q.raw["evidence_pages"] == ["Sanaa - Wikipedia [46697]"]
    # BCP has no per-question article payloads -- retrieval uses the shared cache
    assert q.articles == []


def test_answer_already_a_list_is_preserved(tmp_path):
    p = _write(tmp_path, [{"query_id": "1", "query": "q", "answer": ["a", "b"]}])
    assert load_browsecomp_plus(path=p)[0].answer == ["a", "b"]


def test_missing_optional_fields_do_not_raise(tmp_path):
    p = _write(tmp_path, [{"query_id": "2", "query": "q", "answer": "x"}])
    q = load_browsecomp_plus(path=p)[0]
    assert q.gold_pages == [] and q.articles == []


def test_missing_file_names_the_override(tmp_path):
    """The MoNaCo default silently capped 1315 questions at 200 for weeks;
    a missing BCP file must say how to point at the right one."""
    with pytest.raises(FileNotFoundError) as exc:
        load_browsecomp_plus(path=tmp_path / "nope.json")
    assert "BROWSECOMP_QUESTIONS" in str(exc.value)


def test_subset_routes_browsecomp_plus_to_this_loader(tmp_path, monkeypatch):
    p = _write(tmp_path, [
        {"query_id": "a", "query": "qa", "answer": "x"},
        {"query_id": "b", "query": "qb", "answer": "y"},
    ])
    monkeypatch.setenv("BROWSECOMP_QUESTIONS", str(p))
    subset = PilotSubset(
        subset_id="bcp_test", benchmark="browsecomp_plus", split="full",
        question_ids=["b"],
    )
    got = load_benchmark_questions(subset, split="full")
    assert [q.id for q in got] == ["b"]
    assert isinstance(got[0], BrowseCompQuestion)
