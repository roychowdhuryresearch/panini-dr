"""Bring your own dataset: `panini-dr prepare` + `panini-dr run --dataset` on toy JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from panini_dr.byod import build_corpus, load_prepared, read_docs_jsonl
from panini_dr.cli import app
from panini_dr.eval.byod_dataset import ByodSchemaError, read_questions_jsonl
from tests.e2e.fakes import TOY_QUESTIONS, write_toy_docs_jsonl, write_toy_questions_jsonl

runner = CliRunner()


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_question_schema(tmp_path):
    qs = read_questions_jsonl(_write_jsonl(tmp_path / "q.jsonl", [
        {"id": "a", "question": "Q one?", "answer": "x"},
        {"id": "b", "question": "Q two?", "answer": ["x", "y"]},
        {"id": "c", "question": "Q three?", "answer": [["x", "1"]], "answer_type": "set"},
    ]))
    assert [q.answer_type for q in qs] == ["short", "set", "set"]
    for bad in ({"id": "a", "answer": "x"}, {"id": "a", "question": "?", "answer": 3},
                {"id": "a", "question": "?", "answer": "x", "answer_type": "long"}):
        with pytest.raises(ByodSchemaError):
            read_questions_jsonl(_write_jsonl(tmp_path / "bad.jsonl", [bad]))
    with pytest.raises(ByodSchemaError, match="duplicate"):
        read_questions_jsonl(_write_jsonl(tmp_path / "dup.jsonl", [
            {"id": "a", "question": "?", "answer": "x"}] * 2))


def test_frames_loader_accepts_a_plain_question_key(tmp_path):
    from panini_dr.eval.frames_dataset import load_frames

    path = tmp_path / "f.json"
    path.write_text(json.dumps([{"id": "1", "question": "Plain key?", "answer": "x"}]))
    assert load_frames(path=path)[0].question == "Plain key?"


def test_document_keys(tmp_path):
    docs = read_docs_jsonl(_write_jsonl(tmp_path / "d.jsonl", [
        {"id": "1", "title": "Alpha", "text": "a"},
        {"id": "2", "title": "Shared", "text": "b"},
        {"id": "3", "title": "Shared", "text": "c"},
        {"id": "4", "text": "d"},
    ]))
    assert list(build_corpus(docs)) == ["Alpha", "Shared [2]", "Shared [3]", "4"]


def _prepare(tmp_path: Path, *extra: str) -> Path:
    out = tmp_path / "prepared"
    res = runner.invoke(app, [
        "prepare", "--questions", str(write_toy_questions_jsonl(tmp_path / "questions.jsonl")),
        "--docs", str(write_toy_docs_jsonl(tmp_path / "docs.jsonl")),
        "--out", str(out), "--name", "toy", *extra,
    ])
    assert res.exit_code == 0, res.output
    return out


def test_prepare_bm25_only_and_run(fake_services, tmp_path, monkeypatch):
    out = _prepare(tmp_path, "--no-dense")
    ds = load_prepared(out)
    assert ds.dense_cache is None and ds.manifest["n_questions"] == len(TOY_QUESTIONS)
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    # the paper configuration needs a dense index: refused, with the remedy
    res = runner.invoke(app, ["run", "--dataset", str(out), "--output-dir", str(tmp_path / "r")])
    assert res.exit_code == 2 and "retriever_type=bm25" in res.output
    # BM25-only is possible, as declared drift
    res = runner.invoke(app, [
        "run", "--dataset", str(out), "--output-dir", str(tmp_path / "runs"),
        "--set", "retriever_type=bm25", "--allow-config-drift",
    ])
    assert res.exit_code == 0, res.output
    cell_dir = next((tmp_path / "runs").iterdir())
    cell = json.loads((cell_dir / "cell_result.json").read_text())
    assert cell["benchmark"] == "byod"
    preds = {q["question_id"]: q["predicted_answer"] for q in cell["questions"]}
    assert preds == {q.id: q.answer for q in TOY_QUESTIONS}
    assert cell["config"]["retriever_type"] == "bm25"
    # the closing hint is a complete grading command for this dataset
    assert (f"panini-dr score answers -b byod --run {cell_dir} --questions {out / 'questions.jsonl'} "
            "--cache judge/toy_cache.json --out judge/toy_verdicts.json --backend openai") in res.output


def test_prepare_with_dense_index_and_run_paper_config(fake_services, tmp_path, monkeypatch):
    """The real DenseRetriever builds and loads the index; only the encoder is fake."""
    from tests.e2e.fakes import FakeDenseRetriever

    _fake_encoder(monkeypatch)
    out = _prepare(tmp_path)
    ds = load_prepared(out)
    assert ds.dense_cache is not None
    for name in ("chunk_embeddings.npy", "chunk_ids.pkl", "faiss.index"):
        assert (ds.dense_cache / name).is_file()

    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    res = runner.invoke(app, ["run", "--dataset", str(out), "--output-dir", str(tmp_path / "runs")])
    assert res.exit_code == 0, res.output  # the paper configuration, no drift
    cell_dir = next((tmp_path / "runs").iterdir())
    cell = json.loads((cell_dir / "cell_result.json").read_text())
    preds = {q["question_id"]: q["predicted_answer"] for q in cell["questions"]}
    assert preds == {q.id: q.answer for q in TOY_QUESTIONS}
    meta = json.loads((cell_dir / "run_meta.json").read_text())
    assert meta["config_drift"] == {}
    assert cell["config"]["dense_cache"].endswith("dense_cache")
    assert FakeDenseRetriever.instances == []


def _fake_encoder(monkeypatch):
    """The real DenseRetriever with a deterministic bag-of-words encoder (no GPU, no weights)."""
    pytest.importorskip("faiss")
    pytest.importorskip("torch")
    import numpy as np
    import sentence_transformers

    from panini_dr.retrieval import dense
    from tests.e2e.fakes import _tokens

    monkeypatch.setattr(dense, "DenseRetriever", _RealDense)  # undo the e2e fake

    class _Encoder:
        def __init__(self, *a, **k):
            pass

        def encode(self, texts, **kwargs):
            vecs = []
            for t in texts:
                v = np.zeros(64, dtype=np.float32)
                for tok in _tokens(t):
                    v[hash(tok) % 64] += 1.0
                vecs.append(v / (np.linalg.norm(v) or 1.0))
            return np.stack(vecs)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _Encoder)
    monkeypatch.setattr(dense, "_pick_device", lambda: "cpu")


def test_reprepare_rebuilds_a_stale_dense_index(fake_services, tmp_path, monkeypatch):
    """Same number of chunks, different text: the old embeddings must not be kept."""
    import numpy as np

    _fake_encoder(monkeypatch)
    qs = write_toy_questions_jsonl(tmp_path / "questions.jsonl")
    out = tmp_path / "prepared"

    def prepare(texts):
        docs = _write_jsonl(tmp_path / "docs.jsonl", [
            {"id": str(i), "title": f"Doc {i}", "text": t} for i, t in enumerate(texts)])
        res = runner.invoke(app, ["prepare", "--questions", str(qs), "--docs", str(docs),
                                  "--out", str(out), "--name", "toy"])
        assert res.exit_code == 0, res.output
        return res.output, np.load(out / "dense_cache" / "chunk_embeddings.npy")

    _, first = prepare(["alpha beta gamma", "delta epsilon zeta"])
    stamp = json.loads((out / "dense_cache" / "built_from.json").read_text())
    manifest = json.loads((out / "dataset.json").read_text())
    assert stamp["corpus_sha256"] == manifest["files"]["corpus"]["sha256"]
    assert stamp["chunk_ids_sha256"] == manifest["dense"]["chunk_ids_sha256"] and stamp["n_chunks"] == 2
    log, same = prepare(["alpha beta gamma", "delta epsilon zeta"])
    assert "reusing the dense index" in log and np.array_equal(first, same)
    log, second = prepare(["one two three", "four five six"])  # same ids, same chunk count
    assert "stale" in log and "different corpus" in log
    assert second.shape == first.shape and not np.array_equal(first, second)
    assert load_prepared(out).dense_cache is not None


def test_run_refuses_a_dense_index_from_another_corpus(fake_services, tmp_path, monkeypatch):
    from panini_dr.byod import PreparedDatasetError

    _fake_encoder(monkeypatch)
    out = _prepare(tmp_path)
    stamp_path = out / "dense_cache" / "built_from.json"
    stamp = json.loads(stamp_path.read_text())
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")

    stamp_path.write_text(json.dumps({**stamp, "corpus_sha256": "0" * 64}))
    res = runner.invoke(app, ["run", "--dataset", str(out), "--output-dir", str(tmp_path / "r")])
    assert res.exit_code == 2 and "encoded from another corpus" in res.output
    assert "panini-dr prepare" in res.output and isinstance(res.exception, SystemExit)  # no traceback

    stamp_path.unlink()
    with pytest.raises(PreparedDatasetError, match="built_from.json"):
        load_prepared(out)

    # the run also compares the chunk ids it builds with the ones the index was encoded from
    stamp_path.write_text(json.dumps(stamp))
    from panini_dr.byod import check_dense_cache

    ds = load_prepared(out)
    with pytest.raises(PreparedDatasetError, match="chunks into ids"):
        check_dense_cache(ds, ["not__0", "the__0", "same__0"])


def test_prepared_files_are_checked(tmp_path):
    out = _prepare(tmp_path, "--no-dense")
    (out / "corpus.json").write_text("{}")
    from panini_dr.byod import PreparedDatasetError

    with pytest.raises(PreparedDatasetError, match="sha256"):
        load_prepared(out)


from panini_dr.retrieval.dense import DenseRetriever as _RealDense  # noqa: E402


def test_prepare_and_run_label_the_corpus_as_the_users(fake_services, tmp_path, monkeypatch):
    out = _prepare(tmp_path, "--no-dense")
    monkeypatch.setenv("PANINI_LLM_BASE_URL", "http://llm.invalid/v1")
    res = runner.invoke(app, ["run", "--dataset", str(out), "--output-dir", str(tmp_path / "runs"),
                              "--set", "retriever_type=bm25", "--allow-config-drift"])
    assert res.exit_code == 0, res.output
    log = (next((tmp_path / "runs").iterdir()) / "run.log").read_text()
    assert "Corpus dataset:toy:" in log and "your own dataset" in log
    assert "frames-derived" not in log and "control runs only" not in log
