"""Offline tests of the benchmark preparation scripts (synthetic rows only).

The scripts under ``datasets/`` are loaded by file path: ``datasets`` is not a
package (it would shadow the Hugging Face library of the same name).
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bcp():
    return _load("datasets/bcp/prepare.py", "_panini_prep_bcp")


@pytest.fixture(scope="module")
def frames():
    return _load("datasets/frames/prepare.py", "_panini_prep_frames")


@pytest.fixture(scope="module")
def monaco():
    return _load("datasets/monaco/prepare.py", "_panini_prep_monaco")


@pytest.fixture(scope="module")
def common(bcp):
    import _prep_common  # put on sys.path by the prepare scripts

    return _prep_common


# ---------------------------------------------------------------------------
# FRAMES
# ---------------------------------------------------------------------------


def _frames_row(idx, prompt, answer, links, types="Numerical reasoning | Multiple constraints"):
    return {"": str(idx), "Prompt": prompt, "Answer": answer, "wiki_links": repr(links),
            "reasoning_types": types}


def test_frames_row_conversion(frames):
    rows = [
        _frames_row(0, "The Zorvian ferry opened in 1901. Who designed it?", " Ada Quill ",
                    ["https://en.wikipedia.org/wiki/Zorvian_ferry",
                     "https://en.m.wikipedia.org/wiki/Ada_Quill#Career"]),
        _frames_row(1, "Who founded Blemton?", "Ivo Marsh",
                    ["https://en.wikipedia.org/wiki/Blemton, https://en.wikipedia.org/wiki/Ivo_Marsh",
                     "https://en.wikipedia.org/wiki/Blemton"], types=""),
    ]
    out = frames.split_frames_questions(rows)
    assert out[0]["id"] == 0 and isinstance(out[0]["id"], int)
    assert out[0]["original_prompt"] == "The Zorvian ferry opened in 1901. Who designed it?"
    assert out[0]["context"] == "The Zorvian ferry opened in 1901."
    assert out[0]["test_question"] == "Who designed it?"
    assert out[0]["answer"] == "Ada Quill"
    assert [a["title"] for a in out[0]["articles"]] == ["Zorvian ferry", "Ada Quill"]
    # mobile URL canonicalised, anchor stripped
    assert out[0]["articles"][1]["url"] == "https://en.wikipedia.org/wiki/Ada_Quill"
    assert out[0]["reasoning_types"] == ["Numerical reasoning", "Multiple constraints"]
    assert out[0]["num_hops"] == 2
    # single sentence -> unsplit; comma-packed URLs expanded; duplicates removed
    assert out[1]["context"] is None and out[1]["test_question"] == "Who founded Blemton?"
    assert [a["title"] for a in out[1]["articles"]] == ["Blemton", "Ivo Marsh"]
    assert out[1]["reasoning_types"] == []
    assert frames.build_article_index(out)["Blemton"] == "https://en.wikipedia.org/wiki/Blemton"


def test_frames_output_satisfies_loader(frames, tmp_path, monkeypatch):
    from panini_dr.eval.frames_dataset import load_frames

    rows = [_frames_row(i, f"Fact {i} is stated. What is fact {i}?", f"v{i}",
                        [f"https://en.wikipedia.org/wiki/Page_{i}"]) for i in range(3)]
    tsv = tmp_path / "test.tsv"
    with tsv.open("w") as fh:
        fh.write("\tPrompt\tAnswer\twiki_links\treasoning_types\n")
        for r in rows:
            fh.write("\t".join([r[""], r["Prompt"], r["Answer"], r["wiki_links"], r["reasoning_types"]]) + "\n")
    monkeypatch.setattr(frames.common, "require_split_ids", lambda ids, split: None)
    assert frames.main(["--data-dir", str(tmp_path / "d"), "questions", "--tsv", str(tsv)]) == 0
    qs = load_frames(path=tmp_path / "d" / "frames" / "frames_split.json")
    assert [q.id for q in qs] == ["0", "1", "2"]
    assert qs[1].question == "Fact 1 is stated. What is fact 1?" and qs[1].answer == "v1"


def test_frames_paper_split_check_fails_loudly(frames, tmp_path):
    rows = [_frames_row(0, "Q?", "a", ["https://en.wikipedia.org/wiki/X"])]
    tsv = tmp_path / "test.tsv"
    tsv.write_text("\tPrompt\tAnswer\twiki_links\treasoning_types\n"
                   + "\t".join([rows[0][""], "Q?", "a", rows[0]["wiki_links"], ""]) + "\n")
    # one question cannot cover the 824 ids of the paper split
    assert frames.main(["--data-dir", str(tmp_path / "d"), "questions", "--tsv", str(tsv)]) == 2


# ---------------------------------------------------------------------------
# MoNaCo
# ---------------------------------------------------------------------------


def test_monaco_release_conversion(monaco):
    release = {
        "Which rivers cross Vandria?": {
            "canary": "synthetic", "ex_num": 7, "question": "Which rivers cross Vandria?",
            "decomposition": ["Return rivers", "Return #1 that cross Vandria"],
            "validated_answer": ["Oll", "Tamsy"],
        },
        "List founders and years of the Plume guilds": {
            "canary": "synthetic", "ex_num": 12, "question": "List founders and years of the Plume guilds",
            "decomposition": ["a", "b", "c"],
            "validated_answer": [["Ari Venn", "1702"], ["Bo Lark", "1715"]],
        },
    }
    traces = {"7": {"steps": [{"source_url": "https://en.wikipedia.org/wiki/Vandria#Geography"},
                              {"source_url": ["https://en.wikipedia.org/wiki/Oll_(river)"]},
                              {"evidence": "see https://en.wikipedia.org/wiki/Tamsy"}]}}
    out = monaco.build_questions(monaco.iter_release_rows(release), traces)
    assert out[0] == {"id": "monaco_7", "question": "Which rivers cross Vandria?",
                      "answer": ["Oll", "Tamsy"], "hop": "2step",
                      "gold_pages": ["Oll (river)", "Vandria"]}          # source_url only, sorted
    assert out[1]["id"] == "monaco_12" and out[1]["hop"] == "3step" and out[1]["gold_pages"] == []
    assert out[1]["answer"] == [["Ari Venn", "1702"], ["Bo Lark", "1715"]]

    from panini_dr.eval.monaco_dataset import MonacoQuestion

    q = MonacoQuestion.from_raw(out[1])
    assert q.num_hops == 3 and q.gold_items() == ["Ari Venn", "1702", "Bo Lark", "1715"]


def test_monaco_gold_pages_follow_the_paper_rule(monaco):
    """The URL forms seen in the MoNaCo execution traces, and what the paper's file kept."""
    trace = {"provenance": {"1": [{"source_url": [
        "https://en.wikipedia.org/wiki/Zed_O%27Hara",                              # apostrophe, encoded
        "https://en.wikipedia.org/wiki/Mira_O'Neil",                               # apostrophe, raw
        "https://en.wikipedia.org/wiki/Lake_Ru, https://en.wikipedia.org/wiki/Lake_Vo",  # two URLs, one string
        "https://en.wikipedia.org/wiki/Brannoc,_Duke_of_Ost",                      # comma inside the title
        "https://en.wikipedia.org/wiki/Fen_list\r\n,\r\nhttps://en.wikipedia.org/wiki/Fen_map",
        "https://en.wikipedia.org/wiki/Tor_(band)#Members",                        # fragment dropped
        "https://en.wikipedia.org/wiki/Ula_Vey?wprov=srpw1_0",                     # query string kept
        "https://en.wikipedia.org/wiki/Ada Kell",                                  # literal space
        "https://en.m.wikipedia.org/wiki/Mobile_page",                             # mobile: not gold
        "https://simple.wikipedia.org/wiki/Simple_page",                           # other edition: not gold
        "en.wikipedia.org/wiki/No_scheme",                                         # no scheme: not gold
    ]}]}, "note": "https://en.wikipedia.org/wiki/Outside_source_url"}
    assert monaco.wikipedia_titles(trace) == [
        "Ada Kell", "Brannoc, Duke of Ost", "Fen list", "Fen map", "Lake Ru", "Lake Vo",
        "Mira O'Neil", "Tor (band)", "Ula Vey?wprov=srpw1 0", "Zed O'Hara"]


def test_monaco_duplicate_and_missing_ex_num(monaco, common):
    with pytest.raises(common.PrepareError):
        monaco.build_questions([{"ex_num": 1, "question": "a"}, {"ex_num": 1, "question": "b"}])
    with pytest.raises(common.PrepareError):
        monaco.build_questions([{"question": "no number"}])


def test_monaco_local_release_end_to_end(monaco, tmp_path, monkeypatch):
    rel = tmp_path / "rel" / "monaco_version_1_release.json"
    rel.parent.mkdir()
    rel.write_text(json.dumps({"q": {"ex_num": 3, "question": "q", "decomposition": ["x"],
                                     "validated_answer": ["y"]}}))
    tdir = rel.parent / "execution_traces"
    tdir.mkdir()
    (tdir / "dataset_ex_3.json").write_text(json.dumps({"s": [{"source_url": "https://en.wikipedia.org/wiki/Y_page"}]}))
    monkeypatch.setattr(monaco.common, "require_split_ids", lambda ids, split: None)
    assert monaco.main(["--data-dir", str(tmp_path / "d"), "questions", "--release", str(rel)]) == 0
    rows = json.loads((tmp_path / "d" / "monaco" / "monaco_full1315.json").read_text())
    assert rows == [{"id": "monaco_3", "question": "q", "answer": ["y"], "hop": "1step",
                     "gold_pages": ["Y page"]}]


def test_monaco_question_file_is_written_like_the_papers(monaco, tmp_path, monkeypatch, capsys):
    """Non-ASCII text is written as UTF-8 (ensure_ascii=False), compact, as in the paper's file."""
    rel = tmp_path / "monaco_version_1_release.json"
    rel.write_text(json.dumps({"q": {"ex_num": 4, "question": "Which caf\u00e9?", "decomposition": ["x"],
                                     "validated_answer": ["Barth\u00e9lemy"]}}))
    monkeypatch.setattr(monaco.common, "require_split_ids", lambda ids, split: None)
    assert monaco.main(["--data-dir", str(tmp_path / "d"), "questions", "--release", str(rel)]) == 0
    raw = (tmp_path / "d" / "monaco" / "monaco_full1315.json").read_bytes()
    assert raw == json.dumps(json.loads(raw), ensure_ascii=False).encode("utf-8")
    assert "Barth\u00e9lemy".encode("utf-8") in raw and b"\\u00e9" not in raw
    assert "differs from the paper's question file" in capsys.readouterr().out
    assert len(monaco.PAPER_QUESTIONS_SHA256) == 64


# ---------------------------------------------------------------------------
# BrowseComp-Plus
# ---------------------------------------------------------------------------


def test_bcp_corpus_keys(bcp, common):
    docids = [101, 102, 103, 104]
    texts = ["---\ntitle: Harbor Notes\n---\nbody one",
             "---\ntitle: Harbor Notes\n---\nbody two",
             "no front matter here",
             ""]
    cache, stats = bcp.build_corpus(docids, texts)
    assert list(cache) == ["Harbor Notes [101]", "Harbor Notes [102]", "103"]
    assert stats == {"documents": 3, "skipped_short": 1, "untitled": 1, "shared_title_documents": 2}
    with pytest.raises(common.PrepareError):
        bcp.build_corpus([5, 5], ["a", "b"])


def test_bcp_questions_use_cache_keys(bcp):
    mapping = bcp.docid_to_key_map([1, 2], ["---\ntitle: Gold Page\n---\nx", "plain"])
    rows = [{"query_id": 9, "query": "synthetic question?", "answer": "zed",
             "gold_docs": [{"docid": "1"}, {"docid": "999"}], "evidence_docs": [{"docid": 2}],
             "negative_docs": [{"docid": "1"}]}]
    out, missing = bcp.build_questions(rows, mapping)
    assert out == [{"query_id": "9", "query": "synthetic question?", "answer": "zed",
                    "gold_pages": ["Gold Page [1]"], "evidence_pages": ["2"]}]
    assert missing == 1
    assert bcp.subset_for(out)["question_ids"] == ["9"]

    from panini_dr.eval.browsecomp_dataset import BrowseCompQuestion

    q = BrowseCompQuestion.from_raw(out[0])
    assert q.id == "9" and q.question == "synthetic question?" and q.gold_pages == ["Gold Page [1]"]


def test_bcp_dense_cache_relabelling(bcp, tmp_path):
    np = pytest.importorskip("numpy")
    shard_dir = tmp_path / "idx"
    shard_dir.mkdir()
    reps = np.eye(3, 4, dtype="float32")
    with (shard_dir / "corpus.shard0_of_1.pkl").open("wb") as fh:
        pickle.dump((reps, ["12", "10", "11"]), fh)

    from panini_dr.eval.bcp_dense import load_shards

    loaded, lookup = load_shards(shard_dir)
    mapping = {"10": "Ten [10]", "11": "11", "12": "Twelve [12]"}
    ids = bcp.write_dense_cache(loaded, lookup, mapping, tmp_path / "cache", skip_faiss=True)
    assert ids == ["Twelve [12]__0", "Ten [10]__0", "11__0"]  # row order kept, only labelled
    with (tmp_path / "cache" / "chunk_ids.pkl").open("rb") as fh:
        assert pickle.load(fh) == ids
    assert np.array_equal(np.load(tmp_path / "cache" / "chunk_embeddings.npy"), reps)
    with pytest.raises(KeyError):  # a docid with no document must not shift rows silently
        bcp.write_dense_cache(loaded, lookup, {"10": "a", "11": "b"}, tmp_path / "c2", skip_faiss=True)


_FAKE_DECRYPT = """\
import argparse, json, os
ap = argparse.ArgumentParser()
ap.add_argument("--output", default="decrypted.jsonl")
ap.add_argument("--generate-tsv", default=None)
a = ap.parse_args()
with open(a.output, "w") as fh:
    fh.write(json.dumps({"query_id": "9", "query": "synthetic?", "answer": "zed",
                         "gold_docs": [{"docid": "1"}], "evidence_docs": []}) + "\\n")
"""


def test_bcp_decrypt_args_default_to_the_papers_output(bcp):
    args, out = bcp.decrypt_args(None)
    assert args == ["--output", "data/browsecomp_plus_decrypted.jsonl"] and str(out) == args[1]
    args, out = bcp.decrypt_args(["--generate-tsv", "topics-qrels/queries.tsv"])
    assert args[:2] == ["--output", "data/browsecomp_plus_decrypted.jsonl"] and "--generate-tsv" in args
    assert bcp.decrypt_args(["--output", "x/y.jsonl"]) == (["--output", "x/y.jsonl"], Path("x/y.jsonl"))
    assert bcp.decrypt_args(["--output=z.jsonl"])[1] == Path("z.jsonl")


def test_bcp_decrypt_with_repo_finds_the_output(bcp, tmp_path, monkeypatch):
    """The upstream script defaults to ./decrypted.jsonl; the wrapper must name the output."""
    monkeypatch.setattr(bcp, "_require_hf_datasets", lambda: None)  # the fake script does not need it
    repo = tmp_path / "BrowseComp-Plus"
    (repo / "scripts_build_index").mkdir(parents=True)
    (repo / "scripts_build_index" / "decrypt_dataset.py").write_text(_FAKE_DECRYPT)
    out = bcp._decrypt_with_repo(repo, None)
    assert out == repo / "data" / "browsecomp_plus_decrypted.jsonl"
    assert json.loads(out.read_text().splitlines()[0])["query_id"] == "9"
    assert not (repo / "decrypted.jsonl").exists()


def test_bcp_decrypt_without_hf_datasets_is_a_clean_error(bcp, tmp_path, monkeypatch, capsys):
    import importlib.metadata

    real = importlib.metadata.version

    def version(name):
        if name == "datasets":
            raise importlib.metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    repo = tmp_path / "BrowseComp-Plus"
    (repo / "scripts_build_index").mkdir(parents=True)
    (repo / "scripts_build_index" / "decrypt_dataset.py").write_text(_FAKE_DECRYPT)
    assert bcp.main(["--data-dir", str(tmp_path), "questions", "--bcp-repo", str(repo)]) == 2
    err = capsys.readouterr().err
    assert "`datasets` package" in err and "--extra datasets" in err
    assert not (repo / "data" / "browsecomp_plus_decrypted.jsonl").exists()


def test_bcp_decrypt_failure_is_a_clean_error(bcp, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bcp, "_require_hf_datasets", lambda: None)
    repo = tmp_path / "BrowseComp-Plus"
    (repo / "scripts_build_index").mkdir(parents=True)
    (repo / "scripts_build_index" / "decrypt_dataset.py").write_text("raise SystemExit(3)\n")
    assert bcp.main(["--data-dir", str(tmp_path), "questions", "--bcp-repo", str(repo)]) == 2
    assert "decrypt_dataset.py failed (exit status 3" in capsys.readouterr().err


def test_the_datasets_extra_installs_hf_datasets():
    import tomllib

    extra = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]["datasets"]
    assert any(req.split("==")[0] == "datasets" for req in extra)


def test_bcp_questions_requires_decrypted_input(bcp, tmp_path, capsys):
    assert bcp.main(["--data-dir", str(tmp_path), "--parquet", str(tmp_path / "none.parquet"),
                     "questions"]) == 2
    assert "decrypt" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def test_fingerprint_matches_dense_retriever(common, tmp_path):
    from panini_dr.retrieval.corpus import load_frames_corpus
    from panini_dr.retrieval.dense import DenseRetriever

    articles = {"Beta": "b text " * 30, "Alpha": "a text", "Empty": ""}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(articles))
    corpus = load_frames_corpus(cache_path=path)
    fake = type("F", (), {})()
    fake.corpus, fake.model_name = corpus, "Qwen/Qwen3-Embedding-8B"
    assert common.dense_fingerprint(articles) == DenseRetriever._fingerprint(fake)
    assert common.dense_fingerprint(articles).startswith("Qwen__Qwen3-Embedding-8B__")


def test_paper_fingerprints_are_the_recorded_ones(frames, monaco):
    assert frames.PAPER_DENSE_FINGERPRINT.endswith("99c7a0f7ddf9__efb84805d53b")
    assert monaco.PAPER_DENSE_FINGERPRINT.endswith("7ea9a4d678f8__de3db946f2e8")


def test_split_files_exist_for_every_check():
    for name in ("cf_bcp150.json", "frames_full824_norm.json", "monaco_full1315.json"):
        data = json.loads((REPO / "configs" / "splits" / name).read_text())
        assert data["question_ids"]


@pytest.mark.parametrize("bench", ["bcp", "frames", "monaco"])
def test_every_prepare_subcommand_and_option_has_help(bench):
    import argparse

    mod = _load(f"datasets/{bench}/prepare.py", f"_prep_help_{bench}")
    parser = mod.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    listed = {c.dest: c.help for c in sub._choices_actions}
    for name, p in sub.choices.items():
        assert listed.get(name), f"{bench} {name}: no help in the command list"
        assert p.description, f"{bench} {name}: no description in its --help"
        for action in p._actions:
            if action.option_strings and action.dest != "help":
                assert action.help, f"{bench} {name} {action.option_strings}: no help"
