"""Obs. scorer on a made-up FRAMES-shaped and MoNaCo-shaped toy benchmark."""
import json

import pytest

from panini_dr.eval import obs as O


def _search(chunk_id, title, text):
    return {"name": "search", "args": {"query": "q"},
            "result_full": json.dumps({"results": [{"chunk_id": chunk_id, "title": title, "text": text}]})}


def _trace(run, qid, answer, tool_calls=(), extra=None):
    (run / "traces").mkdir(parents=True, exist_ok=True)
    (run / "traces" / f"q_{qid}.json").write_text(json.dumps(
        {"question_id": qid, "final_answer": answer, "tool_calls": list(tool_calls), "extra": extra or {}}))


@pytest.fixture
def frames_toy(tmp_path):
    corpus = {
        "Zorblat Island": "Zorblat Island\n\nZorblat Island is a made-up island. Its capital is Quuxville.",
        "Mount Fennick": "Mount Fennick\n\nMount Fennick is 4321 metres tall and was first climbed by Ada Brook.",
    }
    (tmp_path / "corpus.json").write_text(json.dumps(corpus))
    qs = [
        dict(id="0", test_question="What is the capital of Zorblat Island?", original_prompt="What is the capital of Zorblat Island?",
             answer="Quuxville", articles=[{"title": "Zorblat Island"}], num_hops=1),
        dict(id="1", test_question="Who named Zorblat Island?", original_prompt="Who named Zorblat Island?",
             answer="Florp", articles=[{"title": "Zorblat Island"}], num_hops=1),
        dict(id="2", test_question="Who first climbed Mount Fennick?", original_prompt="Who first climbed Mount Fennick?",
             answer="Ada Brook", articles=[{"title": "Mount Fennick"}], num_hops=1),
        dict(id="3", test_question="How tall is Mount Fennick?", original_prompt="How tall is Mount Fennick?",
             answer="4321 metres", articles=[{"title": "Mount Fennick"}], num_hops=1),
    ]
    (tmp_path / "frames.json").write_text(json.dumps(qs))
    (tmp_path / "split.json").write_text(json.dumps({"subset_id": "toy", "question_ids": ["0", "1", "2", "3"]}))
    run = tmp_path / "run"
    _trace(run, "0", "Quuxville", [_search("Zorblat Island__0", "Zorblat Island", corpus["Zorblat Island"])])
    _trace(run, "1", "Florp", [_search("Zorblat Island__0", "Zorblat Island", corpus["Zorblat Island"])])
    # q2: the answer only reached the typed resolver (no tool call) -- the resolver channel
    _trace(run, "2", "Ada Brook", extra={"gsw_checker_calls": [{
        "kind": "resolver", "retrieved_chunks": [{"chunk_id": "Mount Fennick__0",
                                                   "text": corpus["Mount Fennick"]}]}]})
    # q3: no trace at all
    return tmp_path


def _ds(root):
    return O.Dataset("frames", questions=str(root / "frames.json"), corpus=str(root / "corpus.json"),
                     split=str(root / "split.json"))


def test_frames_obs_counts_only_supported_correct_answers(frames_toy):
    verdicts = {q: {"correct": True, "pred": "", "extracted": None} for q in ("0", "1", "2")}
    s = O.score_short_obs(_ds(frames_toy), verdicts, run_dirs=[str(frames_toy / "run")],
                          out_dir=str(frames_toy / "out"))
    # official: 3 of 4 judged correct; Obs.: q0 (search) + q2 (resolver channel) supported, q1 not
    assert s["n"] == 4
    assert s["official"] == pytest.approx(0.75)
    assert s["obs"] == pytest.approx(0.5)
    rows = [json.loads(line) for line in (frames_toy / "out" / "obs_frames_rows.jsonl").read_text().splitlines()]
    by = {r["qid"]: r for r in rows}
    assert by["0"]["earned"] == 1.0 and by["1"]["earned"] == 0.0 and by["2"]["earned"] == 1.0
    assert by["3"]["trace_missing"]


def test_resolver_text_must_be_verbatim_corpus_text(frames_toy):
    _trace(frames_toy / "run", "2", "Ada Brook", extra={"gsw_checker_calls": [{
        "kind": "resolver", "retrieved_chunks": [{"chunk_id": "Mount Fennick__0",
                                                   "text": "A model wrote that Ada Brook climbed it."}]}]})
    verdicts = {"2": {"correct": True, "pred": "", "extracted": None}}
    s = O.score_short_obs(_ds(frames_toy), verdicts, run_dirs=[str(frames_toy / "run")])
    assert s["obs"] == 0.0


def test_parse_passage_keeps_nested_bcp_docid_brackets():
    assert O._parse_passage("[Some Page - Wikipedia [65939]__0] body text") == (
        "Some Page - Wikipedia [65939]__0", "body text")
    assert O._parse_passage("no bracket") is None


@pytest.fixture
def monaco_toy(tmp_path):
    corpus = {
        "Flagland": "Flagland\n\nThe flag of Flagland is red and blue. Flagland has a long coast.",
        "Hillshire": "Hillshire\n\nHillshire lies inland. " + "Filler sentence. " * 20 + "Its emblem is a green oak.",
    }
    (tmp_path / "corpus.json").write_text(json.dumps(corpus))
    qs = [dict(id="monaco_1", question="Which colours are on the flag of Flagland?", answer=["Red", "Blue"],
               hop="3step", gold_pages=["Flagland"]),
          dict(id="monaco_2", question="What is on the emblem of Hillshire?", answer=["green oak"],
               hop="2step", gold_pages=["Hillshire"]),
          dict(id="monaco_3", question="Name the coast of Flagland?", answer=["Long Coast"],
               hop="2step", gold_pages=["Flagland"])]
    (tmp_path / "monaco.json").write_text(json.dumps(qs))
    (tmp_path / "split.json").write_text(json.dumps({"subset_id": "toy", "question_ids": [q["id"] for q in qs]}))
    run = tmp_path / "run"
    _trace(run, "monaco_1", "Red, Blue", [_search("Flagland__0", "Flagland", corpus["Flagland"][:60])])
    # monaco_2: the emblem is only in the part of the article the READER paged through
    _trace(run, "monaco_2", "green oak", extra={"agentic_read": {"records": [{
        "chunk_id": "Hillshire__0", "sections_checked": [f"read_full(offset=0) -> chars 0-{len(corpus['Hillshire'])}"]}]}})
    _trace(run, "monaco_3", "  ")                                         # blank answer
    return tmp_path


def _mds(root):
    return O.Dataset("monaco", questions=str(root / "monaco.json"), corpus=str(root / "corpus.json"),
                     split=str(root / "split.json"))


def _set_verdict(glen, credited, f1, plen):
    raw = (f"extracted_final_answers: ...\nfinal answer length: {plen}\noverlapping answers: {'###'.join(credited)}"
           if glen > 1 else f"extracted_final_answer: {credited[0] if credited else 'None'}\nfinal precision: {f1}")
    return {"glen": glen, "passes": [
        {"f1": 0.0, "credited": [], "judge_pred_len": plen, "raw": "", "parsed": True},
        {"f1": f1, "credited": credited, "judge_pred_len": plen, "raw": raw, "parsed": True}]}


def test_monaco_obs_item_rule_with_reader_channel(monaco_toy):
    verdicts = {"monaco_1": _set_verdict(2, ["Red", "Blue"], 1.0, 2),
                "monaco_2": _set_verdict(1, ["green oak"], 1.0, 1)}
    s = O.score_monaco_obs(_mds(monaco_toy), verdicts, run_dirs=[str(monaco_toy / "run")],
                           corpus=str(monaco_toy / "corpus.json"))
    assert s["n"] == 3 and s["with_pred"] == 2
    assert s["official"] == pytest.approx(2 / 3)                       # blank monaco_3 scores 0
    assert s["obs"] == pytest.approx(2 / 3)                            # both credited answers supported
    assert s["earned_gt_official_violations"] == 0


def test_monaco_obs_unsupported_credit_is_removed(monaco_toy):
    # judge credited 'Green' for monaco_1, which no received text contains
    verdicts = {"monaco_1": _set_verdict(2, ["Red", "Green"], 1.0, 2)}
    s = O.score_monaco_obs(_mds(monaco_toy), verdicts, run_dirs=[str(monaco_toy / "run")],
                           corpus=str(monaco_toy / "corpus.json"))
    assert s["official"] == pytest.approx(1 / 3)
    assert 0.0 < s["obs"] < s["official"]


def test_packet_evidence_has_the_three_channels():
    t = {"tool_calls": [_search("A__0", "A", "alpha text")],
         "extra": {"executed_blanks": [{"evidence_chunk_ids": ["B__0", "B__0"]}],
                   "agentic_read": {"records": [{"chunk_id": "C__0", "sections_checked": [
                       "read_full(offset=0) -> chars 0-5", "read_full(offset=3) -> chars 3-9"]}]}}}
    ev = O.build_packet_evidence(t, lambda cid: "bravo chunk" if cid == "B__0" else None,
                                 {"C": "charlie article"})
    assert "### [tool] search(" in ev
    assert ev.count("### [resolver evidence chunk] B__0") == 1
    assert "### [reader read_full] C chars 0-9\ncharlie a" in ev       # overlapping spans merged


def test_monaco_reader_channel_is_what_supports_the_emblem(monaco_toy):
    _trace(monaco_toy / "run", "monaco_2", "green oak")                 # same answer, no reader record
    verdicts = {"monaco_2": _set_verdict(1, ["green oak"], 1.0, 1)}
    s = O.score_monaco_obs(_mds(monaco_toy), verdicts, run_dirs=[str(monaco_toy / "run")],
                           corpus=str(monaco_toy / "corpus.json"))
    assert s["official"] == pytest.approx(1 / 3) and s["obs"] == 0.0
