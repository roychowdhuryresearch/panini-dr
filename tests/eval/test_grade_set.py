import pytest

from panini_dr.eval.grade_set import (
    SetItem, build_prompt, demarkdown, gold_list, grade_set, judge_one, parse_credited,
    question_scores, summarize_set,
)
from panini_dr.eval.judge_cache import JudgeCache
from tests.eval.fakes import MULTI_PROMPT, SINGLE_PROMPT, ScriptedJudge, fake_v2_score


def _multi_reply(plen, answers):
    return (f"extracted_final_answers: ...\nfinal answer length: {plen}\n"
            f"overlapping answers: {'###'.join(answers) if answers else 'NULL'}")


def test_demarkdown_only_touches_numeric_fields():
    raw = "final answer length: **13**\noverlapping answers: **Bold Name**###b\nfinal precision: *0.5*"
    out = demarkdown(raw)
    assert "final answer length: 13" in out and "final precision: 0.5" in out
    assert "**Bold Name**" in out


def test_parse_credited_multi_and_single():
    items, plen, ok = parse_credited("x\nfinal answer length: **4**\noverlapping answers: A###B###NULL", 3, "p", {})
    assert items == ["A", "B"] and plen == 4 and ok
    assert parse_credited("no fields", 3, "p", {}) == ([], None, False)
    assert parse_credited("", 1, "pred", {"judge_score": 1.0}) == (["pred"], 1, True)
    assert parse_credited("", 1, "pred", {"judge_score": 0.0}) == ([], 1, True)


def test_gold_rendering_matches_the_research_pipeline():
    rows = [["Alpha", "1990"], ["Beta", "1991"]]
    assert gold_list(rows) == ["['Alpha', '1990']", "['Beta', '1991']"]
    assert gold_list("solo") == ["solo"]
    prompt, glen = build_prompt(SetItem("q", "Which?", rows, "  Alpha  "), SINGLE_PROMPT, MULTI_PROMPT)
    assert glen == 2 and prompt.startswith("MULTI") and "'Alpha'" in prompt
    assert "['Alpha', '1990']; ['Beta', '1991']" in prompt
    p1, g1 = build_prompt(SetItem("q", "Which?", ["x"], "x"), SINGLE_PROMPT, MULTI_PROMPT)
    assert g1 == 1 and p1.startswith("SINGLE")


def test_max_of_three_and_mean_over_passes():
    replies = [_multi_reply(4, ["a"]), _multi_reply(4, ["a", "b", "c", "d"]), "unparseable"]
    j = ScriptedJudge(lambda p, i: replies[i - 1])
    rec = judge_one(SetItem("q", "Q?", ["a", "b", "c", "d"], "a, b, c, d"), complete=j.complete,
                    single_prompt=SINGLE_PROMPT, multi_prompt=MULTI_PROMPT, score=fake_v2_score)
    f1s = [p["f1"] for p in rec["passes"]]
    assert f1s == pytest.approx([0.25, 1.0, 0.0])
    assert [p["parsed"] for p in rec["passes"]] == [True, True, False]
    s = question_scores(rec)
    assert s["f1_max"] == pytest.approx(1.0) and s["f1_mean"] == pytest.approx(1.25 / 3)


def test_markdown_count_is_read_after_demarkdown():
    j = ScriptedJudge("extracted_final_answers: a, b\nfinal answer length: **2**\noverlapping answers: a###b")
    rec = judge_one(SetItem("q", "Q?", ["a", "b"], "a; b"), complete=j.complete, single_prompt=SINGLE_PROMPT,
                    multi_prompt=MULTI_PROMPT, score=fake_v2_score, passes=1)
    assert rec["passes"][0]["f1"] == pytest.approx(1.0) and rec["passes"][0]["judge_pred_len"] == 2


def test_denominator_counts_blank_missing_and_failed_as_zero(tmp_path):
    def reply(prompt, i):
        if "boom" in prompt:
            raise RuntimeError("judge failed")
        return _multi_reply(2, ["a", "b"])
    j = ScriptedJudge(reply)
    items = [SetItem("q1", "Q1?", ["a", "b"], "a; b"),       # perfect
             SetItem("q2", "Q2?", ["a", "b"], "   "),         # blank: not judged
             SetItem("q3", "Q3?", ["a", "b"], "boom")]        # judge raises
    recs = grade_set(items, complete=j.complete, single_prompt=SINGLE_PROMPT, multi_prompt=MULTI_PROMPT,
                     score=fake_v2_score, model="m", cache=JudgeCache(tmp_path / "c.json"), workers=1)
    assert recs["q2"]["blank"] and "error" in recs["q3"]
    s = summarize_set(recs, ["q1", "q2", "q3", "q4"])                   # q4: no trace at all
    assert s["n"] == 4 and s["f1"] == pytest.approx(0.25)
    assert (s["n_judged"], s["n_blank"], s["n_errors"], s["n_missing"]) == (1, 1, 1, 1)
    # cache: q1 stored, q3 (error) not; a second run re-judges only q3
    j2 = ScriptedJudge(lambda p, i: _multi_reply(2, ["a"]))
    recs2 = grade_set(items, complete=j2.complete, single_prompt=SINGLE_PROMPT, multi_prompt=MULTI_PROMPT,
                      score=fake_v2_score, model="m", cache=JudgeCache(tmp_path / "c.json"), workers=1)
    assert len(j2.calls) == 3 and recs2["q1"].get("cached") and "error" not in recs2["q3"]


def test_offline_raises_on_a_miss_and_serves_hits(tmp_path):
    from panini_dr.eval.grade_set import set_cache_key
    from panini_dr.eval.judge_cache import CacheMissError

    items = [SetItem("q1", "Q1?", ["a", "b"], " a; b "), SetItem("q2", "Q2?", ["a", "b"], "  ")]
    j = ScriptedJudge(_multi_reply(2, ["a", "b"]))
    kw = dict(single_prompt=SINGLE_PROMPT, multi_prompt=MULTI_PROMPT, score=fake_v2_score, model="m")
    with pytest.raises(CacheMissError) as ei:
        grade_set(items, complete=j.complete, cache=JudgeCache(tmp_path / "c.json"), offline=True, **kw)
    assert ei.value.qids == ["q1"] and j.calls == []                 # the blank needs no verdict
    grade_set(items, complete=j.complete, cache=JudgeCache(tmp_path / "c.json"), workers=1, **kw)
    n = len(j.calls)
    recs = grade_set(items, complete=j.complete, cache=JudgeCache(tmp_path / "c.json"), offline=True, **kw)
    assert len(j.calls) == n and recs["q1"]["cached"] and recs["q2"]["blank"]
    key = set_cache_key(items[0], single_prompt=SINGLE_PROMPT, multi_prompt=MULTI_PROMPT, model="m", passes=3)
    assert JudgeCache(tmp_path / "c.json").get(key)["glen"] == 2   # the key hashes the stripped answer
