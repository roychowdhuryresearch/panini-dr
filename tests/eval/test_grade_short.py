import pytest

from panini_dr.eval.grade_short import (
    BLANK_POLICY, ShortItem, count_yes, grade_one, grade_short, majority, summarize_short,
)
from panini_dr.eval.judge_cache import JudgeCache
from tests.eval.fakes import FAKE_TEMPLATE, ScriptedJudge, fake_parse


def _votes(seq):
    """A judge whose i-th call answers seq[(i-1) % len(seq)]."""
    return ScriptedJudge(lambda prompt, i: f"extracted_final_answer: X\ncorrect: {seq[(i - 1) % len(seq)]}")


@pytest.mark.parametrize("seq,expected,votes", [
    (["yes", "yes", "yes"], True, "3/3"),
    (["yes", "yes", "no"], True, "2/3"),
    (["yes", "no", "no"], False, "1/3"),
    (["no", "no", "no"], False, "0/3"),
])
def test_majority_of_three(seq, expected, votes):
    j = _votes(seq)
    _, rec = grade_one(ShortItem("q1", "Which moon?", "Io", "Io"), complete=j.complete,
                       template=FAKE_TEMPLATE, parse=fake_parse, passes=3)
    assert rec["correct"] is expected and rec["votes"] == votes and len(j.calls) == 3


def test_count_yes_accepts_bool_and_strings():
    assert count_yes([{"correct": True}, {"correct": "yes"}, {"correct": "True"},
                      {"correct": False}, {"correct": None}, {"correct": "no"}]) == 3
    assert majority(2, 3) and not majority(1, 2) and not majority(0, 0)


def test_blank_policy_frames_scores_zero_without_a_call():
    j = _votes(["yes"])
    prompt, rec = grade_one(ShortItem("q1", "Q?", "A", "   "), complete=j.complete,
                            template=FAKE_TEMPLATE, parse=fake_parse,
                            blank_policy=BLANK_POLICY["frames"])
    assert prompt == "" and rec["correct"] is False and rec["blank"] and j.calls == []


def test_blank_policy_bcp_grades_no_answer():
    j = _votes(["no"])
    prompt, rec = grade_one(ShortItem("q1", "Q?", "A", ""), complete=j.complete,
                            template=FAKE_TEMPLATE, parse=fake_parse,
                            blank_policy=BLANK_POLICY["bcp"])
    assert "R: (no answer)" in prompt and len(j.calls) == 3 and rec["correct"] is False


def test_grade_short_uses_cache_and_rejudges_changed_answers(tmp_path):
    cache_path = tmp_path / "c.json"
    items = [ShortItem("q1", "Q1?", "Io", "Io"), ShortItem("q2", "Q2?", "Mars", "Venus")]
    j = ScriptedJudge(lambda p, i: "correct: yes" if "R: Io" in p else "correct: no")
    r1 = grade_short(items, complete=j.complete, template=FAKE_TEMPLATE, parse=fake_parse,
                     model="m", cache=JudgeCache(cache_path), workers=2)
    assert r1["q1"]["correct"] and not r1["q2"]["correct"] and len(j.calls) == 6
    j2 = ScriptedJudge(lambda p, i: "correct: yes")
    r2 = grade_short(items, complete=j2.complete, template=FAKE_TEMPLATE, parse=fake_parse,
                     model="m", cache=JudgeCache(cache_path))
    assert j2.calls == [] and r2["q2"]["correct"] is False          # served from cache
    changed = [items[0], ShortItem("q2", "Q2?", "Mars", "Mars")]
    r3 = grade_short(changed, complete=j2.complete, template=FAKE_TEMPLATE, parse=fake_parse,
                     model="m", cache=JudgeCache(cache_path))
    assert len(j2.calls) == 3 and r3["q2"]["correct"] is True         # new answer -> re-judged
    r4 = grade_short(changed, complete=j2.complete, template=FAKE_TEMPLATE, parse=fake_parse,
                     model="other-judge", cache=JudgeCache(cache_path))
    assert len(j2.calls) == 9 and all(r4[q]["correct"] for q in r4)   # new judge -> re-judged


def test_grade_short_records_errors_and_does_not_cache_them(tmp_path):
    def boom(prompt):
        raise RuntimeError("judge down")
    out = grade_short([ShortItem("q1", "Q?", "A", "A")], complete=boom, template=FAKE_TEMPLATE,
                      parse=fake_parse, model="m", cache=JudgeCache(tmp_path / "c.json"))
    assert out["q1"]["correct"] is False and "judge down" in out["q1"]["error"]
    assert not (tmp_path / "c.json").exists()


def test_summary_denominator_is_the_whole_split():
    recs = {"a": {"correct": True}, "b": {"correct": False, "blank": True},
            "c": {"correct": False, "error": "x"}}
    s = summarize_short(recs, ["a", "b", "c", "d"], buckets={"easy": ["a", "b"], "hard": ["c", "d"]})
    assert s["n"] == 4 and s["correct"] == 1 and s["accuracy"] == 0.25
    assert s["n_blank"] == 1 and s["n_errors"] == 1 and s["n_missing"] == 1
    assert s["buckets"] == {"easy": {"n": 2, "correct": 1}, "hard": {"n": 2, "correct": 0}}


def test_offline_raises_on_a_cache_miss_without_calling_the_judge(tmp_path):
    from panini_dr.eval.judge_cache import CacheMissError

    j = _votes(["yes"])
    items = [ShortItem("q1", "Q1?", "Io", "Io"), ShortItem("q2", "Q2?", "Mars", "Venus")]
    with pytest.raises(CacheMissError) as ei:
        grade_short(items, complete=j.complete, template=FAKE_TEMPLATE, parse=fake_parse, model="m",
                    cache=JudgeCache(tmp_path / "c.json"), offline=True)
    assert ei.value.qids == ["q1", "q2"] and j.calls == []
    assert not (tmp_path / "c.json").exists()                       # nothing written


def test_offline_serves_cached_verdicts_and_scores_blanks_without_a_cache(tmp_path):
    cache = JudgeCache(tmp_path / "c.json")
    items = [ShortItem("q1", "Q1?", "Io", "Io"), ShortItem("q2", "Q2?", "Mars", "  ")]
    grade_short(items[:1], complete=_votes(["yes"]).complete, template=FAKE_TEMPLATE, parse=fake_parse,
                model="m", cache=cache)
    j = _votes(["no"])
    recs = grade_short(items, complete=j.complete, template=FAKE_TEMPLATE, parse=fake_parse, model="m",
                       cache=JudgeCache(tmp_path / "c.json"), blank_policy=BLANK_POLICY["frames"],
                       offline=True)
    assert j.calls == [] and recs["q1"]["correct"] and recs["q1"]["cached"]
    assert recs["q2"]["blank"] and recs["q2"]["correct"] is False


def test_offline_bcp_blank_still_needs_a_cached_verdict(tmp_path):
    from panini_dr.eval.judge_cache import CacheMissError

    with pytest.raises(CacheMissError):
        grade_short([ShortItem("q1", "Q?", "A", "")], complete=_votes(["no"]).complete,
                    template=FAKE_TEMPLATE, parse=fake_parse, model="m",
                    blank_policy=BLANK_POLICY["bcp"], offline=True)


def test_short_cache_key_matches_what_grade_short_stores(tmp_path):
    from panini_dr.eval.grade_short import build_short_prompt, short_cache_key

    it = ShortItem("q1", "Q1?", "Io", "")
    assert "R: (no answer)" in build_short_prompt(it, FAKE_TEMPLATE, BLANK_POLICY["bcp"])
    assert build_short_prompt(it, FAKE_TEMPLATE, BLANK_POLICY["frames"]) == ""
    cache = JudgeCache(tmp_path / "c.json")
    grade_short([ShortItem("q1", "Q1?", "Io", "Io")], complete=_votes(["yes"]).complete,
                template=FAKE_TEMPLATE, parse=fake_parse, model="m", cache=cache)
    key = short_cache_key(ShortItem("q1", "Q1?", "Io", "Io"), template=FAKE_TEMPLATE, model="m",
                          passes=3, blank_policy=BLANK_POLICY["frames"])
    assert cache.get(key)["correct"] is True
