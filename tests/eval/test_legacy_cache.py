"""Importing name:qid-keyed verdicts from the research caches (all data made up)."""
import pytest

from panini_dr.eval.grade_set import SetItem, grade_set
from panini_dr.eval.grade_short import BLANK_POLICY, ShortItem, grade_short
from panini_dr.eval.judge_cache import JudgeCache
from panini_dr.eval.legacy_cache import LEGACY_PRED_CHARS, import_set, import_short
from tests.eval.fakes import FAKE_TEMPLATE, MULTI_PROMPT, SINGLE_PROMPT, ScriptedJudge, fake_parse, fake_v2_score


def _never():
    def complete(prompt):
        raise AssertionError("a judge call was made")
    return complete


def _short(legacy, items, policy="bcp", cache=None, passes=3):
    cache = cache if cache is not None else JudgeCache(None)
    rep = import_short(legacy, "run", items, template=FAKE_TEMPLATE, model="m", passes=passes,
                       blank_policy=BLANK_POLICY[policy], put=cache.put)
    return rep, cache


def test_short_import_reproduces_the_verdicts_offline():
    long_answer = "x" * (LEGACY_PRED_CHARS + 50)
    items = [ShortItem("1", "Q1?", "Io", "Io"), ShortItem("2", "Q2?", "Mars", long_answer),
             ShortItem("3", "Q3?", "Zed", "")]
    legacy = {"run:1": dict(correct=True, votes="3/3", extracted="Io", pred="Io"),
              # BCP kept only the first 200 characters of the answer
              "run:2": dict(correct=False, votes="1/3", extracted=None, pred=long_answer[:LEGACY_PRED_CHARS]),
              # BCP graded a blank answer as "(no answer)"
              "run:3": dict(correct=False, votes="0/3", extracted=None, pred="")}
    rep, cache = _short(legacy, items)
    assert rep.imported == ["1", "2", "3"] and not rep.conflicts and rep.verified == ["1", "2", "3"]
    recs = grade_short(items, complete=_never(), template=FAKE_TEMPLATE, parse=fake_parse, model="m",
                       cache=cache, blank_policy=BLANK_POLICY["bcp"], offline=True)
    assert [recs[q]["correct"] for q in "123"] == [True, False, False]
    assert recs["1"]["imported_from"] == "run:1" and recs["2"]["votes"] == "1/3"


def test_short_import_refuses_a_verdict_for_a_different_answer():
    rep, cache = _short({"run:1": dict(correct=True, votes="3/3", pred="Europa")},
                        [ShortItem("1", "Q?", "Io", "Io")])
    assert "1" in rep.conflicts and rep.imported == [] and len(cache) == 0
    rep, _ = _short({"run:1": dict(correct=True, votes="3/3")}, [ShortItem("1", "Q?", "Io", "Io")])
    assert "1" in rep.conflicts                                       # no answer stored: cannot tie it


@pytest.mark.parametrize("rec,why", [
    (dict(correct=True, votes="1/3", pred="Io"), "votes"),
    (dict(correct=True, votes="2/2", pred="Io"), "passes"),
    (dict(correct=True, votes="yes", pred="Io"), "vote string"),
])
def test_short_import_checks_votes(rec, why):
    rep, _ = _short({"run:1": rec}, [ShortItem("1", "Q?", "Io", "Io")])
    assert why in rep.conflicts["1"]


def test_short_import_frames_blanks_and_missing_verdicts():
    items = [ShortItem("1", "Q1?", "Io", "  "), ShortItem("2", "Q2?", "Io", "  "),
             ShortItem("3", "Q3?", "Io", "Io")]
    legacy = {"run:1": dict(correct=False, votes="0/0", extracted=None, pred="", blank=True)}
    rep, cache = _short(legacy, items, policy="frames")
    assert rep.imported == ["1"] and rep.blank == ["1", "2"] and rep.missing == ["3"]
    bad, _ = _short({"run:1": dict(correct=True, votes="3/3", pred="")}, items[:1], policy="frames")
    assert "blank" in bad.conflicts["1"]


def _set(legacy, items, cache=None, passes=3):
    cache = cache if cache is not None else JudgeCache(None)
    rep = import_set(legacy, "run#items", items, single_prompt=SINGLE_PROMPT, multi_prompt=MULTI_PROMPT,
                     model="m", passes=passes, put=cache.put)
    return rep, cache


def _pass(f1, credited, plen=2):
    return dict(f1=f1, precision=f1, recall=f1, credited=credited, judge_pred_len=plen, parsed=True, raw="")


def test_set_import_reproduces_max_of_passes_offline():
    items = [SetItem("m1", "Q1?", ["a", "b"], " a; b "), SetItem("m2", "Q2?", ["Paris"], "Paris"),
             SetItem("m3", "Q3?", ["a"], "   ")]
    legacy = {"run#items:m1": dict(glen=2, passes=[_pass(0.5, ["a"]), _pass(1.0, ["a", "b"]), _pass(0.0, [])]),
              "run#items:m2": dict(glen=1, passes=[_pass(1.0, ["Paris"], 1)] * 3)}
    rep, cache = _set(legacy, items)
    assert rep.imported == ["m1", "m2"] and rep.blank == ["m3"] and not rep.conflicts
    assert rep.verified == ["m2"] and rep.unverified == ["m1"]       # only a single answer echoes the answer
    recs = grade_set(items, complete=_never(), single_prompt=SINGLE_PROMPT, multi_prompt=MULTI_PROMPT,
                     score=fake_v2_score, model="m", cache=cache, offline=True)
    from panini_dr.eval.grade_set import summarize_set
    s = summarize_set(recs, ["m1", "m2", "m3"])
    assert s["f1"] == pytest.approx((1.0 + 1.0 + 0.0) / 3) and s["n_blank"] == 1


@pytest.mark.parametrize("rec,why", [
    (dict(glen=1, passes=[_pass(1.0, ["Lyon"], 1)] * 3), "credited"),
    (dict(glen=2, passes=[_pass(1.0, ["Paris"], 1)] * 3), "gold length"),
    (dict(glen=1, passes=[_pass(1.0, ["Paris"], 1)] * 2), "passes"),
])
def test_set_import_conflicts(rec, why):
    rep, cache = _set({"run#items:m2": rec}, [SetItem("m2", "Q2?", ["Paris"], "Paris")])
    assert why in rep.conflicts["m2"] and len(cache) == 0


def test_imported_and_freshly_judged_verdicts_share_keys(tmp_path):
    """A verdict the grader stores itself and the same verdict imported land on one key."""
    item = ShortItem("1", "Q?", "Io", "Io")
    j = ScriptedJudge("correct: yes")
    fresh = JudgeCache(tmp_path / "fresh.json")
    grade_short([item], complete=j.complete, template=FAKE_TEMPLATE, parse=fake_parse, model="m", cache=fresh)
    fresh.save()
    _, imported = _short({"run:1": dict(correct=True, votes="3/3", extracted=None, pred="Io")}, [item])
    assert set(JudgeCache(tmp_path / "fresh.json")._data) == set(imported._data)
