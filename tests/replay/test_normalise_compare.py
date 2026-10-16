"""Normalisation rules and the strict run comparison."""

from __future__ import annotations

import copy
import json

from panini_dr.replay.compare import compare_cells, replay_gate
from panini_dr.replay.normalise import DROP_KEYS, RULES, Normaliser, diff_paths, rules


def test_wall_clock_keys_are_dropped_at_any_depth():
    n = Normaliser()
    doc = {"wall_time_s": 3.2, "a": {"duration_s": 1.0, "keep": 1, "b": [{"started_at": "x", "v": 2}]}}
    assert n(doc) == {"a": {"keep": 1, "b": [{"v": 2}]}}
    assert set(DROP_KEYS) == {"wall_time_s", "mean_wall_time_s", "duration_s", "started_at", "completed_at",
                              "build_timestamp"}
    assert len(RULES) == len(DROP_KEYS) + 4


def test_paths_and_run_tag_become_placeholders_longest_first():
    n = Normaliser([("/r/out", "<OUT_DIR>"), ("/r/out/sys__m__20261016_021518", "<RUN_DIR>")])
    s = "log at /r/out/sys__m__20261016_021518/run.log, cells in /r/out; tag sys__m__20991231_235959"
    assert n(s) == "log at <RUN_DIR>/run.log, cells in <OUT_DIR>; tag sys__m__<RUN_TAG>"


def test_json_inside_strings_is_normalised_too():
    n = Normaliser()
    s = json.dumps({"duration_s": 2.5, "hits": [1, 2]})
    assert n({"result_full": s}) == {"result_full": {"__json__": {"hits": [1, 2]}}}
    assert n("{not json") == "{not json"


def test_markdown_masks_only_the_rendered_durations():
    n = Normaliser()
    md = "| Wall time | 136.30s |\n### Tool call 1: `search`  _(took 1.61s)_\nanswer: 12.5s"
    assert n.md(md) == ["| Wall time | <WALL> |", "### Tool call 1: `search`  _(took <DURATION>)_", "answer: 12.5s"]


def test_normalisation_is_idempotent():
    n = Normaliser([("/r/run", "<RUN_DIR>")], multiset=False)
    doc = {"wall_time_s": 1, "x": "/r/run/a", "tc": [{"result_full": json.dumps({"duration_s": 1, "k": "[1]"})}],
           "gsw_checker_calls": [{"kind": "b"}, {"kind": "a"}]}
    once = n(doc)
    assert Normaliser([], multiset=False)(once) == once
    md = n.md("| Wall time | 3.0s |\n_(took 2s)_ at /r/run")
    assert Normaliser().md("\n".join(md) + "\n") == md


def test_strict_mode_keeps_list_order_multiset_mode_does_not():
    a = {"gsw_checker_calls": [{"kind": "resolver"}, {"kind": "verifier"}]}
    b = {"gsw_checker_calls": [{"kind": "verifier"}, {"kind": "resolver"}]}
    strict = Normaliser(multiset=False)
    loose = Normaliser(multiset=True)
    assert diff_paths(strict(a), strict(b))
    assert not diff_paths(loose(a), loose(b))
    assert any(r.startswith("(DISABLED") for r in rules(False)) and not any(r.startswith("(DISABLED") for r in rules(True))


def test_diff_paths_reports_json_paths():
    d = diff_paths({"a": [1, {"b": "x"}], "c": 1}, {"a": [1, {"b": "y"}], "d": 1}, "t")
    assert d == ["t.a[1].b: 'x' != 'y'", "t.c: missing on right", "t.d: missing on left"]


# ---------------------------------------------------------------------------
# compare_cells
# ---------------------------------------------------------------------------


def _cell(root, answers, *, wall=1.0, checker_order=("resolver", "verifier")):
    root.mkdir(parents=True)
    (root / "traces").mkdir()
    qs = []
    for qid, ans in answers.items():
        traj = {"question_id": qid, "turns": 2, "tool_calls": [{"name": "search", "duration_s": wall}],
                "wall_time_s": wall, "final_answer": ans,
                "extra": {"stopped_reason": "finished", "gsw_checker_calls": [{"kind": k} for k in checker_order]}}
        (root / "traces" / f"q_{qid}.json").write_text(json.dumps(traj))
        (root / "traces" / f"q_{qid}.md").write_text(f"# q{qid}\n| Wall time | {wall}s |\nanswer {ans}\n")
        qs.append({"question_id": qid, "question": "?", "predicted_answer": ans, "f1": 1.0, "trajectory": traj})
    (root / "cell_result.json").write_text(json.dumps(
        {"system_id": "s", "config": {"k": 1}, "questions": qs, "mean_wall_time_s": wall, "started_at": str(wall)}))
    return root


def test_compare_passes_when_only_ignored_fields_differ(tmp_path):
    L = _cell(tmp_path / "L", {"1": "Veltravia", "2": "Lindqvar"}, wall=1.0)
    R = _cell(tmp_path / "R", {"1": "Veltravia", "2": "Lindqvar"}, wall=7.5)
    rep = compare_cells(L, R, Normaliser(multiset=False), Normaliser(multiset=False))
    assert rep["verdict"] == "PASS" and rep["cell"]["equal"]
    assert all(r["pass"] for r in rep["questions"].values())


def test_compare_fails_on_a_changed_answer_and_on_reordered_lists(tmp_path):
    L = _cell(tmp_path / "L", {"1": "Veltravia", "2": "Lindqvar"})
    R = _cell(tmp_path / "R", {"1": "Veltravia", "2": "Morrowgate"})
    rep = compare_cells(L, R, Normaliser(multiset=False), Normaliser(multiset=False))
    assert rep["verdict"] == "FAIL"
    q2 = rep["questions"]["2"]
    assert not q2["answer_equal"] and not q2["trace_equal"] and not q2["md_equal"] and rep["questions"]["1"]["pass"]
    assert any("predicted_answer" in d for d in q2["diffs"])
    R2 = _cell(tmp_path / "R2", {"1": "Veltravia", "2": "Lindqvar"}, checker_order=("verifier", "resolver"))
    assert compare_cells(L, R2, Normaliser(multiset=False), Normaliser(multiset=False))["verdict"] == "FAIL"
    assert compare_cells(L, R2, Normaliser(multiset=True), Normaliser(multiset=True))["verdict"] == "PASS"


def test_compare_fails_on_a_missing_question_or_trace(tmp_path):
    L = _cell(tmp_path / "L", {"1": "a", "2": "b"})
    R = _cell(tmp_path / "R", {"1": "a"})
    rep = compare_cells(L, R, Normaliser(), Normaliser())
    assert rep["questions"]["2"] == {"pass": False, "why": "missing on right"} and rep["verdict"] == "FAIL"
    R2 = _cell(tmp_path / "R2", {"1": "a", "2": "b"})
    (R2 / "traces" / "q_2.json").unlink()
    assert not compare_cells(L, R2, Normaliser(), Normaliser())["questions"]["2"]["trace_equal"]


def test_replay_gate_needs_zero_misses_leftovers_and_leaks():
    ok = {"n_misses": 0, "n_leftover": 0, "leaks": [], "pick_stats": {"exact_lineage_seq": 3}}
    assert replay_gate(ok)["ok"]
    for bad in ({"n_misses": 1}, {"n_leftover": 2}, {"leaks": [{"address": "x"}]}):
        assert not replay_gate({**copy.deepcopy(ok), **bad})["ok"]


def test_long_diff_lines_are_cut_around_the_change():
    from panini_dr.replay.verify import clip_diff

    a = "-    " + "x" * 500 + "francolin" + "y" * 300
    b = "+    " + "x" * 500 + "francolim" + "y" * 300
    out = clip_diff(["--- recorded", "+++ replay", a, b, " ctx"])
    assert "francolin" in out[2] and "francolim" in out[3] and out[2].startswith("-...")
    assert all(len(x) < 220 for x in out) and out[:2] == ["--- recorded", "+++ replay"]
