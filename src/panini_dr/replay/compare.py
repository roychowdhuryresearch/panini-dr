"""Compare two run directories (e.g. a reference recording and a replay) and say PASS/FAIL.

    python -m panini_dr.replay.compare LEFT RIGHT [--src-root PATH ...] [--ignore-key KEY ...]
                                       [--map OLD=NEW ...] [--replay-summary SUMMARY.json ...]
                                       [--multiset] [--json OUT.json]

LEFT / RIGHT are either a run --output-dir (containing exactly one <system>__<model>__<tag>/ cell
directory) or the cell directory itself (a verify bundle's ``reference/<case>/`` is one).  PASS
requires, for every question:
  * identical final answer (trajectory.final_answer AND predicted_answer, compared raw -- no
    normalisation is ever applied to answers),
  * identical normalised trace (traces/q_<id>.json), identical embedded trajectory and identical
    markdown rendering,
and identical normalised cell_result.json (config + aggregates, trajectories excluded).
Optionally (--replay-summary) also requires zero cassette misses / leaks / leftovers.
Comparison is STRICT by default (list order always matters); --multiset turns on the
UNORDERED_LISTS rule of :mod:`panini_dr.replay.normalise`.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

from panini_dr.replay.normalise import Normaliser, diff_paths, rules


def cell_dir(p: str | Path) -> Path:
    p = Path(p)
    if (p / "cell_result.json").exists():
        return p
    cells = [Path(c).parent for c in glob.glob(str(p / "*" / "cell_result.json"))]
    if len(cells) != 1:
        raise FileNotFoundError(f"{p}: expected exactly one cell dir with cell_result.json, found {len(cells)}")
    return cells[0]


def _load(p: Path) -> Any:
    return json.loads(p.read_text())


def compare_cells(left: Path, right: Path, nl: Normaliser, nr: Normaliser, *, limit: int = 12) -> dict[str, Any]:
    """Compare two cell directories; ``nl`` / ``nr`` normalise each side."""
    L, R = Path(left), Path(right)
    report: dict[str, Any] = {"left": str(L), "right": str(R), "questions": {}, "cell": {}}
    ok = True

    cl, cr = _load(L / "cell_result.json"), _load(R / "cell_result.json")
    ql = {q["question_id"]: q for q in cl.get("questions", [])}
    qr = {q["question_id"]: q for q in cr.get("questions", [])}

    def strip(c: dict) -> dict:
        return {k: v for k, v in c.items() if k != "questions"}

    cd = diff_paths(nl(strip(cl)), nr(strip(cr)), "cell", limit=limit)
    report["cell"] = {"equal": not cd, "diffs": cd}
    ok &= not cd

    ids = list(dict.fromkeys([*ql, *qr]))
    for qid in ids:
        r: dict[str, Any] = {}
        if qid not in ql or qid not in qr:
            r = {"pass": False, "why": "missing on " + ("left" if qid not in ql else "right")}
            report["questions"][qid] = r
            ok = False
            continue
        a_ans, b_ans = ql[qid].get("predicted_answer"), qr[qid].get("predicted_answer")
        a_fin, b_fin = ql[qid]["trajectory"].get("final_answer"), qr[qid]["trajectory"].get("final_answer")
        r["answer_left"], r["answer_right"] = a_ans, b_ans
        r["answer_equal"] = (a_ans == b_ans) and (a_fin == b_fin)
        # QuestionResult (scores etc.) without the trajectory
        qd = diff_paths(nl({k: v for k, v in ql[qid].items() if k != "trajectory"}),
                        nr({k: v for k, v in qr[qid].items() if k != "trajectory"}), "result", limit=limit)
        tl, tr = L / "traces" / f"q_{qid}.json", R / "traces" / f"q_{qid}.json"
        if tl.exists() and tr.exists():
            td = diff_paths(nl(_load(tl)), nr(_load(tr)), "trace", limit=limit)
        else:
            td = [f"trace file missing: left={tl.exists()} right={tr.exists()}"]
        # the embedded trajectory in cell_result must equal the trace file content as well
        ed = diff_paths(nl(ql[qid]["trajectory"]), nr(qr[qid]["trajectory"]), "cell.trajectory", limit=limit)
        ml, mr = L / "traces" / f"q_{qid}.md", R / "traces" / f"q_{qid}.md"
        md_equal = (not ml.exists() and not mr.exists()) or (
            ml.exists() and mr.exists() and nl.md(ml.read_text()) == nr.md(mr.read_text()))
        ex = ql[qid]["trajectory"].get("extra", {})
        r.update({
            "trace_equal": not td and not ed, "result_equal": not qd, "md_equal": md_equal,
            "diffs": (qd + td + ed)[:limit],
            "stopped_reason": ex.get("stopped_reason"),
            "turns": ql[qid]["trajectory"].get("turns"),
            "tool_calls": len(ql[qid]["trajectory"].get("tool_calls", [])),
        })
        r["pass"] = r["answer_equal"] and r["trace_equal"] and r["result_equal"] and md_equal
        ok &= r["pass"]
        report["questions"][qid] = r
    report["verdict"] = "PASS" if ok else "FAIL"
    return report


def replay_gate(summary: dict[str, Any], name: str = "") -> dict[str, Any]:
    """The cassette side of a verdict: no misses, no leftover recorded calls, no leaks."""
    g = {"summary": name, "misses": summary.get("n_misses"), "leftover": summary.get("n_leftover"),
         "leaks": len(summary.get("leaks") or []), "picks": summary.get("pick_stats")}
    g["ok"] = g["misses"] == 0 and g["leftover"] == 0 and g["leaks"] == 0
    return g


def question_line(qid: str, r: dict[str, Any]) -> str:
    if "why" in r:
        return f"  q{qid}: FAIL ({r['why']})"
    return (f"  q{qid}: {'PASS' if r['pass'] else 'FAIL'} answer={'same' if r['answer_equal'] else 'DIFFERENT'} "
            f"trace={'same' if r['trace_equal'] else 'DIFFERENT'} md={'same' if r['md_equal'] else 'DIFFERENT'} "
            f"turns={r['turns']} tool_calls={r['tool_calls']} stop={r['stopped_reason']} | answer={r['answer_left']!r}"
            + ("" if r["answer_equal"] else f" vs {r['answer_right']!r}"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare two run directories (strict by default).")
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("--src-root", action="append", default=[], help="source tree roots to mask as <SRC>")
    ap.add_argument("--ignore-key", action="append", default=[], help="extra key to drop (documented as user rule)")
    ap.add_argument("--map", action="append", default=[], help="OLD=NEW literal string map applied to both sides")
    ap.add_argument("--replay-summary", action="append", default=[], help="cassette replay summary.json to gate on")
    ap.add_argument("--json", default="", help="write the full report as JSON here")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--multiset", action="store_true",
                    help="compare UNORDERED_LISTS as multisets (default: strict order)")
    a = ap.parse_args(argv)

    L, R = cell_dir(a.left), cell_dir(a.right)
    maps = [tuple(m.split("=", 1)) for m in a.map]

    def norm_for(cdir: Path) -> Normaliser:
        reps = [(str(cdir), "<RUN_DIR>"), (str(cdir.parent), "<OUT_DIR>")]
        reps += [(r, "<SRC>") for r in a.src_root]
        return Normaliser(reps, a.ignore_key, maps, multiset=a.multiset)

    report = compare_cells(L, R, norm_for(L), norm_for(R), limit=a.limit)
    report["rules"] = rules(a.multiset)
    gates = [replay_gate(_load(Path(sp)), sp) for sp in a.replay_summary]
    report["replay_gates"] = gates
    if any(not g["ok"] for g in gates):
        report["verdict"] = "FAIL"

    print(f"LEFT  {L}\nRIGHT {R}")
    print("normalisation rules:")
    for rule in report["rules"]:
        print(f"  - {rule}")
    print(f"cell_result (config + aggregates): {'equal' if report['cell']['equal'] else 'DIFFERENT'}")
    for d in report["cell"]["diffs"]:
        print(f"    {d}")
    for qid, r in report["questions"].items():
        print(question_line(qid, r))
        for d in r.get("diffs", []):
            print(f"      {d}")
    for g in gates:
        print(f"  replay gate {g['summary']}: misses={g['misses']} leftover={g['leftover']} leaks={g['leaks']} "
              f"picks={g['picks']} -> {'ok' if g['ok'] else 'FAIL'}")
    print(report["verdict"])
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["cell_dir", "compare_cells", "main", "question_line", "replay_gate"]
