"""Live investigation-memory tests.

(1) EVENT API: the incremental builders (open_episode/add_*) produce an Episode
    that folds identically to a hand-built one.
(2) RETIRE: a plan-update retirement removes actors from ranking but keeps history.
(3) Final shot, scoped budgets and the prior-investigation dossier.

The research repo's parity test against an offline reference implementation
(new_feature/ reconcile.replay over recorded traces) is not ported: its
fixtures are run traces and are not shipped.
"""
from __future__ import annotations

from panini_dr.adapters.ours._investigation_memory import (
    InvestigationMemory,
)
from panini_dr.config import GSW_OMIT_IF_FALSY


def test_event_builders_fold_like_handbuilt():
    plan = {"verb_phrases": [{"id": "vp1", "subject_id": "b_x", "object_id": "e1"}],
            "entities": [{"id": "b_x", "is_target": True}]}
    im = InvestigationMemory(plan)
    ep = im.open_episode(1, "researcher", ["b_x"])
    im.add_attempt(ep, "search", ["c1", "c2"])
    im.add_proposal(ep, "b_x", "Hoover Dam", ["c1"], verdict="accepted", reject_reason=None)
    im.fold(ep)
    im.close_round(1)
    d = im.dump()
    assert d["n_actors"] == 1
    assert d["official_round"] == {"b_x": 1}
    actor = d["actors"][0]
    assert actor["value"] == "Hoover Dam" and actor["real_ev"] and actor["accepted"]


def test_finalizer_prefers_accepted_over_rejected():
    """q227: an accepted, zero-rejection candidate must beat a rejected one even
    if the rejected one was touched more recently."""
    plan = {"verb_phrases": [], "entities": [{"id": "t", "is_target": True}]}
    im = InvestigationMemory(plan)
    # round 1: "Matsuyama" accepted (real evidence, 0 rejections)
    ep = im.open_episode(1, "researcher", ["t"])
    im.add_proposal(ep, "t", "Matsuyama", ["c1"], verdict="accepted", reject_reason=None)
    im.fold(ep); im.close_round(1)
    # round 2 (later): "Nick Taylor" real-evidence but REJECTED
    ep2 = im.open_episode(2, "researcher", ["t"])
    im.add_proposal(ep2, "t", "Nick Taylor", ["c2"], verdict="rejected", reject_reason="wrong")
    im.fold(ep2); im.close_round(2)
    assert im.finalize("t").value == "Matsuyama"             # not the recent rejected one


def test_finalizer_prefers_fewer_rejections():
    """q537: neither accepted, but the 1x-rejected gold beats the 9x-rejected value."""
    plan = {"verb_phrases": [], "entities": [{"id": "t", "is_target": True}]}
    im = InvestigationMemory(plan)
    for i in range(9):                                        # "The Rock" rejected 9x
        ep = im.open_episode(i + 1, "researcher", ["t"])
        im.add_proposal(ep, "t", "The Rock", [f"c{i}"], verdict="rejected", reject_reason="x")
        im.fold(ep); im.close_round(i + 1)
    ep = im.open_episode(10, "researcher", ["t"])            # "Crimson Tide" rejected 1x
    im.add_proposal(ep, "t", "Crimson Tide", ["cg"], verdict="rejected", reject_reason="x")
    im.fold(ep); im.close_round(10)
    assert im.finalize("t").value == "Crimson Tide"


def test_finalizer_g3_real_evidence_only():
    plan = {"verb_phrases": [], "entities": [{"id": "t", "is_target": True}]}
    # real-evidence rejected actor still finalizes (G3 = has real evidence)
    im = InvestigationMemory(plan)
    ep = im.open_episode(1, "researcher", ["t"])
    im.add_proposal(ep, "t", "Hoover Dam", ["c1"], verdict="rejected", reject_reason="weak")
    im.fold(ep); im.close_round(1)
    assert im.finalize("t").value == "Hoover Dam"
    # sentinel-only -> no real evidence -> G3 blocks
    im2 = InvestigationMemory(plan)
    ep2 = im2.open_episode(1, "researcher", ["t"])
    im2.add_proposal(ep2, "t", "Guess", ["insufficient_evidence"], verdict="rejected", reject_reason=None)
    im2.fold(ep2); im2.close_round(1)
    assert im2.finalize("t") is None


def test_retire_excludes_from_ranking_keeps_history():
    plan = {"verb_phrases": [], "entities": [{"id": "b_y", "is_target": True}]}
    im = InvestigationMemory(plan)
    ep = im.open_episode(1, "researcher", ["b_y"])
    im.add_proposal(ep, "b_y", "Paris", ["c9"], verdict="accepted", reject_reason=None)
    im.fold(ep)
    im.close_round(1)
    assert im.mem.finalize("b_y") is not None       # rankable before retire
    im.retire(2, ["b_y"])
    assert im.mem.finalize("b_y") is None            # excluded from ranking
    assert im.dump()["n_actors"] == 1               # but kept in history
    assert im.dump()["retirements"] == [(2, ["b_y"])]


# --- Question-level clock tests (Task 1, sweep V1_R5) ---
def _quiet_round(im, rnd, blank="b_q"):
    """One worked round with zero progress (search that yields nothing new)."""
    ep = im.open_episode(rnd, "researcher", [blank])
    im.add_attempt(ep, "search", ["c_same"])          # same chunk every round
    im.fold(ep); im.close_round(rnd)


def _progress_round(im, rnd, blank="b_q", value="V", chunk="c_new"):
    ep = im.open_episode(rnd, "researcher", [blank])
    im.add_proposal(ep, blank, value, [chunk], verdict="rejected", reject_reason="x")
    im.fold(ep); im.close_round(rnd)


def test_clock_fires_after_5_quiet_rounds_first_round_exempt():
    from panini_dr.adapters.ours._investigation_memory import R_Q
    assert R_Q == 5
    plan = {"verb_phrases": [], "entities": [{"id": "t", "is_target": True}]}
    im = InvestigationMemory(plan)
    _quiet_round(im, 1)                                # exempt (A4 analog)
    for r in range(2, 6):                              # rounds 2-5: streak 1..4
        _quiet_round(im, r)
        assert im.should_terminate() is None, f"fired early at r{r}"
    _quiet_round(im, 6)                                # streak 5
    assert im.should_terminate() is not None


def test_clock_resets_on_progress_anywhere():
    plan = {"verb_phrases": [], "entities": [{"id": "t", "is_target": True}]}
    im = InvestigationMemory(plan)
    _quiet_round(im, 1)
    for r in range(2, 6):
        _quiet_round(im, r)
    # round 6: a NEW real-evidence proposal on a DIFFERENT blank resets the clock
    # (the actor guard would also suppress; retire the actor's acceptance state is
    # irrelevant here — the point is streak reset, so use an accepted actor whose
    # blank then has official_round set, clearing the guard).
    ep = im.open_episode(6, "researcher", ["b_other"])
    im.add_proposal(ep, "b_other", "X", ["c9"], verdict="accepted", reject_reason=None)
    im.fold(ep); im.close_round(6)
    assert im.should_terminate() is None
    for r in range(7, 12):                             # 5 fresh quiet rounds
        _quiet_round(im, r)
    assert im.should_terminate() is not None


def test_clock_guard_suppresses_while_live_unaccepted_evidence():
    plan = {"verb_phrases": [], "entities": [{"id": "t", "is_target": True}]}
    im = InvestigationMemory(plan)
    # round 1: real-evidence candidate proposed but REJECTED -> live, unaccepted
    _progress_round(im, 1, blank="b_live", value="Cand", chunk="c1")
    for r in range(2, 9):
        _quiet_round(im, r)
    assert im.should_terminate() is None               # guard holds (live evidence)
    # acceptance clears the guard -> next quiet rounds can fire
    ep = im.open_episode(9, "researcher", ["b_live"])
    im.add_proposal(ep, "b_live", "Cand", ["c1"], verdict="accepted", reject_reason=None)
    im.fold(ep); im.close_round(9)                     # progress: resets streak too
    for r in range(10, 15):
        _quiet_round(im, r)
    assert im.should_terminate() is not None


def test_read_log_records_read_chunks_in_fold_order():
    plan = {"verb_phrases": [], "entities": [{"id": "t", "is_target": True}]}
    im = InvestigationMemory(plan)
    ep = im.open_episode(1, "researcher", ["t"])
    im.add_attempt(ep, "search", ["A__0", "B__0"])     # searches NOT logged
    im.add_attempt(ep, "read", ["A__0"])
    im.add_attempt(ep, "read", ["B__0"])
    im.fold(ep); im.close_round(1)
    ep2 = im.open_episode(2, "researcher", ["t"])
    im.add_attempt(ep2, "read", ["A__0"])              # duplicates kept (order matters)
    im.fold(ep2); im.close_round(2)
    assert im.read_log == ["A__0", "B__0", "A__0"]


def test_dump_carries_clock_state():
    plan = {"verb_phrases": [], "entities": [{"id": "t", "is_target": True}]}
    im = InvestigationMemory(plan)
    _quiet_round(im, 1)
    d = im.dump()
    assert d["terminated"] is None
    assert d["q_rounds_worked"] == [1]


def test_final_shot_prompt_builder_contents():
    from panini_dr.adapters.ours._investigation_memory import (
        build_final_shot_prompt,
    )
    msgs = build_final_shot_prompt(
        question="Which member turned 5 soonest?",
        resolved={"b_album": ("BoA's first album", "released_on", "2000-08-25")},
        candidates=["- t: candidate 'Jimin' (evidence=True, rejected 1x)"],
        articles=[("BTS", "BTS text here"), ("Jimin", "Jimin text here")],
    )
    assert msgs[0]["role"] == "system" and "ANSWER:" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "Which member turned 5 soonest?" in user
    assert "2000-08-25" in user and "released_on" in user
    assert "Jimin text here" in user and "### ARTICLE: BTS" in user
    assert "rejected 1x" in user


def test_final_shot_answer_parser():
    from panini_dr.adapters.ours._investigation_memory import (
        parse_final_shot_answer,
    )
    assert parse_final_shot_answer("thinking...\nANSWER: Jimin") == "Jimin"
    assert parse_final_shot_answer("ANSWER: NONE") == ""
    assert parse_final_shot_answer("ANSWER:  none ") == ""


def test_final_shot_answer_survives_a_missing_sentinel():
    """The final shot is the never-blank hook, and it dropped answers whenever
    the model forgot the 'ANSWER:' line. Measured on the 200Q run: q802 spent
    14,196 prompt tokens and wrote 552 completion tokens, and the parse returned
    "" — an empty prediction scores 0, so a dropped answer is a pure loss."""
    from panini_dr.adapters.ours._investigation_memory import (
        parse_final_shot_answer,
    )
    # the whole closing line, not a guessed substring — the judge reads prose
    assert parse_final_shot_answer(
        "The album came out in 2004.\nSo the count is 117.") == "So the count is 117."
    # an explicit refusal is still a refusal, sentinel or not
    assert parse_final_shot_answer("I cannot determine this.\nNONE") == ""
    assert parse_final_shot_answer("") == ""
    assert parse_final_shot_answer("   \n\n  ") == ""
    # the sentinel still wins when present, even with text after it
    assert parse_final_shot_answer("ANSWER: Jimin\n(from the roster)") == "Jimin"


def test_finalshot_flag_omitted_when_false():
    assert "gsw_investigation_memory_finalshot" in GSW_OMIT_IF_FALSY


def test_query_log_captures_normalized_search_queries_per_blank():
    plan = {"verb_phrases": [], "entities": [{"id": "t", "is_target": True}]}
    im = InvestigationMemory(plan)
    ep = im.open_episode(1, "researcher", ["b_x"])
    im.add_attempt(ep, "search", ["c1"], query="  BoA First Album ")
    im.add_attempt(ep, "search", ["c2"], query="boa first album")   # dup after norm
    im.add_attempt(ep, "read", ["c1"])                              # reads: no query
    im.fold(ep); im.close_round(1)
    assert im.query_log == {"b_x": ["boa first album"]}


def test_memory_dossier_flag_omitted_when_false():
    assert "gsw_memory_dossier" in GSW_OMIT_IF_FALSY


def test_memory_scoped_dispatch_flag_omitted_when_false():
    assert "gsw_memory_scoped_dispatch" in GSW_OMIT_IF_FALSY


# --- scoped_budget: Actor model -> code-enforced researcher turn budget ----------

def test_scoped_budget_exhausted_blank_gets_tiny_budget():
    from panini_dr.adapters.ours._investigation_memory import scoped_budget

    im = InvestigationMemory({"verb_phrases": [], "entities": [{"id": "t", "is_target": True}]})
    im.mem.dry_hit_at["b_x"] = 2

    assert scoped_budget(im, "b_x", 20) == 3


def test_scoped_budget_live_candidate_missing_link_gets_verify_budget():
    from panini_dr.adapters.ours._investigation_memory import scoped_budget

    plan = {"verb_phrases": [{"id": "vp1", "subject_id": "b_x", "object_id": "e1"},
                             {"id": "vp2", "subject_id": "b_x", "object_id": "e2"}],
            "entities": [{"id": "b_x", "is_target": False}]}
    im = InvestigationMemory(plan)
    ep = im.open_episode(1, "researcher", ["b_x"])
    im.add_proposal(ep, "b_x", "CandidateValue", ["c_ev"], verdict="rejected",
                    reject_reason="needs vp2")
    im.add_binding(ep, "b_x", "vp1", ["c_ev"], ok=True)
    im.fold(ep); im.close_round(1)

    assert scoped_budget(im, "b_x", 20) == 6


def test_scoped_budget_fresh_blank_gets_full_budget():
    from panini_dr.adapters.ours._investigation_memory import scoped_budget

    im = InvestigationMemory({"verb_phrases": [], "entities": [{"id": "t", "is_target": True}]})

    assert scoped_budget(im, "b_unseen", 20) == 20


def test_scoped_budget_malformed_facade_gets_full_budget_never_raises():
    from panini_dr.adapters.ours._investigation_memory import scoped_budget

    class BadFacade:
        @property
        def mem(self):
            raise RuntimeError("bad memory")

    assert scoped_budget(BadFacade(), "b_x", 20) == 20


# --- render_dossier: Actor model -> anti-anchoring dossier (design 2026-07-20) ----

class _FakeChunk:
    def __init__(self, text): self.text = text

class _FakeCorpus:
    def __init__(self, texts): self._t = texts
    def get_chunk(self, cid):
        return _FakeChunk(self._t[cid]) if cid in self._t else None

def _im_with(plan):
    return InvestigationMemory(plan)

def test_render_dossier_none_when_empty():
    from panini_dr.adapters.ours._investigation_memory import render_dossier
    im = _im_with({"verb_phrases": [], "entities": [{"id": "t", "is_target": True}]})
    assert render_dossier(im, "b_unseen", _FakeCorpus({})) is None

def test_render_dossier_sections_and_anti_anchoring():
    from panini_dr.adapters.ours._investigation_memory import render_dossier
    plan = {"verb_phrases": [{"id": "vp1", "subject_id": "b_x", "object_id": "e1"},
                             {"id": "vp2", "subject_id": "b_x", "object_id": "e2"}],
            "entities": [{"id": "b_x", "is_target": False}]}
    im = _im_with(plan)
    ep = im.open_episode(1, "researcher", ["b_x"])
    im.add_attempt(ep, "search", ["c_old"], query="jhope age")
    # a live candidate: real evidence on vp1, vp2 still missing, NOT accepted.
    # (Step-1 note: verdict="accepted" would set accepted=True, which EXCLUDES the
    # actor from OPEN per the spec — use rejected+real-evidence, as in the third
    # test, so this is genuinely the live OPEN candidate.)
    im.add_proposal(ep, "b_x", "1994-02-18", ["c_ev"], verdict="rejected",
                    reject_reason="vp2 still unconfirmed")
    im.add_binding(ep, "b_x", "vp1", ["c_ev"], ok=True)
    # a ruled-out candidate
    im.add_proposal(ep, "b_x", "1999-09-09", ["insufficient_evidence"],
                    verdict="rejected", reject_reason="citation was not grounded")
    im.fold(ep); im.close_round(1)
    corpus = _FakeCorpus({"c_ev": "J-Hope was born February 18, 1994 in Gwangju."})
    txt = render_dossier(im, "b_x", corpus)
    assert txt is not None
    # anti-anchoring invariants
    low = txt.lower()
    assert "leading candidate" not in low and "best answer" not in low
    assert "the answer is" not in low
    # D1: a sentinel-cited value (reject reason "citation was not grounded") is
    # UNGROUNDED — a want-of-citation, not a factual refutation — so it appears in the
    # UNGROUNDED section WITHOUT the format-failure reason (no longer false-RULED-OUT).
    assert "1999-09-09" in txt and "UNGROUNDED" in txt
    assert "citation was not grounded" not in txt
    # tried query present as a fact, not a prohibition
    assert "jhope age" in txt.lower()
    # retrieved evidence text inline (hybrid materials)
    assert "born February 18, 1994" in txt

def test_render_dossier_live_candidate_only_instrumental():
    from panini_dr.adapters.ours._investigation_memory import render_dossier
    plan = {"verb_phrases": [{"id": "vp1", "subject_id": "b_x", "object_id": "e1"},
                             {"id": "vp2", "subject_id": "b_x", "object_id": "e2"}],
            "entities": [{"id": "b_x", "is_target": False}]}
    im = _im_with(plan)
    ep = im.open_episode(1, "researcher", ["b_x"])
    # real evidence, NOT accepted -> live candidate; vp2 missing
    im.add_proposal(ep, "b_x", "CandidateValue", ["c_ev"], verdict="rejected",
                    reject_reason="needs vp2")
    im.add_binding(ep, "b_x", "vp1", ["c_ev"], ok=True)
    im.fold(ep); im.close_round(1)
    txt = render_dossier(im, "b_x", _FakeCorpus({"c_ev": "evidence text"}))
    assert txt is not None and "CandidateValue" in txt
    # every line mentioning the candidate value must be an OPEN/instrumental line
    for line in txt.splitlines():
        if "CandidateValue" in line:
            assert ("FOR OR AGAINST" in line) or ("confirm or refute" in line)

def test_render_dossier_exhausted_open_says_stop():
    from panini_dr.adapters.ours._investigation_memory import render_dossier
    plan = {"verb_phrases": [{"id": "vp1", "subject_id": "b_x", "object_id": "e1"},
                             {"id": "vp2", "subject_id": "b_x", "object_id": "e2"}],
            "entities": [{"id": "b_x", "is_target": False}]}
    im = _im_with(plan)
    ep = im.open_episode(1, "researcher", ["b_x"])
    im.add_proposal(ep, "b_x", "CandidateValue", ["c_ev"], verdict="rejected",
                    reject_reason="needs vp2")
    im.add_binding(ep, "b_x", "vp1", ["c_ev"], ok=True)
    im.fold(ep); im.close_round(1)
    im.mem.dry_hit_at["b_x"] = 2
    im.mem.dry_streak["b_x"] = 7

    txt = render_dossier(im, "b_x", _FakeCorpus({"c_ev": "evidence text"}))

    assert txt is not None
    assert "search-exhausted" in txt
    assert "do NOT keep searching" in txt
    for line in txt.splitlines():
        if "CandidateValue" in line:
            assert "confirm or refute" in line

def test_render_dossier_not_exhausted_open_unchanged():
    from panini_dr.adapters.ours._investigation_memory import render_dossier
    plan = {"verb_phrases": [{"id": "vp1", "subject_id": "b_x", "object_id": "e1"},
                             {"id": "vp2", "subject_id": "b_x", "object_id": "e2"}],
            "entities": [{"id": "b_x", "is_target": False}]}
    im = _im_with(plan)
    ep = im.open_episode(1, "researcher", ["b_x"])
    im.add_proposal(ep, "b_x", "CandidateValue", ["c_ev"], verdict="rejected",
                    reject_reason="needs vp2")
    im.add_binding(ep, "b_x", "vp1", ["c_ev"], ok=True)
    im.fold(ep); im.close_round(1)

    txt = render_dossier(im, "b_x", _FakeCorpus({"c_ev": "evidence text"}))

    assert txt is not None
    assert "find evidence FOR OR AGAINST" in txt

def test_render_dossier_exhausted_no_candidate():
    from panini_dr.adapters.ours._investigation_memory import render_dossier
    plan = {"verb_phrases": [{"id": "vp1", "subject_id": "b_x", "object_id": "e1"}],
            "entities": [{"id": "b_x", "is_target": False}]}
    im = _im_with(plan)
    ep = im.open_episode(1, "researcher", ["b_x"])
    im.add_proposal(ep, "b_x", "DeadEndValue", ["insufficient_evidence"],
                    verdict="rejected", reject_reason="not grounded")
    im.fold(ep); im.close_round(1)
    im.mem.dry_hit_at["b_x"] = 2
    im.mem.dry_streak["b_x"] = 7

    txt = render_dossier(im, "b_x", _FakeCorpus({}))

    assert txt is not None
    exhausted_lines = [line for line in txt.splitlines() if "search-exhausted" in line]
    assert exhausted_lines
    assert any("submit best-effort or defer" in line for line in exhausted_lines)
    assert all("DeadEndValue" not in line for line in exhausted_lines)

def test_render_dossier_never_raises_on_bad_actor(monkeypatch):
    from panini_dr.adapters.ours import _investigation_memory as M
    plan = {"verb_phrases": [], "entities": [{"id": "t", "is_target": True}]}
    im = M.InvestigationMemory(plan)
    ep = im.open_episode(1, "researcher", ["b_x"])
    im.add_proposal(ep, "b_x", "V", ["c1"], verdict="rejected", reject_reason="r")
    im.fold(ep); im.close_round(1)
    class Boom:
        def get_chunk(self, cid): raise RuntimeError("corpus down")
    # a corpus that explodes must yield None, never propagate
    assert M.render_dossier(im, "b_x", Boom()) is None


# --- unfreeze_triggers carries what its name says (2026-08-06) ---------------
# The 2026-08-02 firing audit found it byte-identical to `retirements`: 0
# mismatches over 1,633 pairs, because `retire()` appended the same
# `(round, bids)` tuple to both on consecutive lines. One of two fields carried
# no information.
#
# It is not deleted, because retire()'s docstring says the unfreeze exists "so
# should_skip lets it re-dispatch" — and should_skip is now actually wired
# (--gsw-invmem-skip-stalled). The distinct signal is therefore: which blanks
# were genuinely FROZEN by the guard and got released, as opposed to merely
# retired. A replan that rescues a blank the skip filter would have dropped is
# the event worth counting.


def _im_with(plan=None):
    return InvestigationMemory(plan or {"entities": [], "verb_phrases": [],
                                        "constraints": []})


def test_retiring_an_unfrozen_blank_is_not_an_unfreeze():
    im = _im_with()
    im.retire(3, ["b_never_stuck"])
    assert im.retirements == [(3, ["b_never_stuck"])]
    assert im.unfreeze_triggers == [], (
        "a blank the guard never froze cannot be 'unfrozen'")


def test_retiring_a_stalled_blank_records_the_unfreeze():
    im = _im_with()
    im.mem.stall_hit_at["b_stalled"] = 2
    im.retire(4, ["b_stalled"])
    assert im.retirements == [(4, ["b_stalled"])]
    assert im.unfreeze_triggers == [(4, ["b_stalled"])]
    assert "b_stalled" not in im.mem.stall_hit_at, "the marker must be cleared"


def test_an_exhausted_blank_also_counts():
    im = _im_with()
    im.mem.dry_hit_at["b_dry"] = 5
    im.retire(6, ["b_dry"])
    assert im.unfreeze_triggers == [(6, ["b_dry"])]


def test_only_the_frozen_members_of_a_batch_are_recorded():
    im = _im_with()
    im.mem.stall_hit_at["b_stalled"] = 1
    im.retire(7, ["b_stalled", "b_fine"])
    assert im.retirements == [(7, ["b_stalled", "b_fine"])]
    assert im.unfreeze_triggers == [(7, ["b_stalled"])]


# --- v8: RESOLVED section — accepted values cross into the next dispatch ------
# MEASURED (monaco_920, arms C7/C8 and the baseline alike): every credited
# actor on that question was ACCEPTED, so `render_dossier` -- whose OPEN
# section requires a live UNACCEPTED candidate -- returned an empty body in
# BOTH arms. A later dispatch that had to count the accepted book list could
# not see it, minted the pseudo-id `derived_from_b_martin_book_list`, was
# rejected, and the question ended on the wrong author. An accepted value and
# its evidence ids are exactly what a downstream dispatch needs to cite.


def test_render_dossier_reports_accepted_values_with_their_evidence_ids():
    from panini_dr.adapters.ours._investigation_memory import render_dossier
    plan = {"verb_phrases": [{"id": "vp1", "subject_id": "b_list", "object_id": "e1"}],
            "entities": [{"id": "b_list", "is_target": False}]}
    im = _im_with(plan)
    ep = im.open_episode(1, "researcher", ["b_list"])
    im.add_proposal(ep, "b_list", ["A Game of Thrones", "A Clash of Kings"],
                    ["George R. R. Martin__34"], "accepted", None)
    im.fold(ep)
    out = render_dossier(im, "b_list", _FakeCorpus({}))
    assert out, "an accepted value must make the dossier non-empty"
    assert "RESOLVED" in out
    assert "A Game of Thrones" in out
    assert "George R. R. Martin__34" in out, \
        "the citable evidence id must cross to the next dispatch"


def test_render_dossier_empty_when_only_ungrounded_sentinel_attempts():
    """A blank whose only attempts cited the sentinel has nothing citable to
    hand on -- the dossier must stay silent rather than invent support."""
    from panini_dr.adapters.ours._investigation_memory import render_dossier
    plan = {"verb_phrases": [{"id": "vp1", "subject_id": "b_c", "object_id": "e1"}],
            "entities": [{"id": "b_c", "is_target": False}]}
    im = _im_with(plan)
    ep = im.open_episode(1, "researcher", ["b_c"])
    im.add_proposal(ep, "b_c", 7, ["insufficient_evidence"], "rejected", "needs_grounding")
    im.fold(ep)
    out = render_dossier(im, "b_c", _FakeCorpus({}))
    assert out is None or "RESOLVED" not in out


# --- v9: a LIST blank is never "settled" by a partial commit -----------------
# MEASURED (arm C9, v8, n=55 paired, official V2 judge): multi-value recall
# fell 0.437 -> 0.331 vs baseline while PRECISION held (0.725), and items
# emitted per list answer dropped to 4.8 vs baseline 5.4 (gold 14.9). v8's
# RESOLVED section told a downstream dispatch "build on it; do not re-derive
# it" -- correct for a scalar, wrong for a list, where an accepted value can
# be right AND incomplete. The cost win (turns 24.5 -> 14.5) and the recall
# loss are the SAME early-closure behaviour.


def test_render_dossier_resolved_marks_a_list_blank_as_possibly_incomplete():
    from panini_dr.adapters.ours._investigation_memory import render_dossier
    plan = {"verb_phrases": [{"id": "vp1", "subject_id": "b_list", "object_id": "e1"}],
            "entities": [{"id": "b_list", "is_target": False, "kind": "blank",
                          "value_type": "list"}]}
    im = _im_with(plan)
    ep = im.open_episode(1, "researcher", ["b_list"])
    im.add_proposal(ep, "b_list", ["A Game of Thrones", "A Clash of Kings"],
                    ["George R. R. Martin__34"], "accepted", None)
    im.fold(ep)
    out = render_dossier(im, "b_list", _FakeCorpus({}))
    assert "RESOLVED" in out
    assert "INCOMPLETE" in out, "a list blank must not be presented as finished"
    assert "keep searching" in out.lower() and "fuller list" in out.lower()
    assert "do not re-derive it" not in out, \
        "the scalar 'settled' instruction must not apply to a list blank"


def test_render_dossier_resolved_keeps_settled_wording_for_a_scalar_blank():
    from panini_dr.adapters.ours._investigation_memory import render_dossier
    plan = {"verb_phrases": [{"id": "vp1", "subject_id": "b_year", "object_id": "e1"}],
            "entities": [{"id": "b_year", "is_target": False, "kind": "blank",
                          "value_type": "number"}]}
    im = _im_with(plan)
    ep = im.open_episode(1, "researcher", ["b_year"])
    im.add_proposal(ep, "b_year", 1996, ["chunk__1"], "accepted", None)
    im.fold(ep)
    out = render_dossier(im, "b_year", _FakeCorpus({}))
    assert "RESOLVED" in out and "do not re-derive it" in out
    assert "INCOMPLETE" not in out
