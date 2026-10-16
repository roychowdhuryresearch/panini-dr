"""Live investigation memory (Phase 1 — LOG-ONLY, flag-gated).

A faithful live port of the offline reference in ``new_feature/`` (extract.py +
reconcile.py), frozen and accepted by the full-population Phase 0 safety sweep. The
live system does NOT extract from traces; it is fed events at the sites that already
produce them (proposals in ``_tool_update_blank``, bindings/attempts via ``_dispatch``,
resolver commits + round boundaries in ``_dispatch_subplan_parallel``, retirements in
``invalidate_changed_blanks``). Given the same events, this reproduces the offline
``Memory`` byte-for-byte — the fold/close_round/leading/finalize logic below is copied
verbatim from reconcile.py so the two can never drift.

PHASE 1 CONTRACT: this module only RECORDS and computes a compact dump for
``traj.extra["invmem"]``. It performs NO rule enforcement and drives NO prompt change.
The stall/dry firing rounds it computes are diagnostic only — nothing in the
orchestrator reads them. Enforcement is Phase 2+. When the ``gsw_investigation_memory``
flag is off the orchestrator never constructs this object, so behavior is byte-identical.

Frozen Phase-0 knobs (provenance: new_feature/constants.py, sweep 2026-07-17):
  R_STALL=5, K_DRY=15, GUARD_PENDING_EVIDENCE=True, plus the A5 dry-reset. These were
  the minimal config reaching 0 harmful firings over 947 correct + 83 empty questions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# --- frozen Phase-0 thresholds (log-only in Phase 1; see module docstring) --------
R_STALL = 5
K_DRY = 15
GUARD_PENDING_EVIDENCE = True
GUARD_MAX_REJECTIONS = 2   # == sweep REJ_LIVE_CUTOFF (validated V2 variant)

# Question-level termination knob (Stall->Stop->Final-Shot, spec 2026-07-18 rev3).
# FROZEN by the full-population sweep (new_feature/sweep_question_level.py,
# config V1_R5: 0 fires on 1420 correct traces; R_Q=4 is blocked by q334, the
# same irreducible late-recovery case that froze R_STALL=5).
R_Q = 5

SENTINEL = "insufficient_evidence"

MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}
for _m in list(MONTHS):
    MONTHS[_m[:3]] = MONTHS[_m]
MONTHS["sept"] = MONTHS["september"]   # D9


def _ep_seq_num(episode_id) -> int:    # D5: numeric episode recency for tiebreak
    m = re.search(r"\d+", str(episode_id or ""))
    return int(m.group()) if m else 0


_FORMAT_REJECT_RE = re.compile(r"unknown chunk id", re.I)


def _is_format_rejection(reason) -> bool:   # D1: citation/format failure, not factual
    return bool(reason) and bool(_FORMAT_REJECT_RE.search(str(reason)))


def normalize(value):
    """Value canonicalization (copied verbatim from reconcile.normalize)."""
    if isinstance(value, (list, tuple)):
        return tuple(sorted((normalize(v) for v in value), key=repr))   # D10
    s = str(value).strip()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}"
    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", s)
    if m and m[2].lower() in MONTHS:
        return f"{int(m[3]):04d}-{MONTHS[m[2].lower()]:02d}-{int(m[1]):02d}"
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", s)
    if m and m[1].lower() in MONTHS:
        return f"{int(m[3]):04d}-{MONTHS[m[1].lower()]:02d}-{int(m[2]):02d}"
    try:
        return repr(float(s.replace(",", "")))
    except ValueError:
        pass
    t = re.sub(r"\s+", " ", s.casefold())
    return re.sub(r"^(the|a|an)\s+", "", t)


# --- event dataclasses (from extract.py; a live episode is built incrementally) ---
@dataclass
class Attempt:
    type: str                       # search | read | find
    result_chunks: list = field(default_factory=list)
    query: Optional[str] = None     # search only; dossier "do not repeat" list


@dataclass
class Binding:
    blank_id: str
    connection: str                 # vp_id
    chunk_ids: list
    ok: bool


@dataclass
class Proposal:
    blank_id: str
    value: object
    evidence_chunk_ids: list
    has_real_evidence: bool
    verdict: str                    # accepted | rejected | accepted_with_flag
    reject_reason: Optional[str]


@dataclass
class Episode:
    episode_id: str
    round: int
    worker: str                     # researcher | orchestrator
    dispatched_blanks: list
    ended: str = ""
    attempts: list = field(default_factory=list)
    bindings: list = field(default_factory=list)
    proposals: list = field(default_factory=list)


@dataclass
class Actor:
    actor_id: str
    blank_id: str
    value: object
    normalized_value: object
    first_seen: str
    last_seen: str
    proposed_count: int = 0
    accepted: bool = False
    has_real_evidence: bool = False
    retired: bool = False
    edges: dict = field(default_factory=dict)      # (connection, chunk) -> episode_id
    rejections: list = field(default_factory=list)   # FACTUAL refutations only (D1)
    ungrounded_count: int = 0                        # D1: rejected for want of a citation

    def state(self, required):
        conns = {c for (c, _) in self.edges}
        if required and required <= conns:
            return "fully_connected"
        return "partial" if conns else "proposed"


class Memory:
    """Fold Episodes into M* — copied verbatim from reconcile.Memory (A5 + guard)."""

    def __init__(self, plan_json):
        self.required = {}
        self.target_blank = None
        for vp in plan_json.get("verb_phrases", []):
            for side in ("subject_id", "object_id"):
                b = vp.get(side)
                if b and b.startswith(("b_", "t")):
                    self.required.setdefault(b, set()).add(vp.get("id"))
        # v9: retain each blank's declared value_type. A LIST blank that has an
        # accepted value may still be INCOMPLETE, and the dossier must say so --
        # measured on arm C9, where v8's "do not re-derive it" wording drove
        # multi-value recall 0.437 -> 0.331 at unchanged precision.
        self.value_type = {
            en["id"]: str(en.get("value_type") or "")
            for en in plan_json.get("entities", []) if en.get("id")
        }
        for en in plan_json.get("entities", []):
            if en.get("is_target"):
                self.target_blank = en["id"]
        self.actors = {}
        self.blank_edge_pool = {}
        self.seen_chunks = {}
        self.official = {}
        self.official_round = {}
        self.rounds_worked = {}
        self.progress_rounds = {}
        self.dry_streak = {}
        self.dry_hit_at = {}
        self.stall_hit_at = {}
        # The raw streak behind stall_hit_at. Stored so a consumer can
        # threshold independently: stall_hit_at is fixed at R_STALL and is
        # read by scoped_budget() and the dossier, so re-tuning it to sweep
        # one consumer would silently move the other two.
        self.stall_streak = {}
        self.log = []

    def _has_live_real_evidence(self, blank):
        return any(a.blank_id == blank and a.has_real_evidence
                   and len(a.rejections) < GUARD_MAX_REJECTIONS   # D4/D20: not stuck
                   for a in self.actors.values())

    def _actor(self, blank, value, ep_id):
        key = (blank, normalize(value))
        if key not in self.actors:
            self.actors[key] = Actor(
                actor_id=f"a{len(self.actors)+1}", blank_id=blank, value=value,
                normalized_value=key[1], first_seen=ep_id, last_seen=ep_id)
        return self.actors[key]

    def fold(self, ep):
        progress = {}
        blanks = set(ep.dispatched_blanks) or \
            {p.blank_id for p in ep.proposals} | {b.blank_id for b in ep.bindings}

        for b in blanks:
            seen = self.seen_chunks.setdefault(b, set())
            prog = False

            for at in ep.attempts:                       # chunk novelty -> dry streak
                new = [c for c in at.result_chunks if c not in seen]
                seen.update(at.result_chunks)
                if at.type == "search":
                    if new:
                        self.dry_streak[b] = 0
                    else:
                        self.dry_streak[b] = self.dry_streak.get(b, 0) + 1
            ep_actors = []
            extracted = False        # a grounded value was pulled from a chunk this ep
            for p in ep.proposals:
                if p.blank_id != b:
                    continue
                a = self._actor(b, p.value, ep.episode_id)
                is_new = a.proposed_count == 0
                a.proposed_count += 1
                a.last_seen = ep.episode_id
                accepted = p.verdict in ("accepted", "accepted_with_flag")
                grounded = p.has_real_evidence and not _is_format_rejection(p.reject_reason)  # D1
                if grounded:
                    a.has_real_evidence = True
                    if accepted:                         # D2
                        extracted = True                 # info extracted, not dry (A5)
                    for c in p.evidence_chunk_ids:
                        if c != SENTINEL:
                            k = (None, c)               # general-support edge
                            if k not in a.edges:
                                a.edges[k] = ep.episode_id
                if accepted:
                    a.accepted = True
                    self.official[b] = a
                    self.official_round[b] = ep.round
                    prog = True
                elif grounded:                       # D1: grounded value the verifier refuted
                    a.rejections.append((ep.episode_id, (p.reject_reason or "")[:80]))
                else:                                # D1: sentinel OR format-failure -> ungrounded
                    a.ungrounded_count += 1
                if is_new and grounded and accepted:  # D3: A1 gated on verdict
                    prog = True
                ep_actors.append(a)

            # amendment A5: the dry-search EXHAUST rule measures information EXTRACTED,
            # not chunks retrieved. A grounded proposal this round means the researcher
            # pulled a value from an in-hand chunk, so the blank is not "dry".
            if extracted:
                self.dry_streak[b] = 0
            elif (self.dry_streak.get(b, 0) >= K_DRY and b not in self.dry_hit_at
                    and not (GUARD_PENDING_EVIDENCE and self._has_live_real_evidence(b))):
                self.dry_hit_at[b] = ep.round

            for bind in ep.bindings:                     # amendment A2
                if bind.blank_id != b or not bind.ok:
                    continue
                targets = ep_actors or None
                for c in bind.chunk_ids:
                    k = (bind.connection, c)
                    if targets:
                        for a in targets:
                            if k not in a.edges:
                                a.edges[k] = ep.episode_id
                                prog = True
                    else:
                        pool = self.blank_edge_pool.setdefault(b, {})
                        if k not in pool:
                            pool[k] = ep.episode_id
                            prog = True

            progress[b] = prog
        return progress

    def close_round(self, rnd, progress_by_blank):
        for b, prog in progress_by_blank.items():
            self.rounds_worked.setdefault(b, []).append(rnd)
            if prog:
                self.progress_rounds.setdefault(b, set()).add(rnd)
            worked = self.rounds_worked[b]
            streak = 0
            for r in reversed(worked):
                if r == worked[0]:          # first worked round is exempt (A4)
                    break
                if r in self.progress_rounds.get(b, set()):
                    break
                streak += 1
            self.stall_streak[b] = streak
            if (streak >= R_STALL and b not in self.stall_hit_at
                    and not (GUARD_PENDING_EVIDENCE and self._has_live_real_evidence(b))):
                self.stall_hit_at[b] = rnd

    def leading(self, blank):
        cands = [a for a in self.actors.values()
                 if a.blank_id == blank and not a.retired]
        pool = self.blank_edge_pool.get(blank, {})
        req = self.required.get(blank, set())

        def merged_conns(a):
            conns = {c for (c, _) in a.edges if c}
            conns |= {c for (c, _) in pool if c}
            return conns

        def key(a):
            conns = merged_conns(a)
            fully = 1 if (req and req <= conns) else 0
            # Phase-2 fix: a verifier-ACCEPTED candidate outranks any other, and a
            # candidate with FEWER verifier rejections outranks a heavily-rejected
            # one — before the old proposal/edge/recency tiebreak. Prevents the
            # finalizer from resurrecting a many-times-rejected value (q227/q537).
            return (a.has_real_evidence, a.accepted, -len(a.rejections),
                    fully, len(conns) + len(a.edges),
                    a.proposed_count, _ep_seq_num(a.last_seen))
        return sorted(cands, key=key, reverse=True)

    def finalize(self, blank):
        # Phase-2 fix: if the blank has an ACCEPTED value, that is the answer —
        # never let a rejected candidate override it (q227: accepted "Matsuyama"
        # beats rejected "Nick Taylor").
        off = self.official.get(blank)
        if off is not None and not off.retired:            # D8: accepted official is the answer
            return off
        for a in self.leading(blank):
            if a.has_real_evidence:
                return a
        return None


def result_chunk_ids(name: str, args: dict, result: dict) -> list:
    """Chunk ids a live tool result 'saw' — mirrors the offline extractor's
    ``_result_chunks`` so live attempts feed the dry streak identically."""
    if name == "search":
        return [r.get("chunk_id") for r in (result.get("results") or [])
                if r.get("chunk_id")]
    if name == "read":
        cid = result.get("chunk_id") or args.get("chunk_id")
        return [cid] if (cid and not result.get("error")) else []
    if name == "find":
        cid = args.get("chunk_id")
        matched = bool(result.get("matches") or result.get("snippets")
                       or result.get("results"))
        return [cid] if (cid and matched) else []
    return []


def verdict_from_update_result(result: dict, verifier_status: Optional[str]) -> tuple:
    """Map a live ``_tool_update_blank`` return dict (+ the committed blank's
    verifier_status, read from state after the call) to (verdict, reject_reason),
    mirroring the offline extractor's ``_proposal_from``."""
    ok = bool(result.get("ok"))
    if not ok:
        reason = str(result.get("error") or result.get("gate") or "")[:200]
        return "rejected", reason
    flag = verifier_status or ""
    if "refuted_best_effort" in flag or "unverified" in flag:
        return "accepted_with_flag", None
    return "accepted", None


def scoped_budget(facade, blank_id, full_budget, *, exhausted_turns=3, verify_turns=6):
    """Memory-scoped researcher turn budget (code-enforced leash). Reads only the
    Actor model. Returns full_budget on any doubt; never raises."""
    try:
        mem = facade.mem
        if blank_id in mem.stall_hit_at or blank_id in mem.dry_hit_at:
            return min(full_budget, exhausted_turns)
        ranked = mem.leading(blank_id)
        if ranked and ranked[0].has_real_evidence and not ranked[0].accepted:
            covered = {c for (c, _) in ranked[0].edges if c}
            if mem.required.get(blank_id, set()) - covered:
                return min(full_budget, verify_turns)
        return full_budget
    except Exception:
        return full_budget


def render_dossier(facade, blank_id, corpus, *, max_ruled_out=3, max_queries=8,
                   evidence_text_chars=600, max_retrieved=3):
    """Prior-investigation dossier for ONE blank, from the Actor model, framed as
    evidence (not advice). Anti-anchoring is structural: the live candidate appears
    only inside an OPEN 'FOR OR AGAINST' line; no 'leading candidate'/'best answer'/
    'the answer is' phrasing exists. Returns None when there is nothing to say.
    Never raises into the run."""
    try:
        mem = facade.mem
        actors = [a for a in mem.actors.values()
                  if a.blank_id == blank_id and not a.retired]
        queries = facade.query_log.get(blank_id, [])
        if not actors and not queries:
            return None
        required = mem.required.get(blank_id, set())
        ranked = mem.leading(blank_id)
        open_actor = ranked[0] if (ranked and ranked[0].has_real_evidence
                                   and not ranked[0].accepted) else None
        exhausted = blank_id in mem.stall_hit_at or blank_id in mem.dry_hit_at
        dry = mem.dry_streak.get(blank_id, 0)
        lines = ["Prior investigation of this blank established the following. "
                 "Treat it as evidence to build on — verify, do not assume."]

        # 0. RESOLVED (v8) — the ACCEPTED value and its citable evidence ids.
        #
        # MEASURED on monaco_920 (arms C7/C8 AND the baseline): every credited
        # actor there was accepted, so OPEN -- which requires a live UNACCEPTED
        # candidate -- had nothing to say and the dossier rendered EMPTY in both
        # arms. A later dispatch that had to count that accepted book list could
        # not see it, minted the pseudo-id `derived_from_b_martin_book_list`,
        # was rejected, and the question ended on the wrong author. An accepted
        # value plus the chunk ids that carried it is precisely what a
        # downstream dispatch needs in order to derive from it AND to cite
        # something legitimate. Ungrounded/sentinel-only actors are excluded --
        # nothing citable to hand on.
        accepted = mem.official.get(blank_id)
        if accepted is not None and accepted.has_real_evidence:
            ev_ids = sorted({chk for (_, chk) in accepted.edges if chk})
            cite = f" — cite {ev_ids}" if ev_ids else ""
            is_list = str(getattr(mem, "value_type", {}).get(blank_id, "")) == "list"
            if is_list:
                # v9: for a LIST blank, accepted != complete. v8's scalar
                # wording ("do not re-derive it") converted a wasteful
                # re-dispatch into a premature stop: items emitted per list
                # answer fell to 4.8 (baseline 5.4, gold 14.9) and recall to
                # 0.331 while precision held at 0.725.
                n_items = len(accepted.value) if isinstance(accepted.value, (list, tuple)) else None
                cnt = f" ({n_items} items so far)" if n_items is not None else ""
                lines.append(
                    f"RESOLVED (PARTIAL) — this LIST blank currently holds "
                    f"{accepted.value!r}{cnt}{cite}. It is committed but may be "
                    f"INCOMPLETE: keep searching for members it is missing and "
                    f"re-commit the fuller list, citing those ids alongside any "
                    f"new ones. Stop only when new angles yield no new members."
                )
            else:
                lines.append(
                    f"RESOLVED — this blank was already committed as "
                    f"{accepted.value!r}{cite}. Build on it; do not re-derive it, "
                    f"and cite those ids rather than inventing an evidence id."
                )

        # 1. OPEN — instrumental live candidate; exhaustion-aware
        if open_actor is not None:
            # D13: credit the shared blank_edge_pool the same way leading() does, so the
            # "missing link" text can't contradict the ranking (which counted the pool).
            pool_conns = {c for (c, _) in mem.blank_edge_pool.get(blank_id, {}) if c}
            covered = {c for (c, _) in open_actor.edges if c} | pool_conns
            missing = sorted(required - covered)
            ev = sorted({chk for (_, chk) in open_actor.edges if chk})
            if exhausted:
                lines.append(
                    f"OPEN — connection(s) {missing or 'for this blank'} are "
                    f"search-exhausted ({dry} dry rounds without new evidence); "
                    f"do NOT keep searching. A candidate {open_actor.value!r} "
                    f"(evidence {ev}) is best-supported so far — confirm or refute it "
                    f"once, then submit it or defer rather than searching further.")
            elif missing:
                lines.append(
                    f"OPEN — to resolve this blank, verify connection(s) {missing}: "
                    f"a candidate {open_actor.value!r} has partial evidence ({ev}) "
                    f"but {missing} is unconfirmed — find evidence FOR OR AGAINST it.")
            else:
                lines.append(
                    f"OPEN — verify candidate {open_actor.value!r} (evidence {ev}) "
                    f"— confirm or refute it.")
        elif exhausted:
            lines.append(
                f"OPEN — this blank is search-exhausted ({dry} dry rounds without new "
                f"evidence); further searching is unlikely to help — submit best-effort "
                f"or defer.")

        # 2. RULED OUT (exclude the OPEN actor). D1: a.rejections now holds ONLY factual
        #    refutations of grounded candidates, so most-refuted-first is correct here and
        #    a correct-but-hard-to-ground value can no longer be false-ruled-out (it lands
        #    in UNGROUNDED below instead).
        ruled = [a for a in actors if a.rejections and not a.accepted
                 and a is not open_actor]
        ruled.sort(key=lambda a: (-len(a.rejections), _ep_seq_num(a.last_seen)))  # D5-family
        for a in ruled[:max_ruled_out]:
            reason = (a.rejections[-1][1] if a.rejections and a.rejections[-1] else "") or "no reason recorded"
            lines.append(f"RULED OUT: {a.value!r} — rejected: {reason[:120]}")

        # 2b. UNGROUNDED (D1) — candidates proposed but never chunk-grounded. A verify
        #     signal, not a refutation; flips to a soft negative after repeated failures so
        #     a genuinely wrong guess is still abandoned (preserves the q537 win).
        ung = [a for a in actors if a.ungrounded_count and not a.accepted
               and not a.has_real_evidence and not a.rejections and a is not open_actor]
        ung.sort(key=lambda a: (-a.ungrounded_count, _ep_seq_num(a.last_seen)))
        for a in ung[:max_ruled_out]:
            if a.ungrounded_count >= 3:
                lines.append(f"UNGROUNDED: {a.value!r} — proposed {a.ungrounded_count}x but "
                             f"no supporting chunk found; likely not the answer.")
            else:
                lines.append(f"UNGROUNDED: {a.value!r} — proposed but not chunk-grounded; "
                             f"find a chunk that supports or refutes it.")

        # 3. RETRIEVED — inline text of evidence chunks from the OPEN / accepted actors
        #    ONLY (D24: a ruled-out candidate's chunk must not be re-served as neutral
        #    fact, contradicting its RULED-OUT line), and capped (D25: unbounded
        #    re-injection bloats the dossier and keeps the same stale evidence hot so the
        #    exhaustion detector never fires — the q68/q596 runaway).
        src_actors = [a for a in actors if a is open_actor or a.accepted]
        ev_chunks = []
        for a in src_actors:
            for (_, chk) in a.edges:
                if chk and chk not in ev_chunks:
                    ev_chunks.append(chk)
        for chk in ev_chunks[:max_retrieved]:
            ch = corpus.get_chunk(chk)
            if ch is not None and getattr(ch, "text", ""):
                lines.append(f"RETRIEVED (already in hand) {chk}: "
                             f"{ch.text[:evidence_text_chars]}")

        # 4. TRIED — null-result queries as fact
        if queries:
            qs = "; ".join(repr(q) for q in queries[-max_queries:])
            lines.append(f"Searches already run (returned nothing new): {qs}")

        body = "\n".join(lines)
        return body if len(lines) > 1 else None
    except Exception:  # noqa: BLE001 — memory must never break the run
        return None


# ---- final-shot QA (spec 2026-07-18 rev3; validated 11/24 by finalshot_probe) ----
FINAL_SHOT_SYSTEM = (
    "You are a careful question-answering assistant. You get one research "
    "question, the facts an investigation has already established, candidate "
    "values it considered, and the full text of the articles it read. Answer "
    "the question using ONLY this material. Work step by step, including any "
    "date arithmetic or comparisons the question needs. Give the single best "
    "supported answer even if you are not fully certain; reply NONE only if "
    "the material gives you no basis at all. "
    "Final line must be exactly: ANSWER: <your answer>"
)
FINAL_SHOT_MAX_ART_CHARS = 45000
FINAL_SHOT_MAX_TOTAL_CHARS = 130000
FINAL_SHOT_MAX_ARTICLES = 4


def build_final_shot_prompt(question, resolved, candidates, articles):
    """messages for the final-shot completion.

    resolved:   {blank_id: (subject, relation, value)}
    candidates: pre-rendered candidate lines (may be empty)
    articles:   [(title, full_text)] — newest-read first, <=4, capped here
    """
    known = "\n".join(f"- {b} ({s} {r}): {v}"
                      for b, (s, r, v) in resolved.items()) or "(none)"
    cand = "\n".join(candidates) or "(none)"
    total, parts = 0, []
    for title, text in articles[:FINAL_SHOT_MAX_ARTICLES]:
        txt = (text or "")[:FINAL_SHOT_MAX_ART_CHARS]
        if total + len(txt) > FINAL_SHOT_MAX_TOTAL_CHARS:
            txt = txt[:max(0, FINAL_SHOT_MAX_TOTAL_CHARS - total)]
        total += len(txt)
        if txt:
            parts.append(f"### ARTICLE: {title}\n{txt}")
    user = (f"QUESTION: {question}\n\nESTABLISHED FACTS:\n{known}\n\n"
            f"CANDIDATES CONSIDERED (may be wrong):\n{cand}\n\n"
            + "\n\n".join(parts))
    return [{"role": "system", "content": FINAL_SHOT_SYSTEM},
            {"role": "user", "content": user}]


_FINAL_SHOT_REFUSAL = re.compile(
    r"\b(none|cannot determine|can(?:no|')t determine|unable to determine|"
    r"not enough information|insufficient information|no basis)\b", re.I)


def parse_final_shot_answer(text: str) -> str:
    """The answer from a final-shot reply, sentinel or not.

    The final shot is the never-blank hook: it runs only when the structured
    path has already produced nothing, and an empty prediction scores 0. So a
    reply that answers the question without the ``ANSWER:`` line used to be
    thrown away for free. Measured on the 200Q run: q802 spent 14,196 prompt
    tokens, wrote 552 completion tokens, and parsed to "".

    The sentinel still wins when present. Without it, the last non-empty line
    is taken — unless the reply refuses, which stays a refusal. Abstention is
    load-bearing here (champion answers 4 of 824 on zero evidence, ODS 120);
    this salvages answers the model DID give, it does not manufacture one.
    """
    m = re.search(r"ANSWER:\s*(.+)", text or "")
    if m:
        ans = m.group(1).strip()
        return "" if ans.casefold() in ("none", "") else ans
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    tail = lines[-1]
    return "" if _FINAL_SHOT_REFUSAL.search(tail) else tail


class InvestigationMemory:
    """Orchestrator-side facade. One per question. Siblings build Episodes locally
    (concurrency-safe: each ReActResearcher owns its own Episode); the orchestrator
    folds them at the dispatch join in a deterministic order, then closes the round.
    """

    def __init__(self, plan_json: dict):
        self.mem = Memory(plan_json or {})
        self._round_progress: dict = {}
        self._ep_seq = 0
        self.retirements: list = []      # (round, [blank_ids]) — Phase-1 diagnostic
        # (round, [blank_ids]) for blanks the guard had FROZEN and a replan
        # released — a strict subset of `retirements`, see retire().
        self.unfreeze_triggers: list = []
        self.finalized = None            # Phase-2: actor dump if the finalizer fired
        self.q_rounds_worked: list = []   # question-level clock (rounds folded)
        self.q_progress_rounds: set = set()
        self.read_log: list = []          # chunk ids in read/fold order (final-shot)
        self.terminated = None            # reason string once the clock fired
        self.final_shot = None            # dict once the final-shot QA call ran
        self.query_log: dict = {}        # blank -> ordered deduped normalized queries

    # ---- episode construction (called by ReActResearcher, on its own thread) ----
    def open_episode(self, rnd: int, worker: str, dispatched_blanks) -> Episode:
        self._ep_seq += 1
        pfx = "e" if worker == "researcher" else "o"
        return Episode(episode_id=f"{pfx}{self._ep_seq}", round=int(rnd),
                       worker=worker, dispatched_blanks=list(dispatched_blanks or []))

    @staticmethod
    def add_attempt(ep: Episode, atype: str, result_chunks, query=None) -> None:
        ep.attempts.append(Attempt(type=atype, result_chunks=list(result_chunks or []),
                                   query=query))

    @staticmethod
    def add_binding(ep: Episode, blank_id: str, connection: str, chunk_ids, ok: bool) -> None:
        ep.bindings.append(Binding(blank_id=blank_id, connection=connection,
                                   chunk_ids=list(chunk_ids or []), ok=bool(ok)))

    @staticmethod
    def add_proposal(ep: Episode, blank_id: str, value: Any, evidence_chunk_ids,
                     verdict: str, reject_reason: Optional[str]) -> None:
        ev = list(evidence_chunk_ids or [])
        real = any(c and c != SENTINEL for c in ev)
        ep.proposals.append(Proposal(blank_id=blank_id, value=value,
                                     evidence_chunk_ids=ev, has_real_evidence=real,
                                     verdict=verdict, reject_reason=reject_reason))

    # ---- orchestrator-side folding (single thread, at the dispatch join) --------
    def fold(self, ep: Episode) -> None:
        for at in ep.attempts:            # read-log for the final-shot article set
            if at.type == "read":
                self.read_log.extend(at.result_chunks)
        for at in ep.attempts:            # per-blank query log (dossier "do not repeat")
            if at.type != "search" or not at.query:
                continue
            q = re.sub(r"\s+", " ", str(at.query)).strip().casefold()
            if not q:
                continue
            for b in (ep.dispatched_blanks or []):
                log = self.query_log.setdefault(b, [])
                if q not in log:
                    log.append(q)
        for b, prog in self.mem.fold(ep).items():
            self._round_progress[b] = self._round_progress.get(b, False) or prog

    def close_round(self, rnd: int) -> None:
        any_prog = any(self._round_progress.values())
        self.mem.close_round(int(rnd), self._round_progress)
        self._round_progress = {}
        self.q_rounds_worked.append(int(rnd))
        if any_prog:
            self.q_progress_rounds.add(int(rnd))

    def record_resolver_commit(self, rnd: int, blank_id: str, value: Any,
                               evidence_chunk_ids) -> None:
        """A TypedBlankResolver commit — fold as an accepted proposal (its own
        orchestrator episode). Folded immediately; contributes to the round's
        progress accumulator like any other event."""
        ep = self.open_episode(rnd, "orchestrator", [blank_id])
        self.add_proposal(ep, blank_id, value, evidence_chunk_ids,
                          verdict="accepted", reject_reason=None)
        self.fold(ep)

    def retire(self, rnd: int, blank_ids) -> None:
        """A plan update changed a blank's identity — retire its actors (kept in
        history, excluded from ranking) and UNFREEZE the blank: clear its
        acceptance + stall/exhaust firings so the leash and the dossier stop
        treating it as exhausted."""
        bids = list(blank_ids or [])
        m = self.mem
        for a in m.actors.values():
            if a.blank_id in bids:
                a.retired = True
        unfrozen = []                         # unfreeze trigger (spec Phase-1 note)
        for bid in bids:
            # A blank is FROZEN only if the guard actually fired on it. Recording
            # the whole batch made this field a byte-identical copy of
            # `retirements` (0 mismatches over 1,633 pairs) and carry no
            # information. What is worth counting is a replan releasing a blank
            # the guard had frozen and a replan released.
            if bid in m.stall_hit_at or bid in m.dry_hit_at:
                unfrozen.append(bid)
            m.official.pop(bid, None)
            m.official_round.pop(bid, None)
            m.stall_hit_at.pop(bid, None)
            m.dry_hit_at.pop(bid, None)
            m.dry_streak.pop(bid, None)
            # Cleared with the marker it belongs to, so a reopened blank does not
            # report a stale streak.
            m.stall_streak.pop(bid, None)
        if bids:
            self.retirements.append((int(rnd), bids))
        if unfrozen:
            self.unfreeze_triggers.append((int(rnd), unfrozen))

    # ---- Phase-2 enforcement queries --------------------------------------------
    def should_terminate(self) -> Optional[str]:
        """Question-level clock (sweep config V1_R5): R_Q consecutive worked
        rounds with no fold-progress anywhere, first worked round exempt (A4),
        suppressed while any not-yet-accepted blank holds a live real-evidence
        actor. Plan-updates minting fresh blanks do NOT reset the clock (minting
        is not progress). NOTE: the guard deliberately ignores ``retired`` to
        match the offline sweep semantics; live retire() popping official_round
        only makes the guard MORE suppressive, the safe direction."""
        worked = self.q_rounds_worked
        if len(worked) < 2:
            return None
        streak = 0
        for r in reversed(worked):
            if r == worked[0]:            # first worked round exempt
                break
            if r in self.q_progress_rounds:
                break
            streak += 1
        if streak < R_Q:
            return None
        m = self.mem
        for a in m.actors.values():       # guard (pending evidence; D20 rejection-aware)
            if (a.has_real_evidence and m.official_round.get(a.blank_id) is None
                    and len(a.rejections) < GUARD_MAX_REJECTIONS):   # D20: not stuck
                return None
        return (f"question-exhausted: no grounded progress for {streak} "
                f"consecutive rounds (R_Q={R_Q})")

    def finalize(self, blank_id: str):
        """G2/G3 finalizer: the leading real-evidence actor for a blank (or None).
        Caller enforces G2 (only when the run ended empty)."""
        return self.mem.finalize(blank_id)

    def _actor_dict(self, a) -> dict:
        m = self.mem
        return {
            "blank": a.blank_id, "value": a.value,
            "real_ev": a.has_real_evidence, "accepted": a.accepted,
            "retired": a.retired, "proposed": a.proposed_count,
            "edges": len(a.edges), "conns": sorted({c for (c, _) in a.edges if c}),
            "rejections": len(a.rejections),
            "state": a.state(m.required.get(a.blank_id, set())),
        }

    # ---- compact trace dump (traj.extra["invmem"]) ------------------------------
    def dump(self) -> dict:
        m = self.mem
        actors = [self._actor_dict(a) for a in m.actors.values()]
        return {
            "target_blank": m.target_blank,
            "n_actors": len(m.actors),
            "actors": actors,
            "official_round": dict(m.official_round),
            "stall_hit_at": dict(m.stall_hit_at),
            "dry_hit_at": dict(m.dry_hit_at),
            "dry_streak": {k: v for k, v in m.dry_streak.items() if v},
            # Raw counter behind stall_hit_at, the peer of dry_streak. No consumer
            # reads it since the dispatch filter was removed; kept because it is
            # an observed measurement of real behaviour and it is what made the
            # offline threshold analysis possible.
            "stall_streak": {k: v for k, v in m.stall_streak.items() if v},
            "rounds_worked": {k: list(v) for k, v in m.rounds_worked.items()},
            "retirements": list(self.retirements),
            "unfreeze_triggers": list(self.unfreeze_triggers),
            "finalized": self.finalized,
            "terminated": self.terminated,
            "final_shot": self.final_shot,
            # Accounting of the rendered-briefing mode (removed; never
            # enabled in the paper runs), kept so the dump's shape is unchanged.
            "briefing": [],
            "q_rounds_worked": list(self.q_rounds_worked),
            "q_progress_rounds": sorted(self.q_progress_rounds),
            "knobs": {"R_STALL": R_STALL, "K_DRY": K_DRY,
                      "guard": GUARD_PENDING_EVIDENCE},
        }
