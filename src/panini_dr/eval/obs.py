"""Obs. (observed support) scorer -- the paper's grounding column, parameterised.

    Obs. = the official score, recomputed so that a predicted answer item only
           counts when it is also SUPPORTED: its parts appear (word-boundary
           string match, with numeric/date/word variants) in text the system
           actually received from its tools during that question.

Ported from the research repo's scorer of record:

* BrowseComp-Plus and FRAMES: ``eval_tools/earned_f1_v2.py`` (working tree,
  incl. the 2026-09-23 ``_parse_passage`` fix), harness adapter only --
  every tool result in the trace (search / read / find / lookup), the typed
  resolver's and the grounding checker's retrieved chunks
  (``extra.gsw_checker_calls``, validated verbatim against the corpus), and the
  chunk ids the pipeline committed as evidence. Obs. = mean over all questions
  of the per-question ``earned`` score (judge verdict AND every part supported).
* MoNaCo: the item-level rule of record (package 35, ``build_items_v15.py``)
  over the raw verdicts kept by :mod:`panini_dr.eval.grade_set`: each predicted
  item the judge credited is tested for support (claim parts, gold-component
  fallback, same-document association), and Obs. is the set F1 with only
  supported credited items in the numerator, on the selected pass. The evidence
  is assembled from the run's traces exactly like the research packet builder
  (``build_monaco_audit_inputs.py``: tool results, resolver evidence chunks, and
  the article spans the reader sub-agent paged through), or read from existing
  packet files.

What changed: no run registry and no absolute paths -- every input (traces,
question file, split, corpus, qrels, judge verdicts) is an argument; baseline
adapters are dropped. Matching rules are identical (:mod:`obs_text`).
Deterministic; no model calls.
"""
from __future__ import annotations

import ast
import glob
import json
import os
import pickle
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from panini_dr.eval.obs_text import (  # noqa: F401  (re-exported for callers)
    TitleIndex,
    _CHUNK_IDX,
    _DOCID_SUFFIX_RE,
    _as_dict,
    _literal,
    _search_hits,
    _spaceless,
    _strip_ref,
    docid_of as bcp_docid_of,
    earned_counts,
    final_answer_string,
    gold_items,
    item_matches_gold,
    item_parts_and_kinds,
    item_support,
    load_qrels,
    norm,
    prf,
    split_items,
)


# ----------------------------------------------------------------------------
# corpus text for BrowseComp-Plus (streamed once, subset cached)
# ----------------------------------------------------------------------------
def load_doc_text(need: set, corpus_path: str, cache_path: str | None = None):
    """docid -> text for the docids in `need`, plus normalised-title -> docid
    over the whole corpus (so truncated / title-only references resolve).

    Streams the {key: text} corpus with ijson when installed (the BCP corpus is
    3.3 GB); otherwise falls back to json.load. ``cache_path`` (optional) keeps
    the subset as a pickle between runs."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        if need <= set(cached["text"]):
            return cached["text"], cached["title2doc"]
    text, title2doc = {}, {}

    def _visit(key, val):
        m = _DOCID_SUFFIX_RE.match(key)
        d = m.group("docid") if m else (key if key.isdigit() else None)
        if d is None:
            return
        if m:
            title2doc.setdefault(norm(m.group("title")), d)
        if d in need:
            text[d] = val

    try:
        import ijson  # streaming
    except ImportError:
        ijson = None
    if ijson is not None:
        with open(corpus_path, "rb") as fh:
            for key, val in ijson.kvitems(fh, ""):
                _visit(key, val)
    else:
        for key, val in json.load(open(corpus_path)).items():
            _visit(key, val)
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump({"text": text, "title2doc": title2doc}, fh)
    return text, title2doc


class Dataset:
    """Everything the scorer needs to know about one benchmark.

    docids are numeric BCP docids, or the normalised article title on FRAMES /
    MoNaCo (their chunk ids are '<Title>__<n>'; the gold pages are titles).
    `ref` on an event is the chunk id when the trace has one, so a docid-only
    event can be scored on exactly the chunk that was served.

    Inputs (all explicit):
      bcp:    questions (browsecomp_plus_830.json), buckets ({difficulty: [qid]},
              configs/splits/cf_bcp150_buckets.json), qrels_dir (the
              BrowseComp-Plus checkout's topics-qrels/), corpus (HYBRID_BCP.json),
              doctext_cache (optional pickle path)
      frames: questions (frames_split.json), split (configs/splits/frames_full824_norm.json),
              corpus (the {title: text} FRAMES corpus)
      monaco: questions (monaco_full1315.json), split (configs/splits/monaco_full1315.json),
              corpus (the {title: text} MoNaCo corpus)
    """

    def __init__(self, name, *, questions, corpus, split=None, buckets=None, qrels_dir=None,
                 doctext_cache=None):
        if name not in ("bcp", "frames", "monaco"):
            raise ValueError(f"Obs. scorer supports bcp, frames, monaco; got {name!r}")
        self.name = name
        self.multi = name == "monaco"
        self.gold, self.qids = {}, []
        self.E = {}                      # BCP evidence qrels; empty elsewhere
        self.has_evidence_qrels = name == "bcp"
        self._by_prefix = {}
        self.doc_text, self.title2doc, self.titles = {}, {}, None
        self._spans = {}
        self.gold_pages_missing = 0
        self._questions, self._corpus, self._split = questions, corpus, split
        self._buckets, self._qrels_dir, self._doctext_cache = buckets, qrels_dir, doctext_cache
        if name == "bcp":
            self._load_bcp()
        elif name == "frames":
            self._load_frames()
        else:
            self._load_monaco()
        self._by_prefix = {}
        for q in self.qids:                       # aliases first so the primary text wins on collision
            for alias in self.gold[q].get("aliases", ()):
                self._by_prefix[norm(alias)[:100]] = q
        for q in self.qids:
            self._by_prefix[norm(self.gold[q]["question"])[:100]] = q

    # -- loading -----------------------------------------------------------
    def _load_bcp(self):
        if not (self._buckets and self._qrels_dir):
            raise ValueError("bcp Obs. needs buckets (cf_bcp150_buckets.json) and qrels_dir")
        diff = {q: b for b, ids in json.load(open(self._buckets)).items() for q in ids}
        q830 = {q["query_id"]: q for q in json.load(open(self._questions))}
        E = load_qrels(os.path.join(self._qrels_dir, "qrel_evidence.txt"))
        G = load_qrels(os.path.join(self._qrels_dir, "qrel_golds.txt"))
        self.qids = sorted(diff)
        for q in self.qids:
            self.gold[q] = dict(question=q830[q]["query"], answer=q830[q]["answer"], gold_pages=set(G.get(q, ())),
                                split=diff[q])
            self.E[q] = set(E.get(q, ()))
        need = set()
        for q in self.qids:
            need |= self.E[q] | self.gold[q]["gold_pages"]
        self.doc_text, self.title2doc = load_doc_text(need, self._corpus, self._doctext_cache)
        self.titles = TitleIndex(self.title2doc)

    def _load_titles(self, corpus_path):
        corpus = json.load(open(corpus_path))
        self.doc_text = {}
        for title, text in corpus.items():
            d = norm(title)
            if d and d not in self.doc_text:
                self.doc_text[d] = text
                self.title2doc[d] = d
        self.titles = TitleIndex(self.title2doc)

    def _pages(self, raw, key=None):
        v = _literal(raw) if isinstance(raw, str) else raw
        out = set()
        for x in v or []:
            t = x.get(key) if isinstance(x, dict) else x
            d = self.docid_of(str(t or ""))
            if d:
                out.add(d)
            else:
                self.gold_pages_missing += 1
        return out

    def _split_ids(self):
        return set(json.load(open(self._split))["question_ids"])

    def _load_frames(self):
        self._load_titles(self._corpus)
        ids = self._split_ids()
        for r in json.load(open(self._questions)):
            q = str(r["id"])
            if q not in ids:
                continue
            self.qids.append(q)
            # 184 questions carry a context/test_question split; the baselines were
            # given original_prompt (context + question), so index both texts.
            self.gold[q] = dict(question=r["test_question"], answer=r["answer"],
                                aliases=[r["original_prompt"]] if r.get("original_prompt") != r["test_question"] else [],
                                gold_pages=self._pages(r.get("articles"), "title"), split=f"{r.get('num_hops')}hop")
        self.qids.sort(key=lambda x: int(x) if x.isdigit() else x)

    def _load_monaco(self):
        self._load_titles(self._corpus)
        ids = self._split_ids()
        for r in json.load(open(self._questions)):
            q = str(r["id"])
            if q not in ids:
                continue
            self.qids.append(q)
            self.gold[q] = dict(question=r["question"], answer=r["answer"],
                                gold_pages=self._pages(r.get("gold_pages")), split=str(r.get("hop")))
        self.qids.sort(key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x)

    # -- identity ------------------------------------------------------------
    def docid_of(self, key: str):
        if self.name == "bcp":
            return bcp_docid_of(key)
        k = _strip_ref(key)
        m = _CHUNK_IDX.match(k)
        if m:
            k = m.group("title")
        d = norm(k)
        if not d:
            return None
        if d in self.doc_text:
            return d
        return self.titles.prefix(k, min_len=16) if self.titles else None

    def chunk_ref(self, key: str):
        """(docid, chunk index | None) for a chunk id / link."""
        k = _strip_ref(key)
        if self.name == "bcp":
            return bcp_docid_of(k), None
        m = _CHUNK_IDX.match(k)
        d = self.docid_of(k)
        return d, (int(m.group("idx")) if (m and d) else None)

    def q_by_text(self, text: str):
        t = norm(text)[:100]
        if t in self._by_prefix:
            return self._by_prefix[t]
        for k, v in self._by_prefix.items():
            if t[:60] == k[:60]:
                return v
        return None

    # -- corpus text ---------------------------------------------------------
    def ensure_text(self, need: set):
        if self.name == "bcp":
            extra = need - set(self.doc_text)
            if extra:
                self.doc_text, self.title2doc = load_doc_text(set(self.doc_text) | extra, self._corpus,
                                                              self._doctext_cache)
                self.titles = TitleIndex(self.title2doc)

    def title_of(self, docid) -> str:
        """normalised title of a document (a received document's title is
        received text: it is in the chunk id / search listing / link)."""
        if self.name != "bcp":
            return docid                        # docid == norm(title)
        if not hasattr(self, "_doc2title"):
            self._doc2title = {}
            for t, d in self.title2doc.items():
                self._doc2title.setdefault(d, t)
        return self._doc2title.get(docid, "")

    def text_of(self, docid, idx=None, cap=None):
        art = self.doc_text.get(docid) or ""
        if idx is not None and self.name != "bcp":
            if docid not in self._spans:
                from panini_dr.retrieval.corpus import _split_spans, DEFAULT_CHUNK_CHARS, DEFAULT_CHUNK_OVERLAP
                self._spans[docid] = _split_spans(art, DEFAULT_CHUNK_CHARS, DEFAULT_CHUNK_OVERLAP)
            sp = self._spans[docid]
            if 0 <= idx < len(sp):
                art = art[sp[idx][0]:sp[idx][1]]
        return art[:cap] if cap else art



# ----------------------------------------------------------------------------
# adapters → {qid: dict(pred, events=[{docid, ref, text, channel, src}], offcorpus=[{text, channel}], cited=set)}
#   channel: search | read | find | lookup | cited | self | visit | summary | resolver | grounding
#   src:     delivered (text the model saw) | span (verified citation span) | corpus (docid only)
#            | resolver (text saved verbatim in the trace, validated against the corpus body before it counts)
# ----------------------------------------------------------------------------
def _ev(ds, key, text, channel, src="delivered", spans=None, distilled=None):
    d, idx = ds.chunk_ref(key)
    if not d:
        return None
    e = dict(docid=d, ref=idx, text=text or "", channel=channel, src=src)
    if spans is not None:
        e["spans"] = spans
    if distilled is not None:
        e["distilled"] = distilled
    return e


def _cited_ids(t):
    """chunk ids our pipeline committed as evidence (update_blank /
    bind_connection / report_finding args, dispatch_subplan and
    dispatch_research results, exported evidence map).

    Two channels per orchestrator shape, because a committed citation can reach
    the trace either way. The full system: `update_blank` / `bind_connection`
    args, plus the `resolved` map of a `dispatch_subplan` result (a blank a
    sub-plan resolved is bound inside the dispatch, with no tool call of its
    own at this level). The no-GSW control (2026-09-06 ablation): the
    `report_finding` args of a worker's extended loop, plus the `reports` list
    of a `dispatch_research` result -- a worker whose cheap first look succeeds
    commits a finding WITHOUT the model ever emitting `report_finding`, so the
    dispatch result is the only place those citations appear. Reading only the
    tool-call channel would bias the control's observed support downwards by
    every finding the first look resolved.

    The control does not RELY on its `report_finding` branch: its dispatch result
    returns every report of the dispatch, first-look and loop-resolved alike, so
    that branch adds no citation the `reports` channel lacks. It is kept for a
    future arm with no dispatch wrapper, and it must never add an id the arm did
    not commit: an id here becomes a corpus-sourced event that `_delivered_texts`
    expands into the whole document body, so one spurious real id can flip an
    answer item from unsupported to supported. The control's one such path is the
    closed-assignment echo (`nogsw/researcher.py:362-366`): a second
    `report_finding` in the same turn is answered ok-true with a note, and nothing
    constrains its ids to the committed ones. Those results are skipped.
    """
    ids = []
    for c in t.get("tool_calls") or []:
        name, res = c.get("name"), c.get("result_full") or c.get("result_preview") or ""
        s = res if isinstance(res, str) else json.dumps(res)
        ok = '"ok": true' in s or '"ok":true' in s
        if name in ("update_blank", "bind_connection") and ok:
            ids += list((c.get("args") or {}).get("evidence_chunk_ids") or [])
        elif name == "report_finding" and ok and "already reported" not in s:
            # `s` is the serialised result; that phrase occurs only in the note of
            # the closed-assignment echo, whose ids were never committed (see above)
            ids += list((c.get("args") or {}).get("evidence_chunk_ids") or [])
        elif name in ("dispatch_subplan", "dispatch_research"):
            key = "resolved" if name == "dispatch_subplan" else "reports"
            recs = _as_dict(res).get(key) or {}
            for rec in (recs.values() if isinstance(recs, dict) else recs):
                if isinstance(rec, dict):
                    ids += list(rec.get("evidence_chunk_ids") or [])
    edges = ((t.get("extra") or {}).get("gsw_state") or {}).get("evidence_edges") or {}
    for rec in (edges.values() if isinstance(edges, dict) else edges):
        if isinstance(rec, dict):
            ids += list(rec.get("evidence_chunk_ids") or ([rec.get("chunk_id")] if rec.get("chunk_id") else []))
    return [str(x) for x in ids]


def _parse_passage(ps):
    """Split a grounding passage `[<chunk id>] <text>` into (chunk id, text).

    The id is everything inside the OUTERMOST bracket pair. A BrowseComp-Plus chunk id carries its own
    bracketed docid (`[Tegla Loroupe - Wikipedia [65939]__0] ...`), so the previous rule — the id ends
    at the first `]` — truncated it to `Tegla Loroupe - Wikipedia [65939`, the key never resolved, and
    the passage was silently dropped: 1,047 of 1,103 on the cap-5 BCP run (fixed 2026-09-23; 1,092
    resolve now). FRAMES and MoNaCo ids contain no brackets and parse exactly as before. No reported
    Obs. number changed: every question those passages support was already supported by another channel.
    """
    s = str(ps or "").lstrip()
    if not s.startswith("["):
        return None
    depth = 0
    for i, ch in enumerate(s):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                key = s[1:i].strip()
                return (key, s[i + 1:].lstrip()) if 0 < len(key) <= 300 else None
    return None


def _subagent_events(ds, t):
    """Corpus text our pipeline put in front of a sub-model without emitting a tool call.

    The typed resolver and the grounding checker retrieve and render chunk text into their own
    prompts; the trace saves it verbatim (`gsw_checker_calls[resolver].retrieved_chunks[].text`,
    `[grounding].passages_shown[]`). It is scored as src="resolver", which `_delivered_texts`
    credits only where the saved string is verbatim in the corpus body of the document it is
    attributed to — so this channel can never credit text the system wrote itself."""
    out = []
    for c in (t.get("extra") or {}).get("gsw_checker_calls") or []:
        kind = c.get("kind")
        if kind == "resolver":
            for ch in c.get("retrieved_chunks") or []:
                if not isinstance(ch, dict):
                    continue
                key, txt = str(ch.get("chunk_id") or ch.get("title") or ""), str(ch.get("text") or "")
                e = _ev(ds, key, txt, "resolver", src="resolver") if key and txt else None
                if e:
                    out.append(e)
        elif kind == "grounding":
            for ps in c.get("passages_shown") or []:
                p = _parse_passage(ps)
                if not p:
                    continue
                key, txt = p
                e = _ev(ds, key, txt, "grounding", src="resolver") if key and txt else None
                if e:
                    out.append(e)
    return out


def _tool_events(ds, name, args, res, ev):
    """Shared by the harness and evidence-file adapters: one tool call → events."""
    if name == "search":
        for h in _search_hits(res):
            e = _ev(ds, h.get("chunk_id") or h.get("title") or "", h.get("text") or "", "search")
            if e:
                ev.append(e)
    elif name == "read":
        r = _as_dict(res)
        key = args.get("chunk_id") or args.get("title") or ""
        if not key or not isinstance(r, dict):
            return
        if r.get("article_text"):
            e = _ev(ds, key, r["article_text"], "read")
        elif r.get("content") is not None:
            e = _ev(ds, key, r.get("content") or "", "read")
        elif r.get("citations") or r.get("answer"):
            spans = [str(x.get("span") or "") for x in (r.get("citations") or []) if isinstance(x, dict)]
            e = _ev(ds, key, "", "read", src="span", spans=spans, distilled=str(r.get("answer") or ""))
        else:
            e = None
        if e:
            ev.append(e)
    elif name == "find":
        r = _as_dict(res)
        key = args.get("chunk_id") or ""
        if key and isinstance(r, dict):
            sn = " ".join(str(s.get("snippet") or "") for s in (r.get("snippets") or []) if isinstance(s, dict))
            e = _ev(ds, key, sn, "find") if sn else None
            if e:
                ev.append(e)
    elif name == "lookup":
        key = args.get("title") or args.get("chunk_id") or ""
        txt = res if isinstance(res, str) else json.dumps(res)
        e = _ev(ds, key, txt, "lookup") if key else None
        if e:
            ev.append(e)


def adapt_harness(spec, ds):
    seen = {}
    pats = spec["traces"] if isinstance(spec["traces"], list) else [spec["traces"]]
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            t = json.load(open(p))
            qid = str(t.get("question_id"))
            if qid not in ds.gold or qid in seen:
                continue
            ev = []
            for c in t.get("tool_calls") or []:
                _tool_events(ds, c.get("name"), c.get("args") or {}, c.get("result_full") or c.get("result_preview") or "", ev)
            ev += _subagent_events(ds, t)
            cited = set()
            for cid in _cited_ids(t):
                e = _ev(ds, cid, "", "cited", src="corpus")
                if e:
                    cited.add(e["docid"])
                    ev.append(e)
            seen[qid] = dict(qid=qid, pred=str(t.get("final_answer") or ""), events=ev, offcorpus=[], cited=cited)
    return seen



_EF_HDR = re.compile(r"^### \[(?P<kind>[^\]]+)\] (?P<rest>.*)$", re.M)
_EF_TOOL = re.compile(r"^(?P<name>\w+)\((?P<args>.*)\)\s*$", re.S)
_EF_ARG = re.compile(r"(\w+)=(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|[^,()]+)")
_EF_READFULL = re.compile(r"^(?P<title>.*) chars (?P<s>\d+)-(?P<e>\d+)\s*$", re.S)


def _ef_args(s):
    out = {}
    for k, v in _EF_ARG.findall(s or ""):
        v = v.strip()
        if v[:1] in "\"'":
            try:
                v = ast.literal_eval(v)
            except Exception:
                v = v[1:-1]
        out[k] = v
    return out


def _evidence_text_events(ds, raw):
    """Events of one packet's evidence text (verbatim body of the research
    ``adapt_evidence_file``):
      ### [tool] name(args)\\n<result>       search / read / find / lookup
      ### [resolver evidence chunk] <cid>    a chunk our pipeline committed
      ### [reader read_full] <title> chars a-b   a window the reader read
    Header-less files: blank-line-separated JSON blocks of search results."""
    ev, cited = [], set()
    hdrs = list(_EF_HDR.finditer(raw))
    if hdrs:
        for i, m in enumerate(hdrs):
            body = raw[m.end():hdrs[i + 1].start() if i + 1 < len(hdrs) else len(raw)].strip()
            kind, rest = m.group("kind"), m.group("rest").strip()
            if kind == "tool":
                tm = _EF_TOOL.match(rest)
                if not tm:
                    continue
                _tool_events(ds, tm.group("name"), _ef_args(tm.group("args")), body, ev)
            elif kind == "resolver evidence chunk":
                e = _ev(ds, rest, body, "cited")
                if e:
                    cited.add(e["docid"])
                    ev.append(e)
            elif kind == "reader read_full":
                rm = _EF_READFULL.match(rest)
                e = _ev(ds, rm.group("title") if rm else rest, body, "read")
                if e:
                    ev.append(e)
    else:
        for block in raw.split("\n\n"):
            b = block.strip()
            if not b.startswith("["):
                continue
            try:
                hits = json.loads(b)
            except Exception:
                continue
            _tool_events(ds, "search", {}, hits, ev)
    return ev, cited


def adapt_evidence_file(spec, ds):
    """Existing packet files: <arm>_<qid>.json (answer) + .evidence.txt (or an
    inline ``evidence`` string). ``raw_traces``, when given, is the run the
    packets were built from: its resolver/grounding chunks are added through
    `_subagent_events` (scored as src="resolver")."""
    seen = {}
    raw_by_qid = {}
    for pat in ([spec["raw_traces"]] if isinstance(spec.get("raw_traces"), str) else (spec.get("raw_traces") or [])):
        for rp in sorted(glob.glob(pat)):
            raw_by_qid.setdefault(os.path.basename(rp)[2:-5], rp)
    for p in sorted(glob.glob(spec["traces"])):
        base = os.path.basename(p)
        if base.startswith("_") or not base.endswith(".json"):
            continue
        pk = json.load(open(p))
        qid = str(pk.get("qid"))
        if qid not in ds.gold or qid in seen:
            continue
        if isinstance(pk.get("evidence"), str):
            raw = pk["evidence"]
        else:
            ef = pk.get("evidence_file") or p[:-5] + ".evidence.txt"
            if not os.path.exists(ef):
                ef = os.path.join(os.path.dirname(p), os.path.basename(ef))
            raw = open(ef, errors="ignore").read() if os.path.exists(ef) else ""
        ev, cited = _evidence_text_events(ds, raw)
        rp = raw_by_qid.get(qid)
        if rp:
            ev += _subagent_events(ds, json.load(open(rp)))
        seen[qid] = dict(qid=qid, pred=str(pk.get("answer") or ""), events=ev, offcorpus=[], cited=cited)
    return seen


# ----------------------------------------------------------------------------
# MoNaCo packets, built in memory from the traces exactly like the research
# builder (eval_tools/grounding_audit/build_monaco_audit_inputs.py)
# ----------------------------------------------------------------------------
PACKET_TOOLS = {"search", "read", "find", "ask", "read_raw", "lookup"}
_READ_FULL_RE = re.compile(r"read_full\(offset=(\d+)\)\s*->\s*chars\s*(\d+)-(\d+)")


def build_packet_evidence(t, chunk_text_by_id, articles):
    """The evidence string of one question's packet (three channels: tool
    results; corpus text of every chunk in ``extra.executed_blanks[].evidence_chunk_ids``;
    the article spans the reader sub-agent paged through, from
    ``extra.agentic_read.records[].sections_checked``)."""
    parts, cited = [], []
    for tc in t.get("tool_calls") or []:
        if tc.get("name") not in PACKET_TOOLS:
            continue
        body = tc.get("result_full") or tc.get("result_preview") or ""
        if not str(body).strip():
            continue
        args = tc.get("args") or {}
        argstr = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)[:120]}"
                           for k, v in args.items())
        parts.append(f"### [tool] {tc['name']}({argstr})\n{body}\n")
    for blank in t.get("extra", {}).get("executed_blanks") or []:
        for cid in blank.get("evidence_chunk_ids") or []:
            if cid in cited:
                continue
            cited.append(cid)
            txt = chunk_text_by_id(cid)
            if txt is not None:
                parts.append(f"### [resolver evidence chunk] {cid}\n{txt}\n")
    spans: dict[str, list[tuple[int, int]]] = {}
    for rec in (t.get("extra", {}).get("agentic_read") or {}).get("records") or []:
        cid = rec.get("chunk_id") or ""
        title = cid.rsplit("__", 1)[0]
        if not title:
            continue
        for line in rec.get("sections_checked") or []:
            m = _READ_FULL_RE.search(str(line))
            if m:
                spans.setdefault(title, []).append((int(m.group(2)), int(m.group(3))))
    for title, rngs in spans.items():
        art = articles.get(title) or ""
        if not art:
            continue
        merged: list[list[int]] = []
        for s, e in sorted(rngs):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        for s, e in merged:
            parts.append(f"### [reader read_full] {title} chars {s}-{e}\n{art[s:e]}\n")
    return "===== EVIDENCE THE ARM ACTUALLY RETRIEVED =====\n" + "\n".join(parts)


def adapt_packets_from_traces(spec, ds):
    """MoNaCo: build each question's packet from its trace (in memory) and read
    it like a packet file, plus the trace's resolver/grounding chunks."""
    from panini_dr.retrieval.corpus import load_frames_corpus

    by_id = None
    _articles: dict = {}

    class _LazyArticles:
        def get(self, title, default=None):
            if not _articles:
                _articles.update(json.load(open(spec["corpus"])))
            return _articles.get(title, default)

    articles = _LazyArticles()

    def chunk_text(cid):
        nonlocal by_id
        if by_id is None:
            by_id = {c.chunk_id: c.text for c in load_frames_corpus(cache_path=spec["corpus"]).chunks}
        return by_id.get(cid)

    seen = {}
    pats = spec["traces"] if isinstance(spec["traces"], list) else [spec["traces"]]
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            t = json.load(open(p))
            qid = str(t.get("question_id"))
            if qid not in ds.gold or qid in seen:
                continue
            ev, cited = _evidence_text_events(ds, build_packet_evidence(t, chunk_text, articles))
            ev += _subagent_events(ds, t)
            seen[qid] = dict(qid=qid, pred=str(t.get("final_answer") or ""), events=ev, offcorpus=[], cited=cited)
    return seen


def run_adapter(spec, ds):
    a = spec["adapter"]
    if a == "harness":
        return adapt_harness(spec, ds)
    if a == "evidence_file":
        return adapt_evidence_file(spec, ds)
    if a == "packets_from_traces":
        return adapt_packets_from_traces(spec, ds)
    raise ValueError(a)


# ----------------------------------------------------------------------------
# per-question scoring
# ----------------------------------------------------------------------------
_BODY_MEMO = {}


def _norm_body(ds, docid, ref):
    """norm() of a document's corpus body, memoised: span and resolver validation asks for the same
    bodies thousands of times per run and the corpus does not change inside one."""
    k = (id(ds), docid, ref)
    v = _BODY_MEMO.get(k)
    if v is None:
        v = norm(ds.text_of(docid, ref) or "")
        if len(_BODY_MEMO) < 200000:           # bounded: the tail is cheap to recompute
            _BODY_MEMO[k] = v
    return v


def _delivered_texts(rec, ds, hit_cap):
    """docid → normalised text the system received for that document
    (its title, then every delivered snippet / chunk / verified span)."""
    delivered = defaultdict(list)
    for e in rec["events"]:
        d = e["docid"]
        if e["src"] == "corpus":
            txt = ds.text_of(d, e.get("ref"), hit_cap)
        elif e["src"] == "span":
            body = _norm_body(ds, d, e.get("ref"))
            txt = " ".join(x for x in e.get("spans") or [] if norm(x) and norm(x) in body)
        elif e["src"] == "resolver":
            # saved verbatim in the trace; counted only where the corpus confirms it is that
            # document's own text, so a sub-model's prompt can never launder model-written prose
            body = _norm_body(ds, d, e.get("ref"))
            tn = norm(e["text"])
            txt = e["text"] if tn and tn in body else ""
        else:
            txt = e["text"]
        if txt:
            delivered[d].append(txt)
    return {d: norm(ds.title_of(d) + " " + " ".join(t)) for d, t in delivered.items()}


# v2.2 (2026-09-05): the short-item proximity rule is OFF. It removed one audited false positive but rejected
# genuine bare-number answers whose page lacked a question word nearby (e.g. FRAMES q13 "1960" in the Belgian
# Congo infobox). Word-boundary matching alone removes the digit-substring and word-substring errors.
_V2_SHORT_ITEM_CONTEXT = False
_V2_STOP = set("about above after again against being below between could every first found from having "
               "into itself least might most much never often other over same should since still their there "
               "these those through under until using where which while whose would which what when".split())


def _v2_question_tokens(question: str) -> set[str]:
    return {t for t in norm(question).split() if len(t) >= 5 and t not in _V2_STOP}


def _support(parts, kinds, texts_n, texts_ns=None, qtoks=None):
    """item_support plus the same-document rule: a numeric part is only
    supported by a document that also carries one of the item's text parts
    (bare numbers occur everywhere). v2 adds: a single-part item that is a bare
    number or one short word (<= 6 letters) is supported only by a document that
    also contains a content token of the question (co-occurrence rule)."""
    s = item_support(parts, kinds, texts_n, texts_ns)
    checkable = [(vs, k) for vs, k in zip(parts, kinds) if k != "bool"]
    if s["supported"] and qtoks is not None and len(checkable) == 1 and _V2_SHORT_ITEM_CONTEXT:
        vs, k = checkable[0]
        short = k == "number" or (len(vs[0].split()) == 1 and len(vs[0]) <= 6)
        if short:
            def near(d):
                tn = f" {texts_n.get(d, '')} "
                for v in vs:
                    start = 0
                    while True:
                        i = tn.find(f" {v} ", start)
                        if i < 0: break
                        win = tn[max(0, i - 200): i + len(v) + 202]
                        if any(f" {t} " in win for t in qtoks): return True
                        start = i + 1
                return False
            ok = {d for d in s["docs"] if near(d)}
            if not ok:
                s["supported"] = False
                s["short_item_no_context"] = True
            else:
                s["docs"] = ok
    if s["supported"] and "number" in kinds and "text" in kinds:
        text_docs = set().union(*[h for h, k in zip(s["per_part"], kinds) if k == "text" and h])
        for h, k in zip(s["per_part"], kinds):
            if k == "number" and not (h & text_docs):
                s["supported"] = False
                s["number_off_doc"] = True
                break
    return s


def _derivation_class(s: dict, kinds: list[str]) -> bool:
    """True when the item is unsupported ONLY through parts that are numbers
    or booleans while every text part is on a received document: the answer
    is a comparison / count / arithmetic result whose inputs were observed
    but whose value cannot be checked by string matching. Reported as the
    upper bracket `earned⁺`; the strict `earned` counts it unsupported."""
    if s["supported"]:
        return False
    if s["uncheckable"]:                       # bool-only answer (comparison)
        return True
    text_ok = all(h for h, k in zip(s["per_part"], kinds) if k == "text")
    has_text = any(k == "text" for k in kinds)
    return text_ok and (not has_text or "number" in kinds or "bool" in kinds)


def score_question(qid, rec, J, ds: Dataset, spec):
    g = ds.gold[qid]
    G, E = g["gold_pages"], ds.E.get(qid, set())
    pred = rec["pred"] if rec.get("pred") else ((J or {}).get("pred") or "")
    # single-answer benchmarks: score the string the grader graded (its extracted final answer,
    # else the 'Exact Answer:' field, else the whole prediction) so a prose answer is tested on
    # its answer, not on its formatting. MoNaCo splits the whole prediction into items.
    ans = final_answer_string(pred, (J or {}).get("extracted")) if ds.name in ("bcp", "frames") else pred
    items = split_items(ans, ds.multi)
    blank = not items
    golds = gold_items(ds.name, g)
    glen = len(golds) if ds.multi else 1

    texts_n = _delivered_texts(rec, ds, spec["hit_cap"])
    off_n = {f"off:{i}": norm(o["text"]) for i, o in enumerate(rec.get("offcorpus") or []) if o.get("text")}
    all_n = dict(texts_n, **off_n)
    all_ns, texts_ns = _spaceless(all_n), _spaceless(texts_n)

    det, sup, sup_corpus, uncheck, num_only, answer_docs, comp_unsup = [], [], [], [], [], set(), 0
    derived = []      # unsupported only in its number/bool parts: a derivation, not string-checkable
    for it in items:
        parts, kinds = item_parts_and_kinds(it)
        qtoks = _v2_question_tokens(g.get("question") or "")
        s_all = _support(parts, kinds, all_n, all_ns, qtoks)
        s_cor = _support(parts, kinds, texts_n, texts_ns, qtoks) if off_n else s_all
        det.append(any(item_matches_gold(it, gd) for gd in golds))
        sup.append(bool(s_all["supported"]))
        sup_corpus.append(bool(s_cor["supported"]))
        uncheck.append(bool(s_all["uncheckable"]))
        num_only.append(bool(kinds) and all(k == "number" for k in kinds))
        answer_docs |= {d for d in s_all["docs_any"] if not d.startswith("off:")}
        if not s_all["supported"] and not s_all["uncheckable"] and "number" in kinds:
            comp_unsup += 1
        derived.append(_derivation_class(s_all, kinds))
    sup_plus = [a or b for a, b in zip(sup, derived)]

    # official score and its earned counterpart, per judge pass, max over passes
    passes = (J or {}).get("passes") or []
    official, earned, lo, hi, earned_plus, n_items_mismatch = 0.0, 0.0, 0.0, 0.0, 0.0, False
    if passes:
        best = None
        for ps in passes:
            f1 = float(ps.get("f1") or 0.0)
            if not ds.multi:
                s = bool(sup) and all(sup)                 # single answer: every part supported
                e_f1, e_lo, e_hi = (f1 if s else 0.0,) * 3
                e_plus = f1 if (bool(sup_plus) and all(sup_plus)) else 0.0
            else:
                nc, plen = float(ps.get("nc") or 0), float(ps.get("pred_len") or 0)
                if items and plen and abs(plen - len(items)) > max(1, 0.5 * len(items)):
                    n_items_mismatch = True
                plen = max(plen, len(items)) if items else plen
                if nc > len(items):
                    # the judge matched more items than the deterministic split
                    # finds (prose answer): the invisible items cannot be
                    # supported, so only the visible ones can earn credit
                    n_items_mismatch, nc = True, float(len(items))
                e_nc, e_lo_nc, e_hi_nc = earned_counts(nc, plen, det, sup)
                e_f1 = prf(e_nc, plen, glen)[2]
                e_lo, e_hi = prf(e_lo_nc, plen, glen)[2], prf(e_hi_nc, plen, glen)[2]
                e_plus = prf(earned_counts(nc, plen, det, sup_plus)[0], plen, glen)[2]
            cand = (f1, e_f1, e_lo, e_hi, e_plus)
            if best is None or cand[:2] > best[:2]:
                best = cand
        official, earned, lo, hi, earned_plus = best
    correct = (J or {}).get("correct")
    if correct is None and passes:
        correct = official >= 0.5 if glen == 1 else None
    det_correct = bool(det) and all(det) if not ds.multi else None

    surfaced = {e["docid"] for e in rec["events"]}
    cited = set(rec.get("cited") or set())
    used = cited if spec["cites"] else answer_docs

    def ratio(a, b):
        return (len(a & b) / len(b)) if b else None

    def prec(a, b):
        return (len(a & b) / len(a)) if a else None

    return dict(
        qid=qid, split=g["split"], answer=ans[:300], n_items=len(items), glen=glen, blank=blank,
        crashed=bool(rec.get("crashed")), state_lost=bool(rec.get("state_lost")), judged=bool(passes), correct=correct, det_correct=det_correct,
        official=official, earned=earned, earned_lo=lo, earned_hi=hi, earned_plus=earned_plus,
        n_supported=sum(sup), n_derived=sum(derived), supported_all_plus=bool(sup_plus) and all(sup_plus), n_supported_corpus=sum(sup_corpus), n_uncheckable=sum(uncheck),
        n_number_only=sum(num_only), n_det_correct=sum(det), computed_unsupported=comp_unsup,
        supported_all=bool(sup) and all(sup), n_items_mismatch=n_items_mismatch,
        n_surfaced=len(surfaced), n_offcorpus=len(off_n), n_cited=len(cited), n_answer_docs=len(answer_docs),
        found_gold_R=ratio(surfaced, G), found_evid_R=ratio(surfaced, E) if ds.has_evidence_qrels else None,
        used_gold_P=prec(used, G), used_gold_R=ratio(used, G),
        used_evid_P=prec(used, E) if ds.has_evidence_qrels else None,
        used_evid_R=ratio(used, E) if ds.has_evidence_qrels else None,
        trace_missing=bool(rec.get("missing")),
        items=[dict(text=(" | ".join(map(str, it)) if isinstance(it, (list, tuple)) else str(it))[:160],
                    kinds="".join(k[0] for k in item_parts_and_kinds(it)[1]),
                    det=d, supported=sp, corpus=sc, derived=dv, uncheckable=u)
               for it, d, sp, sc, dv, u in zip(items, det, sup, sup_corpus, derived, uncheck)],
    )


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(rows, has_judge=True):
    n = len(rows)
    if not n:
        return {}
    items = sum(r["n_items"] for r in rows)
    judged = [r for r in rows if r["judged"]]
    corr = [r for r in judged if r["correct"]]
    out = dict(
        n=n, n_judged=len(judged), n_blank=sum(r["blank"] for r in rows), trace_missing=sum(r["trace_missing"] for r in rows),
        n_crashed=sum(r["crashed"] for r in rows), n_state_lost=sum(r["state_lost"] for r in rows),
        official=sum(r["official"] for r in judged) / n if has_judge else None,
        earned=sum(r["earned"] for r in judged) / n if has_judge else None,
        earned_lo=sum(r["earned_lo"] for r in judged) / n if has_judge else None,
        earned_hi=sum(r["earned_hi"] for r in judged) / n if has_judge else None,
        earned_plus=sum(r["earned_plus"] for r in judged) / n if has_judge else None,
        earned_det_plus=sum(bool(r["det_correct"]) and r["supported_all_plus"] for r in rows) / n,
        derived_items=(sum(r["n_derived"] for r in rows) / items) if items else None,
        acc=len(corr) / n if has_judge else None,
        det_acc=sum(bool(r["det_correct"]) for r in rows) / n,
        earned_det=sum(bool(r["det_correct"]) and r["supported_all"] for r in rows) / n,
        supported_q=sum(r["supported_all"] for r in rows) / n,
        supported_items=(sum(r["n_supported"] for r in rows) / items) if items else None,
        supported_items_corpus=(sum(r["n_supported_corpus"] for r in rows) / items) if items else None,
        uncheckable_items=(sum(r["n_uncheckable"] for r in rows) / items) if items else None,
        number_only_items=(sum(r["n_number_only"] for r in rows) / items) if items else None,
        computed_unsupported=sum(r["computed_unsupported"] for r in rows),
        correct_unsupported=sum(1 for r in corr if not r["supported_all"]) / n if has_judge else None,
        n_items_mismatch=sum(r["n_items_mismatch"] for r in rows),
        n_surfaced=mean([r["n_surfaced"] for r in rows]),
        n_offcorpus=sum(r["n_offcorpus"] for r in rows),
        found_gold_R=sum((r["found_gold_R"] or 0) for r in rows) / n,
        found_evid_R=mean([r["found_evid_R"] for r in rows]),
        used_gold_P=mean([r["used_gold_P"] for r in rows]),
        used_gold_R=sum((r["used_gold_R"] or 0) for r in rows) / n,
        used_evid_P=mean([r["used_evid_P"] for r in rows]),
        used_evid_R=mean([r["used_evid_R"] for r in rows]),
        n_used=mean([r["n_cited"] if r["n_cited"] else r["n_answer_docs"] for r in rows]),
    )
    return out


# ----------------------------------------------------------------------------
# MoNaCo: the item-level support rule of record (package 35, build_items_v15.py)
# over the judge's credited items. Functions below are verbatim.
# ----------------------------------------------------------------------------
CLS_SUP = 'supported'
CLS_NOMAP = 'unsupported_no_gold_mapping'
CLS_TEXT = 'unsupported_text_no_gold_string_in_evidence'
CLS_NUM = 'unsupported_numeric_boolean'


def components(g):
    """The components of ONE gold item, without collapsing a tuple into independent strings."""
    if isinstance(g, (list, tuple)):
        out = []
        for y in g:
            out.extend(components(y))
        return out
    return [str(g)]


def gold_text(g):
    return ' '.join(components(g))


def toks(s):
    return {t for t in re.findall(r'[a-z0-9]+', norm(s)) if len(t) >= 3}


def nearest_gold_item(s, gold_items):
    """Return (index, gold item) of the nearest gold ITEM, tuples kept intact."""
    sn = norm(s); stk = toks(s); best, bi, bs = None, None, 0.0
    for i, g in enumerate(gold_items):
        gt = gold_text(g); gn = norm(gt)
        if not gn:
            continue
        if gn == sn or gn in sn or sn in gn:
            return i, g
        u = stk | toks(gt)
        j = len(stk & toks(gt)) / len(u) if u else 0.0
        if j > bs:
            best, bi, bs = g, i, j
    return (bi, best) if bs >= 0.5 else (None, None)


def gold_parts(g):
    """parts/kinds for ONE gold item, component by component, so the scorer's own rules apply:
    word-boundary matching, the same-document rule for numeric parts (a bare number is only
    supported by a document that also carries one of the item's text parts), and the short-item
    co-occurrence rule. Replaces a plain substring test over the concatenated evidence, which
    accepted 'True' inside 'Truelove' and paired a name on one page with a number on another."""
    parts, kinds = [], []
    for c in components(g):
        p, k = item_parts_and_kinds(str(c))
        parts.extend(p); kinds.extend(k)
    return parts, kinds


PROXIMITY_CHARS = 600


def _positions(variants, text):
    """Character positions of any accepted surface variant, word-boundary aware."""
    out = []
    for v in variants:
        n = norm(str(v))
        if not n:
            continue
        pad = f' {text} '; start = 0
        while True:
            i = pad.find(f' {n} ', start)
            if i < 0:
                break
            out.append(i); start = i + 1
    return sorted(out)


LEAD_CHARS = 400


def _title_anchored(parts, kinds, docid, text):
    """A text part that is the document's title, or that occurs in the document's lead (title plus the
    first LEAD_CHARS characters, where a page names its subject and its aliases: 'Citalopram, sold
    under the brand name Celexa'), anchors the whole document: an attribute anywhere on that page is
    taken as associated with the entity. A page titled after a different entity (a cast list) gives
    no anchor, so its values must sit near the name."""
    tn = norm(docid); lead = f' {text[:LEAD_CHARS]} '
    for vs, k in zip(parts, kinds):
        if k != 'text':
            continue
        for v in vs:
            n = norm(str(v))
            if n and (n == tn or f' {n} ' in lead):
                return True
    return False


def parts_supported(parts, kinds, alln, allns, qt):
    """The scorer's own type-aware test over ONE claim's parts, plus association for multi-part
    claims: the parts must appear within PROXIMITY_CHARS of each other on one document, unless a text
    part is that document's title. Returns (supported, detail)."""
    if not parts:
        return False, None
    s = _support(parts, kinds, alln, allns, qt)
    s = dict(s); s['tested_parts'] = parts; s['tested_kinds'] = kinds
    if not s['supported'] or len(parts) < 2:
        return bool(s['supported']), s
    for d in sorted(s.get('docs') or ()):
        t = alln.get(d, '')
        if _title_anchored(parts, kinds, d, t):
            s = dict(s); s['docs'] = {d}; s['association'] = 'title_anchored'; return True, s
        pos = [_positions(vs, t) for vs in parts]
        if any(not p for p in pos):
            continue
        if any(all(any(abs(x - a) <= PROXIMITY_CHARS for x in p) for p in pos) for a in pos[0]):
            s = dict(s); s['docs'] = {d}; s['association'] = 'proximity'; return True, s
    s = dict(s); s['supported'] = False; s['components_not_adjacent'] = True
    return False, s


def gold_components_for(credited, gsel, nparts):
    """The gold components that correspond to the credited claim: the best-overlapping component per
    credited part, never the whole gold row. Crediting 'Hungary' against a row holding area,
    population and capital is a claim about Hungary, not about the row."""
    comps = [c for c in components(gsel) if norm(str(c))]
    if len(comps) <= nparts:
        return comps
    ct = toks(str(credited))
    scored = sorted(comps, key=lambda c: -(len(ct & toks(str(c))) / max(1, len(ct | toks(str(c))))))
    return scored[:max(1, nparts)]


CLS_UNTESTABLE = 'unsupported_association_untestable'
CARRIER_MAX_PARTS = 3
_CLAIM_NUM = re.compile(r'(?<![\w.])[-\u2212]?\d[\d,]*(?:\.\d+)?(?![\w.])')


def claim_parts(claim):
    """The scorer's parts for a claim, with numbers split out as their own numeric parts.

    The split runs on the ORIGINAL claim string, before the scorer's normalisation. Normalisation
    turns '20.7%' into '20 7', so splitting the normalised surface produced two numbers, 20 and 7,
    where the claim makes one. Splitting first keeps '20.7', '316,000' and '1,366,155' intact.

    The scorer already separates a number that follows a dash or colon ('Matthew Fox - 60'); it does
    not separate one that simply follows a name ('Matthew Fox 60', 'Philadelphia Warriors (5)'), and
    those were read as a single text part, which hid the claimed number from every numeric rule and
    let the text fallback drop it.
    """
    raw = str(claim)
    nums = _CLAIM_NUM.findall(raw)
    if not nums:
        return item_parts_and_kinds(raw)
    rest = _CLAIM_NUM.sub(' ', raw)
    out_p, out_k = [], []
    if rest.strip(' \t\u2013\u2014-:;,.()%'):
        rp, rk = item_parts_and_kinds(rest)
        for vs, k in zip(rp, rk):
            out_p.append(vs); out_k.append(k)
    for n in nums:
        np_, nk = item_parts_and_kinds(n.replace('\u2212', '-'))
        out_p.extend(np_); out_k.extend(nk)
    return (out_p, out_k) if out_p else item_parts_and_kinds(raw)


def claim_numbers(claim):
    return [n for n in _CLAIM_NUM.findall(str(claim))]



def _merge(parts_a, kinds_a, parts_b, kinds_b):
    """Append parts of b that are not already in a (compared on the primary surface)."""
    have = {norm(str(v[0])) for v in parts_a}
    p, k = list(parts_a), list(kinds_a)
    for vs, kk in zip(parts_b, kinds_b):
        if norm(str(vs[0])) not in have:
            p.append(vs); k.append(kk); have.add(norm(str(vs[0])))
    return p, k


def claim_supported(credited, cands, gsel, alln, allns, qt):
    """Support for the claim the judge credited.

    INVARIANT: the claim's own numeric parts are part of the test in EVERY path. Text may be
    substituted (by the answer item that carries a bare value, or by the gold item's text components
    when the judge accepted a paraphrase); a claimed number never is. Two ways this was violated
    before: the text fallback swapped in the gold name and dropped the claimed value, and the carrier
    swap replaced the claim with an answer item whose own tokenisation hid it ('2017' inside 'FY2017').

    Booleans are uncheckable in the scorer, so a Boolean claim is never carrier-swapped and never
    counts as supported; it falls to the numeric/Boolean class.

    Returns (supported, mode, unit, detail, untestable=False).
    """
    credited = str(credited)
    mapped = [x for x in cands if norm(str(x)) != norm(credited)]
    c_parts, c_kinds = claim_parts(credited)
    c_num = [(vs, k) for vs, k in zip(c_parts, c_kinds) if k == 'number']
    unit, u_parts, u_kinds = credited, c_parts, c_kinds
    # a bare NUMERIC claim may borrow the entity from the small answer item that carries it
    if c_parts and all(k == 'number' for k in c_kinds) and mapped:
        m_parts, m_kinds = claim_parts(str(mapped[0]))
        if any(k == 'text' for k in m_kinds) and len(m_parts) <= CARRIER_MAX_PARTS:
            unit = mapped[0]
            u_parts, u_kinds = _merge(m_parts, m_kinds, [v for v, _ in c_num], [k for _, k in c_num])
    # or from the gold item's text components
    if u_parts and all(k == 'number' for k in u_kinds) and gsel is not None:
        ents = []
        for c in components(gsel):
            pp, kk = item_parts_and_kinds(str(c))
            if pp and all(k == 'text' for k in kk):
                ents.append((pp, kk))
        if ents:
            ents = sorted(ents, key=lambda e: -sum(len(str(v[0])) for v in e[0]))[:2]
            e_parts = [p for pp, _ in ents for p in pp]; e_kinds = [k for _, kk in ents for k in kk]
            u_parts, u_kinds = _merge(e_parts, e_kinds, u_parts, u_kinds)
            unit = ' | '.join(str(v[0]) for v in e_parts) + ' | ' + str(unit)
    ok, det = parts_supported(u_parts, u_kinds, alln, allns, qt)
    if ok:
        return True, ('claim_value_with_entity' if ' | ' in str(unit) else 'claim_text'), unit, det, False
    # fallback: gold TEXT components stand in for the claim's text parts; its numbers are kept
    if gsel is not None and any(k == 'text' for k in u_kinds):
        n_text = sum(1 for k in u_kinds if k == 'text')
        parts, kinds = [], []
        for c in gold_components_for(unit, gsel, n_text):
            p, k = item_parts_and_kinds(str(c))
            if all(kk == 'text' for kk in k):
                parts.extend(p); kinds.extend(k)
        keep_p = [vs for vs, k in zip(u_parts, u_kinds) if k != 'text']
        keep_k = [k for k in u_kinds if k != 'text']
        parts, kinds = _merge(parts, kinds, keep_p, keep_k)
        if parts and any(k == 'text' for k in kinds):
            ok2, det2 = parts_supported(parts, kinds, alln, allns, qt)
            if ok2:
                return True, 'gold_component_text_with_claimed_values', None, det2, False
            return False, None, None, det2, False
    return False, None, None, det, False


def kind_of(g):
    s = gold_text(g).strip().lower()
    if s in ('true', 'false', 'yes', 'no'):
        return 'boolean'
    try:
        float(s.replace(',', '').rstrip('%')); return 'number'
    except Exception:
        return 'text'



def map_to_item(c, items):
    """judge_items_pipeline.map_to_item (verbatim): the answer item a credited string refers to."""
    cn = norm(c)
    if not cn:
        return None
    for it in items:
        if norm(it) == cn:
            return it
    for it in items:
        n = norm(it)
        if cn in n or n in cn:
            return it
    ct = toks(c); best, bs = None, 0.0
    for it in items:
        t = toks(it); j = len(ct & t) / len(ct | t) if (ct | t) else 0
        if j > bs:
            best, bs = it, j
    return best if bs >= 0.5 else None


def score_monaco_question(qid, rec, verdict, ds, hit_cap=None):
    """Selected-pass official F1 and Obs. for one MoNaCo question (build_items_v15 loop body).

    ``verdict`` is a grade_set record ({glen, passes: [{f1, credited, judge_pred_len, raw, ...}]}).
    Returns a row; blank / unjudged questions score 0."""
    pred = ((rec or {}).get('pred') or '').strip()
    if not pred or not verdict or not verdict.get('passes'):
        return dict(qid=qid, pred=bool(pred), official=0.0, earned=0.0, earned_flat_gold_rule=0.0,
                    credited=0, supported=0, selected_pass=None, classes={})
    g = ds.gold[qid]; gold_items_ = gold_items('monaco', g); glen = len(gold_items_)
    texts = _delivered_texts(rec, ds, hit_cap)
    offc = {f'off:{i}': norm(o['text']) for i, o in enumerate(rec.get('offcorpus') or []) if o.get('text')}
    alln = dict(texts, **offc); allns = _spaceless(alln); allt = ' '.join(alln.values())
    qt = _v2_question_tokens(g.get('question') or '')
    items = split_items(pred, True)

    scored = []
    for pi, ps in enumerate(verdict['passes']):
        credited = [] if (glen > 1 and (ps.get('judge_pred_len') or 0) == 0) else list(ps.get('credited') or [])
        if glen == 1 and credited:
            # single answer: test the string the judge extracted, as BCP/FRAMES do
            mx = re.search(r'extracted_final_answer:\s*(.+)', ps.get('raw') or '')
            if mx and mx.group(1).strip() and mx.group(1).strip().lower() != 'none':
                credited = [mx.group(1).strip()]
        det = []
        for c in credited:
            cands = [c]
            mapped = map_to_item(c, items)
            if mapped and norm(mapped) != norm(c):
                cands.append(mapped)
            gi, gsel = nearest_gold_item(c, gold_items_)
            sup, mode_, sup_text, sup_detail, untestable = claim_supported(c, cands, gsel, alln, allns, qt)
            sup_gold_flat = bool(gsel) and all(norm(x) in allt for x in components(gsel) if norm(x))
            if sup:
                cls = CLS_SUP
            elif untestable:
                cls = CLS_UNTESTABLE
            elif gsel is None:
                cls = CLS_NOMAP
            elif kind_of(gsel) in ('number', 'boolean'):
                cls = CLS_NUM
            else:
                cls = CLS_TEXT
            cls_flat = CLS_SUP if (sup_text is not None or sup_gold_flat) else cls
            det.append(dict(credited=c, mapped_item=mapped, gold_index=gi, cls=cls, cls_flat=cls_flat))
        nsup = sum(1 for d in det if d['cls'] == CLS_SUP)
        nsup_flat = sum(1 for d in det if d['cls_flat'] == CLS_SUP)
        ncred = len(det)
        plen = max((ps.get('judge_pred_len') or 0) or max(len(items), ncred), ncred)
        f1 = float(ps.get('f1') or 0.0)
        if glen == 1:
            e_new = f1 if nsup > 0 else 0.0
            e_flat = f1 if nsup_flat > 0 else 0.0
        else:
            e_new = prf(nsup, plen, glen)[2] if plen else 0.0
            e_flat = prf(nsup_flat, plen, glen)[2] if plen else 0.0
        scored.append(dict(pass_index=pi, f1=f1, earned=e_new, earned_flat=e_flat,
                           plen=plen, ncred=ncred, nsup=nsup, det=det))
    # ONE selected pass, deterministic: highest (official, earned), ties broken by pass index
    sel = max(scored, key=lambda x: (x['f1'], x['earned'], -x['pass_index']))
    return dict(qid=qid, pred=True, official=sel['f1'], earned=sel['earned'],
                earned_flat_gold_rule=sel['earned_flat'], credited=sel['ncred'], supported=sel['nsup'],
                selected_pass=sel['pass_index'], judge_pred_len=sel['plen'],
                classes=dict(Counter(d['cls'] for d in sel['det'])), items=sel['det'])


# ----------------------------------------------------------------------------
# judge verdicts (from panini_dr.eval.grade_short / grade_set records)
# ----------------------------------------------------------------------------
def short_verdicts_to_judge(records: dict, ds) -> dict:
    """grade_short records -> the scorer's judge view (load_judge kinds 'bcp'/'bcpgrader')."""
    out = {}
    for q in ds.qids:
        r = records.get(q)
        if r is None or "error" in r:
            continue
        c = bool(r.get("correct"))
        out[q] = dict(pred=r.get("pred") or "", extracted=r.get("extracted"), glen=1, correct=c,
                      passes=[dict(f1=float(c), p=float(c), r=float(c))])
    return out


# ----------------------------------------------------------------------------
# drivers
# ----------------------------------------------------------------------------
def trace_glob(run_dirs) -> list[str]:
    """``<run>/traces/q_*.json`` (or one level deeper) for each run dir."""
    if isinstance(run_dirs, (str, Path)):
        run_dirs = [run_dirs]
    pats = []
    for d in run_dirs:
        d = str(d)
        if glob.glob(os.path.join(d, "traces", "q_*.json")):
            pats.append(os.path.join(d, "traces", "q_*.json"))
        else:
            pats.append(os.path.join(d, "*", "traces", "q_*.json"))
    return pats


def score_short_obs(ds, verdict_records, *, run_dirs, out_dir=None):
    """BCP / FRAMES Obs. (earned_f1_v2 'ours' path, harness traces)."""
    spec = dict(adapter="harness", hit_cap=None, docids=True, cites=True, traces=trace_glob(run_dirs))
    recs = run_adapter(spec, ds)
    need = {e["docid"] for r in recs.values() for e in r["events"] if e["src"] in ("corpus", "span", "resolver")}
    ds.ensure_text(need)
    J = short_verdicts_to_judge(verdict_records, ds)
    rows = []
    for q in ds.qids:
        rec = recs.get(q) or dict(qid=q, pred="", events=[], offcorpus=[], cited=set(), missing=True)
        rows.append(score_question(q, rec, J.get(q), ds, spec))
    has_judge = bool(J)
    s = dict(all=aggregate(rows, has_judge))
    for sp in sorted({r["split"] for r in rows}):
        s[sp] = aggregate([r for r in rows if r["split"] == sp], has_judge)
    summary = dict(dataset=ds.name, n=len(rows), obs=s["all"].get("earned"), official=s["all"].get("official"),
                   aggregates=s)
    _write(out_dir, ds.name, rows, summary)
    return summary


def score_monaco_obs(ds, verdict_records, *, run_dirs=None, packets_glob=None, corpus=None, out_dir=None):
    """MoNaCo Obs. (build_items_v15 rule) from traces (packets built in memory) or packet files."""
    if packets_glob:
        spec = dict(adapter="evidence_file", traces=packets_glob,
                    raw_traces=trace_glob(run_dirs) if run_dirs else None)
    else:
        spec = dict(adapter="packets_from_traces", traces=trace_glob(run_dirs), corpus=corpus)
    recs = run_adapter(spec, ds)
    rows = [score_monaco_question(q, recs.get(q), verdict_records.get(q), ds) for q in ds.qids]
    n = len(rows)
    cls = Counter()
    for r in rows:
        cls.update(r.get("classes") or {})
    summary = dict(
        dataset="monaco", n=n,
        obs=st.mean(r["earned"] for r in rows) if n else 0.0,
        official=st.mean(r["official"] for r in rows) if n else 0.0,
        obs_flat_gold_rule=st.mean(r["earned_flat_gold_rule"] for r in rows) if n else 0.0,
        with_pred=sum(1 for r in rows if r["pred"]),
        credited_items=sum(cls.values()), credited_classes=dict(cls),
        earned_gt_official_violations=sum(1 for r in rows if r["earned"] > max(r["official"], 0.0) + 1e-9),
    )
    _write(out_dir, "monaco", rows, summary)
    return summary


def _write(out_dir, name, rows, summary):
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"obs_{name}_rows.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=sorted) + "\n")
    with open(os.path.join(out_dir, f"obs_{name}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=sorted)
