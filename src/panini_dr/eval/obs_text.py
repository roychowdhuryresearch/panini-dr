"""Text normalisation, answer-item splitting and string-support tests for the Obs. scorer.

Ported verbatim from the research repo's ``eval_tools/bcp_grounding_ladder.py``
(normalisation, answer variants, docids, qrels, title index, tool-result
parsing) and ``eval_tools/earned_f1_v2.py`` (answer items, part variants,
support test, PRF, earned counts) -- the scorer of record for the paper's
Obs. column. Only the module layout changed; the matching rules are identical.
Deterministic, no model calls.
"""
from __future__ import annotations

import ast
import bisect
import json
from urllib.parse import unquote
import re
import unicodedata
from collections import defaultdict
from datetime import datetime

# ----------------------------------------------------------------------------
# from bcp_grounding_ladder.py
# ----------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_NONALNUM = re.compile(r"[^0-9a-z ]+")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = _NONALNUM.sub(" ", s)
    return _WS.sub(" ", s).strip()


_DATE_IN = ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%B %d, %Y", "%d %B %Y", "%B %d %Y", "%b %d, %Y", "%d %b %Y")
_DATE_OUT = ("%B %d, %Y", "%d %B %Y", "%Y-%m-%d", "%b %d, %Y", "%d %b %Y")


def answer_variants(ans: str) -> list[str]:
    """Normalised forms the answer may take inside a document: the answer
    itself, thousands-separator-free numerals, and every common spelling of
    a date the grader accepted (07/28/1865 == July 28, 1865 == 28 July 1865)."""
    raw = (ans or "").strip()
    out = [norm(raw)]
    if re.search(r"\d", raw):
        out.append(norm(raw.replace(",", "")))
    for fmt in _DATE_IN:
        try:
            d = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        for f2 in _DATE_OUT:
            s = d.strftime(f2)
            out.append(norm(s))
            out.append(norm(re.sub(r"\b0(\d)\b", r"\1", s)))
        break
    low = norm(raw)
    if low.startswith("the "):
        out.append(low[4:])
    return [v for v in dict.fromkeys(out) if v]


def answer_in(variants: list[str], text_n: str) -> str | None:
    """'exact' substring, 'tokens' for all-but-one content tokens, else None."""
    if not text_n:
        return None
    for v in variants:
        if v in text_n:
            return "exact"
    v = variants[0] if variants else ""
    toks = [t for t in v.split() if len(t) > 2]
    if len(toks) >= 3:
        padded = f" {text_n} "
        hit = sum(1 for t in toks if f" {t} " in padded)
        if hit >= len(toks) - 1:
            return "tokens"
    return None


_EXACT = re.compile(r"exact answer\s*[:\-]\s*(.+?)(?:\n|confidence|$)", re.I | re.S)


def final_answer_string(pred: str, extracted: str | None) -> str:
    """The string we look for: the official grader's extraction when it has
    one, else the 'Exact Answer:' field, else the whole prediction."""
    if extracted and extracted.strip().lower() not in ("none", "no answer provided", "null"):
        return extracted.strip()
    m = _EXACT.search(pred or "")
    if m:
        return m.group(1).strip()
    return (pred or "").strip()


BLANK = {"", "none", "unknown", "no answer provided", "timed out", "n/a", "no answer", "null", "not found"}
_ABSTAIN = re.compile(
    r"^(i (cannot|can t|can not|could not|couldn t|do not|don t|am unable|was unable)|insufficient (information|evidence)|"
    r"not enough (information|evidence)|no such |no (person|athlete|movie|film|book|individual|entity|answer) (can|could)|"
    r"unable to (determine|identify|find)|cannot be (determined|identified)|could not be (determined|identified)|"
    r"(the|this) .{0,80} (cannot|could not|can not) be (determined|identified|found)|"
    r"(based on|from) the (available|provided|retrieved) (information|evidence|documents).{0,40}(cannot|no |unable|not))")


def is_abstain(ans: str) -> bool:
    a = norm(ans)
    return a in BLANK or bool(_ABSTAIN.match(a))


_PART_SEP = re.compile(r"\s*(?:;|–|—| - | / |\|)\s*")
_PAREN = re.compile(r"\s*\((?:also (?:called|known as)|aka|a\.k\.a\.|or)\s+([^)]{2,80})\)")


def answer_parts(ans: str) -> list[list[str]]:
    """A compound answer ('Çılgın Bediş; 1996', '2015, Ayoyinka') is grounded
    only when every component is; each component carries its own variant
    list. '(also called X)' parentheticals become alternates, not parts."""
    raw = (ans or "").strip()
    alts = [m.group(1) for m in _PAREN.finditer(raw)]
    raw = _PAREN.sub("", raw)
    parts = [x for x in _PART_SEP.split(raw) if x.strip()]
    if len(parts) == 1 and "," in raw:
        cp = [x.strip() for x in raw.split(",") if x.strip()]
        if len(cp) == 2 and any(re.fullmatch(r"\d{4}", x) for x in cp):
            parts = cp
    out = []
    for i, p in enumerate(parts):
        vs = answer_variants(p)
        if i == 0:
            for a in alts:
                vs += answer_variants(a)
        if vs:
            out.append(vs)
    return out

# ----------------------------------------------------------------------------
# docids
# ----------------------------------------------------------------------------
_DOCID_SUFFIX_RE = re.compile(r"^(?P<title>.*) \[(?P<docid>\d+)\]$", re.DOTALL)


def docid_of(key: str) -> str | None:
    """docid from a cache key / chunk_id / local:// link. Anchored at the END
    because titles themselves contain brackets ('Report [1999] [82002]')."""
    if not key:
        return None
    k = str(key).strip()
    for pre in ("corpus://", "local://"):
        if k.startswith(pre):
            k = k[len(pre):]
    k = k.split("#", 1)[0]
    if "__" in k:
        k = k.rsplit("__", 1)[0]
    k = k.strip()
    m = _DOCID_SUFFIX_RE.match(k)
    if m:
        return m.group("docid")
    return k if k.isdigit() else None


def load_qrels(path):
    out = defaultdict(set)
    with open(path) as fh:
        for ln in fh:
            p = ln.split()
            if len(p) >= 4 and p[3] != "0":
                out[p[0]].add(p[2])
    return dict(out)


# ----------------------------------------------------------------------------
# corpus text (streamed once, subset cached) + title index


class TitleIndex:
    """Resolve a full or truncated title to a unique docid."""

    def __init__(self, title2doc):
        self.t2d = title2doc
        self.keys = sorted(title2doc)

    def prefix(self, title_prefix, min_len=12):
        tn = norm(title_prefix)
        if tn in self.t2d:
            return self.t2d[tn]
        if len(tn) < min_len:
            return None
        i = bisect.bisect_left(self.keys, tn)
        hits = []
        while i < len(self.keys) and self.keys[i].startswith(tn) and len(hits) < 3:
            hits.append(self.keys[i])
            i += 1
        return self.t2d[hits[0]] if len(hits) == 1 else None


# ----------------------------------------------------------------------------
# adapters: each returns {qid: {pred, events:[{docid,text,channel,src}], offcorpus:[{text,channel}]}}
#   channel: search | read | find | cited | visit | summary
#   src:     delivered (text the model saw) | span (verified citation span) | corpus (docid only)
# ----------------------------------------------------------------------------
def _as_dict(res):
    if isinstance(res, str):
        try:
            return json.loads(res)
        except Exception:
            return {}
    return res or {}


def _search_hits(res):
    res = _as_dict(res) if isinstance(res, str) else res
    hits = res.get("results") if isinstance(res, dict) else res
    return [h for h in (hits or []) if isinstance(h, dict)]



# ----------------------------------------------------------------------------
# from earned_f1_v2.py
# ----------------------------------------------------------------------------
_NUM = re.compile(r"^[+-]?\d[\d,]*(?:\.\d+)?\s*%?$")
_BOOL_T = {"true", "yes", "y"}
_BOOL_F = {"false", "no", "n"}
_ITEM_SEP_LINE = re.compile(r"^\s*(?:[-•]|\*(?!\*)|\d+[.)]|\(\d+\))\s*", re.M)
_PAIR_SEP = re.compile(r"\s*(?:≈|~|:|=|→|->| - | – | — |\|)\s*")


def num_variants(s: str) -> list[str]:
    """Numeric spellings a value can take in text: with/without thousands
    separators and %, and rounded to 0/1/2 decimals (14.2857 → 14.29 / 14.3
    / 14). Applied to both sides through norm(), where '.' becomes a space."""
    raw = str(s).strip()
    if not _NUM.match(raw):
        return []
    v = float(raw.replace(",", "").rstrip("%").strip())
    out = [raw, raw.replace(",", ""), raw.replace(",", "").rstrip("%").strip()]
    for nd in (2, 1, 0):
        r = round(v, nd)
        out.append(f"{r:.{nd}f}")
        if nd == 0:
            out.append(str(int(r)))
    if v == int(v):
        out.append(str(int(v)))
        out.append(f"{int(v):,}")
    return [norm(x) for x in dict.fromkeys(out) if norm(x)]


def part_kind(text: str) -> str:
    t = norm(text)
    if t in _BOOL_T or t in _BOOL_F:
        return "bool"
    if _NUM.match(str(text).strip()):
        return "number"
    return "text"


_V2_DATEISH = re.compile(r"^(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}(?:-\d{2})?|[A-Za-z]{3,9}\.? \d{1,2},? \d{4}|\d{1,2} [A-Za-z]{3,9}\.? \d{4})$")
_V2_MONTHS = ["january","february","march","april","may","june","july","august","september","october","november","december"]
_V2_ONES = ["zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty"]
_V2_TENS = {20:"twenty",30:"thirty",40:"forty",50:"fifty",60:"sixty",70:"seventy",80:"eighty",90:"ninety"}
_V2_ORD_ONES = ["zeroth","first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth","eleventh","twelfth","thirteenth","fourteenth","fifteenth","sixteenth","seventeenth","eighteenth","nineteenth","twentieth"]
_V2_ORD_TENS = {20:"twentieth",30:"thirtieth",40:"fortieth",50:"fiftieth",60:"sixtieth",70:"seventieth",80:"eightieth",90:"ninetieth"}


def _v2_number_word(n: int) -> str | None:
    if 0 <= n <= 20: return _V2_ONES[n]
    if n < 100 and n % 10 == 0 and n in _V2_TENS: return _V2_TENS[n]
    if 21 <= n <= 99: return f"{_V2_TENS[n - n % 10]} {_V2_ONES[n % 10]}"
    return None


def _v2_ordinal_word(n: int) -> str | None:
    if 0 <= n <= 20: return _V2_ORD_ONES[n]
    if n < 100 and n % 10 == 0 and n in _V2_ORD_TENS: return _V2_ORD_TENS[n]
    if 21 <= n <= 99: return f"{_V2_TENS[n - n % 10]} {_V2_ORD_ONES[n % 10]}"
    return None


def _v2_variants(text: str) -> list[str]:
    """v2 matching specification (2026-09-05): ordinal and number words, month abbreviations,
    year-month dates. Every variant is a normalised string."""
    raw = str(text).strip(); out: list[str] = []
    low = norm(raw)
    toks = low.split()
    # token-wise: ordinals and small numbers <-> words ("27th amendment" -> "twenty seventh amendment",
    # "kevin durant 4" -> "kevin durant four", "four" -> "4")
    sub = list(toks); changed = False
    for i, tk in enumerate(toks):
        m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)", tk)
        if m:
            w = _v2_ordinal_word(int(m.group(1)))
            if w: sub[i] = w; changed = True
        elif re.fullmatch(r"\d{1,2}", tk):
            w = _v2_number_word(int(tk))
            if w: sub[i] = w; changed = True
        elif tk in _V2_ONES:
            sub[i] = str(_V2_ONES.index(tk)); changed = True
        elif tk in _V2_ORD_ONES:
            sub[i] = str(_V2_ORD_ONES.index(tk)) + "th"; changed = True
    if changed: out.append(" ".join(sub))
    for i, mon in enumerate(_V2_MONTHS):
        ab = mon[:3]
        toks = low.split()
        if ab in toks and mon not in toks:
            out.append(" ".join(mon if t == ab else t for t in toks))
        if mon in toks:
            out.append(" ".join(ab if t == mon else t for t in toks))
    ym = re.fullmatch(r"(\d{4})-(\d{2})", raw)
    if ym:
        y, mo = ym.group(1), int(ym.group(2))
        if 1 <= mo <= 12:
            out += [f"{_V2_MONTHS[mo-1]} {y}", f"{_V2_MONTHS[mo-1][:3]} {y}", f"{y} {mo:02d}"]
    return [norm(v) for v in out if v]


def part_variants(text: str) -> list[str]:
    vs = answer_variants(str(text)) + num_variants(str(text)) + _v2_variants(str(text))
    # surname-first listings ("Schuhmann, Carl"): reversed token order for
    # short text parts; normalisation has already dropped the comma
    for v in list(vs):
        toks = v.split()
        if 2 <= len(toks) <= 3 and not any(t.isdigit() for t in toks):
            vs.append(" ".join(reversed(toks)))
    return [v for v in dict.fromkeys(vs) if v]


def _split_pieces(val) -> list:
    """Raw atomic pieces of one answer element (before variant expansion)."""
    if isinstance(val, (list, tuple)):
        out = []
        for x in val:
            out += _split_pieces(x)
        return out
    if isinstance(val, (bool, int, float)):
        return [val]
    s = str(val).strip()
    if s[:1] in "[{":                                # v2: a literal list/dict answer -> its scalar values
        lit = _literal_list(s)
        if isinstance(lit, dict):
            lit = [v for v in lit.values() if v not in (None, "", [])]
        if isinstance(lit, list) and lit:
            out = []
            for x in lit:
                out += _split_pieces(x)
            return out
    alts = [m.group(1) for m in _PAREN.finditer(s)]
    body = _PAREN.sub("", s)
    pieces = [x for x in _PART_SEP.split(body) if x.strip()]
    if len(pieces) == 1 and len(_PAIR_SEP.split(body)) == 2:      # '1990s ≈ 14 %'
        pieces = [x.strip() for x in _PAIR_SEP.split(body) if x.strip()]
    elif len(pieces) == 1 and "," in body:
        cp = [x.strip() for x in body.split(",") if x.strip()]
        # v2: a comma list with a numeric or date-like piece is a compound answer
        # ("6, Lynn Okamoto", "Legion, 1, Chapter 5"); each piece must be found on its own
        if len(cp) >= 2 and any(re.fullmatch(r"\d{1,4}", x) or _V2_DATEISH.match(x) for x in cp):
            pieces = cp
    if alts and pieces:
        num_alts = [a for a in alts if re.fullmatch(r"\d{1,4}", a.strip())]
        if num_alts and len(pieces) == 1:      # v2: "Kevin Durant (4)" -> name part + number part
            return [pieces[0]] + [int(a) for a in num_alts]
        pieces[0] = (pieces[0], alts)
    return pieces


def item_parts_and_kinds(item):
    """parts (variant lists) and kinds ('text' | 'number' | 'bool') of one item."""
    parts, kinds = [], []
    for pc in _split_pieces(item):
        alts = []
        if isinstance(pc, tuple):
            pc, alts = pc
        if isinstance(pc, bool):
            parts.append([str(pc).lower()]); kinds.append("bool"); continue
        if isinstance(pc, (int, float)):
            txt = f"{pc:g}" if isinstance(pc, float) else str(pc)
            parts.append(part_variants(txt)); kinds.append("number"); continue
        vs = part_variants(pc)
        for a in alts:
            vs += part_variants(a)
        vs = [v for v in dict.fromkeys(vs) if v]
        if vs:
            parts.append(vs); kinds.append(part_kind(pc))
    return parts, kinds


def _literal_list(s: str):
    t = s.strip()
    if not t or t[0] not in "[{(":
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(t)
        except Exception:
            continue
        if isinstance(v, dict):                 # {'year': 1984, 'names': [...]}
            v = _dict_values(v)                 # → 1984, name, name, ...
        if isinstance(v, (list, tuple)):
            return list(v)
    return None


def _dict_values(v) -> list:
    """Values of a dict answer, lists flattened one level: the keys are the
    system's own labels, not facts."""
    out = []
    for x in (v.values() if isinstance(v, dict) else v):
        if isinstance(x, dict):
            out += _dict_values(x)
        elif isinstance(x, (list, tuple)):
            out += [y for y in x if not isinstance(y, (list, tuple, dict))] + _dict_values([y for y in x if isinstance(y, (list, tuple, dict))])
        else:
            out.append(x)
    return out


_MD_HEADER = re.compile(r"^\s*#{1,6}\s+")
_MD_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$")
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_LABEL_LINE = re.compile(r"^[^.]{0,60}:\s*$")


def _md_lines(s: str) -> list:
    """Lines of a markdown/prose answer as item strings: header lines and
    label-only lines ('**Details:**') dropped; a markdown table contributes
    one item per body row (its cells joined with ';', so every cell is a
    part); in a prose line longer than 80 characters the author's own
    emphasised spans (**...**) are the claimed facts and become the items."""
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    out, in_table, header_seen = [], False, False
    for ln in lines:
        if _MD_TABLE_SEP.match(ln):
            in_table, header_seen = True, True
            continue
        if ln.count("|") >= 2:
            if not in_table:            # header row precedes the separator
                in_table = True
                continue
            if not header_seen:
                continue
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            cells = [c for c in cells if c and c.strip("*_ ") not in ("", "-", "—", "n/a", "N/A")]
            if cells:
                out.append(cells)           # one item, every cell a part
            continue
        in_table = header_seen = False
        if _MD_HEADER.match(ln):            # section title, not a claim
            continue
        t = ln
        if _LABEL_LINE.match(t.strip("*_ ")):
            continue
        bold = [b.strip() for b in _MD_BOLD.findall(t) if b.strip() and not b.strip().endswith(":")]
        if len(t) > 80 and bold:
            out += bold
        else:
            out.append(t)
    return out


_AND_SPLIT = re.compile(r"\s+(?:and|&)\s+")
_LEAD_COLON = re.compile(r"^[^:]{0,120}:\s+(?=\S)")


def split_items(pred: str, multi: bool) -> list:
    """Deterministic item boundaries. Single-answer datasets: one item. Sets:
    a JSON/Python literal list, else one item per bulleted/numbered line,
    else ';'-separated, else ','-separated (a ',' inside a number or inside a
    bracketed pair does not split)."""
    s = (pred or "").strip()
    if not s or is_abstain(s):
        return []
    if not multi:
        return [s]
    lit = _literal_list(s)
    if lit is not None:
        return [x for x in lit if x not in (None, "", [])]
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    marked = [ln for ln in lines if _ITEM_SEP_LINE.match(ln)]
    if len(lines) >= 2 and len(marked) >= max(2, len(lines) - 1):
        return [_ITEM_SEP_LINE.sub("", ln).strip().rstrip(",;") for ln in marked]
    if len(lines) >= 2 or ln_has_md(s):
        items = _md_lines(s)
        if len(items) >= 2 or (items and items[0] != s):
            return [x for it in items
                    for x in ([it] if isinstance(it, list) else _one_line_items(it))]
        s = items[0] if items and isinstance(items[0], str) else s
    return _one_line_items(s)


def ln_has_md(s: str) -> bool:
    return "**" in s or s.lstrip().startswith("#") or s.count("|") >= 2


def _one_line_items(s: str) -> list:
    """Items of a single-line answer: 'a; b; c', 'a, b, c', 'prefix: a, b'
    ('The surfaces are: grass, clay'), 'A and B'."""
    s = s.strip()
    if _LEAD_COLON.match(s) and s.count(",") + s.count(";") >= 1:
        s = _LEAD_COLON.sub("", s, count=1)  # drop the restating prefix
    if ";" in s:
        out = []
        for x in s.split(";"):                 # "1984; A, B, C" → 1984, A, B, C
            x = x.strip()
            if x:
                out += _comma_split(x) if x.count(",") >= 2 else [x]
        return out
    out = _comma_split(s)
    if len(out) == 1 and len(_AND_SPLIT.split(s)) == 2:   # 'Polish and Ukrainian'
        a, b = [x.strip() for x in _AND_SPLIT.split(s)]
        if a and b and max(len(a), len(b)) <= 60 and not re.search(r"\b(?:is|are|was|were|has|have|had|did|does|do)\b", s, re.I):
            return [a, b]
    return out


_SENTENCE = re.compile(r"\b(?:is|are|was|were|has|have|had|did|does|do|not)\b", re.I)


def _comma_split(s: str) -> list:
    if _SENTENCE.search(s) and s.count(",") <= 2:      # a sentence, not a list
        return [s]
    prot = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)      # 20,857,303 → 20857303
    prot = re.sub(r"\(([^()]*)\)", lambda m: "(" + m.group(1).replace(",", "\x00") + ")", prot)
    pieces = [x.replace("\x00", ",").strip() for x in prot.split(",")]
    pieces = [re.sub(r"^(?:and|or)\s+", "", x, flags=re.I).strip() for x in pieces]
    pieces = [x for x in pieces if x]
    if len(pieces) >= 2 and all(len(x) <= 80 for x in pieces):
        return pieces
    return [s]


def gold_items(dataset: str, rec) -> list:
    if dataset == "monaco":
        a = rec["answer"]
        try:
            v = ast.literal_eval(a) if isinstance(a, str) else a
        except Exception:
            v = a
        return list(v) if isinstance(v, (list, tuple)) else [v]
    return [rec["answer"]]


# ----------------------------------------------------------------------------
# deterministic gold match (only used to attribute the judge's correct count
# to specific items on MoNaCo; single-answer datasets use the judge directly)
# ----------------------------------------------------------------------------
def _scalar_match(p, g) -> bool:
    pk, gk = part_kind(str(p)), part_kind(str(g))
    if gk == "bool" or isinstance(g, bool):
        gt = str(g).lower() in _BOOL_T or norm(str(g)) in _BOOL_T
        pn = norm(str(p))
        return (pn in _BOOL_T) == gt if (pn in _BOOL_T or pn in _BOOL_F) else False
    if gk == "number" and pk == "number":
        gv = float(str(g).replace(",", "").rstrip("%"))
        pv = float(str(p).replace(",", "").rstrip("%"))
        return abs(gv - pv) <= max(0.5, 0.01 * abs(gv))
    pv, gv = part_variants(str(p)), part_variants(str(g))
    if set(pv) & set(gv):
        return True
    pn, gn = norm(str(p)), norm(str(g))
    if len(gn) >= 4 and (gn in pn or (len(pn) >= 4 and pn in gn)):
        return True
    return answer_in(gv, pn) == "tokens" if len(gn.split()) >= 3 else False


def item_matches_gold(item, gold) -> bool:
    """A tuple gold ([entity, value]) matches when every gold element matches
    some element of the predicted item; a scalar gold matches a scalar or the
    first element of a predicted tuple."""
    gs = list(gold) if isinstance(gold, (list, tuple)) else [gold]
    ps = list(item) if isinstance(item, (list, tuple)) else [item]
    if len(ps) == 1 and len(gs) > 1:
        ps = [x.strip() for x in _PAIR_SEP.split(str(ps[0]))] if isinstance(ps[0], str) else ps
    return all(any(_scalar_match(p, g) for p in ps) for g in gs)


# ----------------------------------------------------------------------------
# support: every checkable part of an item appears in received text
# ----------------------------------------------------------------------------
def _spaceless(texts_n: dict) -> dict:
    return {d: t.replace(" ", "") for d, t in texts_n.items()}


def _v2_answer_in(variants: list[str], text_n: str) -> str | None:
    """v2: exact substring, or all-but-one content tokens only for parts of >= 4 content tokens."""
    if not text_n:
        return None
    padded_t = f" {text_n} "
    for v in variants:
        if not v:
            continue
        if f" {v} " in padded_t:
            return "exact"
        if (len(v.split()) > 1 or len(v) > 6) and f" {v}s " in padded_t:   # plural of a longer part
            return "exact"
    v = variants[0] if variants else ""
    toks = [t for t in v.split() if len(t) > 2]
    if len(toks) >= 4:
        padded = f" {text_n} "
        if sum(1 for t in toks if f" {t} " in padded) >= len(toks) - 1:
            return "tokens"
    return None


def item_support(parts: list[list[str]], kinds: list[str], texts_n: dict, texts_ns: dict | None = None) -> dict:
    """Per part: the set of docs whose delivered text contains it. Bool parts
    are uncheckable (their factual content is in the question, not the
    answer) and never block support; they are counted separately.
    texts_ns: the same texts with spaces removed — a text part of >= 5 chars
    also matches space-free, absorbing corpus tokenisation artefacts such as
    'vitamin B 12' for 'vitamin B12' (never applied to numbers)."""
    per_part, hits_any = [], set()
    for vs, kind in zip(parts, kinds):
        if kind == "bool":
            per_part.append(None)
            continue
        hit = {d for d, tn in texts_n.items() if _v2_answer_in(vs, tn)}
        if not hit and kind == "text":
            two = [v for v in vs if len(v.split()) == 2]
            if two:                                   # v2: an optional middle initial between two name tokens
                pats = [re.compile(r"\b" + re.escape(a) + r" (?:[a-z] )?" + re.escape(b) + r"\b") for a, b in (v.split() for v in two)]
                hit = {d for d, tn in texts_n.items() if any(p.search(tn) for p in pats)}
        if not hit and kind == "text" and texts_ns:
            vss = [v.replace(" ", "") for v in vs if len(v) >= 5 and re.search(r"[a-z]", v)]
            hit = {d for d, ts in texts_ns.items() if any(v in ts for v in vss)}
        per_part.append(hit)
        hits_any |= hit
    checkable = [h for h in per_part if h is not None]
    supported = bool(checkable) and all(checkable)
    return dict(supported=supported, uncheckable=not checkable, per_part=per_part,
                docs=set.intersection(*checkable) if checkable and all(checkable) else set(),
                docs_any=hits_any)


# ----------------------------------------------------------------------------
# F1 arithmetic (official MoNaCo V2 conventions)
# ----------------------------------------------------------------------------
def prf(nc: float, pred_len: float, glen: int):
    if pred_len <= 0 or nc <= 0:
        return 0.0, 0.0, 0.0
    r = min(nc, glen) / glen
    p = nc / max(pred_len, nc)
    return p, r, 2 * p * r / (p + r)


def reconstruct_counts(rec: dict):
    """The judge cache stores P/R/F1 (per pass) but not the matched items.
    Invert the official formula: nc = R·glen, pred_len = nc/P."""
    glen = int(rec.get("glen") or 1)
    passes = rec.get("passes") or [dict(f1=rec.get("f1", 0), precision=rec.get("precision", 0),
                                        recall=rec.get("recall", rec.get("precision", 0)))]
    out = []
    for ps in passes:
        p, r = float(ps.get("precision") or 0), float(ps.get("recall") or 0)
        if glen == 1:
            nc, plen = (1.0 if p > 0 else 0.0), 1.0
            if 0 < p < 1:                     # single-answer partial credit
                nc = p
        else:
            nc = round(r * glen)
            plen = round(nc / p) if p > 0 and nc > 0 else 0
        out.append(dict(f1=float(ps.get("f1") or 0), p=p, r=r, nc=nc, pred_len=plen))
    return glen, out


def earned_counts(nc: float, pred_len: float, det_correct: list[bool], supported: list[bool]):
    """Attribute the judge's nc correct items to specific items. Items our
    deterministic matcher marks correct are taken first; any remainder is
    spread over the unmatched items in proportion to their support rate.
    Returns (earned_nc, lower_bound, upper_bound)."""
    n = len(det_correct)
    if n == 0 or nc <= 0:
        return 0.0, 0.0, 0.0
    D = [i for i in range(n) if det_correct[i]]
    nonD = [i for i in range(n) if not det_correct[i]]
    n_sup = sum(supported)
    lo = max(0.0, nc - (n - n_sup))
    hi = min(nc, n_sup)
    take_D = min(nc, len(D))
    d_sup = sum(supported[i] for i in D)
    earned = d_sup * (take_D / len(D)) if D else 0.0
    rem = nc - take_D
    if rem > 0 and nonD:
        earned += rem * (sum(supported[i] for i in nonD) / len(nonD))
    return min(max(earned, lo), hi), lo, hi


# ----------------------------------------------------------------------------
# datasets: qids, gold answers, gold pages, document identity, corpus text
# ----------------------------------------------------------------------------
_CHUNK_IDX = re.compile(r"^(?P<title>.*)__(?P<idx>\d+)$", re.S)


def _strip_ref(key: str) -> str:
    k = str(key or "").strip()
    for pre in ("corpus://", "local://"):
        if k.startswith(pre):
            k = unquote(k[len(pre):])  # links are percent-encoded (local://Andrew%20Pendlebury#…)
    return k.split("#", 1)[0].strip()


def _literal(s):
    if not isinstance(s, str):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return None

