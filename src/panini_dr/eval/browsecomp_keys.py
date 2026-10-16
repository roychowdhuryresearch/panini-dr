"""Cache keys for the BrowseComp-Plus corpus — one function, two callers.

Our ``HYBRID_*.json`` article caches are flat ``{title: text}`` dicts. That is
safe for FRAMES and MoNaCo, which are Wikipedia-derived and have unique titles by
construction. BrowseComp-Plus is not: 100,195 documents drawn from 34,345
domains. Measured on the real corpus:

* 99,019 documents carry a YAML ``title:`` in front-matter; **1,176 carry none**
* 91,576 distinct titles -> **9.6% of documents share a duplicated title**
* the worst offenders are navigation chrome: "History" x245, "Exiting nps.gov"
  x234, "Screen Rant" x178, "Advanced search" x168

Keying on title alone would silently collapse those documents onto one another.
Appending the docid makes the key unique by construction while keeping the title
readable in traces and logs.

BOTH the corpus builder and the question loader must import this module. If the
two ever compute keys differently, ``gold_pages`` stops matching the cache and
gold-coverage measurement fails silently — which is precisely the missing
instrument that let a wrong-corpus MoNaCo run look like an architecture failure
for half a day on 2026-08-12.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# Front-matter opens the document: "---\ntitle: ...\n". Only the first line of
# the title is taken; some scraped pages wrap it.
_TITLE_RE = re.compile(r"\A---\s*\ntitle:[ \t]*(.+?)[ \t]*\r?\n", re.MULTILINE)

# Long enough to stay human-readable, short enough that a key never dominates a
# log line. Uniqueness does not depend on this -- the docid suffix carries it.
_MAX_TITLE_CHARS = 200


def parse_title(text: str) -> Optional[str]:
    """The document's front-matter title, or ``None`` when it has none."""
    if not text:
        return None
    m = _TITLE_RE.match(text)
    if not m:
        return None
    title = m.group(1).strip()
    return title or None


def cache_key(text: str, docid: Any) -> str:
    """Stable, collision-free cache key for one BrowseComp-Plus document.

    ``"<title> [<docid>]"``, or the bare docid when no title parses. Whitespace
    is flattened so the key stays a single line for line-oriented consumers.
    """
    docid = str(docid)
    title = parse_title(text)
    if not title:
        return docid
    title = re.sub(r"\s+", " ", title).strip()
    if len(title) > _MAX_TITLE_CHARS:
        title = title[:_MAX_TITLE_CHARS].rstrip()
    return f"{title} [{docid}]"
