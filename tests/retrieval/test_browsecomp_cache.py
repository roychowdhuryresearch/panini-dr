"""Cache keys for the BrowseComp-Plus corpus.

WHY THIS EXISTS. Our HYBRID_*.json caches are flat {title: text} dicts, and for
FRAMES/MoNaCo that is safe because both are Wikipedia-derived, where titles are
unique by construction. BrowseComp-Plus is 100,195 documents from 34,345 domains:
measured on the real corpus, 9.6% of documents share a duplicated title and
"History" alone appears 245 times, while 1,176 documents carry no front-matter at
all. A title-keyed cache would silently collapse those into one another.

The gold_pages the loader emits MUST be built by this same function, or
gold-coverage matching breaks silently -- that is exactly the instrument whose
absence let a wrong-corpus MoNaCo run masquerade as an architecture failure for
half a day on 2026-08-12.
"""
from panini_dr.eval.browsecomp_keys import cache_key, parse_title


def test_parses_title_from_front_matter():
    doc = "---\ntitle: Sanaa - Wikipedia\nauthor: X\ndate: 2002-06-16\n---\nbody text"
    assert parse_title(doc) == "Sanaa - Wikipedia"
    assert cache_key(doc, "46697") == "Sanaa - Wikipedia [46697]"


def test_docid_alone_when_no_front_matter():
    assert parse_title("just body text, no front matter") is None
    assert cache_key("just body text, no front matter", "12345") == "12345"


def test_colliding_titles_get_distinct_keys():
    """245 real documents are titled "History"; they must not collapse."""
    a = "---\ntitle: History\n---\nfirst"
    b = "---\ntitle: History\n---\nsecond"
    assert cache_key(a, "111") != cache_key(b, "222")
    assert cache_key(a, "111") == "History [111]"


def test_title_is_sanitised_to_one_line():
    """A newline in the key would corrupt any line-oriented consumer."""
    doc = "---\ntitle: Line one\tand\ttabs\n---\nbody"
    assert "\n" not in cache_key(doc, "9")
    assert "\t" not in cache_key(doc, "9")


def test_overlong_title_is_truncated_but_stays_unique():
    doc = "---\ntitle: " + ("x" * 500) + "\n---\nbody"
    k1 = cache_key(doc, "1")
    k2 = cache_key(doc, "2")
    assert len(k1) < 300
    assert k1 != k2
    assert k1.endswith("[1]")


def test_docid_is_coerced_to_str():
    """The parquet column is int-like; the loader passes whatever it read."""
    assert cache_key("---\ntitle: T\n---\nb", 42) == "T [42]"
