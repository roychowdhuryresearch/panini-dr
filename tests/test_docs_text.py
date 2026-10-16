"""Documentation that went stale once: the verify bundle and the LLM's GPU needs."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_the_verify_directory_is_described_not_reserved():
    for rel in ("datasets/hf_cards/frames/README.md", "datasets/frames/prepare.py"):
        text = (REPO / rel).read_text()
        assert "panini-dr verify" in text, rel
        assert "reserved" not in text.lower(), rel


def test_serve_docs_say_one_96gb_gpu_is_enough():
    for rel in ("serve/README.md", "serve/gpt_oss_120b.sh"):
        text = (REPO / rel).read_text()
        assert "2 × 96 GB" not in text and "2x 96 GB" not in text, rel
        assert "PANINI_LLM_TP=1" in text, rel
    script = (REPO / "serve/gpt_oss_120b.sh").read_text()
    assert 'TP="${PANINI_LLM_TP:-' in script and '--tensor-parallel-size "${TP}"' in script
