"""The LLM client logs streamed text at DEBUG and one summary line per call at INFO.

Without ``-v`` the console shows INFO, so a run's terminal gets one line per LLM
call instead of every flushed piece of streamed text; ``run.log`` keeps DEBUG.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from panini_dr.models.llm_client import LLMClient
from panini_dr.utils import setup_run_logger


def _chunk(content=None, reasoning=None, tool=None, finish=None, usage=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=tool)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=usage
    )


def _stream():
    long_text = "word " * 80  # past the flush threshold, so a text line is logged mid-stream
    return [
        _chunk(reasoning="thinking about it\n"),
        _chunk(content=long_text),
        _chunk(tool=[SimpleNamespace(index=0, id="c1", function=SimpleNamespace(
            name="search", arguments='{"query": "x"}'))]),
        _chunk(finish="tool_calls"),
        SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7)),
    ]


@pytest.fixture
def panini_logger():
    logger = logging.getLogger("panini_dr")
    saved = (list(logger.handlers), logger.level, logger.propagate)
    yield logger
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    for h in saved[0]:
        logger.addHandler(h)
    logger.setLevel(saved[1])
    logger.propagate = saved[2]


def _client() -> LLMClient:
    return LLMClient(model="openai/gpt-oss-120b", base_url="http://llm.invalid/v1", api_key="x")


def test_streamed_pieces_are_debug_and_the_summary_is_info(caplog, monkeypatch):
    # an earlier run in this process may have set up the run logger (propagate=False)
    monkeypatch.setattr(logging.getLogger("panini_dr"), "propagate", True)
    caplog.set_level(logging.DEBUG, logger="panini_dr.llm")
    resp = _client()._consume_stream(iter(_stream()), label="vllm/openai/gpt-oss-120b")
    assert resp.tool_calls and resp.tool_calls[0]["name"] == "search"
    assert (resp.prompt_tokens, resp.completion_tokens) == (11, 7)

    records = [r for r in caplog.records if r.name == "panini_dr.llm"]
    info = [r.getMessage() for r in records if r.levelno == logging.INFO]
    debug = [r.getMessage() for r in records if r.levelno == logging.DEBUG]
    assert len(info) == 1 and info[0].startswith("stream done |"), info
    for prefix in ("stream start", "stream txt", "stream rsn", "stream tool"):
        assert any(m.startswith(prefix) for m in debug), (prefix, debug)


@pytest.mark.parametrize("verbose", [False, True])
def test_console_shows_stream_text_only_with_verbose(tmp_path, capsys, panini_logger, verbose):
    setup_run_logger(tmp_path, verbose=verbose)
    _client()._consume_stream(iter(_stream()), label="vllm/openai/gpt-oss-120b")
    for h in panini_logger.handlers:
        h.flush()

    console = capsys.readouterr().err
    assert "stream done |" in console
    assert ("stream txt |" in console) is verbose
    run_log = (tmp_path / "run.log").read_text()
    assert "stream txt |" in run_log and "stream done |" in run_log  # the file keeps DEBUG
