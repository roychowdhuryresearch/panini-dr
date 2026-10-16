from __future__ import annotations

import importlib
import json
from typing import Any

import pytest

from panini_dr.adapters.ours._planner_exec import BlankResult, GSWPlan
from panini_dr.adapters.ours.gsw_planner_orchestrator_v1 import ReActResearcher
from tests.ours.orchestrator._helpers import (
    _RoutedLLM,
    _ScriptedLLM,
    _StubChunk,
    _StubCorpus,
    _StubHit,
    _StubResponse,
    _StubRetriever,
    _mk_llm_adapter,
    _one_level_plan,
    _tc,
)


def _ctx_module():
    try:
        return importlib.import_module(
            "panini_dr.adapters.ours._context_compaction"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"context compaction module is missing: {exc}")


def _search_result(n: int) -> dict[str, Any]:
    return {
        "results": [
            {
                "chunk_id": f"C{i}",
                "title": f"T{i}",
                "score": 1.0,
                "text": "x" * 800,
            }
            for i in range(n)
        ]
    }


def _msgs_with_searches(k: int):
    """k search tool-calls, ids s0..s(k-1), each with a tool message."""
    ctx = _ctx_module()
    archive = ctx.ToolResultArchive()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]
    for j in range(k):
        cid = f"s{j}"
        res = _search_result(6)
        archive.store(cid, "search", j + 1, {"query": f"q{j}"}, res)
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"query": f"q{j}"}),
                        },
                    }
                ],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": cid, "content": json.dumps(res)}
        )
    return messages, archive


def _msgs_with_reads(k: int):
    """k archived read tool-calls, ids r0..r(k-1), each with full text."""
    ctx = _ctx_module()
    archive = ctx.ToolResultArchive()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]
    for j in range(k):
        cid = f"r{j}"
        res = {
            "chunk_id": f"C{j}",
            "title": f"Read Title {j}",
            "article_text": f"Full article text for read {j}. " + ("evidence " * 400),
        }
        archive.store(cid, "read", j + 1, {"chunk_id": f"C{j}"}, res)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": cid,
                "content": json.dumps(res),
            }
        )
    return messages, archive


def _researcher_for_script(llm: Any, *, compaction: bool | None = None):
    plan = GSWPlan.model_validate(_one_level_plan())
    chunks = {
        f"C{i}": _StubChunk(
            f"C{i}",
            f"Title {i}",
            f"Full article text for chunk {i}. " + ("evidence " * 400),
        )
        for i in range(8)
    }
    kwargs: dict[str, Any] = {
        "llm": llm,
        "corpus": _StubCorpus(chunks),
        "retriever": _StubRetriever({"query": list(chunks.values())}),
        "plan": plan,
        "state": {"t": BlankResult(blank_id="t", status="unknown")},
        "allowed_blank_ids": ["t"],
        "system_prompt": "SYS",
        "question": "Q",
        "max_turns": 8,
    }
    if compaction is not None:
        kwargs["context_compaction_on"] = compaction
    return ReActResearcher(**kwargs)


def _script(num_searches: int = 4) -> list[_StubResponse]:
    return [
        _StubResponse(
            tool_calls=[
                _tc(
                    f"s{i}",
                    "search",
                    {"query": f"query {i}", "top_k": 6},
                )
            ]
        )
        for i in range(num_searches)
    ] + [
        _StubResponse(
            tool_calls=[
                _tc(
                    "u1",
                    "update_blank",
                    {
                        "blank_id": "t",
                        "value": "Recovered Answer",
                        "evidence_chunk_ids": ["C0"],
                    },
                )
            ]
        )
    ]


def _search_tool_messages(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        msg
        for msg in trace["messages"]
        if msg.get("role") == "tool" and str(msg.get("tool_call_id", "")).startswith("s")
    ]


def test_placeholder_caps_chunks_and_keeps_pointers():
    ctx = _ctx_module()
    archive = ctx.ToolResultArchive()
    archive.store("c1", "search", 1, {"query": "qq"}, _search_result(10))
    ph = ctx.build_search_placeholder(archive.get("c1"))
    assert ph["compacted_tool_result"] is True
    assert ph["query"] == "qq"
    assert len(ph["top_chunks"]) == ctx.MAX_PLACEHOLDER_CHUNKS
    assert ph["top_chunks"][0] == {"chunk_id": "C0", "title": "T0"}


def test_read_placeholder_builder_is_not_part_of_search_only_compaction():
    ctx = _ctx_module()
    assert not hasattr(ctx, "build_read_placeholder")


def test_compaction_keeps_last_window_full_and_shrinks():
    ctx = _ctx_module()
    messages, archive = _msgs_with_searches(5)
    before = sum(len(m["content"]) for m in messages)
    n = ctx.compact_messages(messages, archive)
    after = sum(len(m["content"]) for m in messages)
    assert n == 5 - ctx.SEARCH_FULL_WINDOW
    assert after < before * 0.6
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert '"compacted_tool_result"' in tool_msgs[0]["content"]
    assert '"compacted_tool_result"' not in tool_msgs[-1]["content"]


def test_read_results_are_not_compacted_even_when_archived():
    ctx = _ctx_module()
    messages, archive = _msgs_with_reads(3)

    n = ctx.compact_messages(messages, archive)

    assert n == 0
    read_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(read_msgs) == 3
    assert all("compacted_tool_result" not in msg["content"] for msg in read_msgs)
    assert all("Full article text for read" in msg["content"] for msg in read_msgs)
    assert archive.metrics()["read_messages_compacted"] == 0


def test_compaction_is_idempotent():
    ctx = _ctx_module()
    messages, archive = _msgs_with_searches(5)
    n1 = ctx.compact_messages(messages, archive)
    n2 = ctx.compact_messages(messages, archive)
    assert n1 == 3
    assert n2 == 0


def test_empty_search_not_compacted():
    ctx = _ctx_module()
    archive = ctx.ToolResultArchive()
    messages = [{"role": "user", "content": "q"}]
    for j in range(4):
        cid = f"s{j}"
        res = {"results": []}
        archive.store(cid, "search", j + 1, {"query": f"q{j}"}, res)
        messages.append(
            {"role": "tool", "tool_call_id": cid, "content": json.dumps(res)}
        )
    assert ctx.compact_messages(messages, archive) == 0


def test_non_compactable_tools_untouched():
    ctx = _ctx_module()
    archive = ctx.ToolResultArchive()
    messages = [{"role": "tool", "tool_call_id": "u1", "content": '{"ok": true}'}]
    assert ctx.compact_messages(messages, archive) == 0
    assert messages[0]["content"] == '{"ok": true}'


def test_errors_and_stateful_tools_stay_full_with_metrics():
    ctx = _ctx_module()
    archive = ctx.ToolResultArchive()
    messages = []
    cases = [
        ("s_error", "search", {"query": "bad"}, {"error": "retriever failed"}),
        ("u1", "update_blank", {}, {"ok": True, "value": "Answer"}),
        ("g1", "get_state", {}, {"blanks": [{"blank_id": "t"}]}),
        ("b1", "bind_connection", {}, {"ok": True, "bound": {"t": ["C1"]}}),
    ]
    for idx, (cid, tool, args, result) in enumerate(cases, start=1):
        archive.store(cid, tool, idx, args, result)
        messages.append(
            {"role": "tool", "tool_call_id": cid, "content": json.dumps(result)}
        )

    assert ctx.compact_messages(messages, archive) == 0

    assert messages[0]["content"] == '{"error": "retriever failed"}'
    assert messages[1]["content"] == '{"ok": true, "value": "Answer"}'
    assert messages[2]["content"] == '{"blanks": [{"blank_id": "t"}]}'
    assert messages[3]["content"] == '{"ok": true, "bound": {"t": ["C1"]}}'
    metrics = archive.metrics()
    assert metrics["search_messages_compacted"] == 0
    assert metrics["read_messages_compacted"] == 0
    assert metrics["placeholder_chars"] == 0
    assert metrics["estimated_prompt_chars_saved"] == 0
    assert metrics["raw_chars_archived"] > 0


def test_default_off_matches_omitted_flag_and_enabled_compacts_only_messages():
    default_researcher = _researcher_for_script(
        _ScriptedLLM(_script()), compaction=None
    )
    off_researcher = _researcher_for_script(
        _ScriptedLLM(_script()), compaction=False
    )
    on_researcher = _researcher_for_script(
        _ScriptedLLM(_script()), compaction=True
    )

    default_trace = default_researcher.solve()
    off_trace = off_researcher.solve()
    on_trace = on_researcher.solve()

    assert default_trace["messages"] == off_trace["messages"]
    assert all(
        "compacted_tool_result" not in msg["content"]
        for msg in _search_tool_messages(off_trace)
    )
    assert all(
        "compacted_tool_result" not in tc.result_full
        for tc in off_trace["tool_calls"]
    )

    on_search_messages = _search_tool_messages(on_trace)
    assert len(on_search_messages) == 4
    assert "compacted_tool_result" in on_search_messages[0]["content"]
    assert "compacted_tool_result" in on_search_messages[1]["content"]
    assert "compacted_tool_result" not in on_search_messages[-2]["content"]
    assert "compacted_tool_result" not in on_search_messages[-1]["content"]
    assert all(
        "compacted_tool_result" not in tc.result_full
        for tc in on_trace["tool_calls"]
    )
    # every trace carries the reader telemetry; only compaction adds metrics
    for trace in (default_trace, off_trace):
        assert trace["metadata"]["feature_flags"] == {"gsw_context_compaction": False}
        assert "compaction_metrics" not in trace["metadata"]
    metadata = on_trace["metadata"]
    assert metadata["feature_flags"] == {
        "gsw_context_compaction": True,
    }
    metrics = metadata["compaction_metrics"]
    assert metrics["search_messages_compacted"] == 2
    assert metrics["read_messages_compacted"] == 0
    assert metrics["placeholder_chars"] > 0
    assert metrics["raw_chars_archived"] > 0
    assert metrics["estimated_prompt_chars_saved"] > 0


class _PlaceholderReadingLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._n = 0
        self.placeholder_chunk_id: str | None = None

    def chat(self, messages, *, tools=None, max_tokens=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools, **kwargs})
        if self._n < 4:
            response = _StubResponse(
                tool_calls=[
                    _tc(
                        f"s{self._n}",
                        "search",
                        {"query": f"query {self._n}", "top_k": 6},
                    )
                ]
            )
        elif self._n == 4:
            placeholder = None
            for msg in messages:
                content = str(msg.get("content", ""))
                if "compacted_tool_result" in content:
                    placeholder = json.loads(content)
                    break
            assert placeholder is not None
            self.placeholder_chunk_id = placeholder["top_chunks"][0]["chunk_id"]
            response = _StubResponse(
                tool_calls=[
                    _tc("r1", "find", {"chunk_id": self.placeholder_chunk_id,
                                       "pattern": "Full article text"})
                ]
            )
        else:
            response = _StubResponse(
                tool_calls=[
                    _tc(
                        "u1",
                        "update_blank",
                        {
                            "blank_id": "t",
                            "value": "Recovered Answer",
                            "evidence_chunk_ids": [self.placeholder_chunk_id],
                        },
                    )
                ]
            )
        self._n += 1
        return response


def test_compacted_search_placeholder_allows_read_recovery():
    llm = _PlaceholderReadingLLM()
    researcher = _researcher_for_script(llm, compaction=True)

    trace = researcher.solve()

    assert llm.placeholder_chunk_id == "C0"
    read_call = next(tc for tc in trace["tool_calls"] if tc.name == "find")
    assert '"chunk_id": "C0"' in read_call.result_full
    assert "Full article text for chunk 0" in read_call.result_full
    assert trace["stopped"] == "all_resolved"


def test_context_compaction_keeps_read_tool_messages_full_in_react_trace(monkeypatch):
    import panini_dr.adapters.ours._agentic_read as ar

    # the reader answers with the article's opening sentence
    monkeypatch.setattr(ar, "read_article_briefed", lambda **kw: ar.ReadResult(
        answer=kw["full_text"].split(". ")[0],
        citations=[{"span": kw["full_text"].split(". ")[0]}]))
    script = [
        _StubResponse(tool_calls=[_tc("r0", "read", {"chunk_id": "C0", "question": "q0"})]),
        _StubResponse(tool_calls=[_tc("r1", "read", {"chunk_id": "C1", "question": "q1"})]),
        _StubResponse(
            tool_calls=[
                _tc(
                    "u1",
                    "update_blank",
                    {
                        "blank_id": "t",
                        "value": "Recovered Answer",
                        "evidence_chunk_ids": ["C0", "C1"],
                    },
                )
            ]
        ),
    ]
    researcher = _researcher_for_script(_ScriptedLLM(script), compaction=True)

    trace = researcher.solve()

    read_messages = [
        msg
        for msg in trace["messages"]
        if msg.get("role") == "tool" and str(msg.get("tool_call_id", "")).startswith("r")
    ]
    assert len(read_messages) == 2
    assert all("compacted_tool_result" not in msg["content"] for msg in read_messages)
    assert "Full article text for chunk 0" in read_messages[0]["content"]
    assert "Full article text for chunk 1" in read_messages[1]["content"]
    assert trace["metadata"]["compaction_metrics"]["read_messages_compacted"] == 0
    assert trace["stopped"] == "all_resolved"


def test_context_compaction_only_does_not_inject_working_memory_prompt_blocks():
    routes = {
        "You are the **Orchestrator**": [
            _StubResponse(
                tool_calls=[
                    _tc("o1", "dispatch_subplan", {"blank_ids": ["t"]})
                ]
            ),
            _StubResponse(
                tool_calls=[
                    _tc("o2", "submit_answer", {"answer": "Recovered Answer"})
                ]
            ),
        ],
        "`t` (role=`target`": [
            _StubResponse(
                tool_calls=[
                    _tc(
                        "u1",
                        "update_blank",
                        {
                            "blank_id": "t",
                            "value": "Recovered Answer",
                            "evidence_chunk_ids": [],
                        },
                    )
                ]
            )
        ],
    }
    llm = _RoutedLLM(routes)
    adapter = _mk_llm_adapter(
        llm,
        plan_dict=_one_level_plan(),
        extra={"gsw_context_compaction": True},
    )

    traj = adapter.run_question("Q", question_id="ctxc-only")

    assert traj.final_answer == "Recovered Answer"
    assert adapter.gsw_context_compaction is True
    assert traj.extra["gsw_feature_flags"] == {
        "gsw_context_compaction": True,
    }
    assert traj.extra["gsw_compaction_metrics"]["read_messages_compacted"] == 0
    researcher_prompts = [
        call["messages"][0]["content"]
        for call in llm.calls
        if "`t` (role=`target`" in call["messages"][0]["content"]
        and "You are the **Orchestrator**" not in call["messages"][0]["content"]
    ]
    assert researcher_prompts
    joined = "\n".join(researcher_prompts)
    assert "## Working memory focus" not in joined
    assert "## Working memory attempt log" not in joined


def test_synthetic_search_transcript_drops_by_more_than_half():
    ctx = _ctx_module()
    messages, archive = _msgs_with_searches(12)
    before = sum(len(m["content"]) for m in messages)

    n = ctx.compact_messages(messages, archive)
    after = sum(len(m["content"]) for m in messages)

    assert n == 12 - ctx.SEARCH_FULL_WINDOW
    assert after < before * 0.5


def test_find_results_are_not_compacted_even_when_archived():
    ctx = _ctx_module()
    archive = ctx.ToolResultArchive()
    archive.store("f1", "find", 1, {"pattern": "x"}, {"snippets": [{"text": "x"}]})
    messages = [
        {
            "role": "tool",
            "tool_call_id": "f1",
            "content": '{"snippets": [{"text": "x"}]}',
        }
    ]
    assert ctx.compact_messages(messages, archive) == 0
    assert messages[0]["content"] == '{"snippets": [{"text": "x"}]}'


