"""CPU-only end-to-end run of the FULL paper configuration on a toy corpus.

The adapter is built from ``resolve_config()`` with zero drift from
configs/paper.json (hybrid retrieval, jina reranker, every paper flag on). Only
the LLM endpoint, the dense encoder and the reranker HTTP server are faked
(tests/e2e/fakes.py); the planner prompt/parse, plan verifier, orchestrator
loop, typed resolver, verifier, researcher, reader, evidence judge and
synthesis validator all run for real.
"""

from __future__ import annotations

import json

import pytest

from panini_dr.adapters.base import AdapterContext, get_adapter
from panini_dr.config import check_pinned_defaults, config_drift, resolve_config
from panini_dr.eval.harness import run_cell
from panini_dr.retrieval import rerank
from panini_dr.retrieval.corpus import load_frames_corpus
from tests.e2e.fakes import TOY_QUESTIONS, toy_frames_questions

import panini_dr.adapters.ours.gsw_planner_orchestrator_v1  # noqa: F401  (registers)

SYSTEM = "ours_gsw_planner_orchestrator_v1"
GRADER_TRACE_KEYS = ("gsw_state", "gsw_checker_calls", "role_usage", "planner_tool_calls",
                     "stopped_reason")


def _adapter(corpus_path):
    resolved = resolve_config()
    assert config_drift(resolved.flat()) == {}
    extra = dict(resolved.extra)
    extra["corpus"] = load_frames_corpus(cache_path=corpus_path)
    ctx = AdapterContext(
        system_id=SYSTEM, model_id="openai/gpt-oss-120b", model_name="openai/gpt-oss-120b",
        base_url="http://llm.invalid/v1", api_key="",
        max_turns=resolved.max_turns, max_prompt_tokens=resolved.max_prompt_tokens,
        max_completion_tokens=resolved.max_completion_tokens, extra=extra,
    )
    adapter = get_adapter(SYSTEM)(ctx)
    assert check_pinned_defaults(adapter) == {}
    return adapter


@pytest.mark.parametrize("concurrency", [1, 2])
def test_paper_preset_answers_toy_questions(fake_services, toy_corpus_path, tmp_path, concurrency):
    llm, reranker = fake_services
    adapter = _adapter(toy_corpus_path)
    cell = run_cell(adapter, toy_frames_questions(), subset_id="toy", benchmark="frames",
                    raw_trace_dir=tmp_path / "traces", concurrency=concurrency)

    key = {q.id: q for q in TOY_QUESTIONS}
    assert cell.n_total == len(TOY_QUESTIONS)
    for r in cell.questions:
        traj = r.trajectory
        assert r.predicted_answer == key[r.question_id].answer, (r.question_id, traj.extra.get("stopped_reason"))
        assert traj.final_answer == key[r.question_id].answer
        # the trace written to disk is the one graders and the Obs scorer read
        on_disk = json.loads((tmp_path / "traces" / f"q_{r.question_id}.json").read_text())
        assert on_disk["final_answer"] == key[r.question_id].answer
        for k in GRADER_TRACE_KEYS:
            assert k in on_disk["extra"], k
        assert on_disk["tool_calls"], "orchestrator/researcher tool calls are recorded"
        blanks = on_disk["extra"]["gsw_state"]["blanks"]
        target = next(e["id"] for e in key[r.question_id].plan["entities"] if e.get("is_target"))
        assert blanks[target]["status"] == "resolved"
        kinds = {c.get("kind") for c in on_disk["extra"]["gsw_checker_calls"]}
        assert "resolver" in kinds
        roles = on_disk["extra"]["role_usage"]
        for role in ("orchestrator", "planner", "resolver", "verifier",
                     "synthesis_validator", "evidence_judge"):
            assert roles.get(role, {}).get("calls", 0) > 0, (r.question_id, role, roles)
        if key[r.question_id].researcher_blanks:
            for role in ("researcher", "reader"):
                assert roles.get(role, {}).get("calls", 0) > 0, (r.question_id, role, roles)

    health = rerank.RERANK_HEALTH.snapshot()
    assert health["calls"] > 0 and health["failures"] == 0
    assert reranker.calls == health["calls"]


def test_reranker_failure_is_counted_not_silent(fake_services, toy_corpus_path, tmp_path):
    llm, reranker = fake_services
    reranker.fail = True
    adapter = _adapter(toy_corpus_path)
    cell = run_cell(adapter, toy_frames_questions()[:1], subset_id="toy", benchmark="frames",
                    raw_trace_dir=tmp_path / "traces")
    assert cell.n_total == 1
    health = rerank.RERANK_HEALTH.snapshot()
    assert health["failures"] > 0
    assert health["failures"] == health["calls"]
    assert "fake reranker is down" in health["first_errors"][0]


def test_strict_rerank_raises(fake_services, toy_corpus_path):
    llm, reranker = fake_services
    reranker.fail = True
    adapter = _adapter(toy_corpus_path)
    rerank.set_strict(True)
    try:
        with pytest.raises(rerank.RerankUnavailableError):
            adapter.retriever.search("Zorvan Klett", top_k=5)
    finally:
        rerank.set_strict(False)
