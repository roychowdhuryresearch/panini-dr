"""Orchestrator + per-blank researcher adapter.

Pipeline per question:

1. **Plan** — the LLM orchestrator's planning phase: its only tool is
   ``call_planner``, which emits a validated ``GSWPlan`` (``emit_plan``)
   and hands back the plan and its audit.
2. **Topological sort** — ``build_dependency_graph`` +
   ``_compute_levels`` give ``levels: dict[blank_id, int]``.
3. **Orchestrate** — the LLM orchestrator continues the planning
   conversation as one live tool-calling loop: it dispatches researchers
   per blank (``dispatch_subplan``), may ask the plan-updater to revise
   the plan (``request_plan_update``) and submits the answer
   (``submit_answer``). Constraint outputs are auto-computed by the
   cascade after every commit.

Researcher contract:
- ``search``, ``read``, ``find``, ``update_blank``, ``get_state`` tools.
- ``suggest_plan_revision`` escalates to the orchestrator/replanner when
  the plan is missing a retrieval step.
- ``update_blank`` rejects any id outside ``allowed_blank_ids``.
- No ``finish`` tool — researcher ends its own loop when all assigned
  blanks are resolved or max_turns hits.

Falls back to the flat ``ours_gsw_v1`` adapter on planner-side failure
(parse error, topology error). Researcher LLM errors do not trigger
fallback; they return a partial trajectory stamped
``stopped_reason="llm_error"``.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, Optional

from panini_dr.adapters.base import Adapter, AdapterContext, register_adapter
from panini_dr.adapters.ours._blank_resolver import (
    TypedBlankResolver,
    effective_query,
)
from panini_dr.adapters.ours._blank_verifier import (
    BlankVerifier,
    verify_blank,
    _render_edge_line,
)
from panini_dr.adapters.ours._edge_taxonomy import FACTUAL, classify_edge
from panini_dr.adapters.ours import _investigation_memory as _invmem_mod
from panini_dr.adapters.ours._investigation_memory import (
    InvestigationMemory,
    render_dossier,
    scoped_budget,
    result_chunk_ids as _invmem_result_chunk_ids,
    verdict_from_update_result as _invmem_verdict_from_result,
)
from panini_dr.adapters.ours._gsw_workspace import (
    BlankResearchResult,
    GSWWorkspace,
)
from panini_dr.adapters.ours._evidence_ranker import object_matches_value_type
from panini_dr.adapters.ours._connection_manifest import (
    build_connection_manifest,
)
from panini_dr.adapters.ours._context_compaction import (
    ToolResultArchive,
    compact_messages,
)
from panini_dr.adapters.ours._gsw_state import (
    QueryGSWState,
    blank_projection_edges,
    cascade_diagnostics,
    check_evidence_coverage,
    classify_failure,
    collect_supporting_edges,
    evidence_metrics,
    has_real_citation,
    initialize_query_gsw_state,
    judge_relation_llm,
    mirror_results_into_state,
    read_cited_chunks,
    unknown_chunk_ids,
    verify_graph,
)
from panini_dr.adapters.ours._recovery import recovery_diagnostics
from panini_dr.adapters.ours._orchestrator_prompt import (
    MAX_PROSE_NUDGES,
    PROSE_NUDGE,
    build_live_status_line,
    build_planner_bootstrap_prompt,
    build_orchestrator_system_static,
    build_state_message,
    render_plan_brief,
)
from panini_dr.adapters.ours._plan_updater import (
    update_plan,
)
from panini_dr.adapters.ours._planner_emit import (
    PlanEmitError,
    PlanEmitMeta,
    _PlannerFallbackMixin,
    emit_plan,
)
from panini_dr.adapters.ours._planner_exec import (
    BlankResult,
    ExecutionError,
    GSWPlan,
    _coerce_value,
    build_dependency_graph,
    topological_sort_blanks,
)
from panini_dr.adapters.ours._researcher_prompt import build_researcher_prompt
from panini_dr.adapters.ours import _agentic_read
from panini_dr.adapters.ours._gsw_temporal import finish_value
from panini_dr.adapters.ours.gsw_planner_react_v1 import (
    _cascade_auto_compute,
    _compute_levels,
    _stringify,
)
from panini_dr.models.llm_client import LLMClient
from panini_dr.models.trace import ToolCall, Trajectory
from panini_dr.retrieval.corpus import require_corpus
from panini_dr.retrieval.dense import build_retriever

logger = logging.getLogger(__name__)

def _researcher_readiness_telemetry(
    *,
    tool_calls: list[ToolCall],
    turn_usage: list[dict[str, Any]],
    state: dict[str, BlankResult],
    allowed_blank_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Measure when each committed blank's final evidence was already visible.

    This is computed after the run, when the accepted evidence ids are known,
    but uses the recorded action order and exact per-call token usage. It is
    intentionally persisted as compact telemetry rather than duplicating the
    full researcher transcript in ``Trajectory.extra``.
    """
    targets: dict[str, set[str]] = {}
    for blank_id in allowed_blank_ids:
        result = state.get(blank_id)
        if result is None or result.status != "resolved":
            continue
        evidence = {
            str(chunk_id)
            for chunk_id in (result.evidence_chunk_ids or [])
            if str(chunk_id).strip() and str(chunk_id) != "insufficient_evidence"
        }
        if evidence:
            targets[blank_id] = evidence
    if not targets:
        return {}

    seen_chunks: set[str] = set()
    ready_turns: dict[str, int] = {}
    search_events: list[tuple[int, bool]] = []
    for call in tool_calls:
        try:
            result = json.loads(call.result_full or call.result_preview or "{}")
        except (json.JSONDecodeError, TypeError):
            result = {}
        result = result if isinstance(result, dict) else {}
        before = set(seen_chunks)
        seen_chunks.update(_invmem_result_chunk_ids(call.name, call.args or {}, result))
        if call.name == "search":
            search_events.append((int(call.turn), seen_chunks == before))
        for blank_id, evidence in targets.items():
            if blank_id not in ready_turns and evidence.issubset(seen_chunks):
                ready_turns[blank_id] = int(call.turn)

    telemetry: dict[str, dict[str, Any]] = {}
    for blank_id, evidence in targets.items():
        ready_turn = ready_turns.get(blank_id)
        if ready_turn is None:
            continue
        post_ready_prompt_tokens = sum(
            int(row.get("prompt_tokens") or 0)
            for row in turn_usage
            if int(row.get("turn") or 0) > ready_turn
            or (
                int(row.get("turn") or 0) == ready_turn
                and bool(row.get("after_tool"))
            )
        )
        telemetry[blank_id] = {
            "evidence_chunk_ids": sorted(evidence),
            "evidence_ready_turn": ready_turn,
            "post_ready_prompt_tokens": post_ready_prompt_tokens,
            "post_ready_actions": sum(
                1 for call in tool_calls if int(call.turn) > ready_turn
            ),
            "post_ready_searches": sum(1 for turn, _ in search_events if turn > ready_turn),
            "post_ready_zero_new_searches": sum(
                1 for turn, zero_new in search_events if turn > ready_turn and zero_new
            ),
        }
    return telemetry


# ---------------------------------------------------------------------------
# `read` / `read_raw` windowing.
#
# The underlying article can be far larger than a single tool response
# should carry. `_READ_WINDOW` chars are returned per call; when the
# article is longer than that, a `[TRUNCATED: ...]` notice is appended
# naming the next `offset` to pass to keep reading. Articles that fit in
# one window are returned byte-identical to before this notice existed —
# no marker text, no behavior change.
# ---------------------------------------------------------------------------


_READ_WINDOW = 12000


_SECTION_RE = re.compile(r"^#{2,}\s*(.+?)\s*$", re.M)


def _article_outline(
    full: str, win_start: int, win_end: int
) -> list[dict[str, Any]]:
    """Map of an article's sections: ``[{section, start, shown}]``.

    A read window is a *fragment*, and returning only its bytes leaves the model
    unable to navigate (it cannot know a table sits at char 38,078) or to tell a
    partial view from a whole one — which is how a gold-7 count gets answered as
    2. The outline is cheap (a few hundred chars against a 12,000-char body) and
    purely additive: the window text is unchanged.
    """
    out: list[dict[str, Any]] = []
    for m in _SECTION_RE.finditer(full):
        start = m.start()
        out.append({
            "section": m.group(1),
            "start": start,
            "shown": win_start <= start < win_end,
        })
    return out


# How many empty reader results on ONE article before the parent gets the raw
# window back. 2 = the reader gets a genuine second attempt first.
_READER_EMPTY_BEFORE_RAW = 2


def _read_payload(
    full: str, offset: int, *, chunk_id: str | None = None, title: str | None = None
) -> dict[str, Any]:
    """Build a read response: the window, plus a map of what surrounds it."""
    total = len(full)
    start = max(0, min(offset, total))
    end = min(start + _READ_WINDOW, total)
    payload: dict[str, Any] = {}
    if chunk_id is not None:
        payload["chunk_id"] = chunk_id
    if title is not None:
        payload["title"] = title
    payload["article_chars"] = total
    payload["window"] = {"start": start, "end": end}
    # No outline here. The section map exists so a reader can navigate a whole
    # article and choose what to pull; the researcher does not navigate — it
    # reads the head for the infobox and asks `ask` when it needs more. Handing
    # it a map it cannot act on is tokens for nothing, so the map now goes to the
    # reader only (see `_tool_ask` -> read_article_briefed(outline=...)).
    payload["article_text"] = _windowed_article_text(full, offset)
    return payload


def _windowed_article_text(full: str, offset: int = 0) -> str:
    """Slice ``full`` to a ``_READ_WINDOW``-char window starting at
    ``offset`` (clamped to ``[0, len(full)]``), appending a truncation
    notice when the window doesn't reach the end of the article.

    Byte-identical to ``full`` when the whole article fits in one
    window at ``offset=0`` — no notice, no trailing marker.
    """
    total = len(full)
    offset = max(0, min(offset, total))
    end = min(offset + _READ_WINDOW, total)
    window = full[offset:end]
    if end >= total:
        if offset == 0:
            # Whole article fit in a single window — untouched.
            return window
        # Last window of a truncated article: say so, but don't dangle
        # an `offset=` pointing past the end.
        return f"{window}\n\n[END OF ARTICLE]"
    return (
        f"{window}\n\n[TRUNCATED: showing characters {offset}-{end} of "
        f"{total}. Call read with offset={end} to continue reading this "
        "article.]"
    )


def _normalise_question(question: str) -> str:
    """Cache key for the read ledger. Textual only — no stemming, no synonyms.

    A false cache hit returns a stale answer to a genuinely different question,
    which is strictly worse than a miss.
    """
    return re.sub(r"\W+", " ", str(question or "")).lower().strip()


# ---------------------------------------------------------------------------
# Tool definitions for the researcher (5 tools — no `finish`)
# ---------------------------------------------------------------------------


_ORCHESTRATOR_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "dispatch_subplan",
            "description": (
                "Dispatch one researcher per blank_id in the list, running "
                "them in parallel (capped at 8 concurrent). Each researcher "
                "retrieves evidence and calls `update_blank` on its assigned "
                "blank. Returns the resolved state for those blanks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "blank_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Blank ids to resolve in parallel. 1–8 ids. "
                            "Ids must exist in the current plan and must not "
                            "already be status=resolved, and must not be "
                            "constraint-output blanks."
                        ),
                    },
                    "hints": {
                        "type": "string",
                        "description": (
                            "Optional short note passed to the researchers "
                            "(retrieval query hints, bridging anchors, etc.)."
                        ),
                    },
                },
                "additionalProperties": False,
                "required": ["blank_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_plan_update",
            "description": (
                "Invoke the plan-updater LLM to revise the GSWPlan given "
                "retrieved evidence. Use only when the plan itself is wrong "
                "or incomplete (not for weak-retrieval cases)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Short justification for the update.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": (
                            "Short string capturing retrieved facts that "
                            "contradict or extend the current plan."
                        ),
                    },
                },
                "additionalProperties": False,
                "required": ["reason", "evidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": (
                "Return the current plan + blank-fill state. Rarely "
                "needed; the state is already in the system prompt."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "Terminate with the final answer. The target blank must "
                "already be status=resolved; pass its committed value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                },
                "additionalProperties": False,
                "required": ["answer"],
            },
        },
    },
]

#: Every role this adapter meters into ``extra["role_usage"]``. Declared once,
#: because the GSW-off ablation protocol compares the two arms role by role and a
#: role that appears only as a string literal at one call site cannot be
#: reconciled with the control's ledger. The first seven are the protocol's
#: required set; the rest are full-arm-only components the control has no
#: counterpart for, kept separate rather than folded into a neighbouring role.
ROLE_USAGE_ROLES = (
    "orchestrator", "researcher", "verifier", "resolver", "reader", "planner",
    "finalizer", "auditor", "synthesis_validator", "evidence_judge",
)


class _RoleUsage:
    """Per-question prompt/completion tokens and call counts, keyed by role.

    LOCKED, and for the same reason the control's ledger is: one ``_MeteredLLM``
    per role serves every researcher of a question, and those researchers run in
    ``_dispatch_subplan_parallel``'s thread pool, so several threads really do
    reach ``add`` at once. ``r["calls"] += 1`` is three bytecodes with a
    preemption point in the middle; token cost is a measured column of this
    ablation, so a lost update is a wrong number rather than a crash. The lock
    covers the ``setdefault`` too, so two threads meeting an unseen role cannot
    install two counters.

    One instance per question, created in ``run_question`` and reachable only
    from that question — never a counter on ``self``, which the harness's own
    pool would share between concurrent questions.
    """

    def __init__(self) -> None:
        self._d: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def add(self, role: str, prompt: int, completion: int) -> None:
        with self._lock:
            r = self._d.setdefault(
                role, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
            r["prompt_tokens"] += int(prompt or 0)
            r["completion_tokens"] += int(completion or 0)
            r["calls"] += 1

    def to_dict(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {k: dict(v) for k, v in self._d.items()}


class _MeteredLLM:
    """An LLM client wrapper that counts one role's usage. NOTHING ELSE.

    ``chat`` forwards its arguments verbatim, returns the provider's response
    object unchanged and adds no retry, no timeout and no error handling -- a
    provider exception propagates exactly as it would unwrapped, and a call that
    raised is not counted (there is no usage to count). Every other attribute is
    delegated to the wrapped client, so ``supports_seed()``,
    ``default_temperature`` and the rest read through as before.

    The ledger is captured at CONSTRUCTION, not looked up per call: the
    researcher, its reader and its verifier are built inside a pool worker, and
    a per-call thread-local lookup there would meter into an empty per-worker
    ledger (the failure mode ``_read_ledger`` and ``_checker_sink`` document).
    Construct on the question thread, then close over it.
    """

    def __init__(self, llm: Any, role: str, usage: "_RoleUsage") -> None:
        self.llm = llm
        self.role = role
        self.usage = usage

    def chat(self, *args: Any, **kwargs: Any):
        resp = self.llm.chat(*args, **kwargs)
        self.usage.add(self.role,
                       getattr(resp, "prompt_tokens", 0),
                       getattr(resp, "completion_tokens", 0))
        return resp

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes this wrapper does not define, so `llm`,
        # `role`, `usage` and `chat` never recurse.
        return getattr(self.llm, name)


class _PendingResponse:
    """The planning phase's final tool call, replayed as the loop's first turn.

    Shaped like an LLM response so the loop needs no special-casing beyond the
    one branch that produces it: same `.text`, `.tool_calls`, token fields.
    Tokens are zero because the call was already paid for during planning.
    """

    def __init__(self, call: dict):
        self.text = ""
        self.reasoning_content = ""
        self.tool_calls = [call]
        self.prompt_tokens = 0
        self.completion_tokens = 0


#: The planning phase's only tool, and the last tool of the orchestrator's list.
_CALL_PLANNER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "call_planner",
        "description": (
            "Build the worksheet for this question: typed blanks, dependencies, "
            "constraints. Takes NO arguments — you have retrieved nothing, so "
            "anything you could say here would be recalled from memory rather "
            "than known, and 75% of the entities the audit catches in plans were "
            "named by the orchestrator before the planner ever ran. The result "
            "carries the plan and its audit; read the audit and decide what to "
            "do about it."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}


#: gsw_plan_request_contract: the OPTIONAL `plan_request_id` argument spliced
#: into `request_plan_update` ONLY when the flag is on, so a flag-off run's
#: tool schema -- and therefore its prompt -- stays byte-identical to the
#: pre-contract schema (an optional field, purely additive).
_PLAN_REQUEST_ID_FIELD: dict[str, dict[str, str]] = {
    "plan_request_id": {
        "type": "string",
        "description": (
            "REQUIRED whenever a researcher escalated. Copy it verbatim from "
            "`revision_requests[].request_id` in the dispatch_subplan result "
            "you are responding to (e.g. \"806:b_pop_london:0\"). Omitting it "
            "throws away the researcher's concrete structural proposal and "
            "leaves the updater with only your paraphrase. Omit ONLY when no "
            "researcher escalated and this update is entirely your own "
            "initiative."
        ),
    },
}


def _orchestrator_tools(*, plan_request_contract: bool) -> list[dict[str, Any]]:
    """The orchestrator's tool list for this run: ``_ORCHESTRATOR_TOOLS`` plus
    ``call_planner``.

    ``plan_request_contract`` splices ``plan_request_id`` into
    ``request_plan_update`` ONLY when on: every other tool dict in the
    result is the shared module-level object from ``_ORCHESTRATOR_TOOLS``,
    unmutated, so a flag-off run gets the exact
    ``_ORCHESTRATOR_TOOLS + [_CALL_PLANNER_TOOL]`` list.
    """
    tools = list(_ORCHESTRATOR_TOOLS)
    if plan_request_contract:
        tools = [
            (
                {
                    **t,
                    "function": {
                        **t["function"],
                        "parameters": {
                            **t["function"]["parameters"],
                            "properties": {
                                **t["function"]["parameters"]["properties"],
                                **_PLAN_REQUEST_ID_FIELD,
                            },
                        },
                    },
                }
                if t["function"]["name"] == "request_plan_update"
                else t
            )
            for t in tools
        ]
    return tools + [_CALL_PLANNER_TOOL]


def build_commit_tool_schema() -> dict[str, Any]:
    """Build the ``update_blank`` tool schema."""
    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "update_blank",
            "description": (
                "Record a blank's resolved value into shared state. You may "
                "only update blanks in your assigned slice — other ids will "
                "be rejected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "blank_id": {"type": "string"},
                    "value": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "boolean"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                    },
                    "evidence_chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "additionalProperties": False,
                "required": ["blank_id", "value", "evidence_chunk_ids"],
            },
        },
    }
    return schema


# Deterministic final-answer hygiene, applied once at the run_question choke
# point (all 25 scattered `final_answer =` sites flow through it). MEASURED
# need (arm C7, 2026-08-17): two judge losses were formatting, not content --
# monaco_1512 emitted the correct actresses as "A → B" (narrow
# no-break spaces + arrow separators; all-or-nothing judge 1.00 -> 0.00) and
# monaco_920 emitted "J.R.R. Tolkien**" with markdown-bold residue. The
# answer-format contract is comma-separated values with no markup; this can
# only converge answers TOWARD that contract, and it applies identically to
# every arm.
_ANSWER_SPACE_VARIANTS = (" ", " ", " ", " ")


def _normalize_final_answer(ans: str) -> str:
    if not ans:
        return ans
    out = ans
    for ch in _ANSWER_SPACE_VARIANTS:
        out = out.replace(ch, " ")
    out = out.replace("**", "")
    # Separator repair: two-plus arrows/middots with no comma is an
    # enumeration in the wrong dialect (the arrow chronology, the render's
    # middot joiner). A single occurrence stays -- it may be prose.
    for sep in ("→", "·"):
        if out.count(sep) >= 2 and "," not in out:
            out = ", ".join(p.strip() for p in out.split(sep) if p.strip())
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    return out


ASK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ask",
        "description": (
            "Ask one question about one article. A reader with the FULL article "
            "answers it and cites the exact spans it used, or returns "
            "not_found with the sections it checked. Citations are verified "
            "against the article, so an answer with no real evidence is "
            "reported as not_found rather than returned. A not_found is "
            "informative, not a failure — if it lists sections it could not "
            "see, ask again about one of those, or use `find` to locate text "
            "yourself. Repeating the same question about the same chunk "
            "returns the cached result for free."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
                "question": {
                    "type": "string",
                    "description": (
                        "One specific question this article should answer, e.g. "
                        "'which political parties are listed and who leads each?'"
                    ),
                },
            },
            "additionalProperties": False,
            "required": ["chunk_id", "question"],
        },
    },
}


#: `read` IS the reader: the `ask` contract above under the name the model
#: already knows. The parent never sees raw article text; the reader's returned
#: `infobox` is its route to the fact table, and after repeated empty reads of
#: one article `_tool_ask` hands back the raw window instead.
_READ_TOOL: dict[str, Any] = copy.deepcopy(ASK_TOOL)
_READ_TOOL["function"]["name"] = "read"


_RESEARCHER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Retrieve chunks from the FRAMES article corpus most relevant "
                "to a query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "additionalProperties": False,
                "required": ["query"],
            },
        },
    },
    _READ_TOOL,
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": (
                "Search inside a specific chunk's article for a literal "
                "substring (case-insensitive). Returns up to 3 snippets "
                "of ±200 chars around each match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "pattern": {"type": "string"},
                },
                "additionalProperties": False,
                "required": ["chunk_id", "pattern"],
            },
        },
    },
    build_commit_tool_schema(),
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": (
                "Return the current blank-fill state across the whole "
                "plan. Each entry is flagged `writable` — only writable "
                "ids are in your assigned slice."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_plan_revision",
            "description": (
                "Escalate to the orchestrator: the plan is missing a step. "
                "Use ONLY when retrieval reveals that resolving your "
                "assigned blank requires a NEW intermediate blank that "
                "does not exist in the plan yet (not when evidence for "
                "your current blank is just thin — for that, commit your "
                "best-guess value citing the chunk you read instead). "
                "Calling this tool "
                "ENDS your run; your assigned blank stays unresolved and "
                "the orchestrator decides whether to revise the plan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Short note on what was missing. Example: "
                            "'no chunk in the corpus lists postcodes for "
                            "the Wren Clinic'."
                        ),
                    },
                    "hint": {
                        "type": "string",
                        "description": (
                            "Concrete suggestion for the new step, e.g. a "
                            "new blank id and what predicate would link it. "
                            "Example: 'add b_address (entity for the "
                            "hospital address); postcode could be derived "
                            "from address'."
                        ),
                    },
                },
                "additionalProperties": False,
                "required": ["reason", "hint"],
            },
        },
    },
]


_BIND_CONNECTION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bind_connection",
        "description": (
            "Attach the chunk_ids that support ONE connection (vp_id) of a "
            "blank, from its connection manifest. Call once per connection as "
            "you confirm it; then call update_blank to commit the value."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "blank_id": {"type": "string"},
                "vp_id": {"type": "string"},
                "evidence_chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "additionalProperties": False,
            "required": ["blank_id", "vp_id", "evidence_chunk_ids"],
        },
    },
}


_SEARCH_SNIPPET_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def _non_stopword_query_terms(query: str) -> list[str]:
    return [
        term
        for term in re.findall(r"[A-Za-z0-9]+", query.lower())
        if len(term) > 1 and term not in _SEARCH_SNIPPET_STOPWORDS
    ]


def _centered_search_snippet(text: str, query: str, budget: int) -> str:
    """Render a search snippet centered on the first query-term match."""
    if len(text) <= budget:
        return text

    lower = text.lower()
    match_start: int | None = None
    match_len = 0
    for term in _non_stopword_query_terms(query):
        idx = lower.find(term)
        if idx < 0:
            continue
        if match_start is None or idx < match_start:
            match_start = idx
            match_len = len(term)

    if match_start is None:
        return text[:budget]

    midpoint = match_start + (match_len // 2)
    start = max(0, midpoint - (budget // 2))
    end = start + budget
    if end > len(text):
        end = len(text)
        start = max(0, end - budget)

    if start > 0 and not text[start].isspace():
        prev_space = text.rfind(" ", 0, start)
        prev_newline = text.rfind("\n", 0, start)
        prev_ws = max(prev_space, prev_newline)
        if prev_ws >= 0 and start - (prev_ws + 1) <= 20:
            start = prev_ws + 1

    if end < len(text) and not text[end - 1].isspace():
        next_space = text.find(" ", end)
        next_newline = text.find("\n", end)
        candidates = [idx for idx in (next_space, next_newline) if idx >= 0]
        if candidates:
            expanded_end = min(candidates)
            if expanded_end - start <= budget + 20:
                end = expanded_end

    return text[start:end]


def _resolved_target_answer(
    final_answer: str,
    target_id: str,
    state: dict[str, BlankResult],
) -> str:
    """Loop-exit safety net: surface a resolved target as the answer when the
    orchestrator produced none.

    q429: the orchestrator can resolve the target into ``state`` yet run to the
    turn/plan-update cap without ever calling ``submit_answer``, leaving the
    final answer empty. A resolved (gate-passed) target value is strictly better
    than an empty answer, so use it. Never overrides a non-empty answer and
    never fabricates one for an unresolved or empty-valued target.
    """
    if (final_answer or "").strip():
        return final_answer
    res = state.get(target_id)
    if res is not None and res.status == "resolved" and res.value not in (None, ""):
        return _stringify(res.value)
    return final_answer or ""


def _normalise_deterministic_final_answer(
    question: str,
    plan: GSWPlan,
    state: dict[str, BlankResult],
    final_pred: str,
) -> str:
    """Apply narrow, deterministic answer-shape fixes before validation."""
    target = plan.target()
    target_res = state.get(target.id)
    if target_res is None or target_res.status != "resolved":
        return final_pred

    q = (question or "").casefold()
    value = target_res.value
    if isinstance(value, bool) and "shorter or longer" in q:
        return "Shorter." if value else "Longer."

    if isinstance(value, bool) and "longer or shorter" in q:
        return "Longer." if value else "Shorter."

    if isinstance(value, bool):
        comparison_answer = _normalise_comparison_bool_answer(
            question,
            plan,
            target.id,
            value,
        )
        if comparison_answer:
            return comparison_answer

    return final_pred


def _constraint_blank_inputs_for_repair(c: Any, blank_ids: set[str]) -> list[str]:
    """Return blank inputs read by a constraint for repair invalidation."""
    refs: list[str] = []
    if getattr(c, "kind", "") == "derived":
        refs.extend(getattr(c, "args_refs", []) or getattr(c, "args_blanks", []) or [])
    elif getattr(c, "kind", "") in {"argmax", "argmin"}:
        refs.extend(getattr(c, "sort_by_blank_ids", []) or [])
    elif getattr(c, "kind", "") in {"equals", "gt", "lt"}:
        refs.extend(
            ref for ref in (getattr(c, "left_ref", None), getattr(c, "right_ref", None))
            if ref
        )
    elif getattr(c, "kind", "") == "in_list":
        refs.extend(
            ref for ref in (getattr(c, "left_ref", None), getattr(c, "right_ref", None))
            if ref
        )
        refs.extend(getattr(c, "args_blanks", []) or [])
    return [ref for ref in refs if ref in blank_ids]


def _downstream_blanks_for_repair(plan: GSWPlan, seed_blank_id: str) -> list[str]:
    """Return seed plus constraint outputs that depend on the seed."""
    blank_ids = {e.id for e in plan.blank_entities()}
    if seed_blank_id not in blank_ids:
        return []

    reset_ids = {seed_blank_id}
    changed = True
    while changed:
        changed = False
        for c in plan.constraints:
            out_id = c.output_blank_id
            if not out_id or out_id not in blank_ids or out_id in reset_ids:
                continue
            inputs = set(_constraint_blank_inputs_for_repair(c, blank_ids))
            if inputs & reset_ids:
                reset_ids.add(out_id)
                changed = True

    return [seed_blank_id] + sorted(reset_ids - {seed_blank_id})


def _normalise_comparison_bool_answer(
    question: str,
    plan: GSWPlan,
    target_id: str,
    value: bool,
) -> str:
    q = (question or "").casefold()
    wants_greater = any(
        phrase in q
        for phrase in (
            "what is greater",
            "which is greater",
            "what's greater",
            "what is larger",
            "which is larger",
            "what is bigger",
            "which is bigger",
            "what is higher",
            "which is higher",
        )
    )
    wants_smaller = any(
        phrase in q
        for phrase in (
            "what is smaller",
            "which is smaller",
            "what is lower",
            "which is lower",
            "what is less",
            "which is less",
            "what is fewer",
            "which is fewer",
        )
    )
    if not (wants_greater or wants_smaller):
        return ""

    constraint = next(
        (
            c for c in plan.constraints
            if c.output_blank_id == target_id
            and c.kind in {"gt", "lt"}
            and c.left_ref
            and c.right_ref
        ),
        None,
    )
    if constraint is None:
        return ""

    left_is_greater = bool(value) if constraint.kind == "gt" else not bool(value)
    choose_left = left_is_greater if wants_greater else not left_is_greater
    chosen_ref = constraint.left_ref if choose_left else constraint.right_ref
    if not chosen_ref:
        return ""

    description = _describe_plan_ref_for_answer(plan, chosen_ref)
    if not description:
        return ""
    return _sentence_answer(description)


def _describe_plan_ref_for_answer(plan: GSWPlan, ref: str) -> str:
    produced_by = next(
        (
            c for c in plan.constraints
            if c.output_blank_id == ref
            and c.kind == "derived"
            and c.op in {"sum", "avg"}
        ),
        None,
    )
    if produced_by is not None:
        parts = [
            _attribute_part_for_blank(plan, arg)
            for arg in [*produced_by.args_blanks, *produced_by.args_refs]
        ]
        if parts and all(part is not None for part in parts):
            typed_parts = [part for part in parts if part is not None]
            years = {part.get("year", "") for part in typed_parts}
            attrs = {part.get("attribute", "") for part in typed_parts}
            subjects = [
                part["subject"] for part in typed_parts
                if part.get("subject")
            ]
            if len(years) == 1 and len(attrs) == 1 and subjects:
                year = next(iter(years))
                attr = next(iter(attrs))
                year_prefix = f"{year} " if year else ""
                noun = attr if attr.endswith("s") else f"{attr}s"
                return (
                    f"the combined {year_prefix}{noun} of "
                    f"{_join_names(subjects)}"
                )

    part = _attribute_part_for_blank(plan, ref)
    if part is not None:
        year_prefix = f"{part['year']} " if part.get("year") else ""
        return f"the {year_prefix}{part['attribute']} of {part['subject']}"

    try:
        ent = plan.entity_by_id(ref)
    except ExecutionError:
        return ""
    if ent.kind == "filled" and ent.name:
        return ent.name
    return ""


def _attribute_part_for_blank(plan: GSWPlan, ref: str) -> dict[str, str] | None:
    attr_vp = next(
        (
            vp for vp in plan.verb_phrases
            if vp.object_id == ref
            and _attribute_from_phrase(vp.phrase, ref)
        ),
        None,
    )
    if attr_vp is None:
        return None

    try:
        subject = plan.entity_by_id(attr_vp.subject_id)
    except ExecutionError:
        return None
    if subject.kind != "filled" or not subject.name:
        return None

    year = ""
    for vp in plan.verb_phrases:
        if vp.subject_id != subject.id:
            continue
        if vp.phrase not in {"as_of_date", "as_of", "in_year", "during_year"}:
            continue
        try:
            anchor = plan.entity_by_id(vp.object_id)
        except ExecutionError:
            continue
        if anchor.kind == "filled" and anchor.name:
            year = anchor.name
            break

    return {
        "attribute": _attribute_from_phrase(attr_vp.phrase, ref),
        "subject": subject.name,
        "year": year,
    }


def _attribute_from_phrase(phrase: str, ref: str) -> str:
    text = f"{phrase} {ref}".casefold()
    for attr in ("population", "attendance", "capacity", "height", "elevation"):
        if attr in text:
            return attr
    return ""


def _join_names(names: list[str]) -> str:
    if len(names) <= 1:
        return names[0] if names else ""
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _sentence_answer(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _verdict_dict(v) -> dict:
    """Uniform verdict record for sink/attempt logging across ReAct + resolver paths."""
    return {
        "satisfies_all": v.satisfies_all,
        "unique": v.unique,
        "complete": v.complete,
        "ambiguous_candidates": v.ambiguous_candidates,
        "deferred_constraints": v.deferred_constraints,
        "reason": v.reason,
    }


# ---------------------------------------------------------------------------
# ReActResearcher — one per topological level
# ---------------------------------------------------------------------------


class ReActResearcher:
    """A single-level researcher agent.

    Runs a ReAct loop with 5 tools until every blank in
    ``allowed_blank_ids`` has ``status=resolved`` in ``state``, or
    until ``max_turns`` is exhausted. Does NOT return a final answer —
    the orchestrator reads the resolved target out of ``state``.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        corpus: Any,
        retriever: Any,
        plan: GSWPlan,
        state: dict[str, BlankResult],
        allowed_blank_ids: list[str],
        system_prompt: str,
        question: str,
        max_turns: int = 15,
        max_completion_tokens: int = 50000,
        llm_seed: Optional[int] = None,
        # The article reader runs in here off the researcher's own client. Its
        # own client keeps "read harder" separable from "search harder".
        reader_llm: Optional[LLMClient] = None,
        hard_type_filter: bool = False,
        handoff_hints: Optional[dict] = None,
        blank_verifier: Optional[Any] = None,
        edge_grounded_cap: int = 1,
        block_unverified_cascade: bool = False,
        verifier_reasoning: bool = False,
        read_raw_fallback: bool = True,
        read_ledger: Optional[dict] = None,
        context_compaction_on: bool = False,
        codegen: Any = None,
        prior_investigation: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.reader_llm = reader_llm
        self.corpus = corpus
        self.retriever = retriever
        self.plan = plan
        self.state = state
        self.allowed: set[str] = set(allowed_blank_ids)
        self.allowed_order: list[str] = list(allowed_blank_ids)
        self.system_prompt = system_prompt
        self.question = question
        self.max_turns = max_turns
        self.max_completion_tokens = max_completion_tokens
        self.llm_seed = llm_seed
        self.hard_type_filter = hard_type_filter
        self.handoff_hints = handoff_hints
        self.blank_verifier = blank_verifier
        # Shared re-attempt budget for the two edge-grounded gates. 1 reproduces the run
        # of record; raising it is the cap-sensitivity probe.
        self.gsw_edge_grounded_cap = int(edge_grounded_cap)
        self.block_unverified_cascade = block_unverified_cascade
        self.verifier_reasoning = verifier_reasoning
        # The repeat-empty raw-window fallback shipped unconditionally inside
        # the reader arm, which makes 'reader' and 'reader+fallback' the same
        # experiment. Default True preserves the measured behaviour; pass
        # --no-gsw-read-raw-fallback to get the pre-fallback arm.
        self.read_raw_fallback = bool(read_raw_fallback)
        self._read_ledger_ref = read_ledger if read_ledger is not None else {}
        # Task 8: every `read` (reader) call (cache hit, fresh answer/not_found,
        # error) appends one record here — this is what lets an offline pass
        # check whether a `not_found` was genuine (spec §6). Surfaced into
        # ``traj.extra["agentic_read"]`` (see ``_attach_gsw_trace_metadata``).
        self._agentic_read_log: list[dict[str, Any]] = []
        self.context_compaction_on = context_compaction_on
        self._codegen = codegen
        self._prior_investigation = prior_investigation
        self._tool_archive = (
            ToolResultArchive()
            if self._archive_needed()
            else None
        )
        # Effective researcher tool list: the module-level tools (with the
        # optional `reasoning` field on update_blank under verifier_reasoning)
        # plus bind_connection, which attributes evidence chunks to each of a
        # blank's connections (edge-grounded commits).
        if self.verifier_reasoning:
            self._tools = copy.deepcopy(_RESEARCHER_TOOLS)
            for tool in self._tools:
                fn = tool.get("function", {})
                if fn.get("name") != "update_blank":
                    continue
                params = fn["parameters"]
                params["properties"]["reasoning"] = {
                    "type": "string",
                    "description": (
                        "Briefly explain why this value, from which chunk, "
                        "and how it connects to known values."
                    ),
                }
            self._tools.append(copy.deepcopy(_BIND_CONNECTION_TOOL))
        else:
            self._tools = list(_RESEARCHER_TOOLS) + [_BIND_CONNECTION_TOOL]
        # Per-connection evidence sub-state (Task 5). ONLY written by
        # bind_connection — never `self.state`. blank_id -> {vp_id -> [chunk_id]}.
        self._connection_evidence: dict[str, dict[str, list[str]]] = {}
        self._verifier_rejects: dict[str, int] = {}
        self._react_attempts: list[dict[str, Any]] = []
        self._verifier_flagged: dict[str, str] = {}
        # Phase-1 investigation memory: this sibling's episode buffer. The
        # orchestrator sets it to a fresh Episode when the flag is on (before
        # running this researcher in the pool); left None otherwise, so every
        # record hook below is a no-op and behavior is byte-identical.
        self._invmem_episode = None
        # Phase 4 Task 7: per-researcher cascade-block records. Filled by
        # `_tool_update_blank`'s cascade call (via blocked_sink). Returned
        # from the pool worker and MERGED on the MAIN thread — never written
        # to a shared sink from a pool thread. Mirrors `_react_attempts`.
        self._cascade_blocks: list[dict[str, Any]] = []
        # Set by `_tool_suggest_plan_revision` when the researcher
        # escalates. Surfaced in ``solve()``'s return value.
        self._revision_request: Optional[dict[str, Any]] = None

    def _archive_needed(self) -> bool:
        """The tool archive backs context compaction."""
        return bool(getattr(self, "context_compaction_on", False))

    # --- tool implementations (bound to this researcher's state) ----

    def _tool_search(self, *, query: str, top_k: int = 5) -> dict[str, Any]:
        hits = self.retriever.search(query, top_k=top_k)
        snippet_chars = 800
        results = [
            {
                "chunk_id": h.chunk.chunk_id,
                "title": h.chunk.title,
                "score": round(h.score, 3),
                "text": _centered_search_snippet(h.chunk.text, query, snippet_chars),
            }
            for h in hits
        ]
        return {"results": results}

    def _tool_ask(self, *, chunk_id: str, question: str) -> dict[str, Any]:
        """Reader sub-agent: answer ONE question about ONE article.

        This is the `read` tool (and `ask`, the same contract under its own
        name, which a model may still call): the reader reads the whole
        article and answers with verified citations, or returns a scoped
        not_found. After repeated empty reads of one article the raw window
        is handed back instead (``read_raw_fallback``).
        """
        chunk = self.corpus.get_chunk(chunk_id)
        if chunk is None:
            return {"error": f"chunk_id {chunk_id!r} not found"}
        full = self.corpus.article_text(chunk.title)
        q = str(question or "").strip()
        if not q:
            return {"error": "ask requires a `question`"}
        if not full.strip():
            # Spec §4: an article that is missing or empty is an ERROR return, not
            # a reading. Calling the reader on empty text yields a not_found that
            # would then be CACHED, so the whole question inherits a corpus gap as
            # if the page had been read and found wanting.
            return {"error": f"article for chunk_id {chunk_id!r} is missing or empty"}
        # Keyed on the TITLE, not the chunk_id: the reader reads the whole
        # ARTICLE, so two chunk_ids from the same page are two LLM calls over
        # byte-identical text. MoNaCo averages 6.0 reads over 3.0 distinct
        # articles per question, so a chunk-keyed ledger forfeits most of the
        # dedup the cache exists for (spec §3: 71% duplication).
        key = (chunk.title, _normalise_question(q))
        hit = self._read_ledger_ref.get(key)
        if hit is not None:
            payload = hit.to_dict()
            payload["cached"] = True
            self._agentic_read_log.append({
                "chunk_id": chunk_id, "question": q, "outcome": "cached",
                "answer": hit.answer,
                "citations": list(hit.citations),
                "not_found_reason": hit.not_found_reason,
                "sections_checked": list(hit.sections_checked),
                "sections_skipped": list(hit.sections_skipped),
                "n_citations": len(hit.citations),
                "candidates_considered": hit.candidates_considered,
                # A cache hit spends NO tokens — that is the whole point of the
                # ledger, and zero here is what makes spec §3's cost trade
                # computable from the logs.
                "prompt_tokens": 0, "completion_tokens": 0,
            })
            return payload
        # Briefed reader: hand over what we ALREADY know — the article's infobox,
        # its section map, and the passage `search` actually matched, with its
        # char position — instead of a whole-article dump. The single-shot reader
        # got the article and the map but never learned which chunk matched; 24%
        # of its reads concerned a passage >12k chars in and 26% returned
        # not_found. It escalates from there via find / read_full, capped at
        # MAX_READER_TURNS.
        result = _agentic_read.read_article_briefed(
            title=chunk.title,
            full_text=full,
            question=q,
            chunk_text=str(getattr(chunk, "text", "") or ""),
            chunk_start=int(getattr(chunk, "char_start", 0) or 0),
            outline=_article_outline(full, 0, 0),
            llm=self.reader_llm or self.llm,
            seed=self.llm_seed,
        )
        if result.error is None:          # never cache a transient failure
            self._read_ledger_ref[key] = result
        outcome = ("error" if result.error else
                   "answer" if result.answer is not None else "not_found")
        self._agentic_read_log.append({
            "chunk_id": chunk_id, "question": q, "outcome": outcome,
            "answer": result.answer,
            "citations": list(result.citations),
            "not_found_reason": result.not_found_reason,
            "sections_checked": list(result.sections_checked),
            "sections_skipped": list(result.sections_skipped),
            "n_citations": len(result.citations),
            "candidates_considered": result.candidates_considered,
            "turns": result.turns,
            "tool_calls": result.tool_calls,
            "prompt_tokens": int(result.prompt_tokens or 0),
            "completion_tokens": int(result.completion_tokens or 0),
        })
        payload = result.to_dict()
        # Repeat-empty fallback. Measured on FRAMES-824: the 126 questions where
        # the reader returns not_found twice or more carry the ENTIRE deficit
        # (8 fixed / 23 broken, McNemar p=0.011); on the other 461 the arm is
        # exactly level, 386 vs 386. Champion wins those by reading the raw
        # window and mining it with `find`, which this arm removed. So after the
        # second empty read of an ARTICLE, hand the window over instead of a
        # third nothing — the reader has had its chance and the parent has not.
        if outcome == "not_found":
            empties = sum(
                1 for (t_, _q), r_ in self._read_ledger_ref.items()
                if t_ == chunk.title and r_.answer is None and r_.error is None
            )
            if self.read_raw_fallback and empties >= _READER_EMPTY_BEFORE_RAW:
                raw = self._tool_read_raw(title=chunk.title)
                raw["reader_exhausted"] = True
                raw["reason"] = (
                    f"the reader returned nothing on this article {empties} times; "
                    f"here is the raw window — search it yourself with `find`"
                )
                if result.infobox:
                    raw["infobox"] = result.infobox
                return raw
        return payload

    def _tool_read_raw(self, *, title: str, offset: int = 0) -> dict[str, Any]:
        full = self.corpus.article_text(title)
        return _read_payload(full, offset, title=title)

    def _tool_find(self, *, chunk_id: str, pattern: str) -> dict[str, Any]:
        chunk = self.corpus.get_chunk(chunk_id)
        if chunk is None:
            return {"error": f"chunk_id {chunk_id!r} not found"}
        if not pattern or not pattern.strip():
            return {"error": "pattern must be a non-empty string"}
        full = self.corpus.article_text(chunk.title)
        hay = full.lower()
        needle = pattern.lower()
        snippets: list[dict[str, Any]] = []
        start = 0
        while len(snippets) < 3:
            idx = hay.find(needle, start)
            if idx < 0:
                break
            left = max(0, idx - 200)
            right = min(len(full), idx + len(pattern) + 200)
            snippets.append({"offset": idx, "snippet": full[left:right]})
            start = idx + len(pattern)
        return {
            "chunk_id": chunk_id,
            "title": chunk.title,
            "pattern": pattern,
            "match_count": len(snippets),
            "snippets": snippets,
        }

    def _read_cited_chunks(self, evidence_chunk_ids: list[str]) -> list[dict[str, str]]:
        return read_cited_chunks(self.corpus, evidence_chunk_ids)

    def _verifier_feedback(
        self,
        v: Any,
        chunks: list[dict[str, str]],
        *,
        blank_id: Optional[str] = None,
    ) -> str:
        ids = [c["chunk_id"] for c in chunks]
        action = (
            "re-search or read them and commit a value that satisfies ALL "
            "of this blank's relations."
        )
        return (
            f"Rejected: {v.reason}. "
            f"Unmet relations: {v.missing_constraints or '[]'}. "
            f"Ambiguous with: {v.ambiguous_candidates or '[]'}. "
            f"You examined chunk_ids {ids} — {action}"
        )

    # --- per-edge evidence store: blank_id -> {vp_id -> [chunk_id]} ----

    def _bound_edges_for(self, blank_id: str) -> dict[str, list[str]]:
        """{vp_id: [chunk_id]} bound for this blank."""
        return self._connection_evidence.get(blank_id, {})

    def _bind_evidence(
        self, blank_id: str, vp_id: str, chunk_ids: list[str],
    ) -> dict[str, list[str]]:
        """Record one edge's evidence; returns the blank's evidence map."""
        self._connection_evidence.setdefault(blank_id, {})[vp_id] = list(chunk_ids)
        return self._connection_evidence[blank_id]

    def _connection_evidence_snapshot(self, blank_id: str) -> dict[str, Any]:
        """``BlankResult.connection_evidence`` payload: a {vp_id: [chunk_id]} copy."""
        return dict(self._connection_evidence.get(blank_id, {}))

    def _tool_bind_connection(self, *, blank_id, vp_id, evidence_chunk_ids,
                              candidate=None):
        if blank_id not in self.allowed:
            return {"ok": False, "error": f"{blank_id!r} not in your slice."}
        valid_vps = {e["vp_id"] for e in blank_projection_edges(self.plan, blank_id)}
        if vp_id not in valid_vps:
            return {"ok": False, "error": f"vp_id {vp_id!r} is not a connection of "
                    f"{blank_id!r}. Valid: {sorted(valid_vps)}."}
        if not evidence_chunk_ids:
            return {"ok": False, "error": (
                "evidence_chunk_ids is empty — bind at least one chunk_id "
                "exactly as returned by search/read results.")}
        bad = unknown_chunk_ids(self.corpus, list(evidence_chunk_ids))
        if bad:
            return {"ok": False, "error": (
                f"unknown chunk id(s) {bad!r} — cite chunk ids exactly as "
                f"returned by search/read results.")}
        # `candidate` belonged to the removed candidate-evidence matrix; a call
        # that carries it is still validated as before, then ignored.
        if candidate is not None and (
            not isinstance(candidate, str) or not candidate.strip()
        ):
            return {"ok": False, "error": (
                "candidate must be a non-empty string naming the value this "
                "evidence supports — omit it for your current candidate.")}
        row = self._bind_evidence(blank_id, vp_id, list(evidence_chunk_ids))
        out: dict[str, Any] = {"ok": True, "bound": dict(row)}
        return out

    # --- version (b): edge-grounded verification ---------------------

    def _checkable_factual_edges(
        self, blank_id: str, dep_values: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Factual edges incident to ``blank_id`` the researcher owes evidence
        for RIGHT NOW. Scaffold edges (as_of_date / instance_of) and deferred
        edges (partner still unresolved) are excluded."""
        out: list[dict[str, Any]] = []
        for e in blank_projection_edges(self.plan, blank_id):
            if classify_edge(self.plan, e) != FACTUAL:
                continue
            partner = self.plan.entity_by_id(e["other_entity_id"])
            resolved = dep_values.get(e["other_entity_id"])
            checkable = partner.kind == "filled" or (
                resolved is not None and str(resolved).strip())
            if not checkable:
                continue  # deferred — partner unresolved, legitimately skipped
            out.append(e)
        return out

    def _unbound_factual_edges(
        self, blank_id: str, dep_values: dict[str, Any],
    ) -> list[tuple[str, str]]:
        """Checkable factual edges that have no bound evidence yet.
        Returns [(vp_id, rendered_relation_line), ...]."""
        bound = self._bound_edges_for(blank_id)
        return [
            (e["vp_id"], _render_edge_line(self.plan, e, dep_values))
            for e in self._checkable_factual_edges(blank_id, dep_values)
            if not bound.get(e["vp_id"])
        ]

    def _edge_grounded_verify(
        self, blank_id: str, coerced: Any, ev: list[str],
        reasoning: Optional[str] = None,
    ) -> tuple[Optional[str], bool, Any, Optional[dict[str, Any]]]:
        """Version (b) commit check. Two gates sharing ONE re-attempt budget
        (cap=1): (1) every factual+checkable edge must carry bound evidence;
        (2) one holistic verify over the UNION of bound evidence vs the blank's
        NL subquestion. On the 2nd failure of either gate, commit anyway flagged
        ``refuted_best_effort``.

        Returns ``(flagged, committed_partial, v, early_return)``. ``early_return``
        is a tool-response dict when the commit must be bounced back to the
        researcher; ``None`` means fall through to the shared commit."""
        # Effective re-attempt budget for the two edge-grounded gates, which SHARE it.
        # Historically the literal 1; `gsw_edge_grounded_cap` makes it settable for the
        # cap-sensitivity probe (2026-09-07); the paper ran 5.
        EFF_CAP = self.gsw_edge_grounded_cap
        dep_values = {b: br.value for b, br in self.state.items()
                      if br is not None and br.status == "resolved"
                      and br.value is not None and b != blank_id}

        # (1) grounding gate -----------------------------------------------
        unbound = self._unbound_factual_edges(blank_id, dep_values)
        if unbound:
            self._react_attempts.append({"blank_id": blank_id, "value": coerced,
                "source": "react", "note": "needs_edge_grounding",
                "unbound_vp_ids": [vp for vp, _ in unbound], "verdict": None})
            self._verifier_rejects[blank_id] = self._verifier_rejects.get(blank_id, 0) + 1
            if self._verifier_rejects[blank_id] <= EFF_CAP:
                lines = "\n".join(f"  - {vp}: {ln}" for vp, ln in unbound)
                return (None, False, None, {"ok": False, "gate": "needs_grounding",
                    "error": (
                        "Ground each factual connection before committing. "
                        "For each, search it then call bind_connection("
                        f"blank_id={blank_id!r}, vp_id, [chunk_id, ...]):\n"
                        + lines)})
            # cap reached — commit best-effort, skip the holistic check.
            return ("refuted_best_effort", False, None, None)

        # (2) holistic check over the union of bound evidence --------------
        bound = self._bound_edges_for(blank_id)
        union_ids = list(dict.fromkeys(
            list(ev) + [cid for ids in bound.values() for cid in ids]))
        chunks = self._read_cited_chunks(union_ids)
        if not chunks:
            self._verifier_rejects[blank_id] = self._verifier_rejects.get(blank_id, 0) + 1
            self._react_attempts.append({"blank_id": blank_id, "value": coerced,
                "source": "react", "note": "no_usable_cited_chunks", "verdict": None})
            if self._verifier_rejects[blank_id] <= EFF_CAP:
                return (None, False, None, {"ok": False, "gate": "verifier_rejected",
                    "error": ("No usable cited chunks for the verifier — cite "
                              ">=1 retrieved chunk_id that supports this value.")})
            return ("refuted_best_effort", False, None, None)

        verify_kwargs = {}
        if self.verifier_reasoning:
            verify_kwargs["reasoning"] = reasoning
        v = verify_blank(
            self.blank_verifier, self.plan, blank_id, coerced, dep_values,
            holistic_chunks=chunks,
            **verify_kwargs)
        committed_partial = bool(getattr(v, "partial", False))
        if not v.accepted:
            self._verifier_rejects[blank_id] = self._verifier_rejects.get(blank_id, 0) + 1
            self._react_attempts.append({"blank_id": blank_id, "value": coerced,
                "source": "react", "verdict": _verdict_dict(v)})
            if self._verifier_rejects[blank_id] <= EFF_CAP:
                return (None, committed_partial, v, {
                    "ok": False,
                    "gate": "verifier_rejected",
                    "error": self._verifier_feedback(v, chunks, blank_id=blank_id),
                })
            return ("refuted_best_effort", committed_partial, v, None)
        return (None, committed_partial, v, None)

    def _tool_update_blank(self, **kwargs) -> dict[str, Any]:
        """Phase-1 investigation-memory wrapper around the commit handler. Runs
        the real handler unchanged, then (only when the flag is on) records the
        proposal event from (args, result) AFTER the verify path has decided —
        capturing every return path and the handler's internal callers. No-op
        when off: pure pass-through, byte-identical."""
        result = self._tool_update_blank_impl(**kwargs)
        if self._invmem_episode is not None:
            self._invmem_record_proposal(kwargs, result)
        return result

    def _invmem_record_proposal(self, kwargs: dict, result: dict) -> None:
        try:
            blank_id = kwargs.get("blank_id")
            br = self.state.get(blank_id) if blank_id else None
            vs = getattr(br, "verifier_status", None) if br is not None else None
            verdict, reason = _invmem_verdict_from_result(result, vs)
            InvestigationMemory.add_proposal(
                self._invmem_episode, blank_id, kwargs.get("value"),
                kwargs.get("evidence_chunk_ids"), verdict, reason)
        except Exception:  # noqa: BLE001 — memory must never break the run
            pass

    def _tool_update_blank_impl(
        self,
        *,
        blank_id: str,
        value: Any,
        evidence_chunk_ids: list[str],
        reasoning: Optional[str] = None,
        # The signature is the set of argument names a model's update_blank
        # call may carry without a "bad tool call" error, so it is kept as the
        # paper ran it: as_of / valid_from / valid_to (the removed champion
        # temporal layer's fields) are accepted and ignored, and the
        # underscore arguments keep their meaning if a call carries them.
        as_of: str = "",
        valid_from: str = "",
        valid_to: str = "",
        _verifier_status: Optional[str] = None,
        _skip_verifier: bool = False,
    ) -> dict[str, Any]:
        # Slice guard — reject writes outside the assigned slice.
        if blank_id not in self.allowed:
            return {
                "ok": False,
                "error": (
                    f"blank_id {blank_id!r} is not in your assigned slice. "
                    f"You may only update: {sorted(self.allowed)}. "
                    f"Upstream blanks are read-only."
                ),
                "assigned_slice": sorted(self.allowed),
            }
        if blank_id not in self.state:
            return {
                "ok": False,
                "error": f"unknown blank_id {blank_id!r}",
                "known_blank_ids": sorted(self.state.keys()),
            }
        if (
            isinstance(value, str)
            and value.strip().lower() == "insufficient_evidence"
        ):
            return {
                "ok": False,
                "error": (
                    "Passing value='insufficient_evidence' is ALWAYS "
                    "wrong; it is not a value. Retry with your actual "
                    "best-guess as `value`, and cite the chunk it came "
                    "from in `evidence_chunk_ids`."
                ),
            }
        ent = self.plan.entity_by_id(blank_id)
        invalid_value_error = self._invalid_update_value_error(
            value, ent=ent, evidence_chunk_ids=evidence_chunk_ids
        )
        if invalid_value_error:
            return {"ok": False, "error": invalid_value_error}
        coerced = _coerce_value(value, ent.value_type)
        ev = list(evidence_chunk_ids)
        # NO GIVE-UP PATH (removed 2026-09-04). Between 219f20ec and this
        # commit a pure `evidence_chunk_ids=["insufficient_evidence"]` list
        # was honored as a declared give-up: evidence normalized to none, the
        # verifier was skipped, and the value landed with an empty evidence
        # set. On the BCP CF-150 run of record that admitted 223 of 674
        # model-resolved bindings (33.1%) with no evidence, behind 47 of 150
        # answers. Two paired ablations that refused the sentinel scored 68.0%
        # and 68.7% against that run's 65.3%, with zero evidence-free
        # bindings, so the path was not load-bearing and is gone: the sentinel
        # is now just an unknown chunk id and the CE check rejects it like any
        # other. Every admitted binding cites evidence.

        update_reasoning = reasoning if self.verifier_reasoning else None
        if self.hard_type_filter and not object_matches_value_type(
            coerced, ent.value_type
        ):
            # Stage A: value-local type admission — a date blank rejects a
            # non-date, a number blank a non-number, etc. (permissive for
            # entity/text). Needs only the value + its type, so it is snapshot-
            # safe. Off by default → byte-identical.
            return {
                "ok": False,
                "gate": "rejected",
                "error": (
                    f"Type filter: value {coerced!r} does not match the "
                    f"declared type {ent.value_type!r} for blank {blank_id!r}."
                ),
            }
        flagged: Optional[str] = None
        committed_partial = False
        v = None
        # Its verdict fed only the removed evidence-gate 'enforce' mode and
        # grounding fallback. The call stays because it is not inert: an
        # unhashable cited id (a model passing an object in
        # evidence_chunk_ids) raises TypeError in the corpus lookup, which
        # _dispatch reports as a "bad tool call".
        has_real_citation(self.corpus, ev)
        bad = unknown_chunk_ids(self.corpus, ev)
        if bad:
            return {"ok": False, "gate": "rejected", "error": (
                f"unknown chunk id(s) {bad!r} in evidence_chunk_ids — cite "
                "chunk ids exactly as returned by search/read results, or "
                "pass an empty list if you truly found no evidence.")}
        if (
            self.blank_verifier is not None
            and coerced is not None
            and not _skip_verifier
        ):
            # Edge-grounded commit check: proactive per-edge grounding + ONE
            # holistic verify, sharing the gsw_edge_grounded_cap re-attempt
            # budget (see _edge_grounded_verify).
            flagged, committed_partial, v, _early = self._edge_grounded_verify(
                blank_id, coerced, ev, reasoning=update_reasoning)
            if _early is not None:
                return _early
        if _verifier_status is not None:
            flagged = _verifier_status
        result_kwargs = {
            "blank_id": blank_id,
            "value": coerced,
            "status": "resolved" if coerced is not None else "unknown",
            "evidence_chunk_ids": ev,
            "llm_calls": 0,
            "wall_time_s": 0.0,
            "resolved_by": "react",
            "verifier_status": flagged,
            "completeness_status": ("partial" if committed_partial else None),
            "connection_evidence": self._connection_evidence_snapshot(blank_id),
        }
        if self.verifier_reasoning:
            result_kwargs["reasoning"] = update_reasoning
        self.state[blank_id] = BlankResult(**result_kwargs)
        if flagged:
            self._verifier_flagged[blank_id] = flagged
        auto_filled = _cascade_auto_compute(
            self.plan, self.state, block_unverified=self.block_unverified_cascade,
            blocked_sink=self._cascade_blocks, codegen=self._codegen,
        )
        out: dict[str, Any] = {"ok": True, "state": self._state_view()}
        if auto_filled:
            out["auto_computed"] = auto_filled
        return out

    def _tool_get_state(self) -> dict[str, Any]:
        return self._state_view()

    def _state_view(self) -> dict[str, Any]:
        """State view annotated with ``writable`` per blank."""
        blanks: list[dict[str, Any]] = []
        for b_id, br in self.state.items():
            blanks.append(
                {
                    "blank_id": b_id,
                    "status": br.status,
                    "value": br.value,
                    "evidence_chunk_ids": list(br.evidence_chunk_ids or []),
                    "writable": b_id in self.allowed,
                }
            )
        target_id = self.plan.target().id
        return {
            "blanks": blanks,
            "assigned_slice": sorted(self.allowed),
            "target_blank_id": target_id,
        }

    # --- dispatch + loop --------------------------------------------

    # q101 fix 2: cues in a list blank's verification_claim that mean an
    # empty resolution is itself wrong (e.g. "at least one winner"), so a
    # well-evidenced [] must still escalate rather than be accepted as
    # terminal. Deliberately narrow -- most list blanks have no such claim,
    # so the exception below applies by default.
    _NON_EMPTY_LIST_CLAIM_CUES = (
        "must not be empty",
        "cannot be empty",
        "at least one",
        "must include",
        "must contain",
        "non-empty",
        "nonempty",
    )

    def _empty_list_is_valid_terminal(
        self,
        ent: Optional[Any],
        evidence_chunk_ids: Optional[list],
    ) -> bool:
        """q101: a list-type blank the researcher resolved to `[]` with
        real, non-sentinel evidence and no explicit non-emptiness
        requirement in its verification_claim is a legitimate TERMINAL
        answer (e.g. "did the costar ALSO win?" resolves an empty
        who-also-won list) -- not a value to keep bouncing back until the
        turn budget dies. Narrow on purpose: requires a declared
        value_type=='list' blank, at least one real (non-'insufficient_
        evidence') cited chunk id, and no non-emptiness cue in the
        blank's own verification_claim."""
        if ent is None or getattr(ent, "value_type", None) != "list":
            return False
        ids = [
            str(e).strip() for e in (evidence_chunk_ids or [])
            if isinstance(e, str) and str(e).strip()
        ]
        real_ids = [e for e in ids if e.lower() != "insufficient_evidence"]
        if not real_ids:
            return False
        claim = str(getattr(ent, "verification_claim", "") or "").lower()
        if any(cue in claim for cue in self._NON_EMPTY_LIST_CLAIM_CUES):
            return False
        return True

    def _invalid_update_value_error(
        self,
        value: Any,
        *,
        ent: Optional[Any] = None,
        evidence_chunk_ids: Optional[list] = None,
    ) -> str:
        if value is None:
            return (
                "`value` cannot be null. Retry with an actual best-guess "
                "value, citing the chunk ids you read."
            )
        if isinstance(value, list) and not value:
            if self._empty_list_is_valid_terminal(ent, evidence_chunk_ids):
                return ""
            return (
                "`value` cannot be an empty list. Retry with a best-effort "
                "list, citing the chunk ids you read."
            )
        if not isinstance(value, str):
            return ""

        s = value.strip()
        low = s.lower()
        if not s:
            return (
                "`value` cannot be empty. Retry with an actual best-guess "
                "value, citing the chunk ids you read."
            )
        if low in {"unknown", "none", "null", "n/a", "na"}:
            return (
                f"`value={value!r}` is a placeholder, not an answer. Retry "
                "with an actual best-guess value, citing the chunk it "
                "came from."
            )
        if s in self.state:
            return (
                f"`value={value!r}` is a blank id, not an answer. Retry with "
                "the resolved value for that blank."
            )
        if s[:1] in {"{", "["} and s[-1:] in {"}", "]"}:
            try:
                json.loads(s)
            except json.JSONDecodeError:
                return ""
            return (
                "`value` looks like serialized JSON/tool arguments, not an "
                "answer. Retry with the extracted answer value only."
            )
        return ""

    def _tool_suggest_plan_revision(
        self,
        *,
        reason: str,
        hint: str,
    ) -> dict[str, Any]:
        """Researcher escalation: end the run, ask orchestrator to rethink the plan."""
        if not isinstance(reason, str) or not reason.strip():
            return {
                "ok": False,
                "error": (
                    "`reason` must be a non-empty string describing what "
                    "was missing in retrieval."
                ),
            }
        if not isinstance(hint, str) or not hint.strip():
            return {
                "ok": False,
                "error": (
                    "`hint` must be a non-empty string suggesting a new "
                    "intermediate step (e.g. 'add b_address blank')."
                ),
            }
        # Record the request — the loop checks this each turn and exits.
        # Attach the assigned slice so the orchestrator knows which blank
        # triggered the escalation.
        self._revision_request = {
            "blank_id": self.allowed_order[0] if self.allowed_order else None,
            "assigned_blank_ids": list(self.allowed_order),
            "reason": reason.strip(),
            "hint": hint.strip(),
        }
        return {"ok": True, "escalated": True}

    def _dispatch(self, name: str, args_json: str) -> dict[str, Any]:
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as exc:
            return {"error": f"bad tool args: {exc}"}
        try:
            if name == "search":
                result = self._tool_search(**args)
            elif name == "read":
                result = self._tool_ask(**args)
            elif name == "ask":
                result = self._tool_ask(**args)
            elif name == "find":
                result = self._tool_find(**args)
            elif name == "update_blank":
                result = self._tool_update_blank(**args)
            elif name == "bind_connection":
                result = self._tool_bind_connection(**args)
            elif name == "get_state":
                result = self._tool_get_state()
            elif name == "suggest_plan_revision":
                result = self._tool_suggest_plan_revision(**args)
            else:
                return {"error": f"unknown tool {name!r}"}
        except TypeError as exc:
            return {"error": f"bad tool call: {exc}"}
        # Phase-1 investigation memory: record attempt/binding events. Proposals
        # are recorded inside the _tool_update_blank wrapper (to also catch its
        # internal callers), so update_blank is skipped here. No-op when off.
        if self._invmem_episode is not None:
            self._invmem_record_dispatch(name, args, result)
        return result

    def _invmem_record_dispatch(self, name: str, args: dict, result: dict) -> None:
        """Fold a tool result into this sibling's episode (search/read/find ->
        attempt; bind_connection -> binding). Defensive: never raises into the
        tool loop. Phase-1 log-only."""
        ep = self._invmem_episode
        try:
            if name in ("search", "read", "find"):
                InvestigationMemory.add_attempt(
                    ep, name, _invmem_result_chunk_ids(name, args, result),
                    query=(args.get("query") if name == "search" else None))
            elif name == "bind_connection":
                InvestigationMemory.add_binding(
                    ep, args.get("blank_id"), args.get("vp_id"),
                    args.get("evidence_chunk_ids"), bool(result.get("ok")))
        except Exception:  # noqa: BLE001 — memory must never break the run
            pass

    def _unresolved_assigned(self) -> list[str]:
        return [
            b for b in self.allowed_order
            if self.state.get(b) is None
            or self.state[b].status != "resolved"
        ]

    def _all_assigned_resolved(self) -> bool:
        return not self._unresolved_assigned()

    def _budget_warning(self, unresolved: list[str], remaining: int) -> str:
        plural = "turns" if remaining != 1 else "turn"
        return (
            "FINAL BUDGET WARNING: "
            f"{remaining} tool-call {plural} remain for this researcher. "
            f"Unresolved assigned blanks: {unresolved}. Your next tool call "
            "must be `update_blank` for one unresolved assigned blank. Do "
            "not call `search`, `read`, `find`, or `get_state` now. If "
            "evidence is weak, commit a real best-guess value citing the "
            "chunk ids you read."
        )

    def _build_initial_messages(self) -> list[dict[str, Any]]:
        """The ReAct loop's starting message list: the system prompt and the
        question (with this dispatch's hints), plus the memory dossier of a
        re-dispatched blank (``prior_investigation``) when there is one."""
        user = self.question
        if self._prior_investigation:
            user = (
                user + "\n\n## Prior investigation\n"
                + self._prior_investigation
            )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

    def solve(self) -> dict[str, Any]:
        """Run the ReAct loop. Returns a trace dict for the orchestrator."""
        messages: list[dict[str, Any]] = self._build_initial_messages()
        tool_calls: list[ToolCall] = []
        turn_usage: list[dict[str, Any]] = []
        reasoning_chunks: list[str] = []
        prompt_tokens = completion_tokens = 0
        stopped = "max_turns"
        turn = 0
        loop_max_turns = self.max_turns

        for turn in range(1, loop_max_turns + 1):
            unresolved_before_turn = self._unresolved_assigned()
            if not unresolved_before_turn:
                stopped = "all_resolved"
                break
            remaining = self.max_turns - turn + 1
            if remaining <= len(unresolved_before_turn):
                messages.append(
                    {
                        "role": "user",
                        "content": self._budget_warning(
                            unresolved_before_turn, remaining
                        ),
                    }
                )
            try:
                resp = self.llm.chat(
                    messages,
                    tools=self._tools,
                    tool_choice="required",
                    max_tokens=self.max_completion_tokens,
                    seed=self.llm_seed,
                )
            except Exception as exc:  # noqa: BLE001
                stopped = "llm_error"
                reasoning_chunks.append(f"[turn {turn}] LLM error: {exc}")
                break

            prompt_tokens += resp.prompt_tokens
            completion_tokens += resp.completion_tokens
            turn_usage.append(
                {
                    "turn": turn,
                    "prompt_tokens": int(resp.prompt_tokens or 0),
                    "completion_tokens": int(resp.completion_tokens or 0),
                }
            )
            if resp.reasoning_content:
                reasoning_chunks.append(
                    f"[turn {turn}] {resp.reasoning_content}"
                )

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": resp.text or "",
            }
            if resp.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in resp.tool_calls
                ]
            messages.append(assistant_msg)

            if not resp.tool_calls:
                stopped = "no_tool_call"
                break

            for tc in resp.tool_calls:
                t0 = time.time()
                result = self._dispatch(tc["name"], tc["arguments"])
                result_json = json.dumps(result, default=str)
                try:
                    args_parsed = (
                        json.loads(tc["arguments"]) if tc.get("arguments") else {}
                    )
                except json.JSONDecodeError:
                    args_parsed = {"_raw": tc.get("arguments", "")}
                tool_calls.append(
                    ToolCall(
                        turn=turn,
                        name=tc["name"],
                        args=args_parsed,
                        result_preview=result_json[:500],
                        result_full=result_json,
                        duration_s=round(time.time() - t0, 3),
                        error=(
                            result.get("error", "")
                            if isinstance(result, dict)
                            else ""
                        ),
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_json,
                    }
                )
                if self._archive_needed() and tc["name"] in ("search", "read"):
                    self._tool_archive.store(
                        tc["id"], tc["name"], turn, args_parsed, result
                    )
            if self.context_compaction_on:
                compact_messages(messages, self._tool_archive)
            # Researcher escalation overrides the resolved-check —
            # the loop ends regardless of whether assigned blanks are
            # filled, because the researcher decided the plan is wrong.
            if self._revision_request is not None:
                stopped = "plan_revision_requested"
                break
            if self._all_assigned_resolved():
                stopped = "all_resolved"
                break

        resolved_now = [
            b for b in self.allowed
            if self.state.get(b) is not None
            and self.state[b].status == "resolved"
        ]
        trace = {
            "allowed_blank_ids": list(self.allowed_order),
            "turns": turn,
            "stopped": stopped,
            "resolved": resolved_now,
            "unresolved": [b for b in self.allowed if b not in set(resolved_now)],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "turn_usage": turn_usage,
            "tool_calls": tool_calls,
            "messages": messages,
            "reasoning": "\n".join(reasoning_chunks),
            "revision_request": self._revision_request,
        }
        readiness = _researcher_readiness_telemetry(
            tool_calls=tool_calls,
            turn_usage=turn_usage,
            state=self.state,
            allowed_blank_ids=list(self.allowed_order),
        )
        if readiness:
            trace["readiness"] = readiness
        metadata = self._trace_metadata()
        if metadata:
            trace["metadata"] = metadata
        return trace

    def _feature_flags(self) -> dict[str, bool]:
        flags = {
            "gsw_context_compaction": bool(self.context_compaction_on),
        }
        return flags

    def _trace_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {"feature_flags": self._feature_flags()}
        if self.context_compaction_on and self._tool_archive is not None:
            # Every researcher session has its own fresh archive, so its
            # metrics are this session's own counts (floored at 0, as when
            # resumed sessions shared an archive and reported a delta).
            current = self._tool_archive.metrics()
            metadata["compaction_metrics"] = {
                key: max(value, 0) for key, value in current.items()
            }
        metadata["agentic_read_telemetry"] = self.agentic_read_telemetry()
        return metadata

    def agentic_read_telemetry(self) -> dict[str, Any]:
        """Per-call record of every agentic ``read`` this researcher made.

        This is what settles the design's central uncertainty (spec §6): log
        every ``not_found`` together with the ``sections_checked`` it was
        scoped to, so an offline pass can check whether the gold answer was
        on that page anyway. A missing record would silently under-count in
        the safe-looking direction, so every return path in the agentic
        branch of ``_tool_read`` — cache hit, fresh answer/not_found, error —
        appends here before returning.

        The token counters exist because spec §3's cost trade — one extra LLM
        call per (page, question), against 12k windows no longer entering the
        parent's context — is not computable from a call count alone. A cache hit
        contributes 0 to both, which is exactly the saving the ledger claims.
        """
        log = self._agentic_read_log
        return {
            "calls": len(log),
            "cached_hits": sum(1 for r in log if r["outcome"] == "cached"),
            "answers": sum(1 for r in log if r["outcome"] == "answer"),
            "not_found": sum(1 for r in log if r["outcome"] == "not_found"),
            "errors": sum(1 for r in log if r["outcome"] == "error"),
            "prompt_tokens": sum(int(r.get("prompt_tokens", 0) or 0) for r in log),
            "completion_tokens": sum(
                int(r.get("completion_tokens", 0) or 0) for r in log
            ),
            "records": log,
        }


# ---------------------------------------------------------------------------
# Orchestrator adapter
# ---------------------------------------------------------------------------


@register_adapter
class OursGSWPlannerOrchestratorV1Adapter(_PlannerFallbackMixin, Adapter):
    """Deterministic orchestrator + per-level ReActResearcher."""

    system_id: ClassVar[str] = "ours_gsw_planner_orchestrator_v1"
    display_name: ClassVar[str] = "GSW planner + orchestrated per-level researchers"
    description: ClassVar[str] = (
        "Calls the planner, builds the topological level map, and spawns "
        "one ReActResearcher per level (sliced update_blank, no finish "
        "tool). Orchestrator owns control flow; researchers own retrieval. "
        "Falls back to flat ours_gsw_v1 on planner parse / topology failure."
    )

    def __init__(self, ctx: AdapterContext) -> None:
        super().__init__(ctx)
        self.corpus = require_corpus(ctx.extra)
        retriever_override = ctx.extra.get("retriever")
        if retriever_override is not None:
            self.retriever = retriever_override
        else:
            retriever_type = ctx.extra.get("retriever_type", "bm25")
            dense_cache = ctx.extra.get("dense_cache")
            self.retriever = build_retriever(
                retriever_type,
                self.corpus,
                cache_dir=Path(dense_cache) if dense_cache else None,
                lucene_index=ctx.extra.get("lucene_index") or None,
            )
        reranker_spec = ctx.extra.get("reranker")
        if reranker_spec:
            from panini_dr.retrieval.rerank import maybe_wrap_reranker

            self.retriever = maybe_wrap_reranker(self.retriever, reranker_spec)
        self.llm = LLMClient(
            model=ctx.model_name or ctx.model_id,
            base_url=ctx.base_url or None,
            api_key=ctx.api_key or None,
            default_temperature=float(ctx.extra.get("llm_temperature", 0.0)),
            reasoning_effort=str(ctx.extra.get("reasoning_effort", "medium")),
        )
        # Optional deterministic seed for OpenAI/vLLM endpoints. None = no seed.
        self.llm_seed: Optional[int] = ctx.extra.get("llm_seed")
        # Off by default: the validator is useful as instrumentation, but
        # reject-mode adds another stochastic LLM call to the control loop.
        # Valid values: off, log_only, reject, repair.
        self.synthesis_validator_mode = str(
            ctx.extra.get("synthesis_validator_mode", "off")
        ).strip().lower()
        if self.synthesis_validator_mode not in {
            "off", "log_only", "reject", "repair",
        }:
            self.synthesis_validator_mode = "off"
        raw_guided_repair = ctx.extra.get("synthesis_validator_guided_repair", True)
        if isinstance(raw_guided_repair, str):
            self.synthesis_validator_guided_repair = (
                raw_guided_repair.strip().lower() not in {"0", "false", "off", "no"}
            )
        else:
            self.synthesis_validator_guided_repair = bool(raw_guided_repair)
        self._fallback_adapter: Optional[Adapter] = None
        # Per-researcher turn budget (default 20).
        self.level_max_turns = ctx.extra.get("level_max_turns", 20)
        self.gsw_evidence_gate = ctx.extra.get("gsw_evidence_gate", "off")
        self.gsw_hard_type_filter = bool(ctx.extra.get("gsw_hard_type_filter", False))
        # gsw_codegen_compute: route the compute step (op=="diff" over dates)
        # through the seeded skill library (library hits only). Default OFF ->
        # byte-identical.
        self.gsw_codegen_compute = bool(ctx.extra.get("gsw_codegen_compute", False))
        self._codegen_library = None  # lazily built when the flag is on
        self._codegen = None  # per-question CodegenRuntime (set in run_question)
        # gsw_investigation_memory (Phase 1, LOG-ONLY): live event-fed memory that
        # dumps to traj.extra["invmem"]. NO enforcement / NO prompt change; frozen
        # Phase-0 rules are diagnostic only. Default OFF -> _invmem stays None ->
        # every hook is skipped -> byte-identical. Both reset per-question.
        self.gsw_investigation_memory = bool(
            ctx.extra.get("gsw_investigation_memory", False)
        )
        # Phase-2 ENFORCEMENT (score/cost-affecting): dispatch filter + finalizer.
        # Requires the memory to be populated; off -> byte-identical. All enforcement
        # sites gate on ``self._invmem_enforce`` (both flags on + memory present).
        self.gsw_investigation_memory_enforce = bool(
            ctx.extra.get("gsw_investigation_memory_enforce", False)
        ) and self.gsw_investigation_memory
        # Final-shot QA (rev3): one whole-question completion at the empty exit.
        # Requires enforce (which requires memory). Default off.
        self.gsw_investigation_memory_finalshot = bool(
            ctx.extra.get("gsw_investigation_memory_finalshot", False)
        ) and self.gsw_investigation_memory_enforce
        self.gsw_memory_dossier = bool(
            ctx.extra.get("gsw_memory_dossier", False)
        ) and self.gsw_investigation_memory
        self.gsw_memory_scoped_dispatch = bool(
            ctx.extra.get("gsw_memory_scoped_dispatch", False)
        ) and self.gsw_investigation_memory
        # Per-question, and thread-local: the harness runs N run_question calls
        # CONCURRENTLY on this one shared adapter, so a plain instance attribute
        # would cross-contaminate questions. Each question runs on its own thread
        # (the sibling sub-pool never touches self._invmem — it uses the explicitly
        # passed episode), so a threading.local backing store isolates it cleanly.
        # _invmem / _invmem_round below are properties over this store.
        self._invmem_tls = threading.local()
        # Same isolation discipline for the per-question dossier / scoped-budget
        # logs (properties below): appended on the question's own (main) thread
        # inside _dispatch_subplan_parallel, never by pool threads.
        self._memory_log_tls = threading.local()
        self._planner_tool_calls = 0
        self._planner_tool_log = []
        self._orchestrator_turn_log = []
        self._planning_messages = []
        self._pending_call = None
        self._dossier_log: list[dict[str, Any]] = []
        self._scoped_log: list[dict[str, Any]] = []
        self.gsw_read_raw_fallback = bool(ctx.extra.get("gsw_read_raw_fallback", True))
        self.gsw_evidence_grounder = ctx.extra.get(
            "gsw_evidence_grounder", "substring"
        )
        self.gsw_evidence_state = (
            bool(ctx.extra.get("gsw_evidence_state", False))
            or self.gsw_evidence_gate != "off"
        )
        self.gsw_typed_blank_resolver = bool(
            ctx.extra.get("gsw_typed_blank_resolver", False)
        )
        self.gsw_react_fallback = bool(ctx.extra.get("gsw_react_fallback", True))
        self.gsw_max_docs = int(ctx.extra.get("gsw_max_docs", 5))
        self.block_unverified_cascade = bool(
            ctx.extra.get("gsw_block_unverified_cascade", False)
        )
        self.gsw_blank_qualifier = bool(ctx.extra.get("gsw_blank_qualifier", False))
        self.gsw_blank_hints = bool(ctx.extra.get("gsw_blank_hints", False))
        self.gsw_context_compaction = bool(
            ctx.extra.get("gsw_context_compaction", False)
        )
        # Perf-safe lever: trim re-sent search snippets (trim-only; never drops
        # a question).
        # The plan-verifier repair loop: after a plan is emitted, run the
        # deterministic contract gate + LLM repairer until no binding clause
        # is violated (or a clause demotes after its attempt cap). Default
        # off -> _tool_call_planner returns the raw emitted plan exactly as
        # it does today.
        self.gsw_plan_verifier = bool(ctx.extra.get("gsw_plan_verifier", False))
        self.gsw_verifier_scoped = bool(ctx.extra.get("gsw_verifier_scoped", False))
        self.gsw_verifier_reasoning = bool(
            ctx.extra.get("gsw_verifier_reasoning", False)
        )
        self.gsw_plan_update_guard = str(ctx.extra.get("gsw_plan_update_guard", "off"))
        if self.gsw_plan_update_guard not in {"off", "log_only", "enforce"}:
            self.gsw_plan_update_guard = "off"
        # Plan-request contract: hold a request_plan_update call to a contract
        # authored from the researcher's escalation. Default off -> update_plan
        # ignores plan_request/contract_enabled exactly as it does today (see
        # _plan_updater.update_plan: `contract_active = contract_enabled and
        # plan_request is not None`).
        self.gsw_plan_request_contract = bool(
            ctx.extra.get("gsw_plan_request_contract", False)
        )
        # Precomputed once (the flag is fixed for the adapter's lifetime): the
        # exact tool list `_run_llm_orchestrator` sends the LLM every turn (see
        # `_orchestrator_tools`'s docstring).
        self._orchestrator_tools = _orchestrator_tools(
            plan_request_contract=self.gsw_plan_request_contract,
        )
        self.gsw_plan_quality_checks = bool(ctx.extra.get("gsw_plan_quality_checks", False))
        self.gsw_plan_issue_surface = str(ctx.extra.get("gsw_plan_issue_surface", "off"))
        if self.gsw_plan_issue_surface not in {"off", "advisory"}:
            self.gsw_plan_issue_surface = "off"
        if self.gsw_plan_issue_surface == "advisory" and not self.gsw_plan_quality_checks:
            self.gsw_plan_quality_checks = True
        # Per-question plan-update cap (instance override of the ClassVar default).
        try:
            _cap = int(ctx.extra.get("gsw_max_plan_updates", self.MAX_PLAN_UPDATES_PER_RUN))
            if _cap >= 1:
                self.MAX_PLAN_UPDATES_PER_RUN = _cap
        except (TypeError, ValueError):
            pass
        if self.gsw_blank_hints and not self.gsw_typed_blank_resolver:
            logger.warning(
                "gsw_blank_hints=True but gsw_typed_blank_resolver=False: "
                "hints affect planning only, not retrieval. Enable "
                "--gsw-typed-blank-resolver for the intended A/B."
            )
        # Shared budget for the two edge-grounded gates. Defaults to 1, the historical
        # hard-coded value, so an unset key reproduces the run of record exactly.
        self.gsw_edge_grounded_cap = int(ctx.extra.get("gsw_edge_grounded_cap", 1))
        self._checker_sink_tls = threading.local()
        self._read_ledger_tls = threading.local()
        # Per-question per-role token ledger (see the `_role_usage` property).
        # Thread-local for the same reason as the two above: eval/harness.py
        # fans `run_question` over a ThreadPoolExecutor against ONE adapter.
        self._role_usage_tls = threading.local()
        # Phase 4 Task 7: per-question cascade-block sink. Initialized here
        # so mode methods called directly (in tests) don't AttributeError;
        # re-reset per question at the top of `run_question`.
        self._cascade_blocks: list[dict[str, Any]] = []
        # Plan-request contract: registry of researcher escalations for this
        # question (request_id -> PlanRequest, never rewritten after
        # registration) + the ids an update actually cited. THREAD-LOCAL (see
        # the `_plan_requests`/`_consumed_request_ids` properties below,
        # mirroring `_active_question_id`) -- eval/harness.py fans
        # `run_question` over a ThreadPoolExecutor against ONE adapter
        # instance, so per-question state shared on `self` corrupts whichever
        # other question is concurrently in flight. Assigning here goes
        # through the property setter and only touches THIS thread's copy;
        # initialized so direct calls (tests) don't AttributeError; re-reset
        # per question in `run_question` (a request_id is namespaced by
        # question_id, but the `seq` counter and the summary must not carry a
        # prior question's escalations into this one -- see `run_question`
        # for why the reset is still needed even though the property
        # lazy-inits on first access: a pooled thread runs many questions in
        # sequence, not just one).
        self._plan_requests: dict[str, Any] = {}
        self._question_contract = None
        self._consumed_request_ids: set[str] = set()

    def _slice_connection_manifests(
        self,
        plan: GSWPlan,
        slice_blank_ids: list[str],
        state: dict[str, BlankResult],
    ) -> dict[str, dict]:
        """Per-slice connection manifests for the researcher prompt.

        ``resolved_values`` is the resolved blanks' values from the current
        state. It is close to the verify-site ``dep_values`` derivation but
        intentionally omits the ``b != blank_id`` self-exclusion used there:
        the manifest builder only reads partner ids, so the per-blank entry is
        never consulted and the difference is irrelevant here.
        """
        resolved_values = {
            b: br.value
            for b, br in state.items()
            if br is not None and br.status == "resolved" and br.value is not None
        }
        return {
            bid: build_connection_manifest(plan, bid, resolved_values)
            for bid in slice_blank_ids
        }

    def _gsw_trace_feature_flags(self) -> dict[str, bool]:
        # Every prompt/context mechanism that changes what the researcher is
        # sent belongs here. The working memory and the prompt trim were both
        # absent until 2026-08-16, so a paired arm could not be told apart from
        # its control by reading the traces -- the flags had to be recovered
        # from the launch command, which is exactly how a mislabelled arm
        # survives to publication.
        flags = {
            "gsw_context_compaction": bool(self.gsw_context_compaction),
        }
        return flags

    def _attach_gsw_trace_metadata(
        self, traj: Trajectory, researcher_traces: list[dict[str, Any]]
    ) -> None:
        # Each dispatch spawns a fresh ReActResearcher (~2.5 per
        # question), so this MUST sum counters and extend records across
        # every trace rather than overwrite — a later dispatch clobbering
        # an earlier one would silently lose most of the run's telemetry.
        agentic_totals = {
            "calls": 0, "cached_hits": 0, "answers": 0,
            "not_found": 0, "errors": 0,
            # Token cost of the reader itself — without it the run cannot be
            # analysed against spec §3's cost trade at all.
            "prompt_tokens": 0, "completion_tokens": 0,
        }
        agentic_records: list[dict[str, Any]] = []
        for trace in researcher_traces:
            tel = trace.get("metadata", {}).get("agentic_read_telemetry") or {}
            for key in agentic_totals:
                agentic_totals[key] += int(tel.get(key, 0) or 0)
            agentic_records.extend(tel.get("records", []) or [])
        traj.extra["agentic_read"] = {**agentic_totals, "records": agentic_records}

        if not self.gsw_context_compaction:
            return
        traj.extra["gsw_feature_flags"] = self._gsw_trace_feature_flags()

        if self.gsw_context_compaction:
            totals = {
                "search_messages_compacted": 0,
                "read_messages_compacted": 0,
                "placeholder_chars": 0,
                "raw_chars_archived": 0,
                "estimated_prompt_chars_saved": 0,
            }
            for trace in researcher_traces:
                metrics = trace.get("metadata", {}).get("compaction_metrics", {})
                for key in totals:
                    totals[key] += int(metrics.get(key, 0) or 0)
            traj.extra["gsw_compaction_metrics"] = totals

    @property
    def _checker_sink(self) -> list[dict[str, Any]]:
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        sink = getattr(tls, "sink", None)
        if sink is None:
            sink = []
            tls.sink = sink
        return sink

    @_checker_sink.setter
    def _checker_sink(self, sink: list[dict[str, Any]]) -> None:
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        tls.sink = sink

    def _read_ledger(self) -> dict[tuple[str, str], Any]:
        """Per-question read cache, ``(article title, normalised question) ->
        ReadResult``. THREAD-LOCAL so concurrent questions cannot see each
        other's answers; self-heals when __init__ was bypassed (tests construct
        adapters via __new__).

        Callers that hand this to a researcher pool MUST evaluate it on the
        question thread and close over the result — calling it from a pool worker
        mints a fresh empty dict per worker and silently reduces the ledger to
        per-researcher scope (see ``_dispatch_subplan_parallel``)."""
        tls = getattr(self, "_read_ledger_tls", None)
        if tls is None:
            tls = threading.local()
            self._read_ledger_tls = tls
        ledger = getattr(tls, "ledger", None)
        if ledger is None:
            ledger = {}
            tls.ledger = ledger
        return ledger

    @property
    def _role_usage(self) -> "_RoleUsage":
        """This question's per-role token ledger, THREAD-LOCAL.

        Reset at the top of every ``run_question``; self-heals when __init__ was
        bypassed (tests construct adapters via __new__). Logging only -- nothing
        reads it back inside the run.
        """
        tls = getattr(self, "_role_usage_tls", None)
        if tls is None:
            tls = threading.local()
            self._role_usage_tls = tls
        usage = getattr(tls, "usage", None)
        if usage is None:
            usage = _RoleUsage()
            tls.usage = usage
        return usage

    @_role_usage.setter
    def _role_usage(self, usage: "_RoleUsage") -> None:
        tls = getattr(self, "_role_usage_tls", None)
        if tls is None:
            tls = threading.local()
            self._role_usage_tls = tls
        tls.usage = usage

    def _metered(self, client, role: str):
        """``client``, wrapped so its calls are counted under ``role``.

        Instrumentation only: ``_MeteredLLM`` forwards the call and returns the
        provider's response object, so the model, the prompt, the arguments and
        the control flow are exactly what they were. ``None`` passes through
        unchanged -- an absent client must stay absent, because callers read its
        absence as configuration (`reader_llm=None` means "follow the
        researcher's client").

        MUST BE CALLED ON THE QUESTION THREAD. The ledger is resolved here, once,
        from the thread-local; calling this inside a researcher pool worker binds
        that worker's empty ledger instead of the question's (the hazard
        ``_read_ledger`` and ``_checker_sink`` already carry). Hoist before
        ``pool.map`` and close over the result.
        """
        if client is None:
            return None
        return _MeteredLLM(client, role, self._role_usage)

    def _metered_reader_llm(self):
        """The researcher's client, metered as ``reader``: the reader runs
        inside ``ReActResearcher`` on the researcher's client, and only its
        tokens move from the researcher's row to the reader's, which is the
        split the protocol's cost table needs."""
        return self._metered(self.llm, "reader")

    @property
    def _orchestrator_turn_log(self) -> list:
        """Per-turn orchestrator record, THREAD-LOCAL. `traj.reasoning`
        concatenates the planner's, the orchestrator's and every researcher's
        chunks into one string; separating the actors is the point of this."""
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        log = getattr(tls, "orch_turn_log", None)
        if log is None:
            log = []
            tls.orch_turn_log = log
        return log

    @_orchestrator_turn_log.setter
    def _orchestrator_turn_log(self, value: list) -> None:
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        tls.orch_turn_log = value

    @property
    def _planner_tool_log(self) -> list:
        """Per-question planner-tool record, THREAD-LOCAL — same reason as
        _checker_sink. Measured live at concurrency 16: q111 made one call,
        its log held 28 entries and its counter read 3, because other
        questions were spending its budget."""
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        log = getattr(tls, "planner_tool_log", None)
        if log is None:
            log = []
            tls.planner_tool_log = log
        return log

    @_planner_tool_log.setter
    def _planner_tool_log(self, value: list) -> None:
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        tls.planner_tool_log = value

    @property
    def _planner_tool_calls(self) -> int:
        """The CAP counter. A shared counter means one question's calls exhaust
        another's budget — which is how a 200-question sweep produced 27 spurious
        "cap reached" errors on a question that called the planner once."""
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        return int(getattr(tls, "planner_tool_calls", 0) or 0)

    @_planner_tool_calls.setter
    def _planner_tool_calls(self, value: int) -> None:
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        tls.planner_tool_calls = int(value)

    # The planning phase's hand-off to the orchestrator loop: the conversation
    # `_plan_via_tool` held, and the call that ended it. THREAD-LOCAL, like the
    # planner-tool record above: they were plain instance attributes, and the
    # harness runs every question concurrently on this ONE adapter, so a
    # question's loop could read the conversation and pending call ANOTHER
    # question's planning phase had just stored (and that question then found
    # its own hand-off emptied). With real model latency the window between
    # storing and reading the hand-off is milliseconds against seconds-long
    # calls, and no paper trace shows it (2,289 questions checked); with
    # zero-latency replay it fires.

    @property
    def _planning_messages(self) -> list:
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        msgs = getattr(tls, "planning_messages", None)
        if msgs is None:
            msgs = []
            tls.planning_messages = msgs
        return msgs

    @_planning_messages.setter
    def _planning_messages(self, value: list) -> None:
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        tls.planning_messages = value

    @property
    def _pending_call(self) -> Optional[dict]:
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        return getattr(tls, "pending_call", None)

    @_pending_call.setter
    def _pending_call(self, value: Optional[dict]) -> None:
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        tls.pending_call = value

    @property
    def _resolver_attempts(self) -> list[dict[str, Any]]:
        tls = self._checker_sink_tls
        lst = getattr(tls, "attempts", None)
        if lst is None:
            lst = []
            tls.attempts = lst
        return lst

    @property
    def _verifier_flags(self) -> dict[str, str]:
        tls = self._checker_sink_tls
        d = getattr(tls, "verifier_flags", None)
        if d is None:
            d = {}
            tls.verifier_flags = d
        return d

    @property
    def _verifier_reject_counts(self) -> dict[str, int]:
        """Per-blank verifier-reject tally for THIS question (thread-local).
        Aggregated from each researcher's ``_verifier_rejects`` at the merge
        sites. Read post-loop by M3 to detect the over-decomposition class
        (an intermediate blank verifier-rejected >=2x with target unresolved)."""
        tls = self._checker_sink_tls
        d = getattr(tls, "verifier_reject_counts", None)
        if d is None:
            d = {}
            tls.verifier_reject_counts = d
        return d

    @property
    def _active_question_id(self) -> Optional[str]:
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            return None
        return getattr(tls, "question_id", None)

    @_active_question_id.setter
    def _active_question_id(self, question_id: Optional[str]) -> None:
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        tls.question_id = question_id

    @property
    def _question_contract(self) -> Optional[list[dict]]:
        """The contract authored for THIS thread's in-flight question, if any.

        Thread-local for the same reason as `_plan_requests` below: the harness
        fans `run_question` over a ThreadPoolExecutor against ONE adapter, so a
        plain attribute would let one question's contract govern another's
        updates. Carried forward so every update on a question is held to the
        contract its escalation established -- see the call site in
        `request_plan_update`.
        """
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        return getattr(tls, "question_contract", None)

    @_question_contract.setter
    def _question_contract(self, value: Optional[list[dict]]) -> None:
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        tls.question_contract = value

    @property
    def _plan_requests(self) -> dict[str, Any]:
        """Registry of researcher escalations for THIS thread's in-flight
        question (request_id -> PlanRequest). Thread-local -- mirrors
        `_active_question_id` above -- because `eval/harness.py` fans
        `run_question` over a ThreadPoolExecutor against one adapter
        instance; a plain instance dict would let one thread's per-question
        reset (see `run_question`) wipe another thread's in-flight registry
        mid-run."""
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        d = getattr(tls, "plan_requests", None)
        if d is None:
            d = {}
            tls.plan_requests = d
        return d

    @_plan_requests.setter
    def _plan_requests(self, value: dict[str, Any]) -> None:
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        tls.plan_requests = value

    @property
    def _consumed_request_ids(self) -> set[str]:
        """Ids from `_plan_requests` an update actually cited, THIS thread's
        in-flight question only. Thread-local for the same reason as
        `_plan_requests` above."""
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        s = getattr(tls, "consumed_request_ids", None)
        if s is None:
            s = set()
            tls.consumed_request_ids = s
        return s

    @_consumed_request_ids.setter
    def _consumed_request_ids(self, value: set[str]) -> None:
        tls = getattr(self, "_checker_sink_tls", None)
        if tls is None:
            tls = threading.local()
            self._checker_sink_tls = tls
        tls.consumed_request_ids = value

    def _make_codegen_runtime(self, question: str):
        """Build the per-question CodegenRuntime, or ``None`` when the flag is off.

        The library is created once and shared across questions in a run.
        Returns ``None`` off-path so the threaded ``codegen`` argument is
        exactly the pre-flag default.
        """
        if not self.gsw_codegen_compute:
            return None
        from .codegen_library import registry
        from .codegen_library.runtime import CodegenRuntime

        if self._codegen_library is None:
            self._codegen_library = registry.Library.load()
        return CodegenRuntime(
            enabled=True,
            lib=self._codegen_library,
            question=question,
        )

    def _plan_via_tool(self, question: str, traj: Trajectory):
        """Bounded planning phase: the orchestrator asks for the plan itself.

        Its only tool here is `call_planner`, so nothing can dispatch or submit
        before a plan exists — the gate is the tool list, not a prompt rule. The
        conversation it holds during this phase is kept, so in live mode the main
        loop continues the same thread and its first dispatch is informed by the
        plan it chose and the audit it read.

        Returns ``(plan, emit_meta)`` exactly like `emit_plan`, so §2-§4 of
        run_question are unchanged. Raises ``PlanEmitError`` if the phase ends
        with no usable plan, which routes into the existing fallback.
        """
        boot = build_planner_bootstrap_prompt()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": boot},
            {"role": "user", "content": question},
        ]
        last_err = "planner tool produced no plan"
        best = None
        for _ in range(self.MAX_PLANNER_TOOL_CALLS + 1):
            try:
                resp = self._metered(self.llm, "orchestrator").chat(
                    messages,
                    tools=[_CALL_PLANNER_TOOL],
                    tool_choice="required",
                    max_tokens=self.ctx.max_completion_tokens, seed=self.llm_seed,
                )
            except Exception as exc:  # noqa: BLE001
                raise PlanEmitError(kind="llm_error", detail=str(exc)[:200]) from exc
            self._orchestrator_turn_log.append({
                "turn": 0, "phase": "planning",
                "reasoning": (getattr(resp, "reasoning_content", "") or "")[:4000],
                "text": (getattr(resp, "text", "") or "")[:1000],
                "calls": [{"name": tc.get("name"),
                           "arguments": str(tc.get("arguments"))[:2000]}
                          for tc in (resp.tool_calls or [])],
                "prompt_tokens": int(getattr(resp, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(resp, "completion_tokens", 0) or 0),
            })
            assistant = {"role": "assistant", "content": resp.text or ""}
            if resp.tool_calls:
                assistant["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in resp.tool_calls
                ]
            messages.append(assistant)
            if not resp.tool_calls:
                messages.append({"role": "user", "content": PROSE_NUDGE})
                continue
            for tc in resp.tool_calls:
                if tc.get("name") != "call_planner" and best is not None:
                    # Reaching for another tool once a plan exists IS the accept
                    # signal — with tool_choice="required" there is no other way
                    # to say "this plan will do". The call itself is a decision
                    # the orchestrator has already made, so the main loop
                    # EXECUTES it rather than asking for it again: dropping it
                    # lost 39 audit-driven plan-update requests and 28
                    # dispatches across 67 questions, and neither came back.
                    self._pending_call = dict(tc)
                    # Drop the trailing assistant message: its tool_call has no
                    # response yet, and a dangling call reads to the model as
                    # "that already happened".
                    if messages and messages[-1].get("role") == "assistant":
                        messages.pop()
                    self._planning_messages = messages
                    return best
                if tc.get("name") != "call_planner":
                    # No plan yet: refuse. This is what keeps "nothing dispatches
                    # before a plan exists" mechanical rather than advisory.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({
                            "ok": False,
                            "error": (f"{tc.get('name')!r} is unavailable until "
                                      "a plan exists — call call_planner."),
                        }),
                    })
                    continue
                result, action = self._tool_call_planner(question, {}, {})
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": json.dumps(result, default=str)})
                plan = action.get("new_plan")
                if plan is None:
                    last_err = str(result.get("error", last_err))[:200]
                    continue
                # Keep going: the orchestrator has just been handed the audit,
                # and asking for a better plan is exactly what it is for. The
                # loop exits when it reaches for another tool, or at the cap.
                self._planning_messages = messages
                meta = PlanEmitMeta(
                    prompt_tokens=int(getattr(resp, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(resp, "completion_tokens", 0) or 0),
                    raw_response=(self._planner_tool_log[-1].get("planner_raw") or ""),
                    raw_reasoning=(self._planner_tool_log[-1].get("planner_reasoning") or ""),
                    extras={"plan_audit_tags":
                            list(((self._planner_tool_log[-1].get("audit") or {})
                                  .get("tags")) or [])},
                )
                best = (plan, meta)
        if best is not None:
            return best
        raise PlanEmitError(kind="exhausted", detail=last_err)

    def _emit_plan_with_optional_planner_route(
        self,
        question: str,
        emit_kwargs: dict[str, Any],
    ) -> tuple[GSWPlan, PlanEmitMeta]:
        """Emit a plan (``emit_plan``), then run the plan-verifier repair loop
        over it when ``gsw_plan_verifier`` is on. Called by
        ``_tool_call_planner``. (The planner-only model route the name refers
        to was removed; the plan is emitted on the main client.)"""
        plan, meta = emit_plan(question, self._metered(self.llm, "planner"), **emit_kwargs)
        if self.gsw_plan_verifier:
            plan = self._run_plan_repair_loop(plan, question, meta)
        return plan, meta

    def _run_plan_repair_loop(
        self, plan: GSWPlan, question: str, meta: PlanEmitMeta
    ) -> GSWPlan:
        """Run the plan-verifier repair loop over a freshly emitted plan.

        Only called when `self.gsw_plan_verifier` is on -- callers must
        gate this themselves so the flag-off default never imports,
        constructs, or calls anything here. Records the transcript on
        `meta.extras["plan_repair"]` so both callers of
        `_emit_plan_with_optional_planner_route` can surface it
        (`_tool_call_planner` onto its `rec`, `run_question` onto
        `traj.extra`) without duplicating this method.

        Fail-open: `verify_and_repair`'s `raw = propose(...)` call has no
        guard of its own, and `propose()` only catches
        JSONDecodeError/TypeError AFTER `llm.chat()` returns -- an
        exception raised BY `llm.chat()` itself (rate limit, timeout,
        auth, or the Bedrock corrupted-tool-use-streaming failure
        `_plan_repair_prompt.py` documents) propagates straight through.
        Both callers' own try/except only catch PlanEmitError, so an
        unguarded call here would turn a transient repairer hiccup into a
        hard empty-answer miss on a question the ORIGINAL plan would have
        answered -- the exact opposite of what a verifier is for. On
        failure `plan` is left as the originally emitted, already-
        validated plan (the assignment below never runs), and the failure
        is recorded on `meta.extras["plan_repair"]` so it is visible in
        the transcript instead of silent.
        """
        from panini_dr.adapters.ours._plan_repair_loop import verify_and_repair
        from panini_dr.adapters.ours._plan_repair_prompt import make_proposer

        try:
            plan, residual, transcript = verify_and_repair(
                plan, question,
                make_proposer(self._metered(self.llm, "auditor"), self.ctx.max_completion_tokens,
                              self.llm_seed),
            )
            meta.extras["plan_repair"] = {
                "accepted": transcript.ops_accepted,
                "rejected": transcript.ops_rejected,
                "demoted": transcript.demoted,
                "before": transcript.violations_before,
                "after": transcript.violations_after,
            }
        except Exception as exc:  # noqa: BLE001 — fail-open, see above
            meta.extras["plan_repair"] = {"error": f"repair loop failed: {str(exc)[:200]}"}
        return plan

    # Phase-1 investigation memory: thread-local per-question state (see __init__).
    # All accesses are on the question's own thread, so this isolates concurrent
    # run_question calls without threading a param through the orchestration loop.
    @property
    def _invmem(self):
        return getattr(self._invmem_tls, "mem", None)

    @_invmem.setter
    def _invmem(self, value):
        self._invmem_tls.mem = value

    @property
    def _invmem_round(self) -> int:
        return getattr(self._invmem_tls, "round", 0)

    @_invmem_round.setter
    def _invmem_round(self, value: int) -> None:
        self._invmem_tls.round = value

    @property
    def _dossier_log(self) -> list[dict[str, Any]]:
        log = getattr(self._memory_log_tls, "dossier_log", None)
        if log is None:
            log = []
            self._memory_log_tls.dossier_log = log
        return log

    @_dossier_log.setter
    def _dossier_log(self, value: list[dict[str, Any]]) -> None:
        self._memory_log_tls.dossier_log = value

    @property
    def _scoped_log(self) -> list[dict[str, Any]]:
        log = getattr(self._memory_log_tls, "scoped_log", None)
        if log is None:
            log = []
            self._memory_log_tls.scoped_log = log
        return log

    @_scoped_log.setter
    def _scoped_log(self, value: list[dict[str, Any]]) -> None:
        self._memory_log_tls.scoped_log = value

    @property
    def _invmem_enforce(self) -> bool:
        """Phase-2 gate: enforcement flag on AND this question's memory exists."""
        return self.gsw_investigation_memory_enforce and self._invmem is not None

    def _invmem_finalize_answer(self, final_answer: str) -> str:
        """Phase-2 finalizer (G2/G3): on an otherwise-empty exit, if the memory
        holds a real-evidence candidate for the target blank, stringify it through
        the normal answer path and return it (recording the actor on the memory so
        the dump carries invmem['finalized']). Never resurrects a sentinel/empty
        value. Returns the original (empty) answer when nothing qualifies. The
        caller enforces G2 (final_answer == "") and the enforce gate."""
        invmem = self._invmem
        if invmem is None:
            return final_answer
        target = invmem.mem.target_blank
        if not target:
            return final_answer
        actor = invmem.finalize(target)          # G3: real-evidence actors only
        if actor is None:
            return final_answer
        finalized = _stringify(actor.value)
        if not finalized:                        # never resurrect an empty/sentinel
            return final_answer
        invmem.finalized = invmem._actor_dict(actor)
        return finalized

    def _invmem_final_shot(self, question: str, state: dict) -> str:
        """One whole-question completion at the empty exit (spec rev3). Gated by
        the caller on final_answer == "" — empty vs wrong both score 0, so this
        is accuracy-non-negative by construction. Returns "" on NONE/any error."""
        invmem = self._invmem
        if invmem is None:
            return ""
        try:
            resolved = {}
            candidates = []
            for b, br in (state or {}).items():
                if br is None:
                    continue
                if br.status == "resolved" and br.value is not None:
                    resolved[b] = (getattr(br, "subject", "") or "",
                                   getattr(br, "relation", "") or "",
                                   _stringify(br.value))
                else:
                    for a in invmem.mem.leading(b)[:2]:
                        candidates.append(
                            f"- {b} ({getattr(br, 'subject', '')} "
                            f"{getattr(br, 'relation', '')}): candidate {a.value!r} "
                            f"(evidence={a.has_real_evidence}, "
                            f"rejected {len(a.rejections)}x)")
            titles: list[str] = []
            for cid in reversed(invmem.read_log):
                chunk = self.corpus.get_chunk(cid)
                title = chunk.title if chunk is not None else str(cid).rsplit("__", 1)[0]
                if title and title not in titles:
                    titles.append(title)
                if len(titles) == _invmem_mod.FINAL_SHOT_MAX_ARTICLES:
                    break
            articles = [(t, self.corpus.article_text(t)) for t in titles]
            articles = [(t, x) for t, x in articles if x]
            if not articles:
                return ""
            msgs = _invmem_mod.build_final_shot_prompt(
                question, resolved, candidates, articles)
            resp = self._metered(self.llm, "finalizer").chat(msgs, temperature=0.0, max_tokens=4000,
                                 seed=self.llm_seed)
            answer = _invmem_mod.parse_final_shot_answer(
                resp.text or resp.reasoning_content)
            invmem.final_shot = {
                "answer": answer, "articles": titles,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
            }
            return answer
        except Exception as exc:      # any failure anywhere above -> empty, never raise
            invmem.final_shot = {"error": str(exc)[:200]}
            return ""

    def run_question(
        self,
        question: str,
        *,
        question_id: str,
        articles: Optional[list[dict[str, Any]]] = None,
    ) -> Trajectory:
        # Fresh ledger for THIS question, on THIS thread, before anything can
        # call a model; the local reference (not the thread-local) is what the
        # readout below uses, so the write cannot pick up another question's.
        usage = _RoleUsage()
        self._role_usage = usage
        traj = self._run_question_impl(
            question, question_id=question_id, articles=articles
        )
        traj.final_answer = _normalize_final_answer(traj.final_answer or "")
        # Cost accounting for the GSW-off ablation: usage by role, the same
        # shape the control emits. Written here rather than inside the impl so
        # every path -- the orchestrator loop, planner exhaustion, fallback --
        # carries the key. One gap, stated rather than papered over: a question
        # that falls back to `ours_gsw_v1` (`_run_fallback`) runs on a SECOND
        # adapter with its own clients, so its row holds only what this adapter
        # spent before the fallback; `extra["fallback_flag"]` marks those.
        traj.extra["role_usage"] = usage.to_dict()
        return traj

    def _run_question_impl(
        self,
        question: str,
        *,
        question_id: str,
        articles: Optional[list[dict[str, Any]]] = None,
    ) -> Trajectory:
        traj = Trajectory(
            system_id=self.system_id,
            model_id=self.ctx.model_id,
            question_id=question_id,
        )
        traj.extra["gold_articles"] = articles or []
        traj.extra["fallback_flag"] = False
        supports_seed = getattr(self.llm, "supports_seed", lambda: False)
        traj.extra["llm_temperature"] = getattr(self.llm, "default_temperature", None)
        traj.extra["llm_seed"] = self.llm_seed
        traj.extra["llm_seed_effective"] = (
            self.llm_seed is not None and bool(supports_seed())
        )
        traj.extra["synthesis_validator_mode"] = self.synthesis_validator_mode
        traj.extra["synthesis_validator_guided_repair"] = (
            self.synthesis_validator_guided_repair
        )
        start = time.time()
        # Per-question sink for checker (grounding judge / resolver) LLM calls,
        # surfaced in the inspector. Reset each question.
        self._checker_sink: list[dict[str, Any]] = []
        # Per-question read ledger (thread-local, see `_read_ledger`). Scope is
        # the question, not the researcher: a question averages ~2.5 dispatches
        # and cross-dispatch re-reads are the duplication this cache removes.
        # Must be cleared here, not left to accumulate across questions sharing
        # this adapter/thread -- otherwise it becomes a cross-question cache and
        # can return a stale answer to a different question.
        self._read_ledger().clear()
        # gsw_codegen_compute: per-question skill-library runtime (None off-path).
        self._codegen = self._make_codegen_runtime(question)
        # Phase 4 Task 7: per-question cascade-block sink. All MAIN-thread
        # cascade calls append here; pool-thread researcher blocks are merged
        # into it via return-and-merge (see `_dispatch_subplan_parallel`).
        self._cascade_blocks: list[dict[str, Any]] = []
        # Plan-request contract: fresh registry + consumption set for THIS
        # THREAD's question (see __init__ comment -- both are thread-local
        # properties). A pooled thread runs many questions in sequence, so
        # without this reset a question would inherit the previous one's
        # registry (wrong `seq` numbering, stale ids in the summary); without
        # the thread-local backing (see the properties below), concurrent
        # questions on DIFFERENT threads would corrupt each other's registry
        # entirely -- reproduced: two concurrent `run_question` calls, one
        # thread's reset wiped the other's in-flight registry, its cited
        # `plan_request_id` resolved to None, and the contract silently did
        # not fire.
        self._plan_requests: dict[str, Any] = {}
        self._question_contract = None
        self._consumed_request_ids: set[str] = set()
        # Per-question, not per-adapter: one thread runs many questions in
        # sequence, so initialising these in __init__ let a question inherit
        # the previous one's planner calls and orchestrator turns.
        self._planner_tool_calls = 0
        self._planner_tool_log = []
        self._orchestrator_turn_log = []
        self._planning_messages = []
        self._pending_call = None
        self._dossier_log: list[dict[str, Any]] = []
        self._scoped_log: list[dict[str, Any]] = []
        self._checker_sink_tls.attempts = []
        self._checker_sink_tls.verifier_flags = {}
        self._checker_sink_tls.verifier_reject_counts = {}
        self._active_question_id = question_id
        self._checker_sink_tls.question_text = question

        # --- 1. Plan -------------------------------------------------
        try:
            # The plan comes from a bounded PLANNING PHASE in which the
            # orchestrator's only tool is call_planner. It then hands back here,
            # and §2-§4 run on it: topology, plan-quality, state, evidence state,
            # the main loop and the post-loop coverage tail.
            plan, emit_meta = self._plan_via_tool(question, traj)
        except PlanEmitError as exc:
            traj.extra["plan_emit_error"] = f"{exc.kind}:{exc.detail}"
            if exc.kind == "exhausted":
                # All repair attempts ran and failed. Falling back to a
                # flat ReAct adapter here makes things worse: with no
                # plan to ground retrieval, the flat reasoner often
                # hallucinates a confident answer (e.g. "Zero." /
                # "Data not available."). Surface the failure honestly
                # by submitting an empty answer.
                traj.extra["planner_failed"] = True
                traj.extra["stopped_reason"] = "planner_exhausted"
                traj.final_answer = ""
                traj.reasoning = ""
                traj.wall_time_s = round(time.time() - start, 3)
                return traj
            return self._run_fallback(
                question, question_id, articles,
                reason=f"{exc.kind}:{exc.detail}",
            )

        traj.extra["planner_tool_calls"] = self._planner_tool_calls
        traj.extra["planner_tool_log"] = list(self._planner_tool_log)
        traj.extra["orchestrator_turns_log"] = list(self._orchestrator_turn_log)
        traj.extra["plan_json"] = plan.model_dump()
        traj.extra["raw_planner_output"] = emit_meta.raw_response[:2000]
        # `_plan_via_tool` builds `emit_meta` itself and its extras carry only
        # the plan's audit tags; the repair transcript is on
        # `planner_tool_log[-1]["plan_repair"]` (see `_tool_call_planner`).
        plan_audit_tags = list(emit_meta.extras.get("plan_audit_tags") or [])
        traj.extra["planner_audit_tags"] = plan_audit_tags

        # --- 2. Topology --------------------------------------------
        plan_dict = plan.model_dump()
        # Phase-1 investigation memory: construct per-question when flag on;
        # stays None (every hook skipped -> byte-identical) when off.
        self._invmem = (
            InvestigationMemory(plan_dict) if self.gsw_investigation_memory else None
        )
        self._invmem_round = 0
        try:
            plan.target()
            order = topological_sort_blanks(plan)
            deps = build_dependency_graph(plan)
            levels = _compute_levels(deps)
        except ExecutionError as exc:
            traj.extra["topology_error"] = f"{exc.kind}:{exc.detail}"
            traj.extra["stopped_reason"] = "topology_error"
            return self._run_fallback(
                question, question_id, articles,
                reason=f"topology_error:{exc.kind}",
            )
        traj.extra["topological_order"] = order
        traj.extra["topology_levels"] = levels

        # W5: plan-quality assessment (log_only — never acts; W6 acts).
        if self.gsw_plan_quality_checks:
            from dataclasses import asdict
            _pq_ws = GSWWorkspace(question=question, plan=plan, state={}, deps=deps,
                                  levels=levels, order=order, qgsw_state=None)
            _pq_det = _pq_ws.assess_plan_quality_deterministic(use_hints=self.gsw_blank_hints)
            _pq_all = list(_pq_det.issues)
            traj.extra["gsw_plan_quality"] = {
                "stage": "initial_plan",
                "deterministic": {"issues": [asdict(i) for i in _pq_det.issues]},
                "llm_auditor": None,
                "summary": GSWWorkspace.plan_quality_summary(_pq_all),
            }

        _pq_issue_summary: list[dict[str, Any]] = []
        if self.gsw_plan_quality_checks and self.gsw_plan_issue_surface == "advisory":
            _pq_issue_summary = GSWWorkspace.plan_issue_summary(
                deterministic=list(_pq_det.issues),
                auditor=[],
            )
            traj.extra["plan_issue_surface"] = {
                "count": len(_pq_issue_summary),
                "high_count": sum(1 for w in _pq_issue_summary if w["severity"] == "high"),
                "sources": [w["issue_id"].split("_")[1] for w in _pq_issue_summary],
            }

        # --- 3. Init state ------------------------------------------
        state: dict[str, BlankResult] = {
            e.id: BlankResult(blank_id=e.id, status="unknown")
            for e in plan.blank_entities()
        }
        qgsw_state = (
            initialize_query_gsw_state(question, plan, deps)
            if self.gsw_evidence_state
            else None
        )

        # --- 4. The orchestrator loop -------------------------------
        self._run_llm_orchestrator(
            traj=traj, plan=plan, plan_dict=plan_dict,
            state=state, order=order, levels=levels,
            question=question, qgsw_state=qgsw_state,
            plan_issue_summary=_pq_issue_summary,
        )

        traj.extra["orchestrator_turns_log"] = list(self._orchestrator_turn_log)
        traj.extra["planner_tool_calls"] = self._planner_tool_calls
        traj.extra["planner_tool_log"] = list(self._planner_tool_log)

        # --- 4b. Recovery diagnostics -------------------------------
        # MAIN-THREAD ONLY: the orchestrator's ThreadPool has joined.
        # ``gsw_recovery_diag`` is always written. The recovery mechanisms it
        # describes (M1 re-dispatch of a cascade-blocked input, M2 bounded
        # overwrite, M3 holistic answer) were off in the paper and were
        # removed, so its counts are 0; it still records whether the cascade
        # guard left the target blocked and unresolved
        # (``blocked_then_recovered_correct`` pairs that with the target's
        # resolved status, which nothing after the loop can change).
        if qgsw_state is not None:
            try:
                _recovery_target_id = plan.target().id
            except Exception:  # noqa: BLE001
                _recovery_target_id = None
            # Mirror the post-loop ``state`` into ``qgsw_state`` so the cascade
            # diag sees actual resolved/blocked deps.
            _pre_blocked_and_unresolved = False
            if _recovery_target_id is not None:
                mirror_results_into_state(qgsw_state, state, None)
                _pre_blocked_and_unresolved = bool(
                    cascade_diagnostics(
                        self._cascade_blocks, qgsw_state, _recovery_target_id
                    )["blocked_and_unresolved"]
                )
            _tgt_now = (
                state.get(_recovery_target_id)
                if _recovery_target_id is not None else None
            )
            traj.extra["gsw_recovery_diag"] = recovery_diagnostics(
                redispatch_count=0,
                overwrite_traces=[],
                holistic_fired=False,
                was_blocked_and_unresolved=_pre_blocked_and_unresolved,
                target_resolved_now=bool(
                    _tgt_now is not None and _tgt_now.status == "resolved"
                ),
            )

        # --- 5. Populate remaining Trajectory fields ----------------
        # In llm mode the plan may have been revised mid-run; the FINAL
        # plan is in ``traj.extra["plan_json"]`` (set by the mode-method).
        # Use it to compute is_target so blanks added by plan-updater
        # don't trigger a "unknown entity id" lookup error.
        final_plan_dict = traj.extra.get("plan_json") or plan.model_dump()
        target_ids: set[str] = {
            e.get("id")
            for e in (final_plan_dict.get("entities") or [])
            if e.get("is_target")
        }
        traj.extra["executed_blanks"] = [
            {
                "blank_id": res.blank_id,
                "value": res.value,
                "status": res.status,
                "evidence_chunk_ids": list(res.evidence_chunk_ids or []),
                "is_target": res.blank_id in target_ids,
            }
            for res in state.values()
        ]
        traj.extra["plan_emit_meta"] = {
            "prompt_tokens": emit_meta.prompt_tokens,
            "completion_tokens": emit_meta.completion_tokens,
            "repair_used": emit_meta.repair_used,
            "audit_tags": plan_audit_tags,
        }
        # Loop-exit safety net (q429): if the orchestrator resolved the target
        # into state but never submit_answer'd, surface that resolved value
        # rather than returning empty.
        try:
            _final_target_id = plan.target().id
        except Exception:  # noqa: BLE001
            _final_target_id = None
        if _final_target_id is not None:
            traj.final_answer = _resolved_target_answer(
                traj.final_answer, _final_target_id, state
            )
        if qgsw_state is not None:
            # The FINAL plan: a revision may have added blanks (and moved the
            # target) after qgsw_state was initialised from the initial plan.
            # The mirror needs it to create those entries; without it they were
            # missing from every run of record's serialised state (2026-09-04).
            final_plan = (
                GSWPlan.model_validate(traj.extra["plan_json"])
                if traj.extra.get("plan_json") else plan
            )
            mirror_results_into_state(
                qgsw_state, state, traj.final_answer, plan=final_plan
            )
            if self.gsw_evidence_gate != "off":
                judge = (
                    judge_relation_llm(self._metered(self.llm, "evidence_judge"), self.corpus, seed=self.llm_seed,
                                       sink=self._checker_sink)
                    if self.gsw_evidence_grounder == "llm" else None
                )
                verify_graph(final_plan, qgsw_state, self.corpus, judge=judge)
                traj.extra["gsw_evidence_metrics"] = evidence_metrics(qgsw_state)
            for rec in self._resolver_attempts:
                bs = qgsw_state.blanks.get(rec["blank_id"])
                if bs is not None:
                    bs.resolver_attempts.append(
                        {k: v for k, v in rec.items() if k != "blank_id"})
            for _bid, _st in self._verifier_flags.items():
                _bs = qgsw_state.blanks.get(_bid)
                if _bs is not None:
                    _bs.verifier_status = _st
            traj.extra["gsw_state"] = qgsw_state.to_dict()
            traj.extra["gsw_resolution_provenance"] = {
                bid: bs.resolved_by for bid, bs in qgsw_state.blanks.items()
            }
            # Phase 4: deterministic evidence coverage + supporting-edge trace
            # + failure taxonomy. Logging only (non-destructive).
            try:
                target_id = plan.target().id
            except Exception:
                target_id = None
            if target_id is not None:
                traj.extra["gsw_final_supporting_edge_ids"] = (
                    collect_supporting_edges(qgsw_state, target_id)
                )
                cov = check_evidence_coverage(qgsw_state, plan, traj.final_answer)
                traj.extra["gsw_coverage"] = cov.to_dict()
                traj.extra["gsw_failure_bucket"] = classify_failure(
                    qgsw_state, plan, traj.final_answer
                )
                # Phase 4 Task 7: per-question cascade-block diagnostics
                # (Plan-2 mechanism-slice). Pure logging — block records are
                # only produced when block_unverified_cascade is ON; with the
                # flag OFF this is an all-zero/false dict.
                traj.extra["gsw_cascade_diag"] = cascade_diagnostics(
                    self._cascade_blocks, qgsw_state, target_id
                )
            if self._checker_sink:
                for call in self._checker_sink:
                    call.setdefault("question_id", question_id)
                traj.extra["gsw_checker_calls"] = list(self._checker_sink)
        # Phase-1 investigation memory: dump the compact per-question summary
        # (actors, firings, streaks, retirements). LOG-ONLY — read by nothing.
        if self._invmem is not None:
            traj.extra["invmem"] = self._invmem.dump()
        if self._dossier_log:
            traj.extra["dossier"] = self._dossier_log
        if self._scoped_log:
            traj.extra["scoped"] = self._scoped_log
        # Plan-request contract: the consumed/dropped split. Empty when the
        # flag is off (nothing is ever registered, see
        # `_register_revision_request`) or when this question raised no
        # escalations. This is the join `plan_updates[i]["plan_request_id"]`
        # alone can't give you: which registered requests were NEVER cited by
        # any update at all.
        traj.extra["plan_requests"] = self._plan_request_summary()
        traj.wall_time_s = round(time.time() - start, 3)
        return traj

    # ------------------------------------------------------------------
    # Parallel per-blank dispatcher
    # ------------------------------------------------------------------

    MAX_PARALLEL_RESEARCHERS: ClassVar[int] = 8
    # Phase-3.2: cap on plan-updater calls per question. Two revisions
    # is enough for any plausible plan-error recovery; beyond that,
    # the corpus likely lacks the data and further revision is a
    # losing loop. The orchestrator can submit_answer with an empty/
    # best-effort string after the cap is reached.
    #
    # Phase-3.6.2 Fix #3 attempted to raise this to 4 to recover q227 /
    # q308 / q401, but a 2x100Q probe showed wrong_synthesis grew
    # 12.5 → 18 (mean) and judge_acc dropped from 68.5 → 66 — the
    # extra cap budget converted "give up empty" into "submit stale
    # wrong value". Reverted to 2.
    MAX_PLAN_UPDATES_PER_RUN: ClassVar[int] = 2
    # Phase-3.2 follow-up: per-blank dispatch cap. If a single blank
    # has triggered ``MAX_ESCALATIONS_PER_BLANK`` researcher
    # escalations (suggest_plan_revision), further dispatches of that
    # blank are REJECTED. Prevents the "spam dispatch with permuted
    # hints" loop where the orchestrator never calls request_plan_update
    # but keeps re-dispatching a blank whose retrieval target isn't in
    # the corpus.
    MAX_ESCALATIONS_PER_BLANK: ClassVar[int] = 2
    # Phase-3.7 (synthesis validator): if explicitly enabled in
    # reject-mode, a wrong verdict can reject one submit_answer and give
    # the orchestrator one re-think turn. Default mode is off because the
    # validator is another stochastic model call.
    MAX_VALIDATOR_RETRIES: ClassVar[int] = 1
    # Failure-aware verifier loop: one deterministic reopen/re-dispatch
    # after a validator wrong verdict identifies a concrete bad blank.
    MAX_VALIDATOR_GUIDED_REPAIRS_PER_RUN: ClassVar[int] = 1
    VALIDATOR_GUIDED_REPAIR_ERROR_TYPES: ClassVar[set[str]] = {
        "wrong_entity",
        "missing_detail",
    }

    def _read_cited_chunks(self, evidence_chunk_ids: list[str]) -> list[dict[str, str]]:
        return read_cited_chunks(self.corpus, evidence_chunk_ids)

    def _research_blank(
        self,
        *,
        workspace: "GSWWorkspace",
        context: "BlankContext",
        dep_values: dict[str, Any],
        question: str,
        resolver: "TypedBlankResolver",
        verifier: "Optional[BlankVerifier]",
    ) -> BlankResearchResult:
        """W4a/W4b: the per-blank RESOLVER attempt as one structured unit, now
        driven by the workspace's BlankContext. `plan`/`bid` come from the
        workspace; `dep_values` is the fixed pre-loop snapshot (workspace-owned,
        passed in). The resolver/verifier still read hints from `workspace.plan`
        (feeding them FROM the context is a later phase); this is behavior-neutral.
        """
        plan = workspace.plan
        bid = context.blank_id
        res = resolver.resolve_blank(
            plan, bid, dep_values, question,
)
        committed_partial = False
        if res.status == "resolved" and verifier is not None:
            v = verify_blank(
                verifier, plan, bid, res.value, dep_values,
                holistic_chunks=(
                    self._read_cited_chunks(list(res.evidence_chunk_ids))
                    if self.gsw_verifier_scoped else res.retrieved_chunks
                ),
)
            committed_partial = bool(getattr(v, "partial", False))
            if not v.accepted:
                query = effective_query(
                    plan, bid, dep_values, question, use_hints=self.gsw_blank_hints
                ).query
                self._resolver_attempts.append({
                    "blank_id": bid, "query": query,
                    "candidate": res.value, "verifier": _verdict_dict(v)})
                return BlankResearchResult(
                    blank_id=bid, status="unresolved", candidate=res.value,
                    evidence_chunk_ids=list(res.evidence_chunk_ids),
                    local_feedback={
                        "query": query, "candidate": res.value,
                        "reason": v.reason, "chunk_ids": list(res.evidence_chunk_ids),
                    },
                )
        if res.status == "resolved":
            return BlankResearchResult(
                blank_id=bid, status="resolved", value=res.value,
                resolved_by="resolver",
                evidence_chunk_ids=list(res.evidence_chunk_ids),
                completeness_status=("partial" if committed_partial else None),
            )
        # unresolved
        chunk_ids = list(res.evidence_chunk_ids) or [
            c.get("chunk_id", "") for c in res.retrieved_chunks
            if c.get("chunk_id")
        ]
        query = effective_query(
            plan, bid, dep_values, question, use_hints=self.gsw_blank_hints
        ).query
        self._resolver_attempts.append({
            "blank_id": bid, "query": query, "candidate": None,
            "resolver": {"status": res.status, "message": res.message}})
        return BlankResearchResult(
            blank_id=bid, status="unresolved", candidate=None,
            evidence_chunk_ids=chunk_ids,
            local_feedback={
                "query": query, "candidate": None,
                "reason": res.message, "chunk_ids": chunk_ids,
            },
        )

    def _dispatch_subplan_parallel(
        self,
        *,
        plan: GSWPlan,
        plan_dict: dict[str, Any],
        state: dict[str, BlankResult],
        order: list[str],
        levels: dict[str, int],
        question: str,
        blank_ids: list[str],
        hints: str = "",
        level_tag: int = 0,
        qgsw_state: Optional["QueryGSWState"] = None,
        workspace: Optional["GSWWorkspace"] = None,
    ) -> list[dict[str, Any]]:
        """Spawn one ReActResearcher per blank in ``blank_ids`` and run
        them concurrently in a thread pool. Each researcher sees a
        FRESH COPY of the current state (so siblings at the same
        dispatch don't race). After all return, merge each
        researcher's assigned blank back into the shared ``state``,
        then run ``_cascade_auto_compute``.

        Returns a list of per-researcher trace dicts — one per
        blank_id — in the order they were submitted.

        ``level_tag`` is an integer used to label tool_calls +
        messages (the inspector groups by it). In deterministic mode
        this is the topological level; in LLM-orchestrator mode it is
        the dispatch index.
        """
        if not blank_ids:
            return []

        # Phase-1 investigation memory: one round == one dispatch. Resolver
        # commits + react-sibling episodes for THIS call all share this round id.
        if self._invmem is not None:
            self._invmem_round += 1
        _invmem_round = self._invmem_round

        # --- TypedBlankResolver pre-pass + verify + handoff (Phase 3-full) ---
        handoff: dict[str, dict[str, Any]] = {}
        if self.gsw_typed_blank_resolver and blank_ids:
            ws = workspace if workspace is not None else GSWWorkspace(
                question=question, plan=plan, state=state,
                deps=build_dependency_graph(plan), levels=levels, order=order,
                qgsw_state=qgsw_state,
            )
            resolver = TypedBlankResolver(
                retriever=self.retriever, corpus=self.corpus, llm=self._metered(self.llm, "resolver"),
                max_docs=self.gsw_max_docs, hard_type_filter=self.gsw_hard_type_filter,
                max_completion_tokens=self.ctx.max_completion_tokens,
                sink=self._checker_sink,
                use_hints=self.gsw_blank_hints,
            )
            verifier = BlankVerifier(llm=self._metered(self.llm, "verifier"), sink=self._checker_sink,
                                     use_hints=self.gsw_blank_hints,
                                     max_completion_tokens=self.ctx.max_completion_tokens,
                                     scoped=self.gsw_verifier_scoped)
            dep_values = ws.resolved_values()
            contexts = {bid: ws.get_blank_context(bid) for bid in blank_ids}
            remaining: list[str] = []
            for bid in blank_ids:
                result = self._research_blank(
                    workspace=ws, context=contexts[bid],
                    dep_values=dep_values, question=question,
                    resolver=resolver, verifier=verifier,
                )
                if result.status == "resolved":
                    ws.record_research_result(result)
                    if self._invmem is not None:
                        _br = state.get(bid)
                        self._invmem.record_resolver_commit(
                            _invmem_round, bid, result.value,
                            list(getattr(_br, "evidence_chunk_ids", None) or []))
                else:
                    handoff[bid] = result.local_feedback
                    remaining.append(bid)
            _cascade_auto_compute(
                    plan, state, block_unverified=self.block_unverified_cascade,
                    blocked_sink=self._cascade_blocks, codegen=self._codegen,
                )
            if not self.gsw_react_fallback:
                blank_ids = []
            else:
                blank_ids = remaining
            if not blank_ids:
                # Resolver settled everything this dispatch — no react siblings.
                # Close the round so resolver-commit progress is bookkept.
                if self._invmem is not None:
                    self._invmem.close_round(_invmem_round)
                return []
        # -----------------------------------------------------------------------

        # Snapshot state so sibling researchers at the same dispatch
        # see consistent inputs. Taken AFTER the resolver pre-pass so
        # researchers see resolver-resolved siblings.
        pre_snapshot: dict[str, BlankResult] = {
            bid: copy.copy(br) for bid, br in state.items()
        }

        # Dossiers: rendered on the MAIN thread — pool threads cannot see self._invmem (thread-local); only dossier strings cross into the closure.
        dossiers: dict[str, str] = {}
        if self.gsw_memory_dossier and self._invmem is not None:
            for _bid in blank_ids:
                _d = render_dossier(self._invmem, _bid, self.corpus)
                if _d:
                    dossiers[_bid] = _d
                    self._dossier_log.append({
                        "blank": _bid,
                        "round": _invmem_round,
                        "chars": len(_d),
                    })

        scoped = {}
        if self.gsw_memory_scoped_dispatch and self._invmem is not None:
            for _bid in blank_ids:
                _b = scoped_budget(self._invmem, _bid, self.level_max_turns)
                scoped[_bid] = _b
                if _b < self.level_max_turns:
                    self._scoped_log.append({"blank": _bid, "budget": _b})

        # Phase-1 investigation memory: one Episode buffer per sibling, minted on
        # this (main) thread. Each researcher fills its OWN buffer inside the pool
        # thread (no shared mutation); the orchestrator folds them at the join.
        _invmem_eps: dict[str, Any] = {}
        if self._invmem is not None:
            _invmem_eps = {
                bid: self._invmem.open_episode(_invmem_round, "researcher", [bid])
                for bid in blank_ids
            }

        # What this dispatch adds beyond the base question: the
        # orchestrator's hints for the assigned blank.
        dispatch_hint_suffix = (
            f"Orchestrator hints for this assigned blank:\n{hints.strip()}"
            if hints.strip() else ""
        )
        question_with_hints = question
        if dispatch_hint_suffix:
            question_with_hints = f"{question}\n\n{dispatch_hint_suffix}"

        # Same hazard, same fix, for the agentic-read ledger: `_read_ledger()` is
        # THREAD-LOCAL, so evaluating it inside `_solve_one` (a pool worker) mints
        # a fresh empty dict per sibling and the cache degenerates to
        # per-researcher — the scoping spec §3 explicitly rejects. Capture the
        # question thread's dict here and close over it, so every sibling shares
        # one ledger. Concurrent get/setitem on distinct keys is safe under the
        # GIL; the worst case is two siblings both missing and both spending a
        # call.
        read_ledger = self._read_ledger()
        # `_checker_sink` is thread-local too. Evaluated inside `_solve_one`
        # (a pool worker) it handed every sibling a private list that nothing
        # merged, so `gsw_checker_calls` lost every Verifier verdict from
        # multi-blank dispatches (kept 51/59 in single-worker dispatches vs
        # 35/387 in multi-worker ones on the BCP run of record). Hoist once.
        checker_sink = self._checker_sink
        # And once more for the per-role token ledger: `_metered` resolves the
        # THREAD-LOCAL `_role_usage` at construction, so building these inside
        # `_solve_one` would meter every pool sibling into its own empty ledger
        # and the question would report zero researcher, reader and verifier
        # tokens -- the same failure shape as the two hoists above. Wrappers are
        # stateless apart from the ledger they hold, so one per role serves
        # every sibling. Logging only: each wraps exactly the client this
        # dispatch used before.
        researcher_llm = self._metered(self.llm, "researcher")
        researcher_reader_llm = self._metered_reader_llm()
        researcher_verifier_llm = self._metered(self.llm, "verifier")

        def _solve_one(
            blank_id: str,
        ) -> tuple[str, dict[str, Any], dict[str, BlankResult], list[dict[str, Any]], dict[str, str], list[dict[str, Any]], dict[str, int], Optional[dict[str, Any]], bool, list[tuple[str, str, dict[str, str], str]], tuple[Optional[list[dict[str, Any]]], Optional[Any]]]:
            local_state: dict[str, BlankResult] = {
                bid: copy.copy(br) for bid, br in pre_snapshot.items()
            }
            system_prompt = build_researcher_prompt(
                plan_dict, order, levels,
                slice_blank_ids=[blank_id],
                state=local_state,
                level_index=level_tag,
                handoff_hints=handoff,
                connection_manifests=self._slice_connection_manifests(
                    plan, [blank_id], local_state),
                verifier_reasoning=self.gsw_verifier_reasoning,
            )
            researcher = ReActResearcher(
                llm=researcher_llm,          # hoisted; see the comment above
                reader_llm=researcher_reader_llm,
                corpus=self.corpus,
                retriever=self.retriever,
                plan=plan,
                state=local_state,
                allowed_blank_ids=[blank_id],
                verifier_reasoning=self.gsw_verifier_reasoning,
                system_prompt=system_prompt,
                question=question_with_hints,
                max_turns=scoped.get(blank_id, self.level_max_turns),
                max_completion_tokens=self.ctx.max_completion_tokens,
                llm_seed=self.llm_seed,
                hard_type_filter=self.gsw_hard_type_filter,
                handoff_hints=handoff,
                # Edge-grounded commits render the verification claim from
                # the planner's hints (use_hints=True).
                blank_verifier=BlankVerifier(llm=researcher_verifier_llm, sink=checker_sink,
                                             use_hints=True,
                                             max_completion_tokens=self.ctx.max_completion_tokens,
                                             scoped=self.gsw_verifier_scoped),
                edge_grounded_cap=self.gsw_edge_grounded_cap,
                block_unverified_cascade=self.block_unverified_cascade,
                context_compaction_on=self.gsw_context_compaction,
                codegen=self._codegen,
                read_raw_fallback=self.gsw_read_raw_fallback,
                read_ledger=read_ledger,   # pool-shared; see the hoist above
                prior_investigation=dossiers.get(blank_id),
            )
            # runs in the sibling pool thread — self._invmem is thread-local and
            # would read None here; use the closure dict minted on the question thread.
            if _invmem_eps:
                researcher._invmem_episode = _invmem_eps.get(blank_id)
            trace = researcher.solve()
            # Return attempts + flags + cascade blocks — do NOT write to self.*
            # here (pool thread). Merged on the main thread below.
            return (blank_id, trace, local_state, researcher._react_attempts,
                    researcher._verifier_flagged, researcher._cascade_blocks,
                    researcher._verifier_rejects)

        # Single-blank dispatch: skip the pool (no point).
        if len(blank_ids) == 1:
            results = [_solve_one(blank_ids[0])]
        else:
            n_workers = min(len(blank_ids), self.MAX_PARALLEL_RESEARCHERS)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(_solve_one, blank_ids))

        # Merge each researcher's OWN blank back into shared state.
        # (Sibling writes to other blanks inside local_state are
        # irrelevant — update_blank already rejected them via the
        # allowed_blank_ids guard.)
        # Merge results back into shared state on the MAIN thread (thread-safe).
        out_traces: list[dict[str, Any]] = []
        for (blank_id, trace, local_state, _react_attempts,
             _verifier_flagged, _cascade_blocks,
             _verifier_rejects) in results:
            if blank_id in local_state:
                state[blank_id] = local_state[blank_id]
            for tc in trace["tool_calls"]:
                tc.level = level_tag
            out_traces.append(trace)
            # Merge ReAct attempts + verifier flags + cascade blocks here
            # (main thread — thread-safe). Mirrors the _react_attempts merge.
            self._resolver_attempts.extend(_react_attempts)
            for _bid, _st in _verifier_flagged.items():
                self._verifier_flags[_bid] = _st
            for _bid, _n in _verifier_rejects.items():
                self._verifier_reject_counts[_bid] = (
                    self._verifier_reject_counts.get(_bid, 0) + _n)
            self._cascade_blocks.extend(_cascade_blocks)

        # Fire constraint cascade after all parallel researchers settle.
        _cascade_auto_compute(
                    plan, state, block_unverified=self.block_unverified_cascade,
                    blocked_sink=self._cascade_blocks, codegen=self._codegen,
                )
        # Phase-1 investigation memory: fold each sibling's episode at the join in
        # deterministic (blank_id) order, then close the round. Resolver commits
        # for this dispatch were already folded during the pre-pass above.
        if self._invmem is not None:
            for _bid in sorted(blank_ids):
                _ep = _invmem_eps.get(_bid)
                if _ep is not None:
                    self._invmem.fold(_ep)
            self._invmem.close_round(_invmem_round)

        return out_traces

    # ------------------------------------------------------------------
    # Mode: llm (orchestrator ReAct loop; parallel per-blank dispatch)
    # ------------------------------------------------------------------

    def _run_llm_orchestrator(
        self,
        *,
        traj: Trajectory,
        plan: GSWPlan,
        plan_dict: dict[str, Any],
        state: dict[str, BlankResult],
        order: list[str],
        levels: dict[str, int],
        question: str,
        qgsw_state: Optional["QueryGSWState"] = None,
        plan_issue_summary: list[dict[str, Any]] | None = None,
    ) -> None:
        """LLM-driven orchestrator loop.

        One live conversation that continues the planning phase's: every
        turn makes one LLM call with strict tool_choice and dispatches the
        picked tool; the call and its result are appended to the
        conversation. The plan, state, order and levels are all mutable —
        `request_plan_update` can swap them in mid-run.

        Populates ``traj`` in place. The orchestrator itself does NOT
        call retrieval tools — researchers do.
        """
        max_turns = int(self.ctx.extra.get("orchestrator_max_turns", 12))
        constraint_outputs: set[str] = {
            c.output_blank_id for c in plan.constraints if c.output_blank_id
        }

        researcher_traces: list[dict[str, Any]] = []
        plan_json_versions: list[dict[str, Any]] = [plan_dict]
        plan_updates: list[dict[str, Any]] = []
        dispatches: list[dict[str, Any]] = []
        all_tool_calls: list[ToolCall] = []
        all_messages: list[dict[str, Any]] = []
        reasoning_chunks: list[str] = []
        # Phase-3.2 follow-up: per-blank escalation counter.
        # Incremented every time a researcher returns
        # `revision_request` for that blank.
        escalations_per_blank: dict[str, int] = {}
        # Phase-3.7: synthesis-validator retry counter.
        validator_retries: int = 0
        validator_repairs: int = 0
        validator_guided_repairs: int = 0
        validator_verdicts: list[dict[str, Any]] = []

        total_prompt_tokens = total_completion_tokens = 0
        orchestrator_turns = 0
        dispatch_idx = 0
        # One message list for the whole question, the way a researcher holds
        # one list and appends `read` results to it. The system half never
        # changes again, so a prefix cache serves every token up to the newest
        # tool message; the state arrives once, with the installed plan, and
        # thereafter as tool results.
        prose_nudges = 0
        planning = list(getattr(self, "_planning_messages", None) or [])
        self._planning_messages = []
        pending = getattr(self, "_pending_call", None)
        self._pending_call = None
        # Continue the planning conversation (`_plan_via_tool` always leaves
        # it here): the first dispatch should be informed by the plan the
        # orchestrator chose and the audit it read. Only the system prompt is
        # replaced -- the bootstrap one says the tools are unavailable, which
        # is false now that a plan exists.
        live_messages: list[dict[str, Any]] = list(planning)
        live_messages[0] = {
            "role": "system",
            "content": build_orchestrator_system_static(
                plan_dict, order, levels,
                plan_issue_warnings=plan_issue_summary,
                plan_request_contract=self.gsw_plan_request_contract,
            ),
        }
        live_messages.append({
            "role": "user",
            "content": ("PLAN INSTALLED — work from this one.\n\n"
                        f"{render_plan_brief(plan_dict, order, levels)}\n\n"
                        f"{build_state_message(plan_dict, state)}"),
        })
        final_answer = ""
        stopped_reason = "max_turns"
        target_id = plan.target().id
        # Phase-3.4: track consecutive cap-rejected dispatches. Reset
        # on any non-cap-rejected tool call. After 3 in a row, the loop
        # auto-gives-up — see check at end of turn.
        consecutive_cap_rejections = 0
        for turn in range(1, max_turns + 1):
            # Fire any pending constraint cascades before the next call.
            _cascade_auto_compute(
                plan, state, block_unverified=self.block_unverified_cascade,
                blocked_sink=self._cascade_blocks, codegen=self._codegen,
            )

            # W2: rebuild the read-only workspace from the CURRENT loop locals
            # each turn, so it tracks the mid-loop plan/order/levels rebind on
            # request_plan_update. Locals remain the source of truth; the
            # workspace owns context construction (spec §5).
            workspace = GSWWorkspace(
                question=question,
                plan=plan,
                state=state,
                deps=build_dependency_graph(plan),
                levels=levels,
                order=order,
                qgsw_state=qgsw_state,
            )

            if pending is not None:
                # The call that ended the planning phase, honoured here instead
                # of asking the orchestrator to repeat itself.
                resp = _PendingResponse(pending)
                pending = None
            else:
                try:
                    resp = self._metered(self.llm, "orchestrator").chat(
                        live_messages,
                        tools=self._orchestrator_tools,
                        tool_choice="required",
                        max_tokens=self.ctx.max_completion_tokens,
                        seed=self.llm_seed,
                    )
                except Exception as exc:  # noqa: BLE001
                    stopped_reason = "llm_error"
                    reasoning_chunks.append(
                        f"[orchestrator turn {turn}] LLM error: {exc}"
                    )
                    break

            orchestrator_turns += 1
            # One row per orchestrator turn: what it was thinking and what it
            # decided. `traj.reasoning` concatenates every actor's chunks, which
            # is unreadable when you want one actor's thread.
            self._orchestrator_turn_log.append({
                "turn": turn,
                "reasoning": (getattr(resp, "reasoning_content", "") or "")[:4000],
                "text": (getattr(resp, "text", "") or "")[:1000],
                "calls": [{"name": tc.get("name"), "arguments": str(tc.get("arguments"))[:2000]}
                          for tc in (resp.tool_calls or [])],
                "prompt_tokens": int(getattr(resp, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(resp, "completion_tokens", 0) or 0),
            })
            total_prompt_tokens += int(getattr(resp, "prompt_tokens", 0) or 0)
            total_completion_tokens += int(getattr(resp, "completion_tokens", 0) or 0)
            if resp.reasoning_content:
                reasoning_chunks.append(
                    f"[orchestrator turn {turn}] {resp.reasoning_content}"
                )

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": resp.text or "",
                "_level": -1,  # -1 = orchestrator messages
            }
            if resp.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in resp.tool_calls
                ]
            all_messages.append(assistant_msg)
            # `_level` is our own trace bookkeeping and must not be sent.
            live_messages.append(
                {k: v for k, v in assistant_msg.items()
                 if not k.startswith("_")}
            )

            if not resp.tool_calls:
                # A bounded corrective turn: a conversation makes the model
                # likelier to answer in prose, and losing the question to
                # `no_tool_call` is far more expensive than one more call.
                if prose_nudges < MAX_PROSE_NUDGES:
                    prose_nudges += 1
                    live_messages.append({"role": "user", "content": PROSE_NUDGE})
                    continue
                stopped_reason = "no_tool_call"
                break

            # Process each tool_call (usually exactly one).
            broke_out = False
            for tc in resp.tool_calls:
                name = tc.get("name", "")
                try:
                    args = (
                        json.loads(tc["arguments"]) if tc.get("arguments") else {}
                    )
                except json.JSONDecodeError as exc:
                    result: dict[str, Any] = {
                        "ok": False,
                        "error": f"bad tool args: {exc}",
                    }
                else:
                    # Phase-3.2: pass plan-update count (for cap) + last
                    # update summary (for hint enrichment).
                    last_pu_summary = ""
                    if plan_updates:
                        last = plan_updates[-1]
                        if not last.get("rejected"):
                            last_pu_summary = (
                                f"diff={last.get('diff_summary','')}; "
                                f"reason={str(last.get('reason',''))[:80]}"
                            )
                    # Count all attempts, including rejected no-op or invalid
                    # replans. Otherwise the model can burn turns retrying the
                    # same unsupported update forever without hitting the cap.
                    pu_count = len(plan_updates)
                    result, action = self._run_orchestrator_tool(
                        name=name,
                        args=args,
                        plan=plan,
                        plan_dict=plan_dict,
                        state=state,
                        order=order,
                        levels=levels,
                        question=question,
                        dispatch_idx=dispatch_idx,
                        constraint_outputs=constraint_outputs,
                        plan_update_count=pu_count,
                        last_plan_update_summary=last_pu_summary,
                        escalations_per_blank=escalations_per_blank,
                        validator_retries=validator_retries,
                        validator_guided_repairs=validator_guided_repairs,
                        qgsw_state=qgsw_state,
                        workspace=workspace,
                    )
                    if action.get("validator_verdict") is not None:
                        validator_verdicts.append(action["validator_verdict"])
                    if action.get("validator_rejected"):
                        validator_retries += 1
                    if action.get("validator_repaired"):
                        validator_repairs += 1
                    if action.get("validator_guided_repair"):
                        validator_guided_repairs += 1
                    total_prompt_tokens += action.get("prompt_tokens", 0)
                    total_completion_tokens += action.get("completion_tokens", 0)
                    # Update mutable bookkeeping based on the action.
                    if action.get("dispatch_traces") is not None:
                        traces = action["dispatch_traces"]
                        for t in traces:
                            researcher_traces.append(
                                {
                                    "level": dispatch_idx,
                                    "allowed_blank_ids": t["allowed_blank_ids"],
                                    "turns": t["turns"],
                                    "stopped": t["stopped"],
                                    "resolved": t["resolved"],
                                    "unresolved": t["unresolved"],
                                    "prompt_tokens": t["prompt_tokens"],
                                    "completion_tokens": t["completion_tokens"],
                                    "turn_usage": t.get("turn_usage", []),
                                    "readiness": t.get("readiness", {}),
                                    **(
                                        {"metadata": t["metadata"]}
                                        if t.get("metadata")
                                        else {}
                                    ),
                                }
                            )
                            total_prompt_tokens += t["prompt_tokens"]
                            total_completion_tokens += t["completion_tokens"]
                            all_tool_calls.extend(t["tool_calls"])
                            for m in t["messages"]:
                                m_out = dict(m)
                                m_out["_level"] = dispatch_idx
                                all_messages.append(m_out)
                            if t["reasoning"]:
                                reasoning_chunks.append(
                                    f"=== dispatch {dispatch_idx} ===\n{t['reasoning']}"
                                )
                        dispatches.append(
                            {
                                "dispatch_idx": dispatch_idx,
                                "blank_ids": action["blank_ids"],
                                "hints": action.get("hints", ""),
                                "partial": action.get("partial", False),
                                "revision_requests": action.get(
                                    "revision_requests", []
                                ),
                            }
                        )
                        # Phase-3.2 follow-up: bump per-blank escalation
                        # counter for blanks whose researcher escalated.
                        _escalated_now = set()
                        for rr in action.get("revision_requests") or []:
                            bid = rr.get("blank_id")
                            if bid:
                                _escalated_now.add(bid)
                                escalations_per_blank[bid] = (
                                    escalations_per_blank.get(bid, 0) + 1
                                )
                        dispatch_idx += 1
                    if action.get("new_plan") is not None:
                        # Swap plan + plan_dict + order + levels.
                        plan = action["new_plan"]
                        plan_dict = plan.model_dump()
                        try:
                            plan.target()
                            order = topological_sort_blanks(plan)
                            deps = build_dependency_graph(plan)
                            levels = _compute_levels(deps)
                        except Exception as exc:  # noqa: BLE001
                            # Topology failed on the new plan — revert and
                            # log the rejection.
                            plan = action["old_plan"]
                            plan_dict = plan.model_dump()
                            state.clear()
                            state.update(action.get("old_state", {}))
                            result = {
                                "ok": False,
                                "error": (
                                    f"new plan has invalid topology: {exc}. "
                                    "Reverting to previous plan."
                                ),
                            }
                            plan_updates.append(
                                {
                                    "turn": turn,
                                    "reason": action.get("reason"),
                                    "evidence": action.get("evidence"),
                                    "rejected": True,
                                    "error": result["error"],
                                    "plan_request_id": action.get("plan_request_id"),
                                    "plan_request_id_raw": action.get(
                                        "plan_request_id_raw"
                                    ),
                                    "contract": action.get("contract", []),
                                    "contract_satisfied": action.get(
                                        "contract_satisfied", []
                                    ),
                                    "contract_active": action.get(
                                        "contract_active", False
                                    ),
                                    "contract_unmet": action.get(
                                        "contract_unmet", False
                                    ),
                                }
                            )
                        else:
                            constraint_outputs = {
                                c.output_blank_id for c in plan.constraints
                                if c.output_blank_id
                            }
                            target_id = plan.target().id
                            constraint_outputs = {
                                c.output_blank_id for c in plan.constraints
                                if c.output_blank_id
                            }
                            # A blank with no state entry can never be resolved:
                            # the dispatcher looks itself up by id. `request_
                            # plan_update` never hit this because it carries the
                            # old state over; a plan installed by call_planner
                            # starts from nothing, so seed every blank here.
                            for _e in plan.blank_entities():
                                state.setdefault(
                                    _e.id,
                                    BlankResult(blank_id=_e.id, status="unknown"),
                                )
                            if action.get("planner_tool_issues") is not None:
                                # Champion mode discards tool results every turn,
                                # so a tool-provided audit reaches it only
                                # through the block the rebuilt prompt already
                                # renders. call_planner carries no audit of its
                                # own today (that check was removed with the
                                # auditor; a verifier will populate these
                                # fields next) — only override the pre-loop W5
                                # summary when the tool actually found
                                # something, so an empty tool audit does not
                                # blank out real deterministic warnings.
                                _mech = list(action.get("planner_tool_mechanical") or [])
                                _rest = [i for i in action["planner_tool_issues"]
                                         if i not in _mech]
                                if _mech or _rest:
                                    plan_issue_summary = (
                                        GSWWorkspace.plan_issue_summary(
                                            deterministic=_mech, auditor=_rest,
                                        )
                                    )
                            plan_json_versions.append(plan_dict)
                            if action.get("planner_tool_issues") is not None:
                                # A plan installed by call_planner. The
                                # bootstrap prompt says "you have no plan yet ...
                                # dispatch_subplan, get_state and submit_answer
                                # are unavailable" — leaving that standing while a
                                # plan exists contradicts every later turn, and
                                # cost 26% more turns and double the max_turns
                                # deaths. Swap it for the real one.
                                live_messages[0] = {
                                    "role": "system",
                                    "content": build_orchestrator_system_static(
                                        plan_dict, order, levels,
                                        plan_issue_warnings=plan_issue_summary,
                                        plan_request_contract=self.gsw_plan_request_contract,
                                    ),
                                }
                                live_messages.append({
                                    "role": "user",
                                    "content": (
                                        "PLAN INSTALLED — this is the plan you "
                                        "asked for. Work from it.\n\n"
                                        f"{render_plan_brief(plan_dict, order, levels)}"
                                        "\n\n"
                                        f"{build_state_message(plan_dict, state)}"
                                    ),
                                })
                            else:
                                # The old plan stays in the transcript above, so
                                # the new one has to be said out loud or the
                                # orchestrator keeps planning against a plan
                                # that no longer exists.
                                live_messages.append({
                                    "role": "user",
                                    "content": (
                                        "PLAN UPDATED — the plan you were given "
                                        "earlier has been replaced. Ignore it and "
                                        "work from this one.\n\n"
                                        f"{action.get('diff_summary') or 'a new plan was installed'}\n\n"
                                        f"{render_plan_brief(plan_dict, order, levels)}"
                                        "\n\n"
                                        f"{build_state_message(plan_dict, state)}"
                                    ),
                                })
                            # A plan CREATED by call_planner is not a plan
                            # UPDATE. Logging one produced a phantom row whose
                            # every field was None, which reads in an inspector
                            # as an update that explains nothing.
                            if action.get("planner_tool_issues") is None:
                                plan_updates.append({
                                    "turn": turn,
                                    "reason": action.get("reason"),
                                    "evidence": action.get("evidence"),
                                    "diff_summary": action.get("diff_summary"),
                                    "preserved_ids": action.get("preserved_ids"),
                                    "added_ids": action.get("added_ids"),
                                    "dropped_ids": action.get("dropped_ids"),
                                    "invalidated_ids": action.get(
                                        "invalidated_ids", []
                                    ),
                                    "op_summary": action.get("op_summary", ""),
                                    "attempts": action.get("attempts", 1),
                                    "updater_reasoning": action.get("updater_reasoning", ""),
                                    "updater_raw": action.get("updater_raw", ""),
                                    "plan_request_id": action.get("plan_request_id"),
                                    "plan_request_id_raw": action.get(
                                        "plan_request_id_raw"
                                    ),
                                    # Finding 2: contract satisfaction is
                                    # otherwise unmeasurable from a run.
                                    "contract": action.get("contract", []),
                                    "contract_satisfied": action.get(
                                        "contract_satisfied", []
                                    ),
                                    "contract_active": action.get(
                                        "contract_active", False
                                    ),
                                    "contract_unmet": action.get(
                                        "contract_unmet", False
                                    ),
                                    "contract_identity_satisfied": action.get(
                                        "contract_identity_satisfied", False
                                    ),
                                })
                    if action.get("plan_update_rejected"):
                        plan_updates.append(
                            {
                                "turn": turn,
                                "reason": action.get("reason"),
                                "evidence": action.get("evidence"),
                                "rejected": True,
                                "error": action.get("error", ""),
                                "plan_request_id": action.get("plan_request_id"),
                                "plan_request_id_raw": action.get(
                                    "plan_request_id_raw"
                                ),
                                "contract": action.get("contract", []),
                                "contract_satisfied": action.get(
                                    "contract_satisfied", []
                                ),
                                "contract_active": action.get(
                                    "contract_active", False
                                ),
                                "contract_unmet": action.get(
                                    "contract_unmet", False
                                ),
                            }
                        )
                    if action.get("final_answer") is not None:
                        final_answer = action["final_answer"]
                        stopped_reason = action.get("stopped_reason", "finished")
                        broke_out = True

                result_json = json.dumps(result, default=str)
                try:
                    args_parsed = (
                        json.loads(tc["arguments"]) if tc.get("arguments") else {}
                    )
                except json.JSONDecodeError:
                    args_parsed = {"_raw": tc.get("arguments", "")}
                tcall = ToolCall(
                    turn=turn,
                    name=name,
                    args=args_parsed,
                    result_preview=result_json[:500],
                    result_full=result_json,
                    duration_s=0.0,
                    error=(
                        result.get("error", "")
                        if isinstance(result, dict)
                        else ""
                    ),
                    level=-1,
                )
                all_tool_calls.append(tcall)
                all_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_json,
                        "_level": -1,
                    }
                )
                # The result itself — plus the nag the static system prompt
                # gives up by not re-rendering the state table every turn.
                live_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_json,
                })
                live_messages.append({
                    "role": "user",
                    "content": build_live_status_line(
                        plan_dict, state, turn=turn, max_turns=max_turns,
                        target_id=target_id,
                    ),
                })
                # Phase-3.4: track consecutive per-blank cap rejections.
                # The dispatcher returns a result with
                # `error` containing "per-blank dispatch cap reached"
                # (and ok:false) when this happens.
                is_cap_rejection = (
                    name == "dispatch_subplan"
                    and isinstance(result, dict)
                    and result.get("ok") is False
                    and "per-blank dispatch cap reached"
                    in str(result.get("error", ""))
                )
                if is_cap_rejection:
                    consecutive_cap_rejections += 1
                else:
                    consecutive_cap_rejections = 0
                if name == "submit_answer" and isinstance(result, dict):
                    # Kept from the per-turn activity line of the removed
                    # non-live prompt, whose only effect in this loop was this
                    # read: it raises UnboundLocalError when a submit_answer with
                    # unparseable arguments comes before any parsed tool call of
                    # the question (``action`` is first bound by
                    # ``_run_orchestrator_tool`` above).
                    action.get("validator_rejected")
            if broke_out:
                break

            # Stall->Stop (spec 2026-07-18 rev3): the question-level clock says
            # nothing grounded has been learned for R_Q rounds and no live
            # unaccepted evidence remains. Force-exit deterministically — never
            # an error message the LLM could react to (the dispatch filter's
            # measured failure mode). The empty answer then flows through the
            # final-shot + finalizer hooks below.
            if self._invmem_enforce:
                _term = self._invmem.should_terminate()
                if _term is not None:
                    self._invmem.terminated = _term
                    stopped_reason = "invmem_question_exhausted"
                    tgt_res = state.get(target_id)
                    if tgt_res is not None and tgt_res.status == "resolved":
                        final_answer = _stringify(tgt_res.value)
                    else:
                        final_answer = ""
                    break

            # Phase-3.4 safety net: after 3 consecutive cap-rejections,
            # the LLM has clearly ignored the worked example and is
            # stuck. Force-exit instead of burning the whole budget.
            if consecutive_cap_rejections >= 3:
                stopped_reason = "give_up_unanswerable"
                # Best-effort answer: use committed target value if any,
                # else empty.
                tgt_res = state.get(target_id)
                if tgt_res is not None and tgt_res.status == "resolved":
                    final_answer = _stringify(tgt_res.value)
                else:
                    final_answer = ""
                break


        # Final-answer fallback: if orchestrator exhausted its turns
        # but target is resolved, use that committed value.
        tgt_res = state.get(target_id)
        if (
            not final_answer
            and tgt_res is not None
            and tgt_res.status == "resolved"
        ):
            final_answer = _normalise_deterministic_final_answer(
                question,
                plan,
                state,
                _stringify(tgt_res.value),
            )
            # UNGATED 2026-08-13. This is answer FORMATTING -- units, half-up
            # rounding under a years/round cue, integral-float rendering, and
            # reading a bare fraction as a percent under a "what percentage"
            # question. It used to sit behind the (never-enabled) champion
            # temporal layer, so none of this normalisation had ever executed:
            # monaco_448 emitted 0.2857142857 against gold 38.46 for exactly
            # that reason. Idempotent on already-clean strings and a no-op for
            # non-numeric values, so running it always is safe.
            final_answer = finish_value(
                final_answer,
                value_type=plan.target().value_type,
                question=question,
            )
            if stopped_reason == "max_turns":
                stopped_reason = "finished_late"

        traj.turns = orchestrator_turns + sum(
            t.get("turns", 0) for t in researcher_traces
        )
        traj.prompt_tokens = total_prompt_tokens
        traj.completion_tokens = total_completion_tokens
        traj.tool_calls = all_tool_calls
        traj.messages = all_messages
        # Phase-2 finalizer (covers empty-exit paths 3-7): rescue an otherwise-empty
        # answer from a real-evidence candidate in memory. G2 = only when empty;
        # G3 (real-evidence only) + never-sentinel enforced inside the helper.
        if not final_answer and self._invmem_enforce:
            if self.gsw_investigation_memory_finalshot:
                final_answer = self._invmem_final_shot(question, state)
                _fs = getattr(self._invmem, "final_shot", None) or {}
                traj.prompt_tokens += _fs.get("prompt_tokens", 0)
                traj.completion_tokens += _fs.get("completion_tokens", 0)
            if not final_answer:
                final_answer = self._invmem_finalize_answer(final_answer)
        traj.final_answer = final_answer
        traj.reasoning = final_answer
        traj.hidden_reasoning = "\n".join(reasoning_chunks)
        traj.extra["stopped_reason"] = stopped_reason
        traj.extra["researcher_traces"] = researcher_traces
        traj.extra["plan_json_versions"] = plan_json_versions
        traj.extra["plan_updates"] = plan_updates
        traj.extra["dispatches"] = dispatches
        traj.extra["validator_verdicts"] = validator_verdicts
        traj.extra["validator_retries_used"] = validator_retries
        traj.extra["validator_repairs_used"] = validator_repairs
        traj.extra["validator_guided_repairs_used"] = validator_guided_repairs
        traj.extra["orchestrator_mode"] = "llm"
        # Keep the FINAL plan in plan_json (replaces initial).
        traj.extra["plan_json"] = plan_dict
        traj.extra["topological_order"] = order
        traj.extra["topology_levels"] = levels
        self._attach_gsw_trace_metadata(traj, researcher_traces)

    def _try_validator_guided_repair(
        self,
        *,
        verdict: Any,
        proposed_answer: str,
        plan: GSWPlan,
        plan_dict: dict[str, Any],
        state: dict[str, BlankResult],
        order: list[str],
        levels: dict[str, int],
        question: str,
        dispatch_idx: int,
        constraint_outputs: set[str],
        guided_repairs_used: int,
        qgsw_state: Optional["QueryGSWState"] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Reopen one validator-flagged blank and redispatch it once.

        This is deliberately narrower than auto-repair: wrong-entity
        corrections are not submitted directly, but a verifier diagnosis
        can focus the next retrieval worker on the suspect blank.
        """
        if not self.synthesis_validator_guided_repair:
            return None
        if guided_repairs_used >= self.MAX_VALIDATOR_GUIDED_REPAIRS_PER_RUN:
            return None

        error_type = (getattr(verdict, "error_type", "") or "").strip().lower()
        if error_type not in self.VALIDATOR_GUIDED_REPAIR_ERROR_TYPES:
            return None
        confidence = (getattr(verdict, "confidence", "") or "").strip().lower()
        if confidence == "low":
            return None

        flagged_blank = (getattr(verdict, "flagged_blank", "") or "").strip()
        known_blank_ids = {e.id for e in plan.blank_entities()}
        if flagged_blank not in known_blank_ids:
            return None
        if flagged_blank in constraint_outputs:
            return None

        previous = state.get(flagged_blank)
        previous_value = _stringify(previous.value) if previous is not None else ""
        reset_ids = _downstream_blanks_for_repair(plan, flagged_blank)
        if not reset_ids:
            return None

        for bid in reset_ids:
            state[bid] = BlankResult(blank_id=bid, status="unknown")

        reason = (getattr(verdict, "reason", "") or "").strip()
        corrected = (
            getattr(verdict, "corrected_answer", None)
            or getattr(verdict, "suggested_correction", None)
            or ""
        )
        evidence_support = (getattr(verdict, "evidence_support", "") or "").strip()
        hint_parts = [
            "Synthesis validator rejected the previous final answer.",
            f"Previous final answer: {proposed_answer!r}.",
            f"Re-open flagged blank `{flagged_blank}`.",
        ]
        if previous_value:
            hint_parts.append(f"Previous committed value: {previous_value!r}.")
        if error_type:
            hint_parts.append(f"Validator error_type: {error_type}.")
        if reason:
            hint_parts.append(f"Validator reason: {reason}")
        if corrected:
            hint_parts.append(
                "Suggested/corrected answer is a retrieval hint only, "
                f"not evidence: {corrected!r}."
            )
        if evidence_support:
            hint_parts.append(f"Validator evidence note: {evidence_support}")
        hint_parts.append(
            "Retrieve fresh supporting evidence and update only the flagged "
            "blank. Do not repeat the previous value unless a cited chunk "
            "directly supports it."
        )
        hints = " ".join(part for part in hint_parts if part)

        traces = self._dispatch_subplan_parallel(
            plan=plan,
            plan_dict=plan_dict,
            state=state,
            order=order,
            levels=levels,
            question=question,
            blank_ids=[flagged_blank],
            hints=hints,
            level_tag=dispatch_idx,
            qgsw_state=qgsw_state,
        )

        resolved_map: dict[str, Any] = {}
        br = state.get(flagged_blank)
        if br is not None and br.status == "resolved":
            resolved_map[flagged_blank] = {
                "value": br.value,
                "evidence_chunk_ids": list(br.evidence_chunk_ids or []),
            }
        revision_requests: list[dict[str, Any]] = []
        for tr in traces:
            rr = tr.get("revision_request")
            if rr:
                revision_requests.append(
                    self._register_revision_request(
                        rr, question_id=str(self._active_question_id or "")
                    )
                )

        partial = flagged_blank not in resolved_map
        repair_info = {
            "strategy": "redispatch_flagged_blank",
            "flagged_blank": flagged_blank,
            "reset_blank_ids": reset_ids,
            "error_type": error_type,
            "confidence": confidence or None,
            "previous_value": previous_value,
        }
        action: dict[str, Any] = {
            "validator_guided_repair": repair_info,
            "validator_rejected": True,
            "dispatch_traces": traces,
            "blank_ids": [flagged_blank],
            "hints": hints,
            "partial": partial,
            "revision_requests": revision_requests,
        }
        result = {
            "ok": False,
            "error": (
                "validator_guided_repair: reopened "
                f"{flagged_blank!r}, reset downstream blanks {reset_ids}, "
                "and redispatched the flagged blank before the next "
                "orchestrator turn."
            ),
            "validator_guided_repair": repair_info,
            "resolved": resolved_map,
            "partial": partial,
        }
        if revision_requests:
            result["revision_requests"] = revision_requests
        return result, action

    def _register_revision_request(
        self, rr: dict[str, Any], question_id: str
    ) -> dict[str, Any]:
        """Register a researcher escalation and stamp it with a stable id.

        Returns the dict to surface to the orchestrator. Flag off -> returns
        ``rr`` unchanged and registers nothing: the tool schema doesn't
        expose ``plan_request_id`` when off either (see
        ``_orchestrator_tools``), so an unregistered id could never be cited
        anyway, and this keeps a flag-off run's `revision_requests` dicts
        byte-identical to before this feature existed. A malformed escalation
        (blank reason or hint) is passed through unchanged and un-stamped
        even with the flag on: it cannot support a contract, and dispatch
        must not break because a researcher escalated badly.
        """
        if not self.gsw_plan_request_contract:
            return rr
        from panini_dr.adapters.ours._plan_request import (
            plan_request_from_revision,
        )

        pr = plan_request_from_revision(
            rr, question_id=question_id, seq=len(self._plan_requests))
        if pr is None:
            return rr
        self._plan_requests[pr.request_id] = pr
        # "pending" -- true only at registration; do NOT call this field
        # "consumed" (it was, and the name was a defect: this dict is never
        # revisited after registration, so a stamped `"consumed": False`
        # would read as "definitely not consumed" even for a request an
        # update later cited). The real, up-to-date consumed/not split lives
        # in `_plan_request_summary()`, joined against `_consumed_request_ids`
        # at readout.
        return {**rr, "request_id": pr.request_id, "pending": True}

    def _plan_request_summary(self) -> list[dict[str, Any]]:
        """Every escalation this question raised and whether an update cited it.

        Measured on FRAMES-824: 76 questions escalated, 63 were updated, so 13
        researcher requests were dropped silently. This makes that visible.
        """
        return [
            {"request_id": rid, "blank_id": pr.blank_id,
             "consumed": rid in self._consumed_request_ids}
            for rid, pr in self._plan_requests.items()
        ]

    # ------------------------------------------------------------------
    # Orchestrator tool dispatch
    # ------------------------------------------------------------------

    #: Planner calls per question. Bounds the cost of
    #: an orchestrator that keeps re-planning instead of dispatching.
    MAX_PLANNER_TOOL_CALLS: ClassVar[int] = 3

    def _tool_call_planner(
        self, question: str, args: dict[str, Any], action: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Emit a plan on request and hand back the plan and its tags.

        A plan is written before any retrieval runs. The deterministic
        `plan_audit_tags` computed during emission (see `_planner_emit.py`)
        are the quality signal carried back; there is no LLM auditor call
        here.

        Fail-open: a planner error returns an error result and the orchestrator
        may try again, up to the cap.
        """
        # No directives, by design. Sanitising them protected the planner but
        # not the orchestrator: in live mode its own tool call is appended to the
        # conversation verbatim, so a recalled name it wrote stayed in its
        # context for the rest of the question and could resurface in a later
        # request_plan_update or dispatch hint. 75% of plan hallucinations were
        # named in a directive first, and a contaminated memory cannot be
        # un-contaminated downstream — so the channel is gone.
        rec: dict[str, Any] = {}
        if self._planner_tool_calls >= self.MAX_PLANNER_TOOL_CALLS:
            rec["error"] = "cap reached"
            self._planner_tool_log.append(rec)
            return ({"ok": False,
                     "error": (f"planner call cap ({self.MAX_PLANNER_TOOL_CALLS}) "
                               "reached; work with the plan you have.")}, action)
        self._planner_tool_calls += 1

        emit_kwargs = {
            "max_tokens": self.ctx.max_completion_tokens,
            "enable_repair": True,
            "llm_seed": self.llm_seed,
            "blank_qualifier": self.gsw_blank_qualifier,
            "blank_hints": self.gsw_blank_hints,
        }
        try:
            plan, meta = self._emit_plan_with_optional_planner_route(
                question, emit_kwargs)
        except Exception as exc:  # noqa: BLE001 — the orchestrator may retry
            rec["error"] = str(exc)[:200]
            self._planner_tool_log.append(rec)
            return ({"ok": False, "error": f"planner failed: {str(exc)[:200]}"},
                    action)

        # The repair loop (gated on `self.gsw_plan_verifier`) already ran
        # inside `_emit_plan_with_optional_planner_route`, above -- it's the
        # shared funnel the champion path uses too. Copy its transcript onto
        # `rec` so existing consumers of the planner-tool log keep the same
        # shape (`rec["plan_repair"]`); when the flag is off the key is
        # simply absent, matching the old byte-identical-by-default wiring.
        if "plan_repair" in meta.extras:
            rec["plan_repair"] = meta.extras["plan_repair"]

        # The planner's own words, so an inspector can show WHY it emitted this
        # plan and not merely what it emitted.
        rec["kind"] = "create"
        rec["planner_raw"] = (meta.raw_response or "")[:6000]
        rec["planner_reasoning"] = (meta.raw_reasoning or "")[:6000]
        rec["planner_repair_used"] = bool(meta.repair_used)

        audit = {
            "issues": [],
            "tags": list(meta.extras.get("plan_audit_tags") or []),
        }
        # The orchestrator's next decision is WHICH blanks to dispatch together,
        # which is a question about dependency levels — champion renders exactly
        # this into its prompt every turn. Handing back a bare entities/vps/
        # constraints dict left it to infer the order from raw JSON.
        try:
            _order = topological_sort_blanks(plan)
            _levels = _compute_levels(build_dependency_graph(plan))
            brief = render_plan_brief(plan.model_dump(), _order, _levels)
        except Exception:  # noqa: BLE001 — a brief is a nicety, the plan is not
            brief = ""
        rec["brief"] = brief
        rec["audit"] = audit
        rec["plan"] = plan.model_dump()
        self._planner_tool_log.append(rec)

        action["new_plan"] = plan
        action["planner_tool_issues"] = []
        action["planner_tool_mechanical"] = []
        return ({"ok": True, "plan": brief or plan.model_dump(), "audit": audit},
                action)

    def _run_orchestrator_tool(
        self,
        *,
        name: str,
        args: dict[str, Any],
        plan: GSWPlan,
        plan_dict: dict[str, Any],
        state: dict[str, BlankResult],
        order: list[str],
        levels: dict[str, int],
        question: str,
        dispatch_idx: int,
        constraint_outputs: set[str],
        plan_update_count: int = 0,
        last_plan_update_summary: str = "",
        escalations_per_blank: Optional[dict[str, int]] = None,
        validator_retries: int = 0,
        validator_guided_repairs: int = 0,
        qgsw_state: Optional["QueryGSWState"] = None,
        workspace: Optional["GSWWorkspace"] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute one orchestrator-tool call. Returns ``(result, action)``
        where ``result`` is what the orchestrator sees as the tool
        response, and ``action`` describes mutable side effects that
        the outer loop applies (dispatch traces, new plan, final
        answer)."""
        action: dict[str, Any] = {}

        if name == "call_planner":
            if dispatch_idx > 0:
                # Once evidence exists, re-planning belongs to
                # `request_plan_update`: it emits diff ops against the CURRENT
                # plan, carries resolved blanks over by id, and is driven by the
                # evidence the orchestrator supplies. `call_planner` re-emits
                # from the question alone, so mid-run it would rename blanks and
                # silently throw away retrieval that already succeeded.
                return (
                    {"ok": False,
                     "error": ("call_planner creates the initial plan and is "
                               "unavailable once researchers have run — it "
                               "re-emits from the question alone and would "
                               "discard resolved blanks. Use "
                               "request_plan_update, which revises the current "
                               "plan from your evidence and keeps resolved "
                               "values.")},
                    action,
                )
            return self._tool_call_planner(question, args, action)

        if name == "get_state":
            # W2: read the state view THROUGH the workspace. Returns the same
            # dict as the prior inline build (output-identical); the fallback
            # keeps direct-call callers (unit tests) working when no workspace
            # is threaded in.
            if workspace is not None:
                return workspace.state_view(constraint_outputs), action
            view: dict[str, Any] = {"blanks": []}
            for bid, br in state.items():
                view["blanks"].append(
                    {
                        "blank_id": bid,
                        "status": br.status,
                        "value": br.value,
                        "is_target": bid == plan.target().id,
                        "is_constraint_output": bid in constraint_outputs,
                    }
                )
            return view, action

        if name == "dispatch_subplan":
            blank_ids = args.get("blank_ids") or []
            hints = args.get("hints") or ""
            # Phase-3.2: when a plan revision has just happened, prepend
            # a summary so the new researcher knows the prior context
            # (avoids the "researcher rediscovers the same gap" loop).
            if last_plan_update_summary:
                hints = (
                    f"[after plan revision: {last_plan_update_summary}] "
                    + hints
                ).strip()
            if not isinstance(blank_ids, list) or not blank_ids:
                return (
                    {"ok": False, "error": "blank_ids must be a non-empty list"},
                    action,
                )
            # Duplicate blank_ids are NOT collapsed: a dispatch gets one
            # researcher per list entry, duplicates included.
            # Validate each id.
            known_ids = {e.id for e in plan.blank_entities()}
            unknown = [b for b in blank_ids if b not in known_ids]
            if unknown:
                return (
                    {
                        "ok": False,
                        "error": f"unknown blank_ids: {unknown}",
                        "known_blank_ids": sorted(known_ids),
                    },
                    action,
                )
            already = [
                b for b in blank_ids
                if state.get(b) is not None and state[b].status == "resolved"
            ]
            is_output = [b for b in blank_ids if b in constraint_outputs]
            if already:
                return (
                    {
                        "ok": False,
                        "error": (
                            f"blank_ids already resolved: {already}. Use "
                            "`request_plan_update` if their values are wrong."
                        ),
                    },
                    action,
                )
            if is_output:
                return (
                    {
                        "ok": False,
                        "error": (
                            f"blank_ids are constraint outputs (auto-computed): "
                            f"{is_output}. Dispatch their INPUTS instead."
                        ),
                    },
                    action,
                )
            # Dependency order is guidance in the orchestrator prompt, not an
            # enforced dispatch gate (v8 compatibility): a batch may contain a
            # blank whose upstream blank is unresolved.
            if len(blank_ids) > self.MAX_PARALLEL_RESEARCHERS:
                return (
                    {
                        "ok": False,
                        "error": (
                            f"too many blanks in one dispatch (max "
                            f"{self.MAX_PARALLEL_RESEARCHERS}); got "
                            f"{len(blank_ids)}."
                        ),
                    },
                    action,
                )

            # Phase-3.2 follow-up: reject re-dispatch of blanks that
            # have already triggered MAX_ESCALATIONS_PER_BLANK escalations.
            # Prevents the spam-dispatch-with-permuted-hints loop.
            esc = escalations_per_blank or {}
            burned = [
                b for b in blank_ids
                if esc.get(b, 0) >= self.MAX_ESCALATIONS_PER_BLANK
            ]
            if burned:
                return (
                    {
                        "ok": False,
                        "error": (
                            f"per-blank dispatch cap reached for "
                            f"{burned}: each has triggered "
                            f"{self.MAX_ESCALATIONS_PER_BLANK} researcher "
                            "escalations already. Re-dispatching with "
                            "different hints will not help — the corpus "
                            "lacks this data. Either call "
                            "`request_plan_update` to revise the plan "
                            "(if the cap there hasn't been reached), or "
                            "`submit_answer` (empty answer is acceptable)."
                        ),
                        "escalations_per_blank": {b: esc.get(b, 0) for b in blank_ids},
                    },
                    action,
                )

            traces = self._dispatch_subplan_parallel(
                plan=plan, plan_dict=plan_dict,
                state=state, order=order, levels=levels,
                question=question,
                blank_ids=blank_ids,
                hints=hints,
                level_tag=dispatch_idx,
                qgsw_state=qgsw_state,
                workspace=workspace,
            )
            # Build orchestrator-visible summary.
            resolved_map: dict[str, Any] = {}
            any_partial = False
            for bid in blank_ids:
                br = state.get(bid)
                if br is not None and br.status == "resolved":
                    resolved_map[bid] = {
                        "value": br.value,
                        "evidence_chunk_ids": list(br.evidence_chunk_ids or []),
                    }
                else:
                    any_partial = True

            # Collect researcher escalations (Phase-3): a researcher may
            # signal that the plan is missing a step by calling
            # suggest_plan_revision. Surface these to the orchestrator.
            revision_requests: list[dict[str, Any]] = []
            for tr in traces:
                rr = tr.get("revision_request")
                if rr:
                    revision_requests.append(
                        self._register_revision_request(
                            rr, question_id=str(self._active_question_id or "")
                        )
                    )

            result = {
                "ok": True,
                "resolved": resolved_map,
                "partial": any_partial,
            }
            if revision_requests:
                result["revision_requests"] = revision_requests
            action["dispatch_traces"] = traces
            action["blank_ids"] = blank_ids
            action["hints"] = hints
            action["partial"] = any_partial
            action["revision_requests"] = revision_requests
            return result, action

        if name == "request_plan_update":
            reason = args.get("reason", "") or ""
            evidence = args.get("evidence", "") or ""
            # Plan-request contract: look up the escalation this update claims
            # to answer (None when the orchestrator omitted plan_request_id, or
            # cited an id this question never registered). Looked up here --
            # before the cap check -- so every plan_updates record below
            # (accepted or rejected) can name which escalation it answered.
            raw_plan_request_id = args.get("plan_request_id")
            pr = self._plan_requests.get(str(raw_plan_request_id or ""))
            action["plan_request_id"] = pr.request_id if pr is not None else None
            # Final-review fix wave, Finding 4: a resolved id of None is
            # ambiguous -- "the orchestrator omitted plan_request_id" and
            # "the orchestrator cited an id this question never registered"
            # (hallucinated, or crossed questions) both collapse to the same
            # None. Carry the raw argument alongside the resolved one so a
            # lookup MISS (raw truthy, resolved None) is visible on the
            # record, distinct from a genuine omission (raw falsy).
            action["plan_request_id_raw"] = raw_plan_request_id
            # Phase-3.2 cap: prevent runaway loops where the corpus
            # genuinely lacks the requested data and every revision
            # leads back to the same gap.
            if plan_update_count >= self.MAX_PLAN_UPDATES_PER_RUN:
                action["plan_update_rejected"] = True
                action["reason"] = reason
                action["evidence"] = evidence
                action["error"] = (
                    f"max plan updates ({self.MAX_PLAN_UPDATES_PER_RUN}) "
                    f"reached for this question"
                )
                return (
                    {
                        "ok": False,
                        "error": (
                            f"plan_update_cap_reached: already used "
                            f"{plan_update_count}/{self.MAX_PLAN_UPDATES_PER_RUN} "
                            "plan-updater calls. Further revisions are unlikely "
                            "to help — the corpus may genuinely lack this data. "
                            "Either submit_answer with a best-effort answer "
                            "(empty string is acceptable) or dispatch a final "
                            "researcher with explicit hints to commit a "
                            "best-guess from chunks already retrieved."
                        ),
                        "plan_update_count": plan_update_count,
                        "max_plan_updates": self.MAX_PLAN_UPDATES_PER_RUN,
                    },
                    action,
                )
            try:
                old_state = {
                    bid: copy.copy(br) for bid, br in state.items()
                }
                new_plan, diff, _meta = update_plan(
                    old_plan=plan,
                    state=state,
                    question=question,
                    reason=reason,
                    evidence=evidence,
                    llm_client=self._metered(self.llm, "planner"),
                    max_tokens=self.ctx.max_completion_tokens,
                    enable_repair=True,
                    llm_seed=self.llm_seed,
                    blank_hints=self.gsw_blank_hints,
                    plan_request=pr,
                    contract_enabled=self.gsw_plan_request_contract,
                    # Carry-forward: once a contract exists for this question it
                    # governs EVERY later update on it, not just the one that
                    # cited a plan_request_id. Without this, q456's second
                    # (uncited) update ran unguarded and undid the correct plan
                    # the first, gated update had just produced.
                    carried_contract=self._question_contract,
                )
                if pr is not None:
                    # "Consumed" == an update actually cited this escalation
                    # AND update_plan succeeded -- recorded in a separate set,
                    # never by mutating `pr` (PlanRequest is frozen).
                    self._consumed_request_ids.add(pr.request_id)
            except Exception as exc:  # noqa: BLE001
                action["plan_update_rejected"] = True
                action["reason"] = reason
                action["evidence"] = evidence
                action["error"] = str(exc)
                return (
                    {
                        "ok": False,
                        "error": f"plan_update_rejected: {exc}",
                    },
                    action,
                )
            # W6: workspace preserve/invalidate safety net (gated; TRACE-ONLY diag).
            # 'off' skips (byte-identical); 'log_only' records; 'enforce' resets
            # identity-changed preserved blanks + their resolved dependents.
            if self.gsw_plan_update_guard != "off" and workspace is not None:
                _w6 = workspace.invalidate_changed_blanks(
                    old_plan=plan, new_plan=new_plan,
                    preserved_ids=list(diff.preserved_ids),
                    mode=self.gsw_plan_update_guard,
                )
                action["plan_update_guard"] = _w6
            action["new_plan"] = new_plan
            action["old_plan"] = plan
            action["old_state"] = old_state
            # The updater is its own model with its own reasoning, and on the
            # pope question it answered "drop the recalled entity" with an
            # add_blank. Without this, why it chose that op is unrecoverable.
            action["updater_reasoning"] = (_meta.raw_reasoning or "")[:6000]
            action["updater_raw"] = (_meta.raw_response or "")[:6000]
            action["reason"] = reason
            action["evidence"] = evidence
            action["diff_summary"] = diff.summary
            action["preserved_ids"] = diff.preserved_ids
            action["added_ids"] = diff.added_ids
            action["dropped_ids"] = diff.dropped_ids
            action["invalidated_ids"] = diff.invalidated_ids
            # Phase-1 investigation memory: a plan update changed blank identities
            # -> retire their actors (kept in history, excluded from ranking) and
            # log the unfreeze trigger for the touched blanks.
            if self._invmem is not None:
                self._invmem.retire(
                    self._invmem_round,
                    list(diff.invalidated_ids or []) + list(diff.dropped_ids or []))
            action["op_summary"] = diff.op_summary
            action["attempts"] = diff.attempts
            action["prompt_tokens"] = _meta.prompt_tokens
            action["completion_tokens"] = _meta.completion_tokens
            # Final-review fix wave, Finding 2: `diff.contract` /
            # `diff.contract_satisfied` were computed and thrown away here --
            # they reached neither the `plan_updates` records nor
            # `traj.extra`, making the spec's "contract satisfaction rate
            # logged per update" success criterion unmeasurable from a run.
            # `contract_active` also carried explicitly (not inferred from
            # `contract == []`), so an analyst can tell "flag off" apart from
            # "contract active but the author returned an empty contract".
            action["contract"] = diff.contract
            action["contract_satisfied"] = diff.contract_satisfied
            action["contract_active"] = diff.contract_active
            # Fail-open marker: the contract was active but could not be
            # honoured inside the attempt budget, and the structurally-valid
            # op was kept rather than losing the replan (observed live on
            # q351). Without this a trace cannot tell an honoured contract
            # from an abandoned one.
            action["contract_unmet"] = diff.contract_unmet
            # Remember it for the next update on this question (see the
            # carry-forward comment at the update_plan call above).
            if diff.contract:
                self._question_contract = diff.contract
            action["contract_identity_satisfied"] = diff.contract_identity_satisfied
            return (
                {
                    "ok": True,
                    "diff_summary": diff.summary,
                    "preserved_ids": diff.preserved_ids,
                    "added_ids": diff.added_ids,
                    "dropped_ids": diff.dropped_ids,
                    "invalidated_ids": diff.invalidated_ids,
                    "op_summary": diff.op_summary,
                    "attempts": diff.attempts,
                },
                action,
            )

        if name == "submit_answer":
            answer = str(args.get("answer") or "").strip()
            target_id = plan.target().id
            tgt_res = state.get(target_id)
            target_resolved = tgt_res is not None and tgt_res.status == "resolved"
            if not target_resolved:
                # Phase-3.2: when the plan-update cap is reached AND the
                # target is still unresolved, the orchestrator is allowed
                # to give up. Stamps a distinct stopped_reason so the
                # trajectory is visibly "we tried, corpus lacks the data"
                # rather than "we never finished".
                if plan_update_count >= self.MAX_PLAN_UPDATES_PER_RUN:
                    action["final_answer"] = answer  # may be ""
                    action["stopped_reason"] = "give_up_unanswerable"
                    return (
                        {
                            "ok": True,
                            "note": (
                                f"target unresolved AND plan-update cap "
                                f"({self.MAX_PLAN_UPDATES_PER_RUN}) reached; "
                                "submitting best-effort answer."
                            ),
                        },
                        action,
                    )
                tgt_res = state.get(target_id)
                target_resolved = (
                    tgt_res is not None and tgt_res.status == "resolved"
                )
                if not target_resolved:
                    return (
                        {
                            "ok": False,
                            "error": (
                                f"cannot submit_answer: target blank "
                                f"{target_id!r} is not status=resolved "
                                f"(status={tgt_res.status if tgt_res else 'missing'!r}). "
                                "Dispatch or request_plan_update first."
                            ),
                        },
                        action,
                    )
            # Deterministic readout (2026-08-29): the model's submit_answer
            # TRANSCRIPTION drops the '; ' separators of concat-assembled
            # targets in a large fraction of cases (measured 9/13, 4/11,
            # 5/10 across three CF-150 runs). When the retype is
            # CONTENT-EQUIVALENT to the gate-passed worksheet value —
            # differing only in separators/punctuation/case — the worksheet
            # value wins. A content-DIFFERENT submission is kept: the
            # synthesis-validator reject/repair flow legitimately submits
            # corrections that beat a flawed worksheet value.
            if target_resolved and tgt_res.value not in (None, ""):
                _wv = _stringify(tgt_res.value)
                _canon = lambda x: re.sub(r"[^0-9a-z]+", "", str(x).casefold())
                if _canon(answer) == _canon(_wv):
                    # Emit in the answer-format contract (comma-separated):
                    # the worksheet's '; ' joins become ', ', matching what
                    # the synthesis validator's format repair produces.
                    answer = _wv.replace("; ", ", ")
            final_pred = _normalise_deterministic_final_answer(
                question,
                plan,
                state,
                answer or _stringify(tgt_res.value),
            )
            # UNGATED 2026-08-13 -- same seam as the readouts above.
            final_pred = finish_value(
                final_pred,
                value_type=plan.target().value_type,
                question=question,
            )

            # Phase-3.7: optional synthesis validator. In log_only mode
            # it records a verdict; in reject mode, one wrong verdict can
            # reject the submit and surface feedback to the next turn. In
            # repair mode, high-confidence local final-answer fixes can
            # replace the submitted answer immediately; unsafe fixes fall
            # back to the reject path.
            validator_mode = self.synthesis_validator_mode
            should_validate = (
                final_pred.strip()
                and validator_mode in {"log_only", "reject", "repair"}
                and (
                    validator_mode == "log_only"
                    or validator_retries < self.MAX_VALIDATOR_RETRIES
                )
            )
            if should_validate:
                from panini_dr.adapters.ours._synthesis_validator import (
                    auto_repair_candidate,
                    build_rejection_message,
                    validate_synthesis,
                )

                verdict = validate_synthesis(
                    question=question,
                    plan=plan,
                    state=state,
                    proposed_answer=final_pred,
                    corpus=self.corpus,
                    llm=self._metered(self.llm, "synthesis_validator"),
                    llm_seed=self.llm_seed,
                    sink=self._checker_sink,
                )
                action["validator_verdict"] = {
                    "question_id": getattr(self, "_active_question_id", None),
                    "verdict": verdict.verdict,
                    "reason": verdict.reason,
                    "flagged_blank": verdict.flagged_blank,
                    "suggested_correction": verdict.suggested_correction,
                    "corrected_answer": verdict.corrected_answer,
                    "confidence": verdict.confidence,
                    "error_type": verdict.error_type,
                    "evidence_support": verdict.evidence_support,
                    "proposed_answer": final_pred,
                    "mode": validator_mode,
                    "prompt": getattr(verdict, "prompt", ""),
                    "raw_output": getattr(verdict, "raw_output", ""),
                    "reasoning": getattr(verdict, "reasoning", ""),
                }
                if validator_mode == "repair" and verdict.verdict == "wrong":
                    repaired, repair_reason = auto_repair_candidate(
                        verdict, final_pred
                    )
                    action["validator_verdict"]["auto_repair_reason"] = (
                        repair_reason
                    )
                    if repaired is not None:
                        action["validator_verdict"]["auto_repair_accepted"] = True
                        action["validator_repaired"] = True
                        action["final_answer"] = repaired
                        action["stopped_reason"] = "finished_validator_repaired"
                        return (
                            {
                                "ok": True,
                                "validator_repaired": True,
                                "answer": repaired,
                            },
                            action,
                        )
                    guided = self._try_validator_guided_repair(
                        verdict=verdict,
                        proposed_answer=final_pred,
                        plan=plan,
                        plan_dict=plan_dict,
                        state=state,
                        order=order,
                        levels=levels,
                        question=question,
                        dispatch_idx=dispatch_idx,
                        constraint_outputs=constraint_outputs,
                        guided_repairs_used=validator_guided_repairs,
                        qgsw_state=qgsw_state,
                    )
                    if guided is not None:
                        result, guided_action = guided
                        action.update(guided_action)
                        return result, action

                if validator_mode in {"reject", "repair"} and verdict.verdict == "wrong":
                    rejection = build_rejection_message(verdict, final_pred)
                    action["validator_rejected"] = True
                    return (
                        {"ok": False, "error": rejection},
                        action,
                    )

            action["final_answer"] = final_pred
            action["stopped_reason"] = "finished"
            return {"ok": True}, action

        return (
            {"ok": False, "error": f"unknown tool {name!r}"},
            action,
        )
