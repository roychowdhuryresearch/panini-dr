"""Fakes for the CPU-only end-to-end smoke test.

Only the external services are faked; every line of harness code runs:

* ``FakeLLMService`` replaces ``LLMClient.chat`` (the single choke point every
  role goes through). It answers by ROLE, which it reads off the call stack:
  every adapter call passes through ``_MeteredLLM.chat`` carrying ``role``; the
  resolver/researcher/reader frames carry the blank or article being worked on.
  It plays a competent agent on a toy corpus about fictional entities: the
  planner returns a real GSW plan, the orchestrator calls ``call_planner`` then
  dispatches and submits, the resolver/researcher/reader extract values from the
  chunks they were actually shown, and the checkers approve.
* ``FakeDenseRetriever`` stands in for the Qwen3-Embedding-8B index (token
  overlap scores), so ``retriever_type=hybrid`` builds real BM25 + real RRF.
* ``fake_rerank_post`` stands in for the jina-reranker HTTP endpoint, so the real
  ``RerankedRetriever``/``JinaReranker`` code runs.

All text is synthetic.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from panini_dr.models.llm_client import LLMResponse
from panini_dr.retrieval.bm25 import RetrievedChunk

# ---------------------------------------------------------------------------
# Toy corpus and questions
# ---------------------------------------------------------------------------

TOY_DOCS: dict[str, str] = {
    "Zorvan Klett": (
        "Zorvan Klett\n\n[INFOBOX]\nname | Zorvan Klett\noccupation | inventor\n"
        "birth_place | Veltravia\n\n"
        "Zorvan Klett was an inventor of signal lamps. Zorvan Klett was born in Veltravia "
        "in 1871 and later moved to the coast.\n\n## Career\n"
        "He designed the Klett lamp, a beacon used by harbour pilots."
    ),
    "Veltravia": (
        "Veltravia\n\nVeltravia is a small island country in the Amber Sea. "
        "The capital of Veltravia is Morrowgate.\n\n## Economy\n"
        "Fishing and lamp making dominate the economy of Veltravia."
    ),
    "Morrowgate": (
        "Morrowgate\n\nMorrowgate is the capital city of Veltravia. It is known for its "
        "lighthouse district."
    ),
    "Ilsa Brenn": (
        "Ilsa Brenn\n\n[INFOBOX]\nname | Ilsa Brenn\noccupation | painter\n"
        "birth_place | Ostrevan\n\n"
        "Ilsa Brenn is a painter of river landscapes. Ilsa Brenn was born in Ostrevan "
        "and trained in a northern academy.\n\n## Works\n"
        "Her best known canvas is Grey Ford at Dawn."
    ),
    "Ostrevan": (
        "Ostrevan\n\nOstrevan is a landlocked country of high valleys. "
        "The capital of Ostrevan is Lindqvar.\n\n## Geography\n"
        "Three rivers cross Ostrevan from east to west."
    ),
    "Lindqvar": (
        "Lindqvar\n\nLindqvar is the capital of Ostrevan and its largest city."
    ),
    "Pell Arkady": (
        "Pell Arkady\n\nPell Arkady is a composer. Pell Arkady was born in Quessa."
    ),
    "Quessa": "Quessa\n\nQuessa is a coastal country. The capital of Quessa is Harrowmere.",
    "Harrowmere": "Harrowmere\n\nHarrowmere is a port city and the capital of Quessa.",
    "Klett lamp": (
        "Klett lamp\n\nThe Klett lamp is a beacon designed by Zorvan Klett for harbour pilots."
    ),
    "Grey Ford at Dawn": (
        "Grey Ford at Dawn\n\nGrey Ford at Dawn is a landscape painting by Ilsa Brenn."
    ),
    "Amber Sea": "Amber Sea\n\nThe Amber Sea surrounds several island countries, including Veltravia.",
    "Northern Academy of Art": (
        "Northern Academy of Art\n\nThe Northern Academy of Art trains landscape painters."
    ),
    "Harbour pilot": "Harbour pilot\n\nA harbour pilot guides ships into port using beacons.",
    "Signal lamp": "Signal lamp\n\nA signal lamp is a visual signalling device.",
    "River landscape": "River landscape\n\nRiver landscapes are a genre of painting.",
    "Lighthouse district": "Lighthouse district\n\nThe lighthouse district of Morrowgate has twelve towers.",
    "High valleys": "High valleys\n\nHigh valleys are common in landlocked mountain countries.",
    "Port cities": "Port cities\n\nPort cities grow around natural harbours.",
    "Island countries": "Island countries\n\nIsland countries have no land borders.",
}


def _plan_one_hop() -> dict[str, Any]:
    return {
        "entities": [
            {"id": "e_klett", "kind": "filled", "name": "Zorvan Klett", "role": "subject"},
            {
                "id": "t_country1", "kind": "blank", "role": "target", "value_type": "entity",
                "is_target": True,
                "retrieval_hint": "Zorvan Klett born in which country",
                "verification_claim": "Zorvan Klett was born in {t_country1}.",
            },
        ],
        "verb_phrases": [
            {"id": "vp_born1", "phrase": "born_in", "subject_id": "e_klett", "object_id": "t_country1"},
        ],
        "constraints": [],
    }


def _plan_two_hop() -> dict[str, Any]:
    return {
        "entities": [
            {"id": "e_brenn", "kind": "filled", "name": "Ilsa Brenn", "role": "subject"},
            {
                "id": "b_country2", "kind": "blank", "role": "bridge-entity", "value_type": "entity",
                "retrieval_hint": "Ilsa Brenn born in which country",
                "verification_claim": "Ilsa Brenn was born in {b_country2}.",
            },
            {
                "id": "t_capital2", "kind": "blank", "role": "target", "value_type": "entity",
                "is_target": True,
                "retrieval_hint": "capital of {b_country2}",
                "verification_claim": "The capital of {b_country2} is {t_capital2}.",
            },
        ],
        "verb_phrases": [
            {"id": "vp_born2", "phrase": "born_in", "subject_id": "e_brenn", "object_id": "b_country2"},
            {"id": "vp_cap2", "phrase": "has_capital", "subject_id": "b_country2", "object_id": "t_capital2"},
        ],
        "constraints": [],
    }


@dataclass
class ToyQuestion:
    id: str
    question: str
    answer: str
    plan: dict[str, Any]
    #: blank id -> (expected value, research query, reader question)
    facts: dict[str, tuple[str, str, str]]
    #: dispatch order, one blank per dispatch_subplan call
    dispatch: list[str]
    #: blanks whose resolver pre-pass is made to fail (-> ReAct researcher + reader)
    researcher_blanks: set[str] = field(default_factory=set)


TOY_QUESTIONS: list[ToyQuestion] = [
    ToyQuestion(
        id="toy1",
        question="In which country was the inventor Zorvan Klett born?",
        answer="Veltravia",
        plan=_plan_one_hop(),
        facts={"t_country1": ("Veltravia", "Zorvan Klett born", "Where was Zorvan Klett born?")},
        dispatch=["t_country1"],
    ),
    ToyQuestion(
        id="toy2",
        question="What is the capital of the country where the painter Ilsa Brenn was born?",
        answer="Lindqvar",
        plan=_plan_two_hop(),
        facts={
            "b_country2": ("Ostrevan", "Ilsa Brenn born", "Where was Ilsa Brenn born?"),
            "t_capital2": ("Lindqvar", "capital of Ostrevan", "What is the capital of Ostrevan?"),
        },
        dispatch=["b_country2", "t_capital2"],
        researcher_blanks={"b_country2"},
    ),
]


def write_toy_corpus(path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(TOY_DOCS))
    return path


def write_toy_questions_jsonl(path: str | Path) -> Path:
    path = Path(path)
    with path.open("w") as f:
        for q in TOY_QUESTIONS:
            f.write(json.dumps({"id": q.id, "question": q.question, "answer": q.answer}) + "\n")
    return path


def write_toy_docs_jsonl(path: str | Path) -> Path:
    path = Path(path)
    with path.open("w") as f:
        for i, (title, text) in enumerate(TOY_DOCS.items()):
            f.write(json.dumps({"id": f"d{i}", "title": title, "text": text}) + "\n")
    return path


def toy_frames_questions():
    """The toy questions as harness question objects (FRAMES-shaped)."""
    from panini_dr.eval.frames_dataset import FramesQuestion

    return [
        FramesQuestion(id=q.id, question=q.question, answer=q.answer) for q in TOY_QUESTIONS
    ]


# ---------------------------------------------------------------------------
# Retrieval fakes
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


class FakeDenseRetriever:
    """Stand-in for DenseRetriever: deterministic token-overlap scores, CPU only."""

    instances: list["FakeDenseRetriever"] = []

    def __init__(self, corpus, **kwargs: Any) -> None:
        self.corpus = corpus
        self.kwargs = kwargs
        self.n_searches = 0
        FakeDenseRetriever.instances.append(self)

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        self.n_searches += 1
        q = _tokens(query)
        scored = []
        for i, ch in enumerate(self.corpus.chunks):
            overlap = len(q & _tokens(ch.text))
            if overlap:
                scored.append((-overlap, i, ch))
        scored.sort()
        return [RetrievedChunk(chunk=ch, score=float(-s)) for s, _, ch in scored[:top_k]]


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeReranker:
    """Replaces ``httpx.post`` in the rerank module (jina listwise API shape)."""

    def __init__(self) -> None:
        self.calls = 0
        self.fail = False

    def post(self, url: str, json: dict[str, Any] | None = None, **kwargs: Any):  # noqa: A002
        self.calls += 1
        if self.fail:
            raise ConnectionError("fake reranker is down")
        docs = (json or {}).get("documents") or []
        q = _tokens((json or {}).get("query") or "")
        order = sorted(range(len(docs)), key=lambda i: (-len(q & _tokens(docs[i])), i))
        return _FakeHTTPResponse({"results": [{"index": i, "relevance_score": 1.0} for i in order]})


# ---------------------------------------------------------------------------
# The fake LLM
# ---------------------------------------------------------------------------


def _resp(text: str = "", tool_calls: list[dict[str, Any]] | None = None) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=tool_calls or [],
        prompt_tokens=100,
        completion_tokens=20,
        finish_reason="tool_calls" if tool_calls else "stop",
    )


def _frames():
    f = sys._getframe(2)
    while f is not None:
        yield f
        f = f.f_back


def _sentence_with(text: str, value: str) -> str | None:
    for sent in re.split(r"(?<=[.!?])\s+|\n", text):
        if value in sent and len(sent.strip()) > len(value):
            return sent.strip()
    return None


class FakeLLMService:
    """Role-routed scripted LLM. Install with ``install(monkeypatch)``."""

    #: What ``GET {base_url}/models`` lists (``panini-dr run``'s LLM preflight).
    served_models = ("openai/gpt-oss-120b",)

    def __init__(self, questions: list[ToyQuestion] | None = None) -> None:
        self.questions = list(questions or TOY_QUESTIONS)
        self.calls: list[dict[str, Any]] = []
        self.preflight_calls: list[str] = []
        self.down = False
        self._lock = threading.Lock()
        self._counter = 0

    # -- plumbing ---------------------------------------------------------

    def install(self, monkeypatch) -> "FakeLLMService":
        from panini_dr.models.llm_client import LLMClient

        service = self

        def chat(client_self, messages=None, *args: Any, **kwargs: Any) -> LLMResponse:
            if messages is None:
                messages = kwargs.pop("messages", None)
            return service.chat(list(messages or []), **kwargs)

        monkeypatch.setattr(LLMClient, "chat", chat)
        return self

    def get(self, url: str, **kwargs: Any):
        """Replaces ``httpx.get``: the server's ``/v1/models`` (the run's LLM preflight)."""
        import httpx

        self.preflight_calls.append(str(url))
        if self.down:
            raise httpx.ConnectError("fake LLM server is down")
        if not str(url).endswith("/models"):
            return httpx.Response(404, request=httpx.Request("GET", str(url)))
        return httpx.Response(200, json={"object": "list", "data": [
            {"id": m, "object": "model"} for m in self.served_models]},
            request=httpx.Request("GET", str(url)))

    def _id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"call_{self._counter}"

    def _tc(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"id": self._id(), "name": name, "arguments": json.dumps(args)}

    @staticmethod
    def _role() -> tuple[str, dict[str, Any]]:
        role = "unknown"
        ctx: dict[str, Any] = {}
        for f in _frames():
            obj = f.f_locals.get("self")
            name = type(obj).__name__ if obj is not None else ""
            if name == "_MeteredLLM" and role == "unknown":
                role = obj.role
            if name == "TypedBlankResolver" and "blank_id" in f.f_locals:
                ctx.setdefault("resolver", f.f_locals)
            if name == "ReActResearcher" and "researcher" not in ctx:
                ctx["researcher"] = obj
            if f.f_code.co_name == "read_article_briefed":
                ctx.setdefault("reader", f.f_locals)
        return role, ctx

    def _question_for(self, text: str) -> ToyQuestion | None:
        for q in self.questions:
            if q.question in text:
                return q
        return None

    def _question_for_blank(self, blank_id: str) -> ToyQuestion:
        for q in self.questions:
            if blank_id in q.facts:
                return q
        raise KeyError(blank_id)

    # -- dispatch ---------------------------------------------------------

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        role, ctx = self._role()
        tools = kwargs.get("tools") or []
        tool_names = [t.get("function", {}).get("name") for t in tools]
        handler = getattr(self, f"_role_{role}", None)
        out = handler(messages, tool_names, ctx) if handler else _resp("{}")
        with self._lock:
            self.calls.append({"role": role, "tools": tool_names, "n_messages": len(messages),
                               "text": out.text, "tool_calls": out.tool_calls})
        return out

    # -- roles ------------------------------------------------------------

    def _role_planner(self, messages, tools, ctx) -> LLMResponse:
        q = self._question_for("\n".join(str(m.get("content", "")) for m in messages))
        if q is None:
            return _resp("{}")
        return _resp(json.dumps(q.plan))

    def _role_orchestrator(self, messages, tools, ctx) -> LLMResponse:
        q = self._question_for("\n".join(str(m.get("content", "")) for m in messages))
        if q is None:
            return _resp("I cannot tell which question this is.")
        called = [
            tc.get("function", {}).get("name")
            for m in messages if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
        ]
        if "call_planner" not in called:
            return _resp("", [self._tc("call_planner", {})])
        n_dispatched = called.count("dispatch_subplan")
        if n_dispatched < len(q.dispatch):
            return _resp("", [self._tc("dispatch_subplan", {"blank_ids": [q.dispatch[n_dispatched]]})])
        return _resp("", [self._tc("submit_answer", {"answer": q.answer})])

    def _role_resolver(self, messages, tools, ctx) -> LLMResponse:
        loc = ctx.get("resolver") or {}
        blank_id = loc.get("blank_id")
        q = self._question_for_blank(blank_id) if blank_id else None
        if q is None or blank_id in q.researcher_blanks:
            return _resp(json.dumps({"value": None, "supported": False, "confidence": 0.0,
                                     "evidence_chunk_ids": [], "source_span": ""}))
        value = q.facts[blank_id][0]
        for c in loc.get("chunks") or []:
            text = getattr(c, "text", "") or ""
            span = _sentence_with(text, value)
            if span:
                return _resp(json.dumps({
                    "value": value, "object_type": "entity", "source_span": span,
                    "supported": True, "confidence": 0.95,
                    "evidence_chunk_ids": [getattr(c, "chunk_id", "")],
                }))
        return _resp(json.dumps({"value": None, "supported": False, "confidence": 0.0,
                                 "evidence_chunk_ids": []}))

    def _role_verifier(self, messages, tools, ctx) -> LLMResponse:
        return _resp(json.dumps({
            "satisfies_all": True, "unique": True, "complete": True,
            "missing_constraints": [], "deferred_constraints": [],
            "ambiguous_candidates": [], "missing_items": [],
            "reason": "the cited text states it",
        }))

    def _role_researcher(self, messages, tools, ctx) -> LLMResponse:
        researcher = ctx.get("researcher")
        assigned = list(getattr(researcher, "allowed_order", []) or [])
        if not assigned:
            return _resp("nothing assigned")
        blank_id = assigned[0]
        q = self._question_for_blank(blank_id)
        value, query, reader_q = q.facts[blank_id]
        done = [
            (tc.get("function", {}).get("name"), tc.get("function", {}).get("arguments"))
            for m in messages if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
        ]
        names = [n for n, _ in done]
        hit = self._chunk_with_value(messages, value)
        if "search" not in names:
            return _resp("", [self._tc("search", {"query": query, "top_k": 5})])
        if hit is None:
            return _resp("", [self._tc("search", {"query": value, "top_k": 5})])
        if "read" not in names:
            return _resp("", [self._tc("read", {"chunk_id": hit, "question": reader_q})])
        vp_ids = [vp["id"] for vp in q.plan["verb_phrases"]
                  if blank_id in (vp["subject_id"], vp["object_id"])]
        bound = {json.loads(a or "{}").get("vp_id") for n, a in done if n == "bind_connection"}
        for vp_id in vp_ids:
            if vp_id not in bound and self._partner_known(q, vp_id, blank_id, messages):
                return _resp("", [self._tc("bind_connection", {
                    "blank_id": blank_id, "vp_id": vp_id, "evidence_chunk_ids": [hit]})])
        if "update_blank" not in names:
            return _resp("", [self._tc("update_blank", {
                "blank_id": blank_id, "value": value, "evidence_chunk_ids": [hit],
                "reasoning": "the article states it directly"})])
        return _resp("Committed the assigned blank.")

    @staticmethod
    def _partner_known(q: ToyQuestion, vp_id: str, blank_id: str, messages) -> bool:
        vp = next(v for v in q.plan["verb_phrases"] if v["id"] == vp_id)
        partner = vp["object_id"] if vp["subject_id"] == blank_id else vp["subject_id"]
        ent = next(e for e in q.plan["entities"] if e["id"] == partner)
        return ent["kind"] == "filled" or partner in q.dispatch[: q.dispatch.index(blank_id)]

    @staticmethod
    def _chunk_with_value(messages, value: str) -> str | None:
        for m in messages:
            if m.get("role") != "tool":
                continue
            try:
                payload = json.loads(m.get("content") or "")
            except (TypeError, ValueError):
                continue
            for hit in (payload.get("results") or payload.get("hits") or []) if isinstance(payload, dict) else []:
                if value in str(hit.get("text", "")) and hit.get("chunk_id"):
                    return hit["chunk_id"]
        return None

    def _role_reader(self, messages, tools, ctx) -> LLMResponse:
        loc = ctx.get("reader") or {}
        full_text = loc.get("full_text") or ""
        question = loc.get("question") or ""
        for q in self.questions:
            for value, _query, reader_q in q.facts.values():
                if reader_q == question:
                    span = _sentence_with(full_text, value)
                    if span:
                        return _resp(json.dumps({
                            "answer": value, "derivation": None,
                            "citations": [{"span": span, "offset": full_text.find(span)}],
                        }))
        return _resp(json.dumps({"not_found": True, "reason": "not stated on this page"}))

    def _role_evidence_judge(self, messages, tools, ctx) -> LLMResponse:
        return _resp(json.dumps({"grounded": True, "span": ""}))

    def _role_synthesis_validator(self, messages, tools, ctx) -> LLMResponse:
        return _resp(json.dumps({"verdict": "correct", "reason": "matches the worksheet",
                                 "confidence": "high"}))

    def _role_auditor(self, messages, tools, ctx) -> LLMResponse:
        return _resp("{}")

    def _role_finalizer(self, messages, tools, ctx) -> LLMResponse:
        return _resp("")

    def roles(self) -> list[str]:
        with self._lock:
            return [c["role"] for c in self.calls]
