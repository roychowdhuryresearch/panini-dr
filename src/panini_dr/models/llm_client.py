"""Thin LLM client.

Routes chat completions over three paths:

- ``openai`` — OpenAI's cloud API (base_url empty, model starts with ``gpt-``).
- ``vllm`` — any OpenAI-compatible server (``base_url`` set, model name is the
  `--served-model-name` the vLLM process exposes).
- ``bedrock`` — AWS Bedrock via LiteLLM (model starts with ``bedrock/``).
  Mirrors the gsw-memory sleep-time StageTransport recipe:
  ``reasoning_effort`` kwarg + ``allowed_openai_params`` + no ``max_tokens``.

All three paths stream by default so the user sees text / tool-call
arguments as they arrive. The ``LLMResponse`` shape is unchanged; a new
``reasoning_content`` field captures the hidden-reasoning summary when
the model exposes it (Bedrock gpt-oss, GPT-5 reasoning summaries).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

_stream_log = logging.getLogger("panini_dr.llm")
_STREAM_FLUSH_CHARS = 160


# ---------------------------------------------------------------------------
# Qwen3 native XML tool-call format post-parser.
#
# Qwen3 emits tool calls as:
#
#     <tool_call>
#     <function=search>
#     <parameter=query>
#     Joel Oshiro Dyck
#     </parameter>
#     <parameter=top_k>
#     5
#     </parameter>
#     </function>
#     </tool_call>
#
# vLLM's --tool-call-parser hermes expects the JSON form instead and leaves
# this XML as plain text in the response. We extract it ourselves.
# ---------------------------------------------------------------------------

_QWEN3_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_QWEN3_FUNCTION_RE = re.compile(r"<function=([^>]+?)>(.*?)</function>", re.DOTALL)
_QWEN3_PARAM_RE = re.compile(r"<parameter=([^>]+?)>(.*?)</parameter>", re.DOTALL)


def _maybe_coerce_value(raw: str) -> Any:
    """Turn a raw Qwen3 parameter value into int/float/bool/str.

    The XML format uses bare text per parameter. For typed args (``top_k=5``)
    we coerce; anything else stays a string.
    """
    s = raw.strip()
    if not s:
        return s
    # JSON-decodable literals (numbers, booleans, null, quoted strings, lists)
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


def parse_qwen3_xml_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from Qwen3's native XML format.

    Returns OpenAI-shaped entries: ``[{"id": ..., "name": ..., "arguments": <json str>}, ...]``.
    Empty list if no ``<tool_call>`` block is present — safe to call on any text.
    """
    calls: list[dict[str, Any]] = []
    for i, block in enumerate(_QWEN3_TOOL_CALL_RE.findall(text or "")):
        m_fn = _QWEN3_FUNCTION_RE.search(block)
        if not m_fn:
            continue
        name = m_fn.group(1).strip()
        body = m_fn.group(2)
        args: dict[str, Any] = {}
        for k, v in _QWEN3_PARAM_RE.findall(body):
            args[k.strip()] = _maybe_coerce_value(v)
        calls.append(
            {
                "id": f"qwen3_tc_{i}",
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            }
        )
    return calls


def _chunk_for_log(text: str, *, prefix: str, width: int = 240):
    """Yield ``<prefix> | <chunk>`` log lines for a multi-paragraph string.

    Used by the non-streaming Bedrock path to mimic the streaming
    experience: reasoning/text still scrolls past in the terminal (just
    post-hoc rather than token-by-token).
    """
    if not text:
        return
    # Preserve paragraph breaks; wrap long paragraphs at ``width``.
    for paragraph in text.splitlines():
        p = paragraph.strip()
        if not p:
            continue
        for i in range(0, len(p), width):
            yield f"{prefix}| {p[i : i + width]}"


@dataclass(slots=True)
class LLMResponse:
    text: str = ""
    reasoning_content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""


class LLMClient:
    """Wrapper over OpenAI-compatible endpoints + LiteLLM for Bedrock / others."""

    _REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        default_temperature: float = 0.0,
        default_max_tokens: int = 50000,
        reasoning_effort: str = "medium",
    ) -> None:
        self.original_model = (model or "").strip()
        self._base_url = (base_url or "").strip()
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.reasoning_effort = reasoning_effort

        self.provider = self._detect_provider()
        # model name actually sent in the API request (Bedrock normalizes).
        self.model = self._canonical_model_name()

        # Lazy clients: only initialize what we need.
        self._openai_client: Any = None
        self._litellm: Any = None
        if self.provider in {"openai", "vllm"}:
            if OpenAI is None:
                raise RuntimeError("openai package missing; install with `pip install openai`")
            # OPENAI_API_KEY only for OpenAI itself: never send it to a custom
            # base URL (the vLLM server), where it would leak a judge key.
            env_key = os.environ.get("OPENAI_API_KEY") if self.provider == "openai" else None
            resolved_key = api_key or env_key or "dummy-for-vllm"
            self._openai_client = OpenAI(
                base_url=self._base_url or None, api_key=resolved_key
            )
        else:
            try:
                import litellm
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "litellm not installed — run `uv pip install litellm`"
                ) from exc
            self._litellm = litellm

    # ----------------------------------------------------------------------
    # Provider detection (mirrors gsw_memory.query_then_sleep.transport.StageTransport).
    # ----------------------------------------------------------------------

    def _detect_provider(self) -> str:
        if self._base_url:
            return "vllm"
        m = self.original_model
        if m.startswith("bedrock/"):
            return "bedrock"
        if m.startswith("gpt-"):
            return "openai"
        if m.startswith(
            ("together_ai/", "anthropic/", "vertex_ai/", "gemini/", "openai/")
        ):
            return "litellm"
        if "/" in m:
            # treat unknown provider/model as together_ai/
            return "together"
        # Default: treat bare model names as OpenAI (gpt-3.5-turbo, gpt-4o-mini, …).
        return "openai"

    def _canonical_model_name(self) -> str:
        m = self.original_model
        if self.provider == "bedrock":
            if m.startswith("bedrock/converse/") or m.startswith("bedrock/invoke/"):
                return m
            return m.replace("bedrock/", "bedrock/converse/", 1)
        if self.provider == "together" and not m.startswith("together_ai/"):
            return f"together_ai/{m}"
        return m

    # ----------------------------------------------------------------------
    # Capability probes
    # ----------------------------------------------------------------------

    def _is_openai_endpoint(self) -> bool:
        return self.provider == "openai"

    def _is_reasoning_model(self) -> bool:
        lower = (self.original_model or "").lower()
        return self._is_openai_endpoint() and any(
            lower.startswith(p) for p in self._REASONING_MODEL_PREFIXES
        )

    def supports_seed(self) -> bool:
        """Whether ``chat(seed=...)`` reaches the provider request.

        Bedrock / LiteLLM ignore seed today. OpenAI reasoning models are
        routed through Responses API here, where this client does not pass
        a seed field.
        """
        return self.provider in {"openai", "vllm"} and not self._is_reasoning_model()

    def _max_tokens_kwarg(self) -> str:
        return "max_completion_tokens" if self._is_reasoning_model() else "max_tokens"

    # ----------------------------------------------------------------------
    # Public chat API
    # ----------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = True,
        stop: str | list[str] | None = None,
        seed: int | None = None,
    ) -> LLMResponse:
        """One chat round. Streams by default; returns assembled response.

        ``stop`` is plumbed through to the underlying provider so callers
        can halt generation at a protocol token (e.g. Search-o1 uses
        ``stop=["<|end_search_query|>"]`` to yield to real retrieval).

        ``seed`` is passed to the OpenAI Chat Completions ``seed`` field
        (vLLM honors it for deterministic sampling). Bedrock / LiteLLM
        ignore the kwarg today; see :meth:`supports_seed`.
        """
        resolved_max = max_tokens if max_tokens is not None else self.default_max_tokens
        resolved_temp = temperature if temperature is not None else self.default_temperature

        if self.provider in {"openai", "vllm"}:
            return self._chat_openai(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=resolved_temp,
                max_tokens=resolved_max,
                response_format=response_format,
                stream=stream,
                stop=stop,
                seed=seed,
            )

        # bedrock / litellm / together — all go through litellm.
        return self._chat_litellm(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=resolved_temp,
            max_tokens=resolved_max,
            response_format=response_format,
            stream=stream,
            stop=stop,
        )

    # ----------------------------------------------------------------------
    # OpenAI / vLLM path
    # ----------------------------------------------------------------------

    def _chat_openai(
        self,
        messages,
        *,
        tools,
        tool_choice,
        temperature,
        max_tokens,
        response_format,
        stream,
        stop=None,
        seed: int | None = None,
    ) -> LLMResponse:
        # Reasoning models (gpt-5, o-series) expose reasoning summaries only
        # via the Responses API — Chat Completions hides them server-side.
        # So for these we route to the Responses API so hidden_reasoning ≠ "".
        if self._is_reasoning_model():
            return self._chat_openai_responses(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                response_format=response_format,
                stream=stream,
            )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            self._max_tokens_kwarg(): max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format
        if stop:
            kwargs["stop"] = stop
        if seed is not None:
            kwargs["seed"] = seed

        if not stream:
            completion = self._openai_client.chat.completions.create(**kwargs)
            return self._normalize_openai_blocking(completion)

        # Streaming path.
        stream_kwargs = dict(kwargs)
        stream_kwargs["stream"] = True
        stream_kwargs["stream_options"] = {"include_usage": True}
        stream_iter = self._openai_client.chat.completions.create(**stream_kwargs)
        resp = self._consume_stream(stream_iter, label=f"{self.provider}/{self.model}")

        # Known pitfall: vLLM's Chat Completions streaming sometimes reports
        # ``finish_reason=tool_calls`` while emitting the tool-call deltas in
        # a shape the generic consumer doesn't capture (notably when
        # --reasoning-parser is active). Silently losing tool calls means
        # every turn looks like an "early_stop". When we detect this mismatch
        # we re-issue the call non-streaming as a safety net — Bedrock path
        # already uses the same escape hatch for the same reason.
        if (
            resp.finish_reason == "tool_calls"
            and not resp.tool_calls
            and self.provider == "vllm"
        ):
            _stream_log.info(
                "stream miss | %s finish=tool_calls but 0 tool_calls captured — retrying non-streaming",
                self.model,
            )
            completion = self._openai_client.chat.completions.create(**kwargs)
            return self._normalize_openai_blocking(completion)
        return resp

    # ----------------------------------------------------------------------
    # OpenAI Responses API path (for reasoning models: gpt-5, o1, o3, o4)
    # ----------------------------------------------------------------------

    @staticmethod
    def _responses_tools(tools):
        """Translate Chat Completions tool schemas to Responses API shape."""
        if not tools:
            return None
        out = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            out.append(
                {
                    "type": "function",
                    "name": fn.get("name") or t.get("name"),
                    "description": fn.get("description") or t.get("description", ""),
                    "parameters": fn.get("parameters") or t.get("parameters") or {},
                }
            )
        return out

    @staticmethod
    def _messages_to_responses_input(messages):
        """Translate a Chat-Completions message list into Responses API input items.

        - ``system`` / ``user`` → kept as ``{role, content}``.
        - ``assistant`` with ``tool_calls`` → one ``function_call`` item per call.
        - ``assistant`` with ``content`` → kept (may also coexist with tool calls).
        - ``tool`` → ``function_call_output`` item (keyed by ``tool_call_id`` → ``call_id``).
        """
        out: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role in ("system", "user"):
                out.append({"role": role, "content": m.get("content") or ""})
            elif role == "assistant":
                tool_calls = m.get("tool_calls") or []
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    out.append(
                        {
                            "type": "function_call",
                            "call_id": tc.get("id") or "",
                            "name": fn.get("name") or "",
                            "arguments": fn.get("arguments") or "",
                        }
                    )
                if m.get("content"):
                    out.append({"role": "assistant", "content": m["content"]})
            elif role == "tool":
                out.append(
                    {
                        "type": "function_call_output",
                        "call_id": m.get("tool_call_id") or "",
                        "output": m.get("content") or "",
                    }
                )
        return out

    def _chat_openai_responses(
        self,
        messages,
        *,
        tools,
        tool_choice,
        max_tokens,
        response_format,
        stream,
    ) -> LLMResponse:
        """Use client.responses.create with reasoning={'summary':'auto'}.

        Responses API uses ``input`` (a list of items, not a message array) and
        a flat ``tools`` schema. Reasoning summary deltas arrive as their own
        events so we assemble them into ``reasoning_content``.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": self._messages_to_responses_input(messages),
            "max_output_tokens": max_tokens,
            "reasoning": {
                "effort": self.reasoning_effort,
                "summary": "auto",
            },
        }
        resp_tools = self._responses_tools(tools)
        if resp_tools:
            kwargs["tools"] = resp_tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        if response_format:
            # Responses API uses text.format for structured output.
            kwargs["text"] = {"format": response_format}

        if not stream:
            r = self._openai_client.responses.create(**kwargs)
            return self._normalize_responses_blocking(r)

        kwargs["stream"] = True
        stream_iter = self._openai_client.responses.create(**kwargs)
        return self._consume_responses_stream(stream_iter, label=f"openai-resp/{self.model}")

    def _normalize_responses_blocking(self, r) -> LLMResponse:
        """Normalize a blocking Responses API result into LLMResponse."""
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in getattr(r, "output", None) or []:
            item_type = getattr(item, "type", "")
            if item_type == "message":
                for c in getattr(item, "content", None) or []:
                    if getattr(c, "type", "") == "output_text":
                        text_parts.append(getattr(c, "text", "") or "")
            elif item_type == "reasoning":
                for s in getattr(item, "summary", None) or []:
                    reasoning_parts.append(getattr(s, "text", "") or "")
            elif item_type == "function_call":
                tool_calls.append(
                    {
                        "id": getattr(item, "call_id", "") or getattr(item, "id", ""),
                        "name": getattr(item, "name", ""),
                        "arguments": getattr(item, "arguments", "") or "",
                    }
                )
        usage = getattr(r, "usage", None)
        return LLMResponse(
            text="".join(text_parts),
            reasoning_content="".join(reasoning_parts),
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
            raw=r.model_dump() if hasattr(r, "model_dump") else {},
            finish_reason=getattr(r, "status", "") or "",
        )

    def _consume_responses_stream(self, stream_iter, *, label: str) -> LLMResponse:
        """Consume a Responses API event stream."""
        _stream_log.debug("stream start | %s", label)

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_by_item: dict[str, dict[str, str]] = {}
        prompt_tokens = 0
        completion_tokens = 0
        finish_status = ""

        pending_text = ""
        pending_reasoning = ""

        try:
            for event in stream_iter:
                etype = getattr(event, "type", "")

                if etype == "response.output_text.delta":
                    piece = getattr(event, "delta", "") or ""
                    if piece:
                        text_parts.append(piece)
                        pending_text += piece
                        if len(pending_text) >= _STREAM_FLUSH_CHARS or "\n" in pending_text:
                            _stream_log.debug("stream txt | %s", pending_text.replace("\n", " ").rstrip())
                            pending_text = ""

                elif etype == "response.reasoning_summary_text.delta":
                    piece = getattr(event, "delta", "") or ""
                    if piece:
                        reasoning_parts.append(piece)
                        pending_reasoning += piece
                        if len(pending_reasoning) >= _STREAM_FLUSH_CHARS or "\n" in pending_reasoning:
                            _stream_log.debug("stream rsn | %s", pending_reasoning.replace("\n", " ").rstrip())
                            pending_reasoning = ""

                elif etype == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item is not None and getattr(item, "type", "") == "function_call":
                        item_id = getattr(item, "id", "") or getattr(item, "call_id", "")
                        tool_calls_by_item[item_id] = {
                            "id": getattr(item, "call_id", "") or item_id,
                            "name": getattr(item, "name", ""),
                            "arguments": getattr(item, "arguments", "") or "",
                        }
                        if tool_calls_by_item[item_id]["name"]:
                            _stream_log.debug("stream tool | %s(", tool_calls_by_item[item_id]["name"])

                elif etype == "response.function_call_arguments.delta":
                    item_id = getattr(event, "item_id", "")
                    slot = tool_calls_by_item.setdefault(
                        item_id, {"id": "", "name": "", "arguments": ""}
                    )
                    slot["arguments"] += getattr(event, "delta", "") or ""

                elif etype == "response.function_call_arguments.done":
                    item_id = getattr(event, "item_id", "")
                    args = getattr(event, "arguments", "") or ""
                    slot = tool_calls_by_item.setdefault(
                        item_id, {"id": "", "name": "", "arguments": ""}
                    )
                    if args and not slot["arguments"]:
                        slot["arguments"] = args
                    _stream_log.debug(
                        "stream tool | %s ) %s",
                        slot["name"] or "<?>",
                        slot["arguments"].replace("\n", " ")[:400],
                    )

                elif etype == "response.completed":
                    response = getattr(event, "response", None)
                    if response is not None:
                        finish_status = getattr(response, "status", "") or ""
                        usage = getattr(response, "usage", None)
                        if usage is not None:
                            prompt_tokens = getattr(usage, "input_tokens", 0) or prompt_tokens
                            completion_tokens = (
                                getattr(usage, "output_tokens", 0) or completion_tokens
                            )
        except Exception:
            if pending_text:
                _stream_log.debug("stream txt | %s", pending_text.replace("\n", " ").rstrip())
            if pending_reasoning:
                _stream_log.debug("stream rsn | %s", pending_reasoning.replace("\n", " ").rstrip())
            raise

        if pending_text:
            _stream_log.debug("stream txt | %s", pending_text.replace("\n", " ").rstrip())
        if pending_reasoning:
            _stream_log.debug("stream rsn | %s", pending_reasoning.replace("\n", " ").rstrip())

        full_text = "".join(text_parts)
        full_reasoning = "".join(reasoning_parts)
        final_tool_calls: list[dict[str, Any]] = [
            {"id": s["id"], "name": s["name"], "arguments": s["arguments"]}
            for s in tool_calls_by_item.values()
            if s["name"] or s["arguments"]
        ]

        _stream_log.info(
            "stream done | finish=%s text=%d reasoning=%d tools=%d tokens_in=%d tokens_out=%d",
            finish_status or "-",
            len(full_text),
            len(full_reasoning),
            len(final_tool_calls),
            prompt_tokens,
            completion_tokens,
        )

        return LLMResponse(
            text=full_text,
            reasoning_content=full_reasoning,
            tool_calls=final_tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw={},
            finish_reason=finish_status,
        )

    def _normalize_openai_blocking(self, completion) -> LLMResponse:
        choice = completion.choices[0]
        msg = choice.message
        tool_calls: list[dict[str, Any]] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            tool_calls.append(
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            )
        text = msg.content or ""
        finish_reason = choice.finish_reason or ""
        reasoning_content = str(getattr(msg, "reasoning_content", "") or "")

        # Qwen3 XML fallback — narrow safety net: only kicks in when the
        # endpoint returned a <tool_call>...</tool_call> block in the *text*
        # but no structured tool_calls. That covers vLLM deployments without
        # a compatible --tool-call-parser (hermes doesn't match Qwen3's XML).
        # We deliberately do NOT extract from reasoning_content: if the model
        # emitted tool-call XML inside its <think> block, that's a genuine
        # model-output phenomenon that the grid should observe honestly,
        # not paper over. Let such turns surface as wrong_synthesis /
        # early_stop in the failure-mode histogram.
        if not tool_calls and "<tool_call>" in text:
            extracted = parse_qwen3_xml_tool_calls(text)
            if extracted:
                tool_calls = extracted
                text = _QWEN3_TOOL_CALL_RE.sub("", text).strip()
                if not finish_reason or finish_reason == "stop":
                    finish_reason = "tool_calls"

        usage = getattr(completion, "usage", None)
        return LLMResponse(
            text=text,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            raw=completion.model_dump() if hasattr(completion, "model_dump") else {},
            finish_reason=finish_reason,
        )

    # ----------------------------------------------------------------------
    # LiteLLM / Bedrock path
    # ----------------------------------------------------------------------

    def _chat_litellm(
        self,
        messages,
        *,
        tools,
        tool_choice,
        temperature,
        max_tokens,
        response_format,
        stream,
        stop=None,
    ) -> LLMResponse:
        api_params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            api_params["tools"] = tools
            if tool_choice is not None:
                api_params["tool_choice"] = tool_choice
        if response_format:
            api_params["response_format"] = response_format
        # Bedrock's converse API rejects ``stopSequences`` for gpt-oss and
        # other reasoning models ("This model doesn't support the
        # stopSequences field"). Drop it silently — callers still get a
        # valid response, they just can't pre-halt on a protocol token.
        if stop and self.provider != "bedrock":
            api_params["stop"] = stop

        if self.provider == "bedrock":
            # Mirrors sleep-time StageTransport: reasoning-effort kwarg +
            # allowed_openai_params; Bedrock rejects max_tokens.
            api_params["reasoning_effort"] = self.reasoning_effort
            allowed = ["reasoning_effort"]
            if tools is not None:
                allowed.extend(["tools", "tool_choice"])
            if response_format is not None:
                allowed.append("response_format")
            api_params["allowed_openai_params"] = allowed
        else:
            # Non-bedrock litellm backends still accept max_tokens.
            api_params["max_tokens"] = max_tokens

        # Bedrock's tool-use streaming via litellm emits corrupted argument
        # deltas (keys drop, fragments reorder). Non-streaming returns the
        # complete response with intact tool_calls — cheaper safety than
        # streaming. We surface a single per-call log line so visibility
        # isn't lost entirely.
        if self.provider == "bedrock":
            stream = False

        if not stream:
            _stream_log.debug("call start  | %s/%s (non-streaming)", self.provider, self.model)
            completion = self._litellm.completion(**api_params)
            resp = self._normalize_openai_blocking(completion)
            # Even though we can't stream reasoning token-by-token for
            # Bedrock (tool args corrupt mid-stream), we still want the
            # user to see what the model thought. Log the reasoning +
            # text + tool calls post-hoc, chunked for readability.
            for line in _chunk_for_log(resp.reasoning_content, prefix="call rsn   "):
                _stream_log.debug(line)
            for line in _chunk_for_log(resp.text, prefix="call txt   "):
                _stream_log.debug(line)
            for tc in resp.tool_calls:
                _stream_log.debug(
                    "call tool  | %s(%s)",
                    tc.get("name", "<?>"),
                    (tc.get("arguments") or "").replace("\n", " ")[:400],
                )
            _stream_log.info(
                "call done   | %s/%s text=%d reasoning=%d tools=%d tokens_in=%d tokens_out=%d",
                self.provider, self.model,
                len(resp.text),
                len(resp.reasoning_content),
                len(resp.tool_calls),
                resp.prompt_tokens,
                resp.completion_tokens,
            )
            return resp

        api_params["stream"] = True
        stream_iter = self._litellm.completion(**api_params)
        return self._consume_stream(stream_iter, label=f"{self.provider}/{self.model}")

    # ----------------------------------------------------------------------
    # Shared streaming consumer — works for both OpenAI SDK and litellm
    # because both emit delta-shaped chunks with .choices[0].delta and
    # tool-call argument fragments.
    # ----------------------------------------------------------------------

    def _consume_stream(self, stream_iter, *, label: str) -> LLMResponse:
        _stream_log.debug("stream start | %s", label)

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_by_idx: dict[int, dict[str, str]] = {}
        finish_reason = ""
        prompt_tokens = 0
        completion_tokens = 0

        pending_text = ""
        pending_reasoning = ""
        tool_arg_logged_len: dict[int, int] = {}

        try:
            for chunk in stream_iter:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or prompt_tokens
                    completion_tokens = (
                        getattr(usage, "completion_tokens", 0) or completion_tokens
                    )
                if not getattr(chunk, "choices", None):
                    continue

                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)

                content = getattr(delta, "content", None) if delta is not None else None
                if content:
                    text_parts.append(content)
                    pending_text += content
                    if len(pending_text) >= _STREAM_FLUSH_CHARS or "\n" in pending_text:
                        _stream_log.debug("stream txt | %s", pending_text.replace("\n", " ").rstrip())
                        pending_text = ""

                # Reasoning content (Bedrock gpt-oss, some OpenAI models).
                # Litellm surfaces this as delta.reasoning_content; OpenAI
                # sometimes uses delta.reasoning or delta.thinking. We check
                # multiple field names to be resilient.
                for attr in ("reasoning_content", "reasoning", "thinking"):
                    r_piece = getattr(delta, attr, None) if delta is not None else None
                    if r_piece:
                        reasoning_parts.append(str(r_piece))
                        pending_reasoning += str(r_piece)
                        if len(pending_reasoning) >= _STREAM_FLUSH_CHARS or "\n" in pending_reasoning:
                            _stream_log.debug(
                                "stream rsn | %s",
                                pending_reasoning.replace("\n", " ").rstrip(),
                            )
                            pending_reasoning = ""
                        break  # one reasoning surface per delta

                deltas = getattr(delta, "tool_calls", None) if delta is not None else None
                if deltas:
                    for d in deltas:
                        idx = getattr(d, "index", 0) or 0
                        slot = tool_calls_by_idx.setdefault(
                            idx, {"id": "", "name": "", "arguments": ""}
                        )
                        if getattr(d, "id", None):
                            slot["id"] = d.id
                        fn = getattr(d, "function", None)
                        if fn is not None:
                            name_delta = getattr(fn, "name", None)
                            if name_delta:
                                if not slot["name"]:
                                    _stream_log.debug("stream tool | %s(", name_delta)
                                slot["name"] = name_delta
                            args_delta = getattr(fn, "arguments", None)
                            if args_delta:
                                slot["arguments"] += args_delta
                                arg_so_far = slot["arguments"]
                                last = tool_arg_logged_len.get(idx, 0)
                                if len(arg_so_far) - last >= _STREAM_FLUSH_CHARS:
                                    _stream_log.debug(
                                        "stream tool | %s … %s",
                                        slot["name"] or "<?>",
                                        arg_so_far[last:].replace("\n", " "),
                                    )
                                    tool_arg_logged_len[idx] = len(arg_so_far)

                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
        except Exception:
            if pending_text:
                _stream_log.debug("stream txt | %s", pending_text.replace("\n", " ").rstrip())
            if pending_reasoning:
                _stream_log.debug("stream rsn | %s", pending_reasoning.replace("\n", " ").rstrip())
            raise

        if pending_text:
            _stream_log.debug("stream txt | %s", pending_text.replace("\n", " ").rstrip())
        if pending_reasoning:
            _stream_log.debug("stream rsn | %s", pending_reasoning.replace("\n", " ").rstrip())
        for idx, slot in tool_calls_by_idx.items():
            total = len(slot["arguments"])
            last = tool_arg_logged_len.get(idx, 0)
            if total > last:
                _stream_log.debug(
                    "stream tool | %s ) %s",
                    slot["name"] or "<?>",
                    slot["arguments"][last:total].replace("\n", " "),
                )

        full_text = "".join(text_parts)
        full_reasoning = "".join(reasoning_parts)
        final_tool_calls: list[dict[str, Any]] = []
        for idx in sorted(tool_calls_by_idx):
            slot = tool_calls_by_idx[idx]
            if slot["name"] or slot["arguments"]:
                final_tool_calls.append(
                    {"id": slot["id"], "name": slot["name"], "arguments": slot["arguments"]}
                )

        # Qwen3 XML fallback on streamed responses too (saves a retry
        # round-trip). Narrow safety net: text only, not reasoning —
        # see the matching comment in _normalize_openai_blocking for why.
        if not final_tool_calls and "<tool_call>" in full_text:
            extracted = parse_qwen3_xml_tool_calls(full_text)
            if extracted:
                final_tool_calls = extracted
                full_text = _QWEN3_TOOL_CALL_RE.sub("", full_text).strip()
                if not finish_reason or finish_reason == "stop":
                    finish_reason = "tool_calls"

        _stream_log.info(
            "stream done | finish=%s text=%d reasoning=%d tools=%d tokens_in=%d tokens_out=%d",
            finish_reason or "-",
            len(full_text),
            len(full_reasoning),
            len(final_tool_calls),
            prompt_tokens,
            completion_tokens,
        )

        return LLMResponse(
            text=full_text,
            reasoning_content=full_reasoning,
            tool_calls=final_tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw={},
            finish_reason=finish_reason,
        )
