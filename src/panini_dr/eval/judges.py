"""Pluggable LLM-judge backends for the graders of record.

Two transports, one interface (``backend.complete(prompt) -> str``):

``openai``
    Any OpenAI-compatible endpoint (OpenAI itself, vLLM, a gateway). Uses the
    Chat Completions API by default, or the Responses API with
    ``api="responses"``. Base URL / key come from the arguments or from
    ``PANINI_JUDGE_BASE_URL`` / ``PANINI_JUDGE_API_KEY`` (falling back to
    ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``).
``bedrock``
    AWS Bedrock's OpenAI-compatible gateway, Responses API. A short-lived
    bearer token is minted per call from the standard AWS credential chain
    (``aws-bedrock-token-generator``, the ``eval`` extra). The region comes from
    the argument or ``PANINI_BEDROCK_REGION`` / ``AWS_REGION``; there is no
    default region.

There is no default judge model: the caller names one (``--model`` or
``PANINI_JUDGE_MODEL``, see :func:`resolve_judge_model`). The paper's grades
come from an LLM judge with 3 passes per question; re-grading with a different
judge moves the numbers (about +/-1.3 pt on BCP from re-judging alone). The
judge model's name is part of every judge-cache key, so verdicts from two
judges never mix.

Nothing here touches the network at import time, and every transport accepts an
injected ``post``/``client`` so tests never make a real call.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

#: Environment variable naming the judge model when ``--model`` is not given.
JUDGE_MODEL_ENV = "PANINI_JUDGE_MODEL"

DEFAULT_RETRIES = 6


class JudgeBackend(Protocol):
    name: str
    model: str

    def complete(self, prompt: str) -> str: ...


class JudgeBackendError(RuntimeError):
    pass


class JudgeModelMissingError(ValueError):
    """No judge model was named (there is deliberately no default)."""


def resolve_judge_model(model: str | None = None) -> str:
    """The judge model: ``model`` if given, else ``$PANINI_JUDGE_MODEL``.

    Raises :class:`JudgeModelMissingError` when neither is set: the graders
    never pick a judge on the user's behalf.
    """
    m = (model or "").strip() or os.environ.get(JUDGE_MODEL_ENV, "").strip()
    if not m:
        raise JudgeModelMissingError(
            "no judge model: pass --model <name> or set PANINI_JUDGE_MODEL "
            "(any model your judge backend serves; it becomes part of every cache key)"
        )
    return m


def _retry(fn: Callable[[], str], *, tries: int, sleep: Callable[[float], None]) -> str:
    """Call ``fn`` with exponential backoff (the research graders' schedule)."""
    last: BaseException | None = None
    for i in range(max(1, tries)):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- 5xx, timeouts, token refresh
            last = exc
            if i < tries - 1:
                sleep(min(40, 2 ** i) + 0.1 * i)
    raise JudgeBackendError(f"judge call failed after {tries} tries: {last}") from last


def _responses_text(payload: dict[str, Any]) -> str:
    """Concatenate the text parts of a Responses-API reply."""
    return "".join(
        c.get("text", "")
        for o in payload.get("output", []) or []
        for c in (o.get("content") or [])
        if isinstance(c, dict)
    )


class OpenAICompatibleBackend:
    """Chat Completions (default) or Responses API on any OpenAI-compatible server."""

    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api: str = "chat",
        retries: int = DEFAULT_RETRIES,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if api not in {"chat", "responses"}:
            raise ValueError("api must be 'chat' or 'responses'")
        self.model = model
        self.api = api
        self.retries = retries
        self._sleep = sleep
        self._client = client
        self._base_url = (
            base_url
            or os.environ.get("PANINI_JUDGE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or None
        )
        self._api_key = (
            api_key
            or os.environ.get("PANINI_JUDGE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or None
        )
        self._lock = threading.Lock()

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                from openai import OpenAI

                if not self._api_key and not self._base_url:
                    raise JudgeBackendError(
                        "openai judge backend needs PANINI_JUDGE_API_KEY/OPENAI_API_KEY "
                        "or a --judge-base-url for an OpenAI-compatible server"
                    )
                self._client = OpenAI(base_url=self._base_url, api_key=self._api_key or "unused")
            return self._client

    def _once(self, prompt: str) -> str:
        client = self._get_client()
        if self.api == "responses":
            r = client.responses.create(model=self.model, input=prompt)
            text = getattr(r, "output_text", None)
            if text is None and isinstance(r, dict):
                text = _responses_text(r)
            return text or ""
        # gpt-5-class / o-series reject temperature=0; the research graders
        # omitted it for them and pinned 0.0 otherwise.
        kw = {} if self.model.startswith(("gpt-5", "o1", "o3", "o4")) else {"temperature": 0.0}
        r = client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], **kw
        )
        return r.choices[0].message.content or ""

    def complete(self, prompt: str) -> str:
        return _retry(lambda: self._once(prompt), tries=self.retries, sleep=self._sleep)


def bedrock_region(region: str | None = None) -> str:
    r = region or os.environ.get("PANINI_BEDROCK_REGION") or os.environ.get("AWS_REGION")
    if not r:
        raise JudgeBackendError(
            "bedrock judge backend needs a region: pass --bedrock-region or set "
            "PANINI_BEDROCK_REGION (or AWS_REGION) to a region that serves the judge model"
        )
    return r


class BedrockResponsesBackend:
    """Bedrock's OpenAI-compatible gateway, Responses API."""

    name = "bedrock"

    def __init__(
        self,
        model: str,
        *,
        region: str | None = None,
        retries: int = DEFAULT_RETRIES,
        timeout_s: float = 900.0,
        post: Callable[..., Any] | None = None,
        token_provider: Callable[[str], str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.region = bedrock_region(region)
        self.retries = retries
        self.timeout_s = timeout_s
        self._post = post
        self._token_provider = token_provider
        self._sleep = sleep

    @property
    def bedrock_model_id(self) -> str:
        return self.model if self.model.startswith("openai.") else f"openai.{self.model}"

    @property
    def url(self) -> str:
        return f"https://bedrock-mantle.{self.region}.api.aws/openai/v1/responses"

    def _token(self) -> str:
        if self._token_provider is not None:
            return self._token_provider(self.region)
        try:
            from aws_bedrock_token_generator import provide_token
        except ImportError as exc:  # pragma: no cover - optional extra
            raise JudgeBackendError(
                "bedrock judge backend needs aws-bedrock-token-generator "
                "(install the 'eval' extra)"
            ) from exc
        return provide_token(region=self.region)

    def _once(self, prompt: str) -> str:
        post = self._post
        if post is None:
            import httpx

            post = httpx.post
        r = post(
            self.url,
            headers={"Authorization": f"Bearer {self._token()}", "Content-Type": "application/json"},
            json={"model": self.bedrock_model_id, "input": prompt},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return _responses_text(r.json())

    def complete(self, prompt: str) -> str:
        return _retry(lambda: self._once(prompt), tries=self.retries, sleep=self._sleep)


def make_backend(
    kind: str,
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    api: str = "chat",
    region: str | None = None,
) -> JudgeBackend:
    """Build a judge backend by name (``openai`` or ``bedrock``)."""
    if not model:
        raise JudgeModelMissingError("a judge model name is required")
    if kind == "openai":
        return OpenAICompatibleBackend(model, base_url=base_url, api_key=api_key, api=api)
    if kind == "bedrock":
        return BedrockResponsesBackend(model, region=region)
    raise ValueError(f"unknown judge backend {kind!r}; expected 'openai' or 'bedrock'")


__all__ = [
    "BedrockResponsesBackend",
    "JudgeBackend",
    "JUDGE_MODEL_ENV",
    "JudgeBackendError",
    "JudgeModelMissingError",
    "OpenAICompatibleBackend",
    "bedrock_region",
    "make_backend",
    "resolve_judge_model",
]
