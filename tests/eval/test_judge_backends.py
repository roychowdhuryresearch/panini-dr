import pytest

from panini_dr.eval.judges import (
    BedrockResponsesBackend, JudgeBackendError, JudgeModelMissingError, OpenAICompatibleBackend,
    make_backend, resolve_judge_model,
)


class _Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self.payload


def test_bedrock_backend_builds_the_responses_call():
    seen = {}

    def post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, json=json)
        return _Resp({"output": [{"content": [{"text": "correct: "}, {"text": "yes"}]}]})

    be = BedrockResponsesBackend("gpt-judge", region="test-region-1", post=post,
                                 token_provider=lambda region: f"tok-{region}")
    assert be.complete("PROMPT") == "correct: yes"
    assert seen["url"] == "https://bedrock-mantle.test-region-1.api.aws/openai/v1/responses"
    assert seen["json"] == {"model": "openai.gpt-judge", "input": "PROMPT"}
    assert seen["headers"]["Authorization"] == "Bearer tok-test-region-1"


def test_bedrock_backend_retries_then_fails():
    calls = []

    def post(*a, **k):
        calls.append(1)
        return _Resp({}, status=503)

    be = BedrockResponsesBackend("m", region="r", post=post, token_provider=lambda r: "t",
                                 retries=3, sleep=lambda s: None)
    with pytest.raises(JudgeBackendError):
        be.complete("p")
    assert len(calls) == 3


def test_bedrock_backend_needs_a_region(monkeypatch):
    monkeypatch.delenv("PANINI_BEDROCK_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    with pytest.raises(JudgeBackendError):
        BedrockResponsesBackend("m")
    monkeypatch.setenv("PANINI_BEDROCK_REGION", "env-region")
    assert BedrockResponsesBackend("m").region == "env-region"


class _FakeOpenAI:
    def __init__(self):
        self.kw = None
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.kw = kw

                class _M:
                    content = "judged"

                class _C:
                    message = _M()

                class _R:
                    choices = [_C()]
                return _R()

        class _Chat:
            completions = _Completions()

        class _Responses:
            def create(self, **kw):
                outer.kw = kw

                class _R:
                    output_text = "via responses"
                return _R()

        self.chat = _Chat()
        self.responses = _Responses()


def test_openai_backend_chat_pins_temperature_except_for_reasoning_models():
    c = _FakeOpenAI()
    assert OpenAICompatibleBackend("gpt-4.1", client=c).complete("p") == "judged"
    assert c.kw["temperature"] == 0.0 and c.kw["messages"] == [{"role": "user", "content": "p"}]
    OpenAICompatibleBackend("o3-my-judge-model", client=c).complete("p")
    assert "temperature" not in c.kw


def test_openai_backend_responses_api():
    c = _FakeOpenAI()
    assert OpenAICompatibleBackend("m", client=c, api="responses").complete("p") == "via responses"
    assert c.kw == {"model": "m", "input": "p"}


def test_make_backend_validates():
    with pytest.raises(ValueError):
        make_backend("nope", "m")
    with pytest.raises(ValueError):
        make_backend("openai", "")
    assert make_backend("openai", "m", base_url="http://judge.invalid/v1").model == "m"


def test_there_is_no_default_judge_model(monkeypatch):
    monkeypatch.delenv("PANINI_JUDGE_MODEL", raising=False)
    with pytest.raises(JudgeModelMissingError, match="PANINI_JUDGE_MODEL"):
        resolve_judge_model(None)
    with pytest.raises(JudgeModelMissingError):
        resolve_judge_model("  ")
    monkeypatch.setenv("PANINI_JUDGE_MODEL", "my-judge-model")
    assert resolve_judge_model(None) == "my-judge-model"
    assert resolve_judge_model("other-judge") == "other-judge"     # the flag wins over the env
