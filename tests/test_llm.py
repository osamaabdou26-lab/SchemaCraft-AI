"""Provider client tests: request shaping and error mapping, with the SDK call stubbed."""

from types import SimpleNamespace

import pytest

from core.config import ANTHROPIC, OPENAI
from core.llm import AnthropicClient, LLMError, MissingAPIKeyError, OpenAIClient, build_client

ENVELOPE = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"],
            "additionalProperties": False}
MESSAGES = [{"role": "user", "content": "hi"}]


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        build_client("OpenAI", "gpt-4o-mini", 0.2)


def test_override_key_used(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(build_client("Anthropic", "claude-haiku-4-5", 0.2, "sk-ant-test"), AnthropicClient)


@pytest.mark.parametrize(
    "provider, model, key, match",
    [
        ("OpenAI", "gpt-4o", "sk-ant-abc123", "Anthropic \\(Claude\\) key"),
        ("Anthropic", "claude-sonnet-5", "sk-proj-abc123", "looks like an OpenAI key"),
        ("OpenAI", "gpt-4o", "sk-proj-abc 123", "spaces or line breaks"),
    ],
)
def test_key_mismatch_detected(provider, model, key, match):
    with pytest.raises(LLMError, match=match):
        build_client(provider, model, 0.2, key)


def test_quoted_key_is_unwrapped(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", '"sk-proj-abc123"')
    client = build_client("OpenAI", "gpt-4o", 0.2)
    assert client._client.api_key == "sk-proj-abc123"


def test_unknown_model():
    with pytest.raises(LLMError, match="Unknown model"):
        build_client("OpenAI", "nope", 0.2, "sk-test")


def _openai_response(content, finish="stop", refusal=None):
    message = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)])


def test_openai_request_shape_and_parse():
    client = OpenAIClient(OPENAI.models[1], 0.3, "sk-test")
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _openai_response('{"a": "x"}')

    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert client.complete_json("sys", MESSAGES, ENVELOPE, "env") == {"a": "x"}
    assert captured["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["temperature"] == 0.3


@pytest.mark.parametrize(
    "response, match",
    [
        (_openai_response("not json"), "malformed JSON"),
        (_openai_response("[1]"), "expected an object"),
        (_openai_response("", ), "empty response"),
        (_openai_response('{"a"', finish="length"), "token limit"),
        (_openai_response(None, refusal="no"), "declined"),
    ],
)
def test_openai_bad_outputs(response, match):
    client = OpenAIClient(OPENAI.models[1], 0.3, "sk-test")
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )
    with pytest.raises(LLMError, match=match):
        client.complete_json("sys", MESSAGES, ENVELOPE, "env")


def _anthropic_response(text, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)])


def _stub_anthropic(client, response, captured):
    def create(**kwargs):
        captured.update(kwargs)
        return response

    client._client = SimpleNamespace(messages=SimpleNamespace(create=create))


def test_anthropic_opus_omits_temperature_and_enables_fallbacks():
    client = AnthropicClient(ANTHROPIC.models[0], 0.5, "sk-test")
    captured = {}
    _stub_anthropic(client, _anthropic_response('{"a": "x"}'), captured)
    assert client.complete_json("sys", MESSAGES, ENVELOPE, "env") == {"a": "x"}
    body = captured["extra_body"]
    assert body["output_config"]["format"]["schema"] == ENVELOPE
    assert body["fallbacks"] == "default"
    assert "temperature" not in body and "temperature" not in captured
    assert captured["extra_headers"]["anthropic-beta"].startswith("server-side-fallback")


def test_anthropic_haiku_sends_temperature():
    client = AnthropicClient(ANTHROPIC.models[2], 0.5, "sk-test")
    captured = {}
    _stub_anthropic(client, _anthropic_response('{"a": "x"}'), captured)
    client.complete_json("sys", MESSAGES, ENVELOPE, "env")
    assert captured["extra_body"]["temperature"] == 0.5
    assert "fallbacks" not in captured["extra_body"]


@pytest.mark.parametrize("stop, match", [("refusal", "declined"), ("max_tokens", "token limit")])
def test_anthropic_stop_reasons(stop, match):
    client = AnthropicClient(ANTHROPIC.models[1], 0.5, "sk-test")
    _stub_anthropic(client, _anthropic_response("{}", stop=stop), {})
    with pytest.raises(LLMError, match=match):
        client.complete_json("sys", MESSAGES, ENVELOPE, "env")


def test_anthropic_auth_error_mapped():
    import anthropic
    import httpx2

    client = AnthropicClient(ANTHROPIC.models[1], 0.5, "sk-test")
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.AuthenticationError("bad key", response=httpx2.Response(401, request=request), body=None)

    def create(**_):
        raise error

    client._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    with pytest.raises(LLMError, match="rejected the API key"):
        client.complete_json("sys", MESSAGES, ENVELOPE, "env")


@pytest.mark.parametrize(
    "code, match",
    [("insufficient_quota", "no API credit"), ("rate_limit_exceeded", "Wait a minute")],
)
def test_openai_429_quota_vs_rate_limit(code, match):
    import httpx2
    import openai

    client = OpenAIClient(OPENAI.models[1], 0.3, "sk-test")
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    error = openai.RateLimitError(
        "429", response=httpx2.Response(429, request=request), body={"code": code, "message": "x"}
    )

    def create(**_):
        raise error

    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(LLMError, match=match):
        client.complete_json("sys", MESSAGES, ENVELOPE, "env")


def test_gemini_uses_compatible_endpoint_and_max_tokens():
    from core.config import GEMINI

    client = build_client("Google Gemini", "gemini-2.5-flash", 0.2, "AIzaTestKey")
    assert isinstance(client, OpenAIClient)
    assert str(client._client.base_url).startswith(GEMINI.base_url.rstrip("/"))
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _openai_response('```json\n{"a": "x"}\n```')

    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert client.complete_json("sys", MESSAGES, ENVELOPE, "env") == {"a": "x"}
    assert captured["max_tokens"] == 16000 and "max_completion_tokens" not in captured


def _bad_request(message):
    import httpx2
    import openai

    request = httpx2.Request("POST", "https://example.test/chat/completions")
    return openai.BadRequestError(message, response=httpx2.Response(400, request=request), body=None)


def test_gemini_falls_back_to_json_mode_when_schema_rejected():
    client = build_client("Google Gemini", "gemini-2.5-flash", 0.2, "AIzaTestKey")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if kwargs["response_format"]["type"] == "json_schema":
            raise _bad_request("Invalid JSON payload: unknown field 'strict'")
        return _openai_response('{"a": "x"}')

    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert client.complete_json("sys", MESSAGES, ENVELOPE, "env") == {"a": "x"}
    assert [c["response_format"]["type"] for c in calls] == ["json_schema", "json_object"]
    assert '"required": ["a"]' in calls[1]["messages"][0]["content"]


def test_gemini_invalid_key_400_mapped():
    client = build_client("Google Gemini", "gemini-2.5-flash", 0.2, "AIzaTestKey")

    def create(**_):
        raise _bad_request("API key not valid. Please pass a valid API key.")

    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(LLMError, match="rejected the API key.*aistudio"):
        client.complete_json("sys", MESSAGES, ENVELOPE, "env")


@pytest.mark.parametrize(
    "provider, model, key",
    [("OpenAI", "gpt-4o", "AIzaTestKey"), ("Google Gemini", "gemini-2.5-flash", "sk-proj-abc")],
)
def test_gemini_key_mismatch(provider, model, key):
    with pytest.raises(LLMError, match="Gemini"):
        build_client(provider, model, 0.2, key)
