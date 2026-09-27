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
    assert isinstance(build_client("Anthropic", "claude-haiku-4-5", 0.2, "sk-test"), AnthropicClient)


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
