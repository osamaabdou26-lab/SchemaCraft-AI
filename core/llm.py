"""Provider-neutral LLM access with structured (JSON-schema constrained) output.

Both providers are driven through their native structured-output modes:

* OpenAI: ``response_format={"type": "json_schema", "strict": True, ...}``
* Anthropic: ``output_config={"format": {"type": "json_schema", ...}}``

Each call returns a parsed ``dict`` that already matches the requested envelope
schema, so callers never have to strip markdown fences or prose.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from .config import PROVIDERS, ModelSpec, ProviderSpec

# Requests are non-streaming; 16k output tokens keeps them under SDK HTTP timeouts
# while leaving room for large schemas and 10 mock records.
MAX_OUTPUT_TOKENS = 16000
REQUEST_TIMEOUT_S = 180.0

# Anthropic server-side refusal fallback (routes a declined request to another model).
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

Message = dict[str, str]


class LLMError(Exception):
    """A provider call failed. The message is safe to show to the user."""


class MissingAPIKeyError(LLMError):
    """No API key is configured for the selected provider."""


class LLMClient(Protocol):
    """Anything that can turn a conversation into a JSON object matching a schema."""

    def complete_json(
        self,
        system: str,
        messages: list[Message],
        output_schema: dict[str, Any],
        schema_name: str,
    ) -> dict[str, Any]: ...


def resolve_api_key(provider: ProviderSpec, override: str | None = None) -> str:
    """Return the API key from the explicit override or the provider's env var."""
    key = (override or "").strip() or os.environ.get(provider.env_var, "").strip()
    if not key:
        raise MissingAPIKeyError(
            f"{provider.env_var} is not set. Export it in your shell or add it to a "
            f".env file, or paste a key in the sidebar for this session."
        )
    return key


def _parse_json_payload(text: str | None, provider: str) -> dict[str, Any]:
    """Parse the provider's text payload into a JSON object, with a clear error."""
    if not text or not text.strip():
        raise LLMError(f"{provider} returned an empty response.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"{provider} returned malformed JSON: {exc.msg} (pos {exc.pos}).") from exc
    if not isinstance(payload, dict):
        raise LLMError(f"{provider} returned JSON of type {type(payload).__name__}, expected an object.")
    return payload


class OpenAIClient:
    """OpenAI Chat Completions with strict JSON-schema structured outputs."""

    def __init__(self, model: ModelSpec, temperature: float, api_key: str) -> None:
        import openai  # Imported lazily so the app starts even if one SDK is missing.

        self._openai = openai
        self._model = model
        self._temperature = temperature
        self._client = openai.OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_S, max_retries=2)

    def complete_json(self, system, messages, output_schema, schema_name):
        openai = self._openai
        request: dict[str, Any] = {
            "model": self._model.id,
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
            "messages": [{"role": "system", "content": system}, *messages],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": output_schema},
            },
        }
        if self._model.supports_temperature:
            request["temperature"] = self._temperature

        try:
            response = self._client.chat.completions.create(**request)
        except openai.AuthenticationError as exc:
            raise LLMError("OpenAI rejected the API key (401). Check OPENAI_API_KEY.") from exc
        except openai.PermissionDeniedError as exc:
            raise LLMError(f"OpenAI denied access to model '{self._model.id}' (403).") from exc
        except openai.NotFoundError as exc:
            raise LLMError(f"OpenAI model '{self._model.id}' was not found (404).") from exc
        except openai.RateLimitError as exc:
            raise LLMError("OpenAI rate limit or quota exceeded (429). Try again shortly.") from exc
        except openai.BadRequestError as exc:
            raise LLMError(f"OpenAI rejected the request: {exc.message}") from exc
        except openai.APITimeoutError as exc:
            raise LLMError("OpenAI request timed out. Try a smaller prompt or fewer records.") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Could not reach the OpenAI API. Check your network connection.") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"OpenAI API error ({exc.status_code}): {exc.message}") from exc

        if not response.choices:
            raise LLMError("OpenAI returned no choices.")
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMError(f"OpenAI declined the request: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise LLMError("OpenAI output hit the token limit. Simplify the domain or request fewer records.")
        return _parse_json_payload(choice.message.content, "OpenAI")


class AnthropicClient:
    """Anthropic Messages API with JSON-schema structured outputs."""

    def __init__(self, model: ModelSpec, temperature: float, api_key: str) -> None:
        import anthropic  # Imported lazily so the app starts even if one SDK is missing.

        self._anthropic = anthropic
        self._model = model
        self._temperature = temperature
        self._client = anthropic.Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_S, max_retries=2)

    def complete_json(self, system, messages, output_schema, schema_name):
        anthropic = self._anthropic
        # `output_config`, `fallbacks` and `temperature` go through extra_body: the 1.x
        # SDK removed the typed `temperature` argument, and older releases lack typed
        # `output_config`, so this form works across SDK versions.
        extra_body: dict[str, Any] = {
            "output_config": {"format": {"type": "json_schema", "schema": output_schema}}
        }
        extra_headers: dict[str, str] = {}
        if self._model.refusal_fallbacks:
            extra_body["fallbacks"] = "default"
            extra_headers["anthropic-beta"] = _FALLBACK_BETA

        request: dict[str, Any] = {
            "model": self._model.id,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "system": system,
            "messages": messages,
            "extra_body": extra_body,
        }
        if extra_headers:
            request["extra_headers"] = extra_headers
        if self._model.supports_temperature:
            extra_body["temperature"] = self._temperature

        try:
            response = self._client.messages.create(**request)
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key (401). Check ANTHROPIC_API_KEY.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError(f"Anthropic denied access to model '{self._model.id}' (403).") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Anthropic model '{self._model.id}' was not found (404).") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Anthropic rate limit exceeded (429). Try again shortly.") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"Anthropic rejected the request: {exc.message}") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMError("Anthropic request timed out. Try a smaller prompt or fewer records.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API. Check your network connection.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("Claude declined this request. Rephrase the domain description.")
        if response.stop_reason == "max_tokens":
            raise LLMError("Claude output hit the token limit. Simplify the domain or request fewer records.")
        text = "".join(block.text for block in response.content if block.type == "text")
        return _parse_json_payload(text, "Anthropic")


def build_client(
    provider_name: str,
    model_id: str,
    temperature: float,
    api_key_override: str | None = None,
) -> LLMClient:
    """Construct a client for the chosen provider/model, validating the key first."""
    provider = PROVIDERS.get(provider_name)
    if provider is None:
        raise LLMError(f"Unknown provider '{provider_name}'.")
    model = next((m for m in provider.models if m.id == model_id), None)
    if model is None:
        raise LLMError(f"Unknown model '{model_id}' for {provider_name}.")

    api_key = resolve_api_key(provider, api_key_override)
    try:
        if provider is PROVIDERS["OpenAI"]:
            return OpenAIClient(model, temperature, api_key)
        return AnthropicClient(model, temperature, api_key)
    except ImportError as exc:
        package = "openai" if provider.name == "OpenAI" else "anthropic"
        raise LLMError(f"The '{package}' package is not installed. Run: pip install -r requirements.txt") from exc
