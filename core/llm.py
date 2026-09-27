"""Provider-neutral LLM access with structured (JSON-schema constrained) output.

Each provider is driven through its native structured-output mode:

* OpenAI: ``response_format={"type": "json_schema", "strict": True, ...}``
* Anthropic: ``output_config={"format": {"type": "json_schema", ...}}``
* Google Gemini (free tier): the same OpenAI request against Gemini's
  OpenAI-compatible endpoint, falling back to plain JSON mode if needed.

Each call returns a parsed ``dict`` that already matches the requested envelope
schema, so callers never have to strip markdown fences or prose.
"""

from __future__ import annotations

import json
import os
import re
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
    raw = (override or "").strip() or os.environ.get(provider.env_var, "").strip()
    # Keys pasted from docs or .env files often keep their surrounding quotes.
    key = raw.strip("\"'").strip()
    if not key:
        raise MissingAPIKeyError(
            f"{provider.env_var} is not set. Export it in your shell or add it to a "
            f".env file, or paste a key in the sidebar for this session."
        )
    if any(ch.isspace() for ch in key):
        raise LLMError("The API key contains spaces or line breaks. Copy it again without them.")

    # Catch the most common 401 cause before any request: a key for the other provider.
    is_anthropic_key = key.startswith("sk-ant-")
    if provider.name == "OpenAI" and is_anthropic_key:
        raise LLMError(
            "This is an Anthropic (Claude) key (it starts with 'sk-ant-'). "
            "Switch the LLM provider to Anthropic, or use an OpenAI key."
        )
    if provider.name == "Anthropic" and key.startswith("sk-") and not is_anthropic_key:
        raise LLMError(
            "This looks like an OpenAI key; Anthropic keys start with 'sk-ant-'. "
            "Switch the LLM provider to OpenAI, or use an Anthropic key."
        )
    if provider.name == "OpenAI" and key.startswith("AIza"):
        raise LLMError("This is a Google Gemini key. Switch the LLM provider to Google Gemini.")
    if provider.name == "Google Gemini" and key.startswith("sk-"):
        raise LLMError(
            "This is an OpenAI or Anthropic key; Gemini keys start with 'AIza'. "
            f"Get a free one at {provider.key_url}."
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
    """OpenAI Chat Completions with strict JSON-schema structured outputs.

    Also serves OpenAI-compatible providers (Google Gemini) via ``provider.base_url``.
    """

    def __init__(
        self, model: ModelSpec, temperature: float, api_key: str, provider: ProviderSpec | None = None
    ) -> None:
        import openai  # Imported lazily so the app starts even if one SDK is missing.

        self._openai = openai
        self._model = model
        self._temperature = temperature
        self._provider = provider or PROVIDERS["OpenAI"]
        self._label = self._provider.name
        self._compat = self._provider.base_url is not None
        self._client = openai.OpenAI(
            api_key=api_key, base_url=self._provider.base_url, timeout=REQUEST_TIMEOUT_S, max_retries=2
        )

    def _request(self, system, messages, output_schema, schema_name, strict_schema: bool) -> dict[str, Any]:
        if strict_schema:
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": output_schema},
            }
        else:
            # Plain JSON mode: the envelope shape is spelled out in the system prompt instead.
            response_format = {"type": "json_object"}
            system = (
                f"{system}\n\nReturn only a JSON object that matches this JSON Schema:\n"
                f"{json.dumps(output_schema)}"
            )
        request: dict[str, Any] = {
            "model": self._model.id,
            "messages": [{"role": "system", "content": system}, *messages],
            "response_format": response_format,
        }
        # OpenAI now names the output cap max_completion_tokens; compatible APIs use max_tokens.
        request["max_tokens" if self._compat else "max_completion_tokens"] = MAX_OUTPUT_TOKENS
        if self._model.supports_temperature:
            request["temperature"] = self._temperature
        return request

    def _call(self, request: dict[str, Any]):
        openai, label = self._openai, self._label
        try:
            return self._client.chat.completions.create(**request)
        except openai.AuthenticationError as exc:
            raise LLMError(f"{label} rejected the API key (401). Check {self._provider.env_var}.") from exc
        except openai.PermissionDeniedError as exc:
            raise LLMError(f"{label} denied access to model '{self._model.id}' (403).") from exc
        except openai.NotFoundError as exc:
            hint = " Pick another model in the sidebar." if self._provider.discover_models else ""
            raise LLMError(
                f"{label} model '{self._model.id}' was not found or is not available to your key (404).{hint}"
            ) from exc
        except openai.RateLimitError as exc:
            # OpenAI uses 429 both for "no credit" and for real rate limiting.
            if getattr(exc, "code", None) == "insufficient_quota":
                raise LLMError(
                    "Your OpenAI account has no API credit (429 insufficient_quota). API usage is "
                    "billed separately from ChatGPT: add credit at "
                    "https://platform.openai.com/settings/organization/billing and retry, "
                    "or switch to Google Gemini, which has a free tier."
                ) from exc
            if self._provider.free_tier:
                raise LLMError(
                    f"{label} free-tier limit reached (429). Wait a minute and retry, "
                    f"or pick a lighter model such as Flash-Lite."
                ) from exc
            raise LLMError(f"{label} rate limit exceeded (429). Wait a minute and try again.") from exc
        except openai.BadRequestError as exc:
            # Gemini reports an invalid key as 400 rather than 401.
            if "api key" in str(exc).lower():
                raise LLMError(
                    f"{label} rejected the API key. Check {self._provider.env_var} "
                    f"(create one at {self._provider.key_url})."
                ) from exc
            raise
        except openai.APITimeoutError as exc:
            raise LLMError(f"{label} request timed out. Try a smaller prompt or fewer records.") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"Could not reach the {label} API. Check your network connection.") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"{label} API error ({exc.status_code}): {exc.message}") from exc

    def complete_json(self, system, messages, output_schema, schema_name):
        openai, label = self._openai, self._label
        request = self._request(system, messages, output_schema, schema_name, strict_schema=True)
        try:
            response = self._call(request)
        except openai.BadRequestError as exc:
            if not self._compat:
                raise LLMError(f"{label} rejected the request: {exc.message}") from exc
            # Compatible APIs may not accept strict JSON-schema mode; retry in plain JSON mode.
            fallback = self._request(system, messages, output_schema, schema_name, strict_schema=False)
            try:
                response = self._call(fallback)
            except openai.BadRequestError as exc2:
                raise LLMError(f"{label} rejected the request: {exc2.message}") from exc2

        if not response.choices:
            raise LLMError(f"{label} returned no choices.")
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMError(f"{label} declined the request: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise LLMError(f"{label} output hit the token limit. Simplify the domain or request fewer records.")
        return _parse_json_payload(_strip_code_fence(choice.message.content), label)


def _strip_code_fence(text: str | None) -> str | None:
    """Plain JSON mode on some providers still wraps output in ```json fences."""
    if text is None:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rsplit("```", 1)[0]
    return stripped


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


# Model families that cannot do chat + JSON output (embeddings, media, speech...).
_NON_CHAT_MARKERS = (
    "embedding", "image", "imagen", "veo", "tts", "audio", "live", "aqa",
    "learnlm", "robotics", "computer-use", "native", "vision",
)
_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9.\-]*$")


def _model_sort_key(model_id: str) -> tuple:
    """Newest Flash models first, then Flash-Lite, then everything else."""
    match = re.search(r"gemini-(\d+)(?:\.(\d+))?", model_id)
    version = (int(match.group(1)), int(match.group(2) or 0)) if match else (0, 0)
    is_alias = model_id.endswith("-latest")
    unstable = any(tag in model_id for tag in ("preview", "exp"))
    return (
        "flash" not in model_id,
        "lite" in model_id,
        not is_alias,
        unstable,
        (-version[0], -version[1]),
        model_id,
    )


def filter_chat_models(model_ids: list[str]) -> list[str]:
    """Keep Gemini chat models from a raw model listing, best choices first."""
    cleaned = {m.removeprefix("models/") for m in model_ids}
    usable = [
        m for m in cleaned
        if m.startswith("gemini-") and _MODEL_ID_RE.match(m)
        and not any(marker in m for marker in _NON_CHAT_MARKERS)
    ]
    return sorted(usable, key=_model_sort_key)


def list_available_models(provider_name: str, api_key_override: str | None = None) -> list[str]:
    """Return the chat model IDs the key can use (for providers with discover_models)."""
    provider = PROVIDERS.get(provider_name)
    if provider is None:
        raise LLMError(f"Unknown provider '{provider_name}'.")
    if not provider.discover_models:
        return [m.id for m in provider.models]

    api_key = resolve_api_key(provider, api_key_override)
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=provider.base_url, timeout=20.0, max_retries=1)
    try:
        ids = [m.id for m in client.models.list()]
    except openai.BadRequestError as exc:
        if "api key" in str(exc).lower():
            raise LLMError(f"{provider.name} rejected the API key. Check it and try again.") from exc
        raise LLMError(f"Could not list {provider.name} models: {exc.message}") from exc
    except openai.APIError as exc:
        raise LLMError(f"Could not list {provider.name} models: {exc}") from exc

    models = filter_chat_models(ids)
    if not models:
        raise LLMError(f"{provider.name} returned no usable chat models for this key.")
    return models


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
    if model is None and provider.discover_models and _MODEL_ID_RE.match(model_id or ""):
        model = ModelSpec(model_id, model_id)  # a model discovered from the provider's listing
    if model is None:
        raise LLMError(f"Unknown model '{model_id}' for {provider_name}.")

    api_key = resolve_api_key(provider, api_key_override)
    try:
        if provider.name == "Anthropic":
            return AnthropicClient(model, temperature, api_key)
        return OpenAIClient(model, temperature, api_key, provider)
    except ImportError as exc:
        package = "anthropic" if provider.name == "Anthropic" else "openai"
        raise LLMError(f"The '{package}' package is not installed. Run: pip install -r requirements.txt") from exc
