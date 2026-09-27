"""Static configuration: supported providers, models and pipeline limits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """One selectable model and the request options it accepts."""

    id: str
    label: str
    # Current Claude Opus/Sonnet models reject sampling parameters with a 400,
    # so temperature is only sent to models that accept it.
    supports_temperature: bool = True
    # Opt into Anthropic's server-side refusal fallback for models that support it.
    refusal_fallbacks: bool = False


@dataclass(frozen=True)
class ProviderSpec:
    """An LLM provider, the environment variable holding its key, and its models."""

    name: str
    env_var: str
    models: tuple[ModelSpec, ...]


OPENAI = ProviderSpec(
    name="OpenAI",
    env_var="OPENAI_API_KEY",
    models=(
        ModelSpec("gpt-4o", "GPT-4o"),
        ModelSpec("gpt-4o-mini", "GPT-4o mini"),
    ),
)

ANTHROPIC = ProviderSpec(
    name="Anthropic",
    env_var="ANTHROPIC_API_KEY",
    models=(
        ModelSpec("claude-opus-5", "Claude Opus 5", supports_temperature=False, refusal_fallbacks=True),
        ModelSpec("claude-sonnet-5", "Claude Sonnet 5", supports_temperature=False),
        ModelSpec("claude-haiku-4-5", "Claude Haiku 4.5"),
    ),
)

PROVIDERS: dict[str, ProviderSpec] = {p.name: p for p in (OPENAI, ANTHROPIC)}

# Prompt limits: short enough to reject empty/junk input, long enough for real specs.
MIN_PROMPT_CHARS = 15
MAX_PROMPT_CHARS = 6000

MIN_RECORDS = 1
MAX_RECORDS = 10

# Self-correction retries per stage (on top of the first attempt).
DEFAULT_MAX_RETRIES = 2
MAX_RETRIES_LIMIT = 4

# How many validation errors are echoed back to the LLM per correction round.
MAX_FEEDBACK_ERRORS = 12

# Accepted `$schema` dialect URIs (with and without the trailing '#').
DRAFT7_URI = "http://json-schema.org/draft-07/schema#"
DRAFT202012_URI = "https://json-schema.org/draft/2020-12/schema"
SUPPORTED_DIALECTS = frozenset(
    {DRAFT7_URI, DRAFT7_URI.rstrip("#"), DRAFT202012_URI, DRAFT202012_URI + "#"}
)
