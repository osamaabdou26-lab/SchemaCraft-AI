"""The three-stage generation pipeline with automatic self-correction.

    description ──▶ [1] JSON Schema ──▶ [2] Pydantic models ──▶ [3] mock data
                         ▲   │               ▲   │                 ▲   │
                         └───┘ retry         └───┘ retry           └───┘ retry

Each stage validates its output programmatically. On failure the validation
errors are appended to the conversation and the LLM is asked to fix them, up to
``max_retries`` extra attempts. A schema that never validates aborts the run,
since the other stages depend on it; the later stages degrade gracefully and
report their remaining errors instead.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from . import prompts
from .config import MAX_FEEDBACK_ERRORS, MAX_PROMPT_CHARS, MAX_RECORDS, MIN_PROMPT_CHARS, MIN_RECORDS
from .llm import LLMClient, Message
from .validation import (
    load_models,
    schema_warnings,
    validate_records,
    validate_records_with_model,
    validate_schema,
)

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]


class InvalidPromptError(ValueError):
    """The user's description cannot be turned into a schema."""


class PipelineError(RuntimeError):
    """A stage could not produce a valid result within the retry budget."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


@dataclass
class StageAttempt:
    """One LLM round-trip of one stage, kept for the pipeline log."""

    stage: str
    attempt: int
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class GenerationResult:
    """Everything the UI needs to render the three output tabs."""

    schema: dict[str, Any]
    pydantic_code: str = ""
    root_model: str = ""
    records: list[Any] = field(default_factory=list)
    schema_warnings: list[str] = field(default_factory=list)
    model_errors: list[str] = field(default_factory=list)
    data_errors: list[str] = field(default_factory=list)
    model_crosscheck_errors: list[str] = field(default_factory=list)
    attempts: list[StageAttempt] = field(default_factory=list)

    @property
    def models_ok(self) -> bool:
        return bool(self.pydantic_code) and not self.model_errors

    @property
    def data_ok(self) -> bool:
        return bool(self.records) and not self.data_errors


def validate_description(description: str) -> str:
    """Normalize and bounds-check the user's domain description."""
    text = (description or "").strip()
    if len(text) < MIN_PROMPT_CHARS:
        raise InvalidPromptError(
            f"Describe your domain in at least {MIN_PROMPT_CHARS} characters, "
            f"e.g. the entities, their fields and any constraints."
        )
    if len(text) > MAX_PROMPT_CHARS:
        raise InvalidPromptError(
            f"The description is {len(text):,} characters; the limit is {MAX_PROMPT_CHARS:,}."
        )
    return text


def _truncate(errors: list[str]) -> list[str]:
    if len(errors) <= MAX_FEEDBACK_ERRORS:
        return errors
    hidden = len(errors) - MAX_FEEDBACK_ERRORS
    return errors[:MAX_FEEDBACK_ERRORS] + [f"...and {hidden} more errors of the same kind."]


def _parse_json_string(raw: Any, what: str) -> tuple[Any, list[str]]:
    """Decode a JSON document the LLM returned as a string field."""
    if not isinstance(raw, str):
        return None, [f"{what} must be a JSON string, got {type(raw).__name__}."]
    try:
        return json.loads(raw), []
    except json.JSONDecodeError as exc:
        return None, [f"{what} is not valid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}."]


class SchemaCraftPipeline:
    """Runs the three stages against any ``LLMClient``."""

    def __init__(
        self,
        llm: LLMClient,
        max_retries: int = 2,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self._llm = llm
        self._max_attempts = max(0, max_retries) + 1
        self._progress = on_progress or (lambda _msg: None)
        self._attempts: list[StageAttempt] = []

    # ------------------------------------------------------------------ #
    # Generic self-correcting loop
    # ------------------------------------------------------------------ #

    def _run_stage(
        self,
        stage: str,
        artifact: str,
        system: str,
        first_message: str,
        envelope: dict[str, Any],
        envelope_name: str,
        check: Callable[[dict[str, Any]], tuple[Any, list[str]]],
    ) -> tuple[Any, list[str]]:
        """Call the LLM, validate with ``check``, and feed errors back until clean.

        Returns the last parsed value and its remaining errors (empty on success).
        """
        messages: list[Message] = [{"role": "user", "content": first_message}]
        value: Any = None
        errors: list[str] = []

        for attempt in range(1, self._max_attempts + 1):
            suffix = "" if attempt == 1 else f" (self-correction {attempt - 1}/{self._max_attempts - 1})"
            self._progress(f"{stage}{suffix}…")

            envelope_out = self._llm.complete_json(system, messages, envelope, envelope_name)
            value, errors = check(envelope_out)
            self._attempts.append(StageAttempt(stage, attempt, list(errors)))
            if not errors:
                return value, []

            log.info("%s attempt %d failed: %s", stage, attempt, errors)
            messages += [
                {"role": "assistant", "content": json.dumps(envelope_out, ensure_ascii=False)},
                {"role": "user", "content": prompts.correction_message(artifact, _truncate(errors))},
            ]
        return value, errors

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #

    def generate_schema(self, description: str) -> dict[str, Any]:
        def check(out: dict[str, Any]) -> tuple[Any, list[str]]:
            if out.get("is_valid_domain") is False:
                reason = str(out.get("rejection_reason") or "it does not describe a data structure.")
                raise InvalidPromptError(f"The request could not be turned into a schema: {reason}")
            schema, errors = _parse_json_string(out.get("schema_json"), "schema_json")
            return schema, errors or validate_schema(schema)

        schema, errors = self._run_stage(
            "Generating JSON Schema", "JSON Schema", prompts.SCHEMA_SYSTEM,
            prompts.schema_user_message(description), prompts.SCHEMA_ENVELOPE, "json_schema_result", check,
        )
        if errors:
            raise PipelineError(
                f"The JSON Schema still failed validation after {self._max_attempts} attempts.", errors
            )
        return schema

    def generate_models(self, schema: dict[str, Any]) -> tuple[str, str, Any, list[str]]:
        """Return (code, root_model, ModelLoadResult | None, errors)."""
        state: dict[str, Any] = {}

        def check(out: dict[str, Any]) -> tuple[Any, list[str]]:
            code = out.get("code")
            root = out.get("root_model")
            if not isinstance(code, str) or not code.strip():
                return None, ["code is empty."]
            if not isinstance(root, str) or not root.strip():
                return None, ["root_model is empty."]
            loaded = load_models(code, root.strip())
            state.update(code=code, root=root.strip(), loaded=loaded)
            return loaded, loaded.errors

        loaded, errors = self._run_stage(
            "Writing Pydantic v2 models", "models.py", prompts.MODELS_SYSTEM,
            prompts.models_user_message(schema), prompts.MODELS_ENVELOPE, "pydantic_models_result", check,
        )
        return state.get("code", ""), state.get("root", ""), loaded, errors

    def generate_records(self, schema: dict[str, Any], count: int) -> tuple[list[Any], list[str]]:
        def check(out: dict[str, Any]) -> tuple[Any, list[str]]:
            records, errors = _parse_json_string(out.get("records_json"), "records_json")
            if errors:
                return None, errors
            return records, validate_records(schema, records, expected_count=count)

        records, errors = self._run_stage(
            "Generating mock data", "records_json array", prompts.DATA_SYSTEM,
            prompts.data_user_message(schema, count), prompts.DATA_ENVELOPE, "mock_data_result", check,
        )
        return records if isinstance(records, list) else [], errors

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #

    def run(self, description: str, record_count: int) -> GenerationResult:
        """Execute all three stages. Raises InvalidPromptError / PipelineError / LLMError."""
        description = validate_description(description)
        if not MIN_RECORDS <= record_count <= MAX_RECORDS:
            raise InvalidPromptError(f"Record count must be between {MIN_RECORDS} and {MAX_RECORDS}.")
        self._attempts = []

        schema = self.generate_schema(description)
        result = GenerationResult(schema=schema, schema_warnings=schema_warnings(schema))

        code, root, loaded, model_errors = self.generate_models(schema)
        result.pydantic_code, result.root_model, result.model_errors = code, root, model_errors

        result.records, result.data_errors = self.generate_records(schema, record_count)

        # Cross-check: the data that passed jsonschema should also parse through the models.
        if result.data_ok and loaded is not None and loaded.ok:
            self._progress("Cross-checking mock data with the Pydantic models…")
            result.model_crosscheck_errors = validate_records_with_model(loaded.models[root], result.records)

        result.attempts = list(self._attempts)
        self._progress("Done.")
        return result
