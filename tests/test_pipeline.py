"""Pipeline tests driven by a scripted fake LLM (no network)."""

import copy

import pytest
from conftest import (
    GOOD_RECORDS,
    GOOD_SCHEMA,
    data_envelope,
    models_envelope,
    schema_envelope,
)

from core.pipeline import InvalidPromptError, PipelineError, SchemaCraftPipeline

DESCRIPTION = "A student with an ID, email, status, GPA and enrolled courses."


def _responses(schema=None, models=None, data=None):
    return {
        "json_schema_result": schema or [schema_envelope(GOOD_SCHEMA)],
        "pydantic_models_result": models or [models_envelope()],
        "mock_data_result": data or [data_envelope(GOOD_RECORDS)],
    }


def test_happy_path(fake_llm_factory):
    llm = fake_llm_factory(_responses())
    progress = []
    result = SchemaCraftPipeline(llm, on_progress=progress.append).run(DESCRIPTION, 2)

    assert result.schema == GOOD_SCHEMA
    assert result.models_ok and result.root_model == "Student"
    assert result.data_ok and result.records == GOOD_RECORDS
    assert result.model_crosscheck_errors == []
    assert [c[0] for c in llm.calls] == ["json_schema_result", "pydantic_models_result", "mock_data_result"]
    assert progress[-1] == "Done."


def test_schema_self_correction_feeds_errors_back(fake_llm_factory):
    broken = copy.deepcopy(GOOD_SCHEMA)
    del broken["additionalProperties"]
    llm = fake_llm_factory(_responses(schema=[schema_envelope("{not json"), schema_envelope(broken),
                                              schema_envelope(GOOD_SCHEMA)]))
    result = SchemaCraftPipeline(llm, max_retries=2).run(DESCRIPTION, 2)

    schema_calls = [msgs for name, msgs in llm.calls if name == "json_schema_result"]
    assert len(schema_calls) == 3
    assert "not valid JSON" in schema_calls[1][-1]["content"]
    assert "additionalProperties" in schema_calls[2][-1]["content"]
    assert schema_calls[2][-2]["role"] == "assistant"
    assert [a.ok for a in result.attempts if a.stage.startswith("Generating JSON")] == [False, False, True]


def test_schema_retry_budget_exhausted(fake_llm_factory):
    llm = fake_llm_factory(_responses(schema=[schema_envelope({"type": "object"})]))
    with pytest.raises(PipelineError) as info:
        SchemaCraftPipeline(llm, max_retries=1).run(DESCRIPTION, 2)
    assert info.value.errors
    assert len(llm.calls) == 2


def test_llm_rejects_non_domain_prompt(fake_llm_factory):
    llm = fake_llm_factory(_responses(schema=[schema_envelope("{}", valid=False, reason="It is a greeting.")]))
    with pytest.raises(InvalidPromptError, match="greeting"):
        SchemaCraftPipeline(llm).run("hello there, how are you today?", 1)


@pytest.mark.parametrize("text", ["", "   ", "too short", "x" * 7000])
def test_description_bounds(fake_llm_factory, text):
    with pytest.raises(InvalidPromptError):
        SchemaCraftPipeline(fake_llm_factory(_responses())).run(text, 1)


@pytest.mark.parametrize("count", [0, 11])
def test_record_count_bounds(fake_llm_factory, count):
    with pytest.raises(InvalidPromptError):
        SchemaCraftPipeline(fake_llm_factory(_responses())).run(DESCRIPTION, count)


def test_data_self_correction(fake_llm_factory):
    bad = copy.deepcopy(GOOD_RECORDS)
    bad[0]["gpa"] = 7
    llm = fake_llm_factory(_responses(data=[data_envelope(bad), data_envelope(GOOD_RECORDS)]))
    result = SchemaCraftPipeline(llm).run(DESCRIPTION, 2)
    assert result.data_ok
    data_calls = [msgs for name, msgs in llm.calls if name == "mock_data_result"]
    assert "Record 0 at 'gpa'" in data_calls[1][-1]["content"]


def test_persistent_data_and_model_errors_degrade_gracefully(fake_llm_factory):
    llm = fake_llm_factory(_responses(
        models=[models_envelope(code="import os")],
        data=[data_envelope(GOOD_RECORDS[:1])],
    ))
    result = SchemaCraftPipeline(llm, max_retries=0).run(DESCRIPTION, 2)
    assert not result.models_ok and any("import of 'os'" in e for e in result.model_errors)
    assert not result.data_ok and any("Expected exactly 2" in e for e in result.data_errors)
    assert result.schema == GOOD_SCHEMA


def test_provider_failure_after_schema_keeps_partial_result():
    from core.llm import ProviderOverloadedError

    class FlakyLLM:
        def complete_json(self, system, messages, output_schema, schema_name):
            if schema_name == "json_schema_result":
                return schema_envelope(GOOD_SCHEMA)
            if schema_name == "pydantic_models_result":
                raise ProviderOverloadedError("Google Gemini is overloaded right now (503).")
            return data_envelope(GOOD_RECORDS)

    result = SchemaCraftPipeline(FlakyLLM()).run(DESCRIPTION, 2)
    assert result.schema == GOOD_SCHEMA
    assert not result.models_ok and "503" in result.model_errors[0]
    assert result.data_ok and result.model_crosscheck_errors == []
