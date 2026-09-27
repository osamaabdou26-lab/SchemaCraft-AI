# 🧬 SchemaCraft AI

SchemaCraft AI is a developer-first tool that converts natural language domain descriptions into production-ready JSON Schemas, Pydantic v2 models, and type-validated mock data in seconds. Built to accelerate API design and backend data modeling using LLMs.

Describe a data domain in plain English and get three validated artifacts:

1. **JSON Schema** (Draft 7; 2020-12 documents are also accepted by the validator) → `schema.json`
2. **Pydantic v2 models** → `models.py`
3. **Mock sample data** (1–10 records) that passes `jsonschema` validation → `sample_data.json`

## How it works

```
description ─▶ [1] JSON Schema ─▶ [2] Pydantic models ─▶ [3] mock data ─▶ Pydantic cross-check
                    ▲    │             ▲    │                ▲    │
                    └────┘ retry       └────┘ retry          └────┘ retry
```

- **Structured output only.** OpenAI runs in strict `json_schema` response mode and Anthropic runs with `output_config.format`. Each stage returns a fixed JSON envelope, so there is never any markdown or prose to strip out.
- **Few-shot prompts.** Every stage's system prompt includes a worked example (schema, models and records for a library loan). A unit test checks that the example passes the app's own validators.
- **Self-correction loop.** Each output is checked in code: the schema against its metaschema plus rules for required metadata, `required ⊆ properties` and resolvable `$ref`s; the models by a safe import; the data with `jsonschema` and format checking. Any errors are sent back to the model, which gets a configurable number of retries.
- **Safe model check.** Before the generated Python is imported, an AST allowlist vets it. Only allowlisted imports, classes, type aliases and `model_rebuild()` calls are permitted: no I/O, `eval`/`exec`/`open`, dunder access or `while` loops. The generated records are then validated through the root model as a cross-check.

## Setup

Requires Python 3.10+.

```bash
cd schemacraft-ai
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then add OPENAI_API_KEY and/or ANTHROPIC_API_KEY
streamlit run app.py               # opens http://localhost:8501
```

Instead of using `.env`, you can export the keys in your shell or paste one into the sidebar (it is kept in memory for that session only). If no key is set, the app shows a warning and makes no API call.

## Models

| Provider  | Models                                                 | Temperature |
|-----------|--------------------------------------------------------|-------------|
| OpenAI    | `gpt-4o`, `gpt-4o-mini`                                | yes         |
| Anthropic | `claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5` | Haiku only¹ |

¹ Current Claude Opus/Sonnet models reject sampling parameters, so the slider is disabled for them. `claude-opus-5` also opts in to Anthropic's server-side refusal fallback (`fallbacks: "default"`). Add or remove models in `core/config.py`.

## Project layout

```
app.py              Streamlit UI (sidebar, presets, output tabs, downloads)
core/config.py      Providers, models, limits
core/presets.py     Quick-example domain descriptions
core/prompts.py     System prompts, few-shot example, output envelopes
core/llm.py         OpenAI / Anthropic clients with structured output and error mapping
core/validation.py  Schema, data and generated-code validation
core/pipeline.py    Three-stage pipeline with self-correction
tests/              Offline tests (fake LLM + Streamlit AppTest)
```

## Tests

```bash
python -m pytest -q
```

The tests need no network access and no API key.
