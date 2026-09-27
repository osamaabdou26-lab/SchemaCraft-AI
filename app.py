"""SchemaCraft AI — API & JSON Schema generator from plain text.

Streamlit front end. All generation, validation and LLM logic lives in ``core/``;
this module only collects settings, runs the pipeline and renders the results.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.config import (
    DEFAULT_MAX_RETRIES,
    MAX_PROMPT_CHARS,
    MAX_RECORDS,
    MAX_RETRIES_LIMIT,
    MIN_RECORDS,
    PROVIDERS,
)
from core.llm import LLMError, MissingAPIKeyError, build_client
from core.pipeline import GenerationResult, InvalidPromptError, PipelineError, SchemaCraftPipeline
from core.presets import PRESETS

load_dotenv()  # Pick up OPENAI_API_KEY / ANTHROPIC_API_KEY from a local .env file.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("schemacraft")

st.set_page_config(page_title="SchemaCraft AI", page_icon="🧬", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      .sc-badge { display:inline-block; padding:0.25rem 0.7rem; border-radius:999px;
                  font-weight:600; font-size:0.85rem; margin-right:0.4rem; }
      .sc-ok   { background:#123d2a; color:#4ade80; border:1px solid #1f6f4a; }
      .sc-fail { background:#3d1212; color:#f87171; border:1px solid #7a2323; }
      .sc-warn { background:#3d3212; color:#facc15; border:1px solid #7a6423; }
    </style>
    """,
    unsafe_allow_html=True,
)


def badge(text: str, kind: str) -> str:
    """Return an inline HTML status pill (kind: ok | fail | warn)."""
    return f'<span class="sc-badge sc-{kind}">{text}</span>'


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

st.session_state.setdefault("description", "")
st.session_state.setdefault("result", None)


def apply_preset(text: str) -> None:
    """Button callback: load a preset into the text area before it is rendered."""
    st.session_state["description"] = text


# --------------------------------------------------------------------------- #
# Sidebar: configuration
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("⚙️ Configuration")

    provider_name = st.selectbox("LLM provider", list(PROVIDERS))
    provider = PROVIDERS[provider_name]
    model = st.selectbox(
        "Model",
        provider.models,
        format_func=lambda m: f"{m.label}  ·  {m.id}",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        disabled=not model.supports_temperature,
        help="Lower is more deterministic. Schemas benefit from low values.",
    )
    if not model.supports_temperature:
        st.caption(f"ℹ️ {model.label} does not accept a temperature setting; it is ignored.")

    record_count = st.slider("Sample records", MIN_RECORDS, MAX_RECORDS, 3)
    max_retries = st.slider(
        "Self-correction retries",
        0,
        MAX_RETRIES_LIMIT,
        DEFAULT_MAX_RETRIES,
        help="Extra attempts per stage when validation fails; errors are fed back to the model.",
    )

    st.divider()
    st.subheader("🔑 API key")
    env_key_present = bool(os.environ.get(provider.env_var, "").strip())
    if env_key_present:
        st.success(f"`{provider.env_var}` found in environment.")
        api_key_override = None
    else:
        st.warning(f"`{provider.env_var}` is not set.")
        api_key_override = st.text_input(
            "Paste a key for this session",
            type="password",
            help="Used only in memory for this browser session; never written to disk.",
        )

# --------------------------------------------------------------------------- #
# Main: input
# --------------------------------------------------------------------------- #

st.title("🧬 SchemaCraft AI")
st.caption("Describe a data domain in plain English → JSON Schema, Pydantic v2 models and validated mock data.")

st.markdown("**Quick examples**")
preset_cols = st.columns(len(PRESETS))
for col, (label, text) in zip(preset_cols, PRESETS.items()):
    col.button(label, on_click=apply_preset, args=(text,), width="stretch")

st.text_area(
    "Domain description",
    key="description",
    height=200,
    max_chars=MAX_PROMPT_CHARS,
    placeholder=(
        "e.g. A university management system with students, courses, enrollments, "
        "and status constraints…"
    ),
)

generate = st.button("✨ Generate", type="primary", width="stretch")

# --------------------------------------------------------------------------- #
# Run the pipeline
# --------------------------------------------------------------------------- #

if generate:
    st.session_state["result"] = None
    try:
        llm = build_client(provider_name, model.id, temperature, api_key_override)
        with st.status("Running SchemaCraft pipeline…", expanded=True) as status:
            pipeline = SchemaCraftPipeline(llm, max_retries=max_retries, on_progress=status.write)
            result = pipeline.run(st.session_state["description"], record_count)
            status.update(label="Generation complete", state="complete", expanded=False)
        st.session_state["result"] = result
    except MissingAPIKeyError as exc:
        st.warning(f"🔑 {exc}")
    except InvalidPromptError as exc:
        st.warning(f"✍️ {exc}")
    except PipelineError as exc:
        st.error(f"❌ {exc}")
        if exc.errors:
            with st.expander("Remaining validation errors"):
                st.code("\n".join(exc.errors), language="text")
    except LLMError as exc:
        st.error(f"🌐 {exc}")
    except Exception:  # Last-resort guard so the UI never shows a raw traceback.
        log.exception("Unexpected failure during generation")
        st.error("An unexpected error occurred. Check the server logs for details.")

# --------------------------------------------------------------------------- #
# Output tabs
# --------------------------------------------------------------------------- #


def render_schema_tab(result: GenerationResult) -> None:
    schema_text = json.dumps(result.schema, indent=2, ensure_ascii=False)
    st.markdown(badge("✔ Valid JSON Schema", "ok"), unsafe_allow_html=True)
    for warning in result.schema_warnings:
        st.caption(f"⚠️ {warning}")

    code_col, tree_col = st.columns([3, 2])
    with code_col:
        st.code(schema_text, language="json")  # st.code has a built-in copy button.
    with tree_col:
        st.markdown("**Interactive tree**")
        st.json(result.schema, expanded=2)

    st.download_button(
        "⬇️ Download schema.json", schema_text, file_name="schema.json", mime="application/json"
    )


def render_models_tab(result: GenerationResult) -> None:
    if result.models_ok:
        st.markdown(
            badge("✔ Compiles", "ok") + badge(f"Root model: {result.root_model}", "ok"),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(badge("✖ Model validation failed", "fail"), unsafe_allow_html=True)
        with st.expander("Errors", expanded=True):
            st.code("\n".join(result.model_errors), language="text")

    if not result.pydantic_code:
        st.info("No model code was produced.")
        return
    st.code(result.pydantic_code, language="python")
    st.download_button(
        "⬇️ Download models.py", result.pydantic_code, file_name="models.py", mime="text/x-python"
    )


def render_data_tab(result: GenerationResult) -> None:
    if result.data_ok:
        pills = badge(f"✔ jsonschema.validate passed ({len(result.records)} records)", "ok")
        if result.models_ok:
            pills += (
                badge("✔ Pydantic cross-check passed", "ok")
                if not result.model_crosscheck_errors
                else badge("⚠ Pydantic cross-check failed", "warn")
            )
        st.markdown(pills, unsafe_allow_html=True)
    else:
        st.markdown(badge("✖ Schema validation failed", "fail"), unsafe_allow_html=True)
        with st.expander("Validation errors", expanded=True):
            st.code("\n".join(result.data_errors), language="text")

    if result.model_crosscheck_errors:
        with st.expander("Pydantic cross-check errors"):
            st.caption("The data satisfies the JSON Schema but not the generated models.")
            st.code("\n".join(result.model_crosscheck_errors), language="text")

    if not result.records:
        st.info("No records were produced.")
        return

    data_text = json.dumps(result.records, indent=2, ensure_ascii=False)
    table_view, json_view = st.tabs(["Table", "JSON"])
    with table_view:
        try:
            st.dataframe(pd.json_normalize(result.records), width="stretch")
        except Exception:  # Records that are not objects cannot be flattened.
            st.info("These records cannot be shown as a table; see the JSON view.")
    with json_view:
        st.code(data_text, language="json")

    st.download_button(
        "⬇️ Download sample_data.json", data_text, file_name="sample_data.json", mime="application/json"
    )


def render_log(result: GenerationResult) -> None:
    with st.expander(f"🔁 Pipeline log ({len(result.attempts)} LLM calls)"):
        for attempt in result.attempts:
            icon = "✅" if attempt.ok else "🔁"
            st.markdown(f"{icon} **{attempt.stage}** — attempt {attempt.attempt}")
            if attempt.errors:
                st.code("\n".join(attempt.errors[:10]), language="text")


result: GenerationResult | None = st.session_state.get("result")
if result is not None:
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Schema", "Valid")
    m2.metric("Pydantic models", "Compiles" if result.models_ok else "Errors")
    m3.metric("Mock data", f"{len(result.records)} valid" if result.data_ok else "Errors")
    m4.metric("LLM calls", len(result.attempts))

    tab_schema, tab_models, tab_data = st.tabs(["📐 JSON Schema", "🐍 Pydantic v2 Code", "🧪 Mock Data"])
    with tab_schema:
        render_schema_tab(result)
    with tab_models:
        render_models_tab(result)
    with tab_data:
        render_data_tab(result)
    render_log(result)
