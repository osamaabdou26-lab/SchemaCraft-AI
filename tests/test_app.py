"""End-to-end UI tests with Streamlit's AppTest and a fake LLM (no network)."""

from pathlib import Path

import pytest
from conftest import GOOD_RECORDS, GOOD_SCHEMA, FakeLLM, data_envelope, models_envelope, schema_envelope
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def _click(at, label):
    next(b for b in at.button if b.label == label).click().run()


def test_missing_key_warns(app):
    assert any("OPENAI_API_KEY" in w.value for w in app.sidebar.warning)
    _click(app, "🎓 University Student Portal")
    assert "student" in app.text_area[0].value.lower()
    _click(app, "✨ Generate")
    assert any("OPENAI_API_KEY is not set" in w.value for w in app.warning)
    assert not app.exception


def test_full_generation_renders_tabs(app, monkeypatch):
    llm = FakeLLM({
        "json_schema_result": [schema_envelope(GOOD_SCHEMA)],
        "pydantic_models_result": [models_envelope()],
        "mock_data_result": [data_envelope(GOOD_RECORDS)],
    })
    monkeypatch.setattr("core.llm.build_client", lambda *a, **k: llm)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.sidebar.slider[1].set_value(2).run()  # record count
    at.text_area[0].input("A student with an ID, email, status, GPA and courses.").run()
    _click(at, "✨ Generate")

    assert not at.exception, at.exception
    assert not at.error
    assert [t.label for t in at.tabs][:3] == ["📐 JSON Schema", "🐍 Pydantic v2 Code", "🧪 Mock Data"]
    code_blocks = [c.language for c in at.code]
    assert "json" in code_blocks and "python" in code_blocks
    html = " ".join(m.value for m in at.markdown)
    assert "jsonschema.validate passed (2 records)" in html
    assert "Pydantic cross-check passed" in html
