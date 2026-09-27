"""Unit tests for schema, data and Pydantic-code validation."""

import copy

from conftest import GOOD_CODE, GOOD_RECORDS, GOOD_SCHEMA

from core.validation import (
    check_code_safety,
    load_models,
    validate_records,
    validate_records_with_model,
    validate_schema,
)


def test_good_schema_is_valid():
    assert validate_schema(GOOD_SCHEMA) == []


def test_2020_12_dialect_accepted():
    schema = copy.deepcopy(GOOD_SCHEMA)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$defs"] = schema.pop("definitions")
    schema["properties"]["courses"]["items"]["$ref"] = "#/$defs/Course"
    assert validate_schema(schema) == []


def test_missing_metadata_reported():
    schema = copy.deepcopy(GOOD_SCHEMA)
    del schema["$schema"], schema["additionalProperties"]
    errors = validate_schema(schema)
    assert any("$schema" in e and "additionalProperties" in e for e in errors)


def test_metaschema_violation_reported():
    schema = copy.deepcopy(GOOD_SCHEMA)
    schema["properties"]["gpa"]["minimum"] = "zero"
    assert any("not a valid JSON Schema" in e for e in validate_schema(schema))


def test_required_must_exist_in_properties():
    schema = copy.deepcopy(GOOD_SCHEMA)
    schema["required"].append("ghost")
    assert any("ghost" in e for e in validate_schema(schema))


def test_unresolvable_ref_reported():
    schema = copy.deepcopy(GOOD_SCHEMA)
    schema["properties"]["courses"]["items"]["$ref"] = "#/definitions/Nope"
    assert any("Unresolvable $ref" in e for e in validate_schema(schema))


def test_non_object_schema():
    assert validate_schema(["not", "a", "schema"])


def test_records_valid():
    assert validate_records(GOOD_SCHEMA, GOOD_RECORDS, expected_count=2) == []


def test_records_errors_are_located():
    bad = copy.deepcopy(GOOD_RECORDS)
    bad[1]["status"] = "expelled"
    bad[0]["email"] = "not-an-email"
    bad[0]["extra"] = 1
    errors = validate_records(GOOD_SCHEMA, bad, expected_count=3)
    assert any("Expected exactly 3" in e for e in errors)
    assert any("Record 1 at 'status'" in e for e in errors)
    assert any("Record 0 at 'email'" in e for e in errors)
    assert any("extra" in e for e in errors)


def test_records_must_be_list():
    assert validate_records(GOOD_SCHEMA, {"a": 1})


def test_good_code_loads_and_validates_records():
    loaded = load_models(GOOD_CODE, "Student")
    assert loaded.ok, loaded.errors
    assert validate_records_with_model(loaded.models["Student"], GOOD_RECORDS) == []
    bad = [dict(GOOD_RECORDS[0], gpa=9)]
    assert validate_records_with_model(loaded.models["Student"], bad)


def test_missing_root_model():
    loaded = load_models(GOOD_CODE, "Nope")
    assert any("root_model 'Nope'" in e for e in loaded.errors)


def test_syntax_error_reported():
    assert any("SyntaxError" in e for e in check_code_safety("class A(:\n  pass"))


def test_unsafe_code_rejected_and_not_executed(tmp_path):
    marker = tmp_path / "pwned"
    attacks = [
        f"import os\nos.system('touch {marker}')",
        f"open('{marker}', 'w')",
        "from pydantic import BaseModel\nclass A(BaseModel):\n    x: int = ().__class__.__base__.__subclasses__()",
        "import subprocess",
        "from . import thing",
        "while True:\n    pass",
        "def helper():\n    return 1",
        "print('side effect')",
        "__import__('os')",
    ]
    for code in attacks:
        assert check_code_safety(code), code
        assert not load_models(code, "A").ok
    assert not marker.exists()


def test_model_rebuild_and_validators_allowed():
    code = GOOD_CODE + (
        "\n\nfrom pydantic import field_validator\n\n"
        "class Wrapper(BaseModel):\n"
        '    """Wrapper."""\n'
        "    student: Student\n\n"
        "    @field_validator('student')\n"
        "    @classmethod\n"
        "    def check(cls, v):\n"
        "        return v\n\n"
        "Wrapper.model_rebuild()\n"
    )
    loaded = load_models(code, "Wrapper")
    assert loaded.ok, loaded.errors


def test_few_shot_example_is_self_consistent():
    """The example baked into the prompts must pass our own validators."""
    from core import prompts

    assert validate_schema(prompts._EXAMPLE_SCHEMA) == []
    assert validate_records(prompts._EXAMPLE_SCHEMA, prompts._EXAMPLE_RECORDS) == []
    loaded = load_models(prompts._EXAMPLE_CODE, "BookLoan")
    assert loaded.ok, loaded.errors
    assert validate_records_with_model(loaded.models["BookLoan"], prompts._EXAMPLE_RECORDS) == []
