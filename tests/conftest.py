"""Shared fixtures: a scripted fake LLM and a known-good schema/code/data triple."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GOOD_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Student",
    "description": "A student.",
    "type": "object",
    "properties": {
        "student_id": {"type": "string", "pattern": "^S[0-9]{7}$"},
        "email": {"type": "string", "format": "email"},
        "status": {"type": "string", "enum": ["active", "graduated"]},
        "gpa": {"type": "number", "minimum": 0, "maximum": 4},
        "courses": {"type": "array", "items": {"$ref": "#/definitions/Course"}},
    },
    "required": ["student_id", "email", "status", "gpa", "courses"],
    "additionalProperties": False,
    "definitions": {
        "Course": {
            "type": "object",
            "properties": {"code": {"type": "string"}, "credits": {"type": "integer", "minimum": 1}},
            "required": ["code", "credits"],
            "additionalProperties": False,
        }
    },
}

GOOD_CODE = '''"""Models."""
from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class Status(str, Enum):
    """Status."""

    ACTIVE = "active"
    GRADUATED = "graduated"


class Course(BaseModel):
    """A course."""

    model_config = ConfigDict(extra="forbid")
    code: str
    credits: int = Field(..., ge=1)


class Student(BaseModel):
    """A student."""

    model_config = ConfigDict(extra="forbid")
    student_id: str = Field(..., pattern=r"^S[0-9]{7}$")
    email: EmailStr
    status: Status
    gpa: float = Field(..., ge=0, le=4)
    courses: List[Course]
'''

GOOD_RECORDS = [
    {"student_id": "S1234567", "email": "a@example.com", "status": "active", "gpa": 3.5,
     "courses": [{"code": "CS101", "credits": 3}]},
    {"student_id": "S7654321", "email": "b@example.com", "status": "graduated", "gpa": 2.9, "courses": []},
]


def schema_envelope(schema: Any, valid: bool = True, reason: str = "") -> dict[str, Any]:
    raw = schema if isinstance(schema, str) else json.dumps(schema)
    return {"is_valid_domain": valid, "rejection_reason": reason, "schema_json": raw}


def models_envelope(code: str = GOOD_CODE, root: str = "Student") -> dict[str, Any]:
    return {"root_model": root, "code": code}


def data_envelope(records: Any) -> dict[str, Any]:
    return {"records_json": records if isinstance(records, str) else json.dumps(records)}


class FakeLLM:
    """Returns queued responses per envelope name and records every call."""

    def __init__(self, responses: dict[str, list[dict[str, Any]]]) -> None:
        self.responses = {k: list(v) for k, v in responses.items()}
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def complete_json(self, system, messages, output_schema, schema_name):
        self.calls.append((schema_name, [dict(m) for m in messages]))
        queue = self.responses[schema_name]
        return queue.pop(0) if len(queue) > 1 else queue[0]


@pytest.fixture
def fake_llm_factory():
    return FakeLLM
