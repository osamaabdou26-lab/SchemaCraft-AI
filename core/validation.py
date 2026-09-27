"""Programmatic validation of every generated artifact.

* JSON Schema: metaschema check (Draft 7 / 2020-12) plus SchemaCraft's own
  structural rules (required metadata, ``required`` subset of ``properties``,
  resolvable ``$ref``s).
* Mock data: ``jsonschema`` validation of each record, with format checking.
* Pydantic code: a static AST allowlist, then an isolated import of the module
  and validation of the mock records through the root model.
"""

from __future__ import annotations

import ast
import json
import sys
import types
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from jsonschema import Draft7Validator, FormatChecker, validators
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ValidationError

from .config import SUPPORTED_DIALECTS

REQUIRED_ROOT_KEYS = ("$schema", "title", "type", "properties", "required", "additionalProperties")

# --------------------------------------------------------------------------- #
# JSON Schema
# --------------------------------------------------------------------------- #


def _validator_class(schema: dict[str, Any]) -> type:
    """Pick the validator matching ``$schema``; Draft 7 when unspecified."""
    return validators.validator_for(schema, default=Draft7Validator)


def _iter_object_schemas(node: Any, path: str = "#") -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (json-pointer, subschema) for every subschema that declares properties."""
    if isinstance(node, dict):
        if isinstance(node.get("properties"), dict):
            yield path, node
        for key, value in node.items():
            if key in ("enum", "const", "examples", "default"):
                continue  # literal values, not subschemas
            yield from _iter_object_schemas(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _iter_object_schemas(item, f"{path}/{index}")


def _iter_refs(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            yield ref
        for key, value in node.items():
            if key not in ("enum", "const", "examples", "default"):
                yield from _iter_refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_refs(item)


def _resolve_local_ref(schema: dict[str, Any], ref: str) -> bool:
    """Return True if a local '#/...' JSON pointer resolves inside ``schema``."""
    if ref == "#":
        return True
    node: Any = schema
    for raw in ref[2:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        else:
            return False
    return True


def validate_schema(schema: Any) -> list[str]:
    """Return a list of problems with a generated schema; empty means valid."""
    if not isinstance(schema, dict):
        return [f"The schema must be a JSON object, got {type(schema).__name__}."]

    errors: list[str] = []
    missing = [k for k in REQUIRED_ROOT_KEYS if k not in schema]
    if missing:
        errors.append(f"Root is missing required metadata keys: {', '.join(missing)}.")

    dialect = schema.get("$schema")
    if dialect is not None and dialect not in SUPPORTED_DIALECTS:
        errors.append(
            f"Unsupported $schema '{dialect}'. Use 'http://json-schema.org/draft-07/schema#' "
            f"or 'https://json-schema.org/draft/2020-12/schema'."
        )
    if schema.get("type") != "object":
        errors.append(f"Root 'type' must be 'object', got {schema.get('type')!r}.")
    if "title" in schema and not (isinstance(schema["title"], str) and schema["title"].strip()):
        errors.append("Root 'title' must be a non-empty string.")
    if "additionalProperties" in schema and schema["additionalProperties"] is not False:
        errors.append("Root 'additionalProperties' must be false.")

    # Metaschema validation: catches wrong keyword types, bad enums, etc.
    try:
        _validator_class(schema).check_schema(schema)
    except SchemaError as exc:
        location = "/".join(str(p) for p in exc.path) or "(root)"
        errors.append(f"Schema is not a valid JSON Schema at '{location}': {exc.message}")

    # Structural rules the metaschema does not enforce.
    for pointer, obj in _iter_object_schemas(schema):
        required = obj.get("required", [])
        if isinstance(required, list):
            unknown = [name for name in required if name not in obj["properties"]]
            if unknown:
                errors.append(f"At '{pointer}': required lists undefined properties {unknown}.")

    for ref in sorted(set(_iter_refs(schema))):
        if ref.startswith("#") and not _resolve_local_ref(schema, ref):
            errors.append(f"Unresolvable $ref '{ref}'.")

    return errors


def schema_warnings(schema: dict[str, Any]) -> list[str]:
    """Non-fatal quality hints (shown in the UI, not sent back to the LLM)."""
    warnings = []
    for pointer, obj in _iter_object_schemas(schema):
        if pointer != "#" and "additionalProperties" not in obj:
            warnings.append(f"Object at '{pointer}' does not set additionalProperties.")
    return warnings


# --------------------------------------------------------------------------- #
# Mock data
# --------------------------------------------------------------------------- #


def validate_records(schema: dict[str, Any], records: Any, expected_count: int | None = None) -> list[str]:
    """Validate every record against the schema; return human-readable errors."""
    if not isinstance(records, list):
        return [f"Mock data must be a JSON array of records, got {type(records).__name__}."]

    errors: list[str] = []
    if expected_count is not None and len(records) != expected_count:
        errors.append(f"Expected exactly {expected_count} records, got {len(records)}.")

    try:
        validator = _validator_class(schema)(schema, format_checker=FormatChecker())
        for index, record in enumerate(records):
            for err in sorted(validator.iter_errors(record), key=lambda e: list(e.path)):
                location = "/".join(str(p) for p in err.absolute_path) or "(root)"
                errors.append(f"Record {index} at '{location}': {err.message}")
    except SchemaError as exc:
        errors.append(f"Schema could not be applied to the data: {exc}")
    except Exception as exc:  # unresolvable $refs raise version-specific referencing errors
        errors.append(f"Schema could not be applied to the data: {type(exc).__name__}: {exc}")
    return errors


# --------------------------------------------------------------------------- #
# Pydantic code
# --------------------------------------------------------------------------- #

ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "__future__", "pydantic", "typing", "typing_extensions", "datetime", "enum",
        "uuid", "decimal", "re", "ipaddress", "annotated_types",
    }
)

FORBIDDEN_NAMES = frozenset(
    {
        "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars",
        "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help",
        "memoryview", "__builtins__", "__loader__", "__spec__",
    }
)

_ALLOWED_TOP_LEVEL = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Assign, ast.AnnAssign)
_FORBIDDEN_NODES = (ast.While, ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await, ast.Try)


def _is_docstring(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def _is_model_rebuild(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "model_rebuild"
    )


def check_code_safety(code: str) -> list[str]:
    """Statically vet generated code before it is ever executed.

    Only declarative model code passes: allowlisted imports, class definitions,
    type aliases and ``model_rebuild()`` calls. Dunder attribute access and
    introspection builtins are rejected, which closes the usual sandbox escapes.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError on line {exc.lineno}: {exc.msg}"]

    errors: list[str] = []
    for stmt in tree.body:
        if not (isinstance(stmt, _ALLOWED_TOP_LEVEL) or _is_docstring(stmt) or _is_model_rebuild(stmt)):
            errors.append(
                f"Line {stmt.lineno}: top-level {type(stmt).__name__} is not allowed; "
                f"only imports, classes, type aliases and model_rebuild() calls."
            )

    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORT_ROOTS:
                    errors.append(f"Line {line}: import of '{alias.name}' is not allowed.")
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module or node.module.split(".")[0] not in ALLOWED_IMPORT_ROOTS:
                errors.append(f"Line {line}: import from '{node.module or '.'}' is not allowed.")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"Line {line}: use of '{node.id}' is not allowed.")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            errors.append(f"Line {line}: dunder attribute access '.{node.attr}' is not allowed.")
        elif isinstance(node, _FORBIDDEN_NODES):
            errors.append(f"Line {line}: {type(node).__name__} statements are not allowed.")
    return errors


@dataclass
class ModelLoadResult:
    """Outcome of importing generated Pydantic code."""

    models: dict[str, type[BaseModel]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_models(code: str, root_model: str) -> ModelLoadResult:
    """Safety-check, then import the code as a throwaway module and collect its models."""
    result = ModelLoadResult(errors=check_code_safety(code))
    if result.errors:
        return result

    module_name = f"schemacraft_generated_{uuid.uuid4().hex}"
    module = types.ModuleType(module_name)
    # Registering the module lets Pydantic resolve postponed (string) annotations.
    sys.modules[module_name] = module
    try:
        exec(compile(code, f"<{module_name}>", "exec"), module.__dict__)  # noqa: S102 - vetted above
        models = {
            name: obj
            for name, obj in vars(module).items()
            if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
            and obj.__module__ == module_name
        }
        for model in models.values():
            model.model_rebuild(_types_namespace=module.__dict__)
        result.models = models
    except Exception as exc:  # any import-time failure is a code-quality error to report
        result.errors.append(f"Importing the models failed: {type(exc).__name__}: {exc}")
        return result
    finally:
        sys.modules.pop(module_name, None)

    if not result.models:
        result.errors.append("The code defines no pydantic.BaseModel subclasses.")
    elif root_model not in result.models:
        result.errors.append(
            f"root_model '{root_model}' is not defined. Defined models: {sorted(result.models)}."
        )
    return result


def validate_records_with_model(model: type[BaseModel], records: list[Any]) -> list[str]:
    """Cross-check mock data through the Pydantic root model (JSON mode parsing)."""
    errors: list[str] = []
    for index, record in enumerate(records):
        try:
            model.model_validate_json(json.dumps(record))
        except ValidationError as exc:
            for err in exc.errors():
                location = "/".join(str(p) for p in err["loc"]) or "(root)"
                errors.append(f"Record {index} at '{location}': {err['msg']}")
    return errors
