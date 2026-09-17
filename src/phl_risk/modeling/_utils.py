"""Small standard-library helpers for immutable declarations."""

from __future__ import annotations

import json
from typing import TypeAlias

from phl_risk.exceptions import PlanError

JSONValue: TypeAlias = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
Columns: TypeAlias = str | list[str] | tuple[str, ...]


def name(value: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{context} must be a non-empty string; got {value!r}")
    return value


def columns(value: Columns, context: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (list, tuple)):
        raise PlanError(f"{context} must be a column string, list or tuple")
    result = tuple(name(item, context) for item in value)
    if not result and not allow_empty:
        raise PlanError(f"{context} must not be empty")
    if len(set(result)) != len(result):
        raise PlanError(f"Duplicate columns in {context}: {result!r}")
    return result


def describe(kind: str, value: dict[str, JSONValue]) -> str:
    return kind + "\n" + json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
