"""Dependency loading and detached JSON values shared by experiment modules."""

import importlib
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from phl_risk.exceptions import ExperimentError, OptionalDependencyError


def require(module: str, extra: str):
    try:
        backend = importlib.import_module(module)
    except (ImportError, OSError) as exc:
        raise OptionalDependencyError(
            f"{module} is unavailable; install phl-risk[{extra}]. Original error: {exc}"
        ) from exc

    return backend


def detached(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (ValueError, TypeError) as exc:
        raise ExperimentError(f"Expected finite JSON-compatible configuration: {exc}") from exc


SHANGHAI = ZoneInfo("Asia/Shanghai")


def shanghai_now() -> str:
    return datetime.now(SHANGHAI).isoformat()


def shanghai_time(value: str) -> str:
    """Display an aware historical timestamp in Shanghai without rewriting its record."""
    return datetime.fromisoformat(value).astimezone(SHANGHAI).isoformat()


def slug(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError("Name must be a non-empty string")
    return re.sub(r"[^\w-]+", "-", value, flags=re.ASCII).strip("-")[:80] or "run"


def partition_names(value) -> list[str]:
    """Materialize once so iterators cannot lose their values during validation."""
    from phl_risk.exceptions import RunError

    if isinstance(value, str):
        raise RunError("comparison_partitions must be a sequence of names")
    try:
        names = list(value)
    except TypeError as error:
        raise RunError("comparison_partitions must be a sequence of names") from error
    if any(not isinstance(p, str) or not p for p in names):
        raise RunError("comparison_partitions must be a sequence of names")
    if len(set(names)) != len(names):
        raise RunError("Duplicate comparison_partitions")
    return names
