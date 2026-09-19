"""Dependency loading and detached JSON values shared by the execution layer."""

import importlib
import json
import re
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from phl_risk.exceptions import ExperimentError, OptionalDependencyError


def require(module: str, extra: str):
    try:
        backend = importlib.import_module(module)
    except (ImportError, OSError) as exc:
        raise OptionalDependencyError(
            f"{module} is unavailable; install phl-risk[{extra}]. Original error: {exc}"
        ) from exc

    if module == "lightgbm":
        from ._compat import prepare_backend

        backend = prepare_backend(backend)
    return backend


def detached(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (ValueError, TypeError) as exc:
        raise ExperimentError(f"Expected finite JSON-compatible configuration: {exc}") from exc


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError("Name must be a non-empty string")
    return re.sub(r"[^\w-]+", "-", value, flags=re.ASCII).strip("-")[:80] or "run"


def dependency_versions() -> dict:
    result = {}
    for package in ("phl-risk", "lightgbm", "scikit-learn", "hydra-core", "pandas", "numpy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result
