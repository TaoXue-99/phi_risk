"""Backend-neutral serialization of parameters and callable identities."""

import hashlib
import inspect
from collections.abc import Mapping
from pathlib import Path

from phl_risk.exceptions import RunError

from ._utils import detached


def parameter_snapshot(value):
    """Record callable identity, never executable code or arbitrary repr strings."""
    if callable(value):
        try:
            source_hash = hashlib.sha256(inspect.getsource(value).encode()).hexdigest()
        except (OSError, TypeError):
            source_hash = None
        module = getattr(value, "__module__", None)
        name = getattr(value, "__qualname__", getattr(value, "__name__", None))
        kind = f"{type(value).__module__}.{type(value).__qualname__}"
        identity = f"{module}.{name}" if module and name else f"{kind}:{name}" if name else kind
        return {"callable": identity, "source_sha256": source_hash}
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise RunError("Recorded parameter keys must be strings")
        return {k: parameter_snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [parameter_snapshot(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return detached(value)
