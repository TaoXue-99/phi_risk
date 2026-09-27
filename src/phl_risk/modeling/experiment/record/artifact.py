"""Backend-independent artifact writers, paths and atomic filesystem primitives."""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from phl_risk.exceptions import RunError


def atomic_write(path: Path, writer, *, exclusive=False):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as file:
            temporary = Path(file.name)
        writer(temporary)
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunError(f"Invalid artifact {path}: {exc}") from exc


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def relative_artifact_path(relative: str) -> Path:
    """A portable artifact path cannot escape or replace the Run manifest."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise RunError("Artifact path must be a non-empty relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path == Path(".") or path.parts[0] == "run.json":
        raise RunError("Artifact path escapes Run or conflicts with run.json")
    return path


def artifact_path(root: Path, relative: str) -> Path:
    path = relative_artifact_path(relative)
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise RunError("Artifact path escapes Run directory")
    return resolved


@dataclass(frozen=True)
class Artifact:
    """An execution-owned serializer; Store calls it with a temporary output path."""

    name: str
    path: str
    format: str
    write: Callable[[Path], object]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise RunError("Artifact name must be a non-empty string")
        if not isinstance(self.format, str) or not self.format:
            raise RunError("Artifact format must be a non-empty string")
        if not callable(self.write):
            raise RunError("Artifact requires a serializer callable")
        relative_artifact_path(self.path)


def verify_artifact(root: Path, descriptor: dict) -> Path:
    try:
        path = artifact_path(root, descriptor["path"])
        if digest(path) != descriptor["sha256"]:
            raise ValueError("checksum mismatch")
        return path
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RunError(f"Invalid artifact: {exc}") from exc
