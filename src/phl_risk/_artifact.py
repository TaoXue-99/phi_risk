"""Versioned Python-native artifacts. Load only files from trusted producers.

joblib uses pickle: validation of the envelope is not a sandbox for untrusted files.
"""

import os
import platform
import tempfile
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib

from . import __version__
from ._data import check_fitted
from .exceptions import ArtifactError

ARTIFACT_VERSION = 1
DEPENDENCIES = ("pandas", "numpy", "scikit-learn", "joblib", "optbinning")


def dependency_versions():
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def save_artifact(obj, path):
    check_fitted(obj)
    hook = getattr(obj, "_check_artifact", None)
    if hook is not None:
        hook()
    envelope = dict(
        artifact_type=type(obj).__name__,
        artifact_version=ARTIFACT_VERSION,
        phl_risk_version=__version__,
        python_version=platform.python_version(),
        dependencies=dependency_versions(),
        payload=obj,
    )
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
        joblib.dump(envelope, temporary, compress=3)
        os.replace(temporary, path)
    except Exception as exc:
        raise ArtifactError(f"Could not save artifact to {path}: {exc}") from exc
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_artifact(cls, path):
    try:
        envelope = joblib.load(path)
    except Exception as exc:
        raise ArtifactError(f"Could not load artifact {path}: {exc}") from exc
    if not isinstance(envelope, dict):
        raise ArtifactError("Expected a phl_risk artifact envelope")
    required = {
        "artifact_type",
        "artifact_version",
        "phl_risk_version",
        "python_version",
        "dependencies",
        "payload",
    }
    if not required <= envelope.keys() or not isinstance(envelope["dependencies"], dict):
        raise ArtifactError("Invalid artifact metadata")
    if envelope["artifact_version"] != ARTIFACT_VERSION:
        raise ArtifactError(f"Unsupported artifact version: {envelope['artifact_version']}")
    obj = envelope["payload"]
    if envelope["artifact_type"] != cls.__name__ or not isinstance(obj, cls):
        raise ArtifactError(f"Expected {cls.__name__}, got {envelope['artifact_type']}")
    if not getattr(obj, "_is_fitted_", False):
        raise ArtifactError("Artifact payload is not fitted")
    current = {
        "python": platform.python_version(),
        "phl_risk": __version__,
        **dependency_versions(),
    }
    trained = {
        "python": envelope["python_version"],
        "phl_risk": envelope["phl_risk_version"],
        **envelope["dependencies"],
    }
    differences = [
        f"{name}: trained={old}, current={current.get(name)}"
        for name, old in trained.items()
        if old != current.get(name)
    ]
    if differences:
        warnings.warn(
            "Artifact environment differs: " + "; ".join(differences), UserWarning, stacklevel=2
        )
    return obj
