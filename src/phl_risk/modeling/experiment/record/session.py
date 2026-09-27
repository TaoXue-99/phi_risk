"""A write-once recording context around arbitrary native Python code."""

import hashlib
import inspect
import json
import platform
import shutil
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

from phl_risk import __version__
from phl_risk.exceptions import RunError

from .._utils import detached
from .artifact import Artifact, artifact_path, atomic_write, read_json
from .run import Run, RunRecord
from .store import write_json


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


class RunSession:
    """Mutable only while the context is active; ``record`` exposes its final Run.

    Each artifact is serialized when logged, not when the context exits. This
    prevents subsequent model or source-file mutations from changing its snapshot.
    """

    def __init__(self, store, *, name, params, metadata, comparison_partitions, model, config=None):
        if params is not None and not isinstance(params, Mapping):
            raise RunError("params must be a mapping")
        if config is not None and params is not None:
            raise RunError("Use config or params, not both")
        self._config_artifacts = {}
        self._store = store
        self._name = name
        self._facts = {
            "model": model
            or {"family": "unspecified", "backend": "unspecified", "backend_version": "unrecorded"},
            "input": {},
            "configuration": {
                "supplied": parameter_snapshot(params or {}),
                "resolved": parameter_snapshot(params or {}),
            },
            "metadata": parameter_snapshot(metadata or {}),
            "environment": {"python": platform.python_version(), "phl-risk": __version__},
        }
        if config is not None:
            from .configuration import configuration_snapshot

            configuration, self._config_artifacts = configuration_snapshot(config)
            self._facts["configuration"] = configuration
        if comparison_partitions is not None:
            if isinstance(comparison_partitions, str) or any(
                not isinstance(p, str) or not p for p in comparison_partitions
            ):
                raise RunError("comparison_partitions must be a sequence of names")
            self._facts["input"]["comparison_partitions"] = list(comparison_partitions)
        self._result = {"metrics": {}}
        self._artifacts = {}
        self._paths = set()
        self._state = "new"
        self.path = None

    def __enter__(self):
        if self._state != "new":
            raise RunError("A Run context can only be entered once")
        self.path = self._store.allocate(self._name, self._facts)
        self._state = "active"
        self._start = perf_counter()
        try:
            for name, (filename, content) in self._config_artifacts.items():
                self.log_artifact(
                    name,
                    filename,
                    lambda path, text=content: path.write_text(text, encoding="utf-8"),
                    format="yaml",
                )
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise
        return self

    def _active(self):
        if self._state != "active":
            raise RunError("Run is immutable outside its active recording context")

    @property
    def run_id(self):
        if self.path is None:
            raise RunError("Run has not started")
        return self.path.name

    @property
    def record(self):
        if self._state not in ("completed", "failed"):
            raise RunError("A final record is available only after the context exits")
        return Run.load(self.path)

    def _update_facts(self, **updates):
        self._active()
        current = read_json(self.path / "run.json")
        current.update(detached(updates))
        write_json(self.path / "run.json", RunRecord.from_dict(current).to_dict())

    def log_model_info(self, *, family, backend, backend_version):
        self._active()
        current = read_json(self.path / "run.json")["model"]
        if current["backend_version"] != "unrecorded" and current["backend"] != backend:
            raise RunError("Model backend differs from the method space")
        self._update_facts(
            model={"family": family, "backend": backend, "backend_version": backend_version}
        )

    def log_input(self, **metadata):
        """Record explicit data descriptors, never raw training data."""
        self._active()
        values = read_json(self.path / "run.json")["input"]
        if (
            "features" in values
            and "features" in metadata
            and values["features"] != list(metadata["features"])
        ):
            raise RunError("Feature metadata conflicts with the model snapshot")
        values.update(parameter_snapshot(metadata))
        self._update_facts(input=values)

    def log_params(self, params):
        """Add actual execution arguments without silently changing earlier facts."""
        self._active()
        if self._config_artifacts:
            raise RunError(
                "With config=, prepare all parameters before start_run; config is immutable"
            )
        configuration = read_json(self.path / "run.json")["configuration"]
        values = parameter_snapshot(params)
        if not isinstance(values, dict):
            raise RunError("params must be a mapping")
        for key, value in values.items():
            if key in configuration["resolved"] and configuration["resolved"][key] != value:
                raise RunError(f"Parameter already recorded with a different value: {key}")
        configuration["resolved"].update(values)
        self._update_facts(configuration=configuration)

    def log_metrics(self, metrics):
        self._active()
        values = detached(metrics)
        if not isinstance(values, dict):
            raise RunError("metrics must map partition names to metric dictionaries")
        merged = detached(self._result["metrics"])
        for partition, scores in values.items():
            if not isinstance(scores, dict):
                raise RunError("Each metric partition must be a mapping")
            old = merged.setdefault(partition, {})
            if old.keys() & scores.keys():
                raise RunError("A metric cannot be logged twice in the same Run")
            old.update(scores)
        # Store validates finite scalar scores on commit; detached rejects NaN now.
        self._result["metrics"] = merged

    def log_result(self, **values):
        self._active()
        if {"metrics", "recording_seconds"} & values.keys() or self._result.keys() & values.keys():
            raise RunError("Duplicate or reserved result key")
        self._result.update(parameter_snapshot(values))

    def log_artifact(self, name, filename, writer, *, format):
        """Snapshot through a serializer receiving a temporary pathlib.Path."""
        self._active()
        item = Artifact(name, filename, format, writer)
        if Path(filename).parts[0] == ".pending":
            raise RunError("Reserved internal artifact path")
        destination = artifact_path(self.path, filename)
        if name in self._artifacts or destination in self._paths:
            raise RunError("Duplicate artifact name/path")
        pending = self.path / ".pending"
        pending.mkdir(exist_ok=True)
        staged = pending / str(len(self._artifacts))
        atomic_write(staged, item.write)
        self._artifacts[name] = Artifact(
            name, filename, format, lambda target, source=staged: shutil.copyfile(source, target)
        )
        self._paths.add(destination)

    def log_json(self, name, value):
        content = json.dumps(
            parameter_snapshot(value), ensure_ascii=False, indent=2, allow_nan=False
        )
        self.log_artifact(
            name, f"{name}.json", lambda p: p.write_text(content, encoding="utf-8"), format="json"
        )

    def log_file(self, name, source, *, filename=None):
        source = Path(source)
        self.log_artifact(
            name, filename or source.name, lambda p: shutil.copyfile(source, p), format="file"
        )

    def __exit__(self, exc_type, error, traceback):
        try:
            if error is not None:
                self._state = "failed"
                try:
                    self._store.fail(self.path, error)
                except Exception as persist_error:
                    error.add_note(f"Failed to persist status: {persist_error}")
                return False
            self._result["recording_seconds"] = perf_counter() - self._start
            self._store.complete(self.path, self._result, tuple(self._artifacts.values()))
            self._state = "completed"
        except BaseException as failure:
            self._state = "failed"
            try:
                self._store.fail(self.path, failure)
            except Exception as persist_error:
                failure.add_note(f"Failed to persist status: {persist_error}")
            raise
        finally:
            shutil.rmtree(self.path / ".pending", ignore_errors=True)
        return False
