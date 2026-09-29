"""Model-independent immutable facts and verified artifact access."""

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from phl_risk.exceptions import ExperimentError, RunError

from .._utils import detached
from .artifact import read_json, relative_artifact_path, verify_artifact

ARTIFACT_VERSION = 3


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")


def _validate(facts):
    required = {
        "artifact_version",
        "run_id",
        "name",
        "status",
        "created_at",
        "started_at",
        "model",
        "input",
        "configuration",
        "environment",
        "artifacts",
    }
    if not required <= _mapping(facts, "record").keys():
        raise ValueError("missing record fields")
    if facts["artifact_version"] != ARTIFACT_VERSION:
        raise ValueError("unsupported artifact version; use the original version for old records")
    for key in ("run_id", "name"):
        _text(facts[key], key)
    if Path(facts["run_id"]).name != facts["run_id"] or facts["run_id"] in (".", ".."):
        raise ValueError("invalid Run identity")
    for key in ("created_at", "started_at"):
        if datetime.fromisoformat(facts[key]).tzinfo is None:
            raise ValueError("timestamps must include timezone")
    model = _mapping(facts["model"], "model")
    for key in ("family", "backend", "backend_version"):
        _text(model[key], f"model.{key}")
    inputs = _mapping(facts["input"], "input")
    features = inputs.get("features", [])
    if not isinstance(features, list) or any(not isinstance(f, str) or not f for f in features):
        raise ValueError("invalid feature metadata")
    if len(set(features)) != len(features):
        raise ValueError("duplicate feature metadata")
    for name, partition in _mapping(inputs.get("partitions", {}), "partitions").items():
        _text(name, "partition name")
        if type(partition["rows"]) is not int or partition["rows"] < 0:
            raise ValueError("invalid partition rows")
    if "comparison_partitions" in inputs:
        from .._utils import partition_names

        partition_names(inputs["comparison_partitions"])
    if "task" in facts:
        task = _mapping(facts["task"], "task")
        if set(task) - {"objective", "metric", "feval"}:
            raise ValueError("task only supports objective, metric and feval display identities")

        def valid_label(value):
            return (
                value is None
                or isinstance(value, str)
                or isinstance(value, dict)
                and isinstance(value.get("callable"), str)
                or isinstance(value, list)
                and all(valid_label(v) for v in value)
            )

        if not all(valid_label(value) for value in task.values()):
            raise ValueError("task values must be names, callable identities, or lists")
    config = _mapping(facts["configuration"], "configuration")
    _mapping(config["supplied"], "supplied configuration")
    _mapping(config["resolved"], "resolved configuration")
    _mapping(facts["environment"], "environment")
    artifacts = _mapping(facts["artifacts"], "artifacts")
    paths = set()
    for name, desc in artifacts.items():
        _text(name, "artifact name")
        _text(desc["format"], "artifact format")
        path = relative_artifact_path(desc["path"])
        if path in paths:
            raise ValueError("duplicate artifact path")
        paths.add(path)
        if not isinstance(desc["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", desc["sha256"]):
            raise ValueError("invalid artifact checksum")
    status = facts["status"]
    if status == "completed":
        result = _mapping(facts["result"], "result")
        metrics = _mapping(result["metrics"], "metrics")
        for partition, scores in metrics.items():
            _text(partition, "metric partition")
            for metric, value in _mapping(scores, "partition metrics").items():
                _text(metric, "metric name")
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("metrics must be finite numbers")
        if datetime.fromisoformat(facts["completed_at"]) < datetime.fromisoformat(
            facts["started_at"]
        ):
            raise ValueError("completion precedes start")
    elif status in ("running", "failed"):
        if "result" in facts or artifacts:
            raise ValueError("incomplete Run cannot claim results/artifacts")
        if status == "failed":
            _text(facts["error"]["type"], "error type")
            if not isinstance(facts["error"]["message"], str):
                raise ValueError("invalid error message")
            datetime.fromisoformat(facts["failed_at"])
    else:
        raise ValueError("invalid status")


@dataclass(frozen=True)
class RunRecord:
    """Backend-independent snapshot. Every nested value returned is a detached copy."""

    _json: str

    @classmethod
    def from_dict(cls, facts: dict) -> "RunRecord":
        try:
            facts = detached(facts)
            _validate(facts)
            return cls(json.dumps(facts, allow_nan=False))
        except (ValueError, TypeError, KeyError, AttributeError, ExperimentError) as exc:
            raise RunError(f"Invalid Run record: {exc}") from exc

    def to_dict(self) -> dict:
        return json.loads(self._json)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        facts = self.to_dict()
        if name not in facts:
            raise AttributeError(name)
        return facts[name]

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self.input.get("features", ()))

    @property
    def n_features(self) -> int | None:
        return len(self.features) if "features" in self.input else None

    @property
    def metrics(self) -> dict:
        return self.to_dict().get("result", {}).get("metrics", {})


@dataclass(frozen=True)
class Run:
    """A model-independent record with artifact access, never a backend loader."""

    path: Path
    record: RunRecord

    @classmethod
    def load(cls, path: str | Path, *, verify: bool = True) -> "Run":
        """Load metadata; optionally verify all files (default retains strict loading)."""
        path = Path(path).expanduser().resolve()
        record = RunRecord.from_dict(read_json(path / "run.json"))
        if record.run_id != path.name:
            raise RunError("Run identity differs from directory")
        run = cls(path, record)
        if verify:
            run.verify_artifacts()
        return run

    def verify_artifacts(self) -> None:
        """Explicit full integrity check, including large model/checkpoint files."""
        for descriptor in self.record.artifacts.values():
            verify_artifact(self.path, descriptor)

    def __getattr__(self, name):
        return getattr(self.record, name)

    def artifact_path(self, name: str) -> Path:
        if self.status != "completed":
            raise RunError("Artifacts require a completed Run")
        if name not in self.artifacts:
            raise RunError(f"Run has no artifact {name!r}")
        return verify_artifact(self.path, self.artifacts[name])

    def read_json(self, name: str) -> dict:
        if self.artifacts.get(name, {}).get("format") != "json":
            raise RunError(f"Artifact {name!r} is not JSON")
        return read_json(self.artifact_path(name))

    @property
    def config(self) -> dict:
        return self.configuration["resolved"]

    @property
    def supplied_config(self) -> dict:
        return self.configuration["supplied"]
