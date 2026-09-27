"""Atomic records and arbitrary artifacts; no model or training knowledge."""

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from phl_risk.exceptions import ExperimentError, RunError

from .._utils import shanghai_now, slug
from .artifact import Artifact, artifact_path, atomic_write, digest, read_json
from .run import ARTIFACT_VERSION, RunRecord


def write_json(path: Path, value, *, exclusive=False):
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    atomic_write(path, lambda p: p.write_text(content, encoding="utf-8"), exclusive=exclusive)


class ExperimentStore:
    """Own identities, lifecycle and atomic commits, not backend serialization."""

    def __init__(self, root: Path, name: str, *, method=None, create=True, direct=False):
        if slug(name) != name or name in (".", ".."):
            raise ExperimentError("Experiment name must be a safe directory name")
        self.path = Path(root).expanduser().resolve()
        if not direct:
            self.path /= name
        self.method = method
        if create:
            self.path.mkdir(parents=True, exist_ok=True)
        manifest = self.path / ("method.json" if method else "experiment.json")
        if not create and not manifest.is_file():
            raise ExperimentError(f"Method is not initialized: {self.path}")
        expected = {"artifact_version": ARTIFACT_VERSION, "name": name}
        if method:
            expected.update(method=method["name"], family=method["family"], prefix=method["prefix"])
        if create and not manifest.exists():
            try:
                write_json(manifest, {**expected, "created_at": shanghai_now()}, exclusive=True)
            except FileExistsError:
                pass
        stored = read_json(manifest)
        if not isinstance(stored, dict) or any(stored.get(k) != v for k, v in expected.items()):
            raise ExperimentError(
                "Experiment name/artifact version mismatch; use a new directory. "
                "Old records must be read with their original version."
            )
        if create and method:
            (self.path / "runs").mkdir(exist_ok=True)

    def allocate(self, name: str, facts: dict) -> Path:
        if self.method:
            return self._allocate_numbered(name, facts)
        label = slug(name)
        (self.path / "runs").mkdir(exist_ok=True)
        for _ in range(10):
            now = shanghai_now()
            stamp = now.replace("-", "").replace(":", "").split(".")[0]
            path = self.path / "runs" / f"{stamp}_{uuid4().hex}_{label}"
            record = RunRecord.from_dict(
                {
                    **facts,
                    "artifact_version": ARTIFACT_VERSION,
                    "run_id": path.name,
                    "name": name,
                    "created_at": now,
                    "started_at": now,
                    "status": "running",
                    "artifacts": {},
                }
            )
            try:
                path.mkdir()
            except FileExistsError:
                continue
            write_json(path / "run.json", record.to_dict())
            return path
        raise RunError("Could not allocate a unique Run directory")

    def _allocate_numbered(self, name, facts):
        with FileLock(str(self.path / ".sequence.lock"), timeout=30):
            counter = self.path / "sequence.json"
            sequence = read_json(counter)["last"] if counter.exists() else 0
            if type(sequence) is not int or sequence < 0:
                raise RunError("Invalid run sequence")
            # Recover the high-water mark even if only the counter was removed.
            prefix = self.method["prefix"] + "_run_"
            for existing in (self.path / "runs").iterdir():
                if existing.name.startswith(prefix):
                    try:
                        sequence = max(sequence, int(existing.name[len(prefix) :].split("_")[0]))
                    except ValueError:
                        raise RunError(f"Invalid numbered Run directory: {existing}") from None
            sequence += 1
            label = f"{prefix}{sequence:02d}"
            now = shanghai_now()
            stamp = datetime.fromisoformat(now).strftime("%Y_%m_%d_%H%M%S")
            path = self.path / "runs" / f"{label}_{stamp}"
            record = RunRecord.from_dict(
                {
                    **facts,
                    "artifact_version": ARTIFACT_VERSION,
                    "run_id": path.name,
                    "run": label,
                    "name": name or label,
                    "created_at": now,
                    "started_at": now,
                    "status": "running",
                    "artifacts": {},
                }
            )
            # Commit reservation before directory creation: interrupted/failed IDs are never reused.
            write_json(counter, {"last": sequence})
            path.mkdir()
            write_json(path / "run.json", record.to_dict())
            return path

    def record_paths(self):
        paths = list((self.path / "runs").glob("*/run.json"))
        if not self.method:
            for directory in self.path.iterdir():
                if directory.is_dir() and (directory / "method.json").is_file():
                    paths.extend((directory / "runs").glob("*/run.json"))
        return paths

    def run_path(self, run_id):
        matches = [p.parent for p in self.record_paths() if p.parent.name == run_id]
        if len(matches) != 1:
            raise RunError(f"Run identity not uniquely found: {run_id}")
        return matches[0]

    def complete(self, path: Path, result: dict, artifacts: tuple[Artifact, ...]):
        current = read_json(path / "run.json")
        if current["status"] != "running":
            raise RunError("Run is immutable after completion or failure")
        descriptors, destinations = {}, set()
        for artifact in artifacts:
            destination = artifact_path(path, artifact.path)
            if artifact.name in descriptors or destination in destinations:
                raise RunError("Duplicate artifact name/path")
            descriptors[artifact.name] = {
                "path": artifact.path,
                "format": artifact.format,
                "sha256": "0" * 64,
            }
            destinations.add(destination)
        # Validate all facts before any serializer is called.
        completed = {
            **current,
            "result": result,
            "artifacts": descriptors,
            "status": "completed",
            "completed_at": shanghai_now(),
        }
        RunRecord.from_dict(completed)
        for artifact in artifacts:
            destination = artifact_path(path, artifact.path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(destination, artifact.write)
            descriptors[artifact.name]["sha256"] = digest(destination)
        completed["completed_at"] = shanghai_now()
        write_json(path / "run.json", RunRecord.from_dict(completed).to_dict())

    def fail(self, path: Path, error: Exception):
        current = read_json(path / "run.json")
        if current["status"] == "running":
            failed = {
                **current,
                "status": "failed",
                "failed_at": shanghai_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            write_json(path / "run.json", RunRecord.from_dict(failed).to_dict())

    def list_records(self):
        records = []
        for manifest in self.record_paths():
            path = manifest.parent
            record = RunRecord.from_dict(read_json(path / "run.json"))
            if record.run_id != path.name:
                raise RunError(f"Run identity differs from directory: {path}")
            records.append(record)
        return tuple(sorted(records, key=lambda r: (r.started_at, r.run_id)))
