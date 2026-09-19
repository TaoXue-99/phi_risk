"""Filesystem persistence with exclusive run allocation and atomic commit manifests."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from phl_risk.exceptions import ExperimentError, RunError

from ._utils import require, slug, utc_now

ARTIFACT_VERSION = 1
ARTIFACTS = (
    "config.yaml",
    "overrides.yaml",
    "features.json",
    "metrics.json",
    "eval_history.json",
    "feature_importance.csv",
    "model.txt",
)


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


class ExperimentStore:
    """Persist contracts and runs; completed directories are never rewritten."""

    def __init__(self, root: Path, name: str, contract: dict):
        if slug(name) != name or name in (".", ".."):
            raise ExperimentError(
                "Experiment name must be a safe directory name (letters/digits/_/-)"
            )
        self.path = Path(root).expanduser().resolve() / name
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "runs").mkdir(exist_ok=True)
        manifest = self.path / "experiment.json"
        expected = {"artifact_version": ARTIFACT_VERSION, "name": name, "contract": contract}
        if not manifest.exists():
            try:
                self.write_json(manifest, {**expected, "created_at": utc_now()}, exclusive=True)
            except FileExistsError:
                pass  # Another creator committed the same contract first.
        stored = read_json(manifest)
        if not isinstance(stored, dict) or any(stored.get(k) != v for k, v in expected.items()):
            raise ExperimentError(
                "Existing experiment name/version/ModelPlan/DataPlan contract differs"
            )

    @staticmethod
    def write_json(path: Path, value, *, exclusive=False):
        content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        atomic_write(path, lambda p: p.write_text(content, encoding="utf-8"), exclusive=exclusive)

    @staticmethod
    def write_yaml(path: Path, value):
        yaml = require("yaml", "lightgbm")
        content = yaml.safe_dump(value, allow_unicode=True, sort_keys=False)
        atomic_write(path, lambda p: p.write_text(content, encoding="utf-8"))

    def allocate(self, name: str, metadata: dict) -> Path:
        label = slug(name)
        for _ in range(10):
            stamp = utc_now().replace("-", "").replace(":", "").split(".")[0]
            run_id = f"{stamp}_{uuid4().hex}_{label}"
            path = self.path / "runs" / run_id
            try:
                path.mkdir()
            except FileExistsError:
                continue
            self.write_json(
                path / "run.json",
                {
                    **metadata,
                    "artifact_version": ARTIFACT_VERSION,
                    "run_id": run_id,
                    "name": name,
                    "created_at": utc_now(),
                    "status": "running",
                },
            )
            return path
        raise RunError("Could not allocate a unique run directory")

    def complete(self, path: Path, metadata: dict, fit, importance):
        current = read_json(path / "run.json")
        if current["status"] != "running":
            raise RunError("Run is immutable after completion or failure")
        self.write_yaml(path / "config.yaml", metadata["resolved_config"])
        self.write_yaml(path / "overrides.yaml", metadata["overrides"])
        for filename, key in [
            ("features.json", "features"),
            ("metrics.json", "metrics"),
            ("eval_history.json", "eval_history"),
        ]:
            self.write_json(path / filename, metadata[key])
        atomic_write(path / "feature_importance.csv", lambda p: importance.to_csv(p, index=False))
        atomic_write(
            path / "model.txt",
            lambda p: fit.booster.save_model(str(p), num_iteration=fit.best_iteration),
        )
        hashes = {name: digest(path / name) for name in ARTIFACTS}
        self.write_json(
            path / "run.json",
            {
                **current,
                **metadata,
                "status": "completed",
                "completed_at": utc_now(),
                "sha256": hashes,
            },
        )

    def fail(self, path: Path, error: Exception):
        current = read_json(path / "run.json")
        if current["status"] == "running":
            self.write_json(
                path / "run.json",
                {
                    **current,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "failed_at": utc_now(),
                },
            )

    def list_runs(self):
        from .run import LightGBMRun

        runs = []
        for path in sorted((self.path / "runs").iterdir()):
            if not path.is_dir() or not (path / "run.json").exists():
                continue  # A concurrent writer may have just allocated its directory.
            metadata = read_json(path / "run.json")
            if not isinstance(metadata, dict) or metadata.get("status") not in (
                "running",
                "failed",
                "completed",
            ):
                raise RunError(f"Invalid artifact status: {path}")
            if metadata["status"] == "completed":
                runs.append(LightGBMRun.load(path))
        return tuple(sorted(runs, key=lambda r: (r.completed_at, r.run_id)))

    def set_reference(self, run_id: str):
        self.write_json(self.path / "reference.json", {"run_id": run_id})

    def reference(self):
        path = self.path / "reference.json"
        return read_json(path)["run_id"] if path.exists() else None
