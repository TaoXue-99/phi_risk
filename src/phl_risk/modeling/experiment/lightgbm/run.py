"""Read-only run snapshots; native Boosters are loaded only on request."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from phl_risk.exceptions import RunError

from ._utils import require
from .store import ARTIFACT_VERSION, ARTIFACTS, digest, read_json


@dataclass(frozen=True)
class LightGBMRun:
    """A completed attempt. Mutable values are returned as detached copies.

    ``model`` returns a fresh Booster each time, so caller mutations cannot alter
    either the stored model or a later read of this Run.
    """

    _path: Path
    _json: str

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMRun":
        path = Path(path).expanduser().resolve()
        try:
            metadata = read_json(path / "run.json")
            required = {
                "run_id",
                "name",
                "created_at",
                "status",
                "features",
                "n_features",
                "resolved_config",
                "overrides",
                "metrics",
                "best_iteration",
                "best_score",
                "training_seconds",
                "sha256",
                "eval_history",
                "completed_at",
                "metadata",
            }
            if not isinstance(metadata, dict) or not required <= metadata.keys():
                raise ValueError("missing manifest fields")
            if (
                metadata.get("artifact_version") != ARTIFACT_VERSION
                or metadata["status"] != "completed"
            ):
                raise ValueError("unsupported version or incomplete run")
            if metadata["run_id"] != path.name:
                raise ValueError("run identity differs from directory")
            if metadata["n_features"] != len(metadata["features"]) or not metadata["features"]:
                raise ValueError("invalid feature metadata")
            for name in ARTIFACTS:
                if metadata["sha256"].get(name) != digest(path / name):
                    raise ValueError(f"checksum mismatch: {name}")
            for name, key in [
                ("features.json", "features"),
                ("metrics.json", "metrics"),
                ("eval_history.json", "eval_history"),
            ]:
                if read_json(path / name) != metadata[key]:
                    raise ValueError(f"inconsistent {name}")
            yaml = require("yaml", "lightgbm")
            for name, key in [("config.yaml", "resolved_config"), ("overrides.yaml", "overrides")]:
                if yaml.safe_load((path / name).read_text(encoding="utf-8")) != metadata[key]:
                    raise ValueError(f"inconsistent {name}")
            return cls(path, json.dumps(metadata, allow_nan=False))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise RunError(f"Invalid Run artifact {path}: {exc}") from exc

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        payload = json.loads(self._json)
        if name not in payload:
            raise AttributeError(name)
        value = payload[name]
        return tuple(value) if name in ("features", "overrides") else value

    @property
    def path(self) -> Path:
        return self._path

    @property
    def run_path(self) -> Path:
        return self.path

    @property
    def model_path(self) -> Path:
        return self.path / "model.txt"

    @property
    def model(self):
        lgb = require("lightgbm", "lightgbm")
        try:
            if digest(self.model_path) != json.loads(self._json)["sha256"]["model.txt"]:
                raise ValueError("model checksum mismatch")
            model = lgb.Booster(model_file=str(self.model_path))
            if tuple(model.feature_name()) != self.features:
                raise ValueError("model feature schema differs")
            return model
        except Exception as exc:
            raise RunError(f"Cannot load model artifact {self.model_path}: {exc}") from exc

    @property
    def feature_importance(self) -> pd.DataFrame:
        try:
            path = self.path / "feature_importance.csv"
            if digest(path) != json.loads(self._json)["sha256"][path.name]:
                raise ValueError("importance checksum mismatch")
            return pd.read_csv(path, dtype={"feature": str}, keep_default_na=False)
        except (OSError, ValueError) as exc:
            raise RunError(f"Cannot load feature importance artifact: {exc}") from exc
