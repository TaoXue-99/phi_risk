"""Coordinate data contracts, native attempts, persisted results and human comparison."""

import platform
from collections.abc import Mapping, Sequence
from pathlib import Path

from phl_risk.exceptions import ExperimentError, OptionalDependencyError, RunError
from phl_risk.modeling import DataPlan, ModelPlan

from ._utils import dependency_versions, require
from .config import resolve_config
from .dataset import LGBDatasetBuilder
from .evaluation import evaluate, feature_importance
from .run import LightGBMRun
from .runtime import resolve_data, select_features, validate_partitions
from .store import ExperimentStore
from .trainer import LightGBMTrainer


class LightGBMExperiment:
    """One binary modeling contract with independently persisted training attempts."""

    def __init__(
        self, *, name: str, root: str | Path, data, model_plan: ModelPlan, data_plan: DataPlan
    ):
        self._runtime = resolve_data(data, model_plan, data_plan)
        self._model_plan, self._data_plan = model_plan, data_plan
        contract = {
            "model_plan": model_plan.to_dict(),
            "data_plan": data_plan.to_dict(),
            "partition_order": list(data_plan.split.partitions.definitions),
            "hash_encoding": "sha256-json-typed-v1",
        }
        self._store = ExperimentStore(Path(root), name, contract)
        self.name = name
        # Detect corrupt completed artifacts when reopening, without loading Boosters.
        self._store.list_runs()

    @property
    def path(self) -> Path:
        return self._store.path

    @property
    def runs(self) -> tuple[LightGBMRun, ...]:
        return self._store.list_runs()

    def get_run(self, reference: str) -> LightGBMRun:
        runs = self.runs
        exact = [r for r in runs if r.run_id == reference]
        matches = exact or [r for r in runs if r.name == reference]
        if len(matches) != 1:
            raise ExperimentError(
                f"Run reference {reference!r} is "
                f"{'ambiguous; use run_id' if matches else 'not found'}"
            )
        return matches[0]

    def run(
        self,
        *,
        name: str,
        config: Mapping,
        features: Sequence[str] | None = None,
        overrides: Sequence[str] = (),
        _selection: dict | None = None,
    ) -> LightGBMRun:
        resolved = resolve_config(config)
        selected = select_features(self._data_plan, features)
        validation = resolved["train"]["validation_partition"]
        validate_partitions(self._runtime, validation)
        if isinstance(overrides, str) or any(not isinstance(v, str) for v in overrides):
            raise ExperimentError("overrides must be a sequence of strings")
        backend = require("lightgbm", "lightgbm")
        require("yaml", "lightgbm")
        metadata = {
            "resolved_config": resolved,
            "overrides": list(overrides),
            "features": list(selected),
            "n_features": len(selected),
            "metadata": {
                "versions": dependency_versions(),
                "lightgbm_compatibility": list(backend._phl_risk_compatibility),
                "python_version": platform.python_version(),
                "target": self._runtime.target,
                "weight": self._runtime.weight,
                "partition_counts": {k: len(v) for k, v in self._runtime.partitions.items()},
                "categorical_features": [f for f in selected if f in self._runtime.categorical],
                "feature_names": list(selected),
                "feature_count": len(selected),
                "random_seed": resolved["model"]["params"].get("seed"),
                "schema": self._runtime.schema,
            },
        }
        if _selection is not None:
            metadata["metadata"]["feature_selection"] = _selection
        path = self._store.allocate(name, metadata)
        try:
            datasets = LGBDatasetBuilder().build(self._runtime, selected, validation)
            fit = LightGBMTrainer().fit(datasets, resolved)
            if tuple(fit.booster.feature_name()) != selected:
                raise RunError("Backend changed feature names; refusing to commit inconsistent Run")
            metrics = evaluate(
                fit,
                self._runtime,
                selected,
                validation,
                num_threads=resolved["model"]["params"].get("num_threads"),
            )
            importance = feature_importance(fit, selected)
            metadata.update(
                metrics=metrics,
                best_iteration=fit.best_iteration,
                best_score=fit.best_score,
                eval_history=fit.eval_history,
                training_seconds=fit.training_seconds,
            )
            self._store.complete(path, metadata, fit, importance)
        except Exception as exc:
            try:
                self._store.fail(path, exc)
            except Exception as persist_error:
                exc.add_note(f"Could not persist failed status: {persist_error}")
            if isinstance(exc, (ExperimentError, OptionalDependencyError)):
                raise
            raise RunError(f"Run {path.name} failed: {exc}") from exc
        return LightGBMRun.load(path)

    def set_reference(self, reference: str) -> None:
        """Persist a uniquely resolved reference; names may require a run_id."""
        self._store.set_reference(self.get_run(reference).run_id)

    def compare(
        self,
        *,
        reference: str | None = None,
        include_oot: bool = False,
        include_test: bool = False,
        params="changed",
    ):
        """Return completed attempts and differences from the reference as a DataFrame."""
        from .comparison import compare_runs

        runs = self.runs
        chosen = reference if reference is not None else self._store.reference()
        baseline = self.get_run(chosen) if chosen is not None else (runs[0] if runs else None)
        return compare_runs(
            runs, baseline, include_oot=include_oot, include_test=include_test, params=params
        )

    def select_features(self, *, base_run: str, candidate_counts, step=0.1, method="rfe"):
        """Generate train-only RFE candidates and evaluate each via native training."""
        from .feature_selection import select_candidates

        return select_candidates(
            self, base_run=base_run, candidate_counts=candidate_counts, step=step, method=method
        )
