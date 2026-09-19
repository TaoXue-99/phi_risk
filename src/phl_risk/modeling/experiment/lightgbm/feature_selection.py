"""Train-only sklearn RFE proposes subsets; native Experiment runs evaluate them."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sklearn import config_context
from sklearn.feature_selection import RFE

from phl_risk.exceptions import ExperimentError

from ._utils import require
from .run import LightGBMRun

if TYPE_CHECKING:
    from .experiment import LightGBMExperiment


@dataclass(frozen=True)
class RFEResult:
    """Ordinary candidate runs, with an optional non-binding tolerance recommendation."""

    experiment: "LightGBMExperiment"
    base_run_id: str
    runs: tuple[LightGBMRun, ...]

    def compare(self, **kwargs):
        kwargs.setdefault("reference", self.base_run_id)
        frame = self.experiment.compare(**kwargs)
        ids = {self.base_run_id, *(r.run_id for r in self.runs)}
        return frame.loc[frame["run_id"].isin(ids)].reset_index(drop=True)

    def within_tolerance(
        self, *, metric: str | None = None, tolerance: float = 0.001
    ) -> LightGBMRun:
        """Recommend the smallest candidate near the best candidate AUC; change no state."""
        if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
            raise ExperimentError("tolerance must be finite and non-negative")
        validation_metrics = {
            f"{run.resolved_config['train']['validation_partition']}_auc" for run in self.runs
        }
        if len(validation_metrics) != 1:
            raise ExperimentError("Candidates must use the same validation partition")
        validation_metric = next(iter(validation_metrics))
        metric = validation_metric if metric is None else metric
        if metric != validation_metric:
            raise ExperimentError("Use the configured validation AUC metric for recommendations")
        if any(metric not in run.metrics for run in self.runs):
            raise ExperimentError(f"Unknown candidate metric: {metric}")
        best = max(run.metrics[metric] for run in self.runs)
        eligible = [run for run in self.runs if run.metrics[metric] >= best - tolerance]
        return min(eligible, key=lambda r: (r.n_features, -r.metrics[metric], r.run_id))


def select_candidates(
    experiment: "LightGBMExperiment", *, base_run: str, candidate_counts, step=0.1, method="rfe"
) -> RFEResult:
    if method != "rfe":
        raise ExperimentError('V1 feature selection supports method="rfe" only')
    baseline = experiment.get_run(base_run)
    counts = tuple(candidate_counts)
    if (
        not counts
        or any(type(c) is not int or not 1 <= c <= baseline.n_features for c in counts)
        or len(set(counts)) != len(counts)
    ):
        raise ExperimentError(
            "candidate_counts must be unique integers between 1 and base feature count"
        )
    if not ((type(step) is int and step >= 1) or (type(step) is float and 0 < step < 1)):
        raise ExperimentError("RFE step must be an integer >=1 or a fraction in (0,1)")
    if baseline.n_features < 2:
        raise ExperimentError("sklearn RFE requires at least two input features")
    lgb = require("lightgbm", "lightgbm")
    runtime = experiment._runtime
    train = runtime.partitions["train"]
    x = train[list(baseline.features)].copy()
    # sklearn RFE slices a numerical ndarray. Ordinal codes are used only for
    # candidate ranking, never substituted into the final native run's schema.
    for col in baseline.features:
        if col in runtime.categorical:
            x[col] = x[col].cat.codes.astype(float).replace(-1, float("nan"))
    params = baseline.resolved_config["model"]["params"].copy()
    positional = {
        "mc",
        "monotone_constraint",
        "monotone_constraints",
        "monotonic_cst",
        "fc",
        "feature_contri",
        "feature_contrib",
        "feature_penalty",
        "fp",
        "interaction_constraints",
        "forced_splits",
        "forced_splits_file",
        "forced_splits_filename",
        "forcedsplits_filename",
        "fs",
        "forcedbins_filename",
        "cegb_penalty_feature_lazy",
        "cegb_penalty_feature_coupled",
    }
    unsupported = sorted(
        key for key in positional if key in params and params[key] not in (None, [], "")
    )
    if unsupported:
        raise ExperimentError(
            f"RFE V1 cannot remap feature-indexed parameters: {unsupported}. "
            "Use an unconstrained base run for candidate generation, then explicitly "
            "align constraints to the selected feature order in a native exp.run()."
        )
    # Avoid conflicting defaults in the sklearn wrapper for canonical native names.
    aliases = {
        "seed": "random_state",
        "num_threads": "n_jobs",
        "min_data_in_leaf": "min_child_samples",
        "lambda_l1": "reg_alpha",
        "lambda_l2": "reg_lambda",
        "feature_fraction": "colsample_bytree",
        "bagging_fraction": "subsample",
        "bagging_freq": "subsample_freq",
        "min_gain_to_split": "min_split_gain",
    }
    for native, wrapper in aliases.items():
        if native in params:
            params[wrapper] = params.pop(native)
    params["n_estimators"] = (
        baseline.best_iteration or baseline.resolved_config["train"]["num_boost_round"]
    )
    params["importance_type"] = "gain"
    weights = {} if runtime.weight is None else {"sample_weight": train[runtime.weight].to_numpy()}
    runs = []
    for count in counts:
        selector = RFE(lgb.LGBMClassifier(**params), n_features_to_select=count, step=step)
        # Local sklearn context uses the documented **fit_params route, regardless
        # of the caller's global metadata-routing preference; it restores on exit.
        try:
            with config_context(enable_metadata_routing=False):
                selector.fit(x.to_numpy(dtype=float), train[runtime.target].to_numpy(), **weights)
        except (ValueError, lgb.basic.LightGBMError) as exc:
            raise ExperimentError(f"RFE candidate {count} failed: {exc}") from exc
        selected = tuple(
            f for f, keep in zip(baseline.features, selector.support_, strict=True) if keep
        )
        run = experiment.run(
            name=f"rfe_{count}",
            config=baseline.resolved_config,
            features=selected,
            overrides=baseline.overrides,
            _selection={
                "method": "rfe",
                "base_run_id": baseline.run_id,
                "candidate_count": count,
                "step": step,
                "n_estimators": params["n_estimators"],
                "categorical_ranking": "train_ordinal_codes",
            },
        )
        runs.append(run)
    return RFEResult(experiment, baseline.run_id, tuple(runs))
