"""Resolve a complete execution configuration without depending on Hydra."""

import math
from collections.abc import Mapping
from copy import deepcopy

from phl_risk.exceptions import ExperimentError

from ._utils import detached

DEFAULT_CONFIG = {
    "model": {
        "params": {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "seed": 2026,
            "verbosity": -1,
        }
    },
    "train": {
        "num_boost_round": 5000,
        "validation_partition": "valid",
        "early_stopping": {
            "enabled": True,
            "stopping_rounds": 200,
            "first_metric_only": True,
            "min_delta": 0.0,
        },
        "log_evaluation": {"enabled": True, "period": 100},
    },
}


def _merge(default, supplied, path=""):
    if not isinstance(supplied, Mapping):
        raise ExperimentError(f"{path or 'config'} must be a mapping")
    result = deepcopy(default)
    for key, value in supplied.items():
        if key not in default:
            raise ExperimentError(f"Unknown configuration key: {path}{key}")
        if path + str(key) == "model.params":
            if not isinstance(value, Mapping):
                raise ExperimentError("model.params must be a mapping")
            result[key].update(_normalize_aliases(dict(value)))
        elif isinstance(default[key], dict):
            result[key] = _merge(default[key], value, path + key + ".")
        else:
            result[key] = value
    return result


def _normalize_aliases(params: dict) -> dict:
    # Normalize aliases of framework defaults before merging, so an explicit alias
    # cannot be silently shadowed by our canonical default.
    groups = {
        "seed": ("random_state", "random_seed"),
        "boosting_type": ("boosting", "boost"),
        "verbosity": ("verbose",),
    }
    for canonical, aliases in groups.items():
        present = [key for key in (canonical, *aliases) if key in params]
        if not present:
            continue
        value = params[present[0]]
        if any(params[key] != value for key in present[1:]):
            raise ExperimentError(f"Parameter aliases conflict: {present}")
        for key in present:
            params.pop(key)
        params[canonical] = value
    return params


def resolve_config(config: Mapping) -> dict:
    result = detached(_merge(DEFAULT_CONFIG, config))
    params, train = result["model"]["params"], result["train"]
    for key, expected in [("objective", "binary"), ("metric", "auc")]:
        if params[key] != expected:
            raise ExperimentError(f"LightGBM Experiment V1 requires {key}={expected!r}")
    # These parameters would override Dataset/Trainer arguments or alter prediction semantics.
    forbidden = {
        "task_type",
        "is_predict_raw_score",
        "predict_rawscore",
        "is_predict_leaf_index",
        "leaf_index",
        "contrib",
        "label",
        "weight",
        "blacklist",
        "ignore_feature",
        "group",
        "group_id",
        "query",
        "query_column",
        "query_id",
        "application",
        "app",
        "loss",
        "objective_type",
        "metrics",
        "metric_types",
        "num_iterations",
        "num_iteration",
        "n_iter",
        "num_tree",
        "num_trees",
        "num_round",
        "num_rounds",
        "nrounds",
        "num_boost_round",
        "n_estimators",
        "max_iter",
        "early_stopping_round",
        "early_stopping_rounds",
        "early_stopping",
        "n_iter_no_change",
        "early_stopping_min_delta",
        "first_metric_only",
        "categorical_feature",
        "cat_feature",
        "categorical_column",
        "cat_column",
        "categorical_features",
        "feature_name",
        "label_column",
        "weight_column",
        "ignore_column",
        "group_column",
        "task",
        "num_class",
        "num_classes",
        "predict_raw_score",
        "raw_score",
        "predict_leaf_index",
        "pred_leaf",
        "predict_contrib",
        "pred_contrib",
        "is_predict_contrib",
    }
    bad = sorted(forbidden.intersection(params))
    if bad:
        raise ExperimentError(f"Parameters controlled by Experiment/DataPlan: {bad}")
    if not isinstance(train["validation_partition"], str) or train["validation_partition"] in (
        "",
        "train",
        "oot",
    ):
        raise ExperimentError("validation_partition must be a non-training, non-OOT partition")
    early, log = train["early_stopping"], train["log_evaluation"]
    for key, value, minimum in [
        ("num_boost_round", train["num_boost_round"], 1),
        ("stopping_rounds", early["stopping_rounds"], 1),
        ("period", log["period"], 0),
    ]:
        if type(value) is not int or value < minimum:
            raise ExperimentError(f"{key} must be an integer >= {minimum}")
    for key, value in [
        ("early_stopping.enabled", early["enabled"]),
        ("first_metric_only", early["first_metric_only"]),
        ("log_evaluation.enabled", log["enabled"]),
    ]:
        if type(value) is not bool:
            raise ExperimentError(f"{key} must be boolean")
    delta = early["min_delta"]
    if type(delta) not in (int, float) or not math.isfinite(delta) or delta < 0:
        raise ExperimentError("min_delta must be finite and non-negative")
    boosting = params.get("boosting", params.get("boost", params["boosting_type"]))
    if boosting == "dart" and early["enabled"]:
        raise ExperimentError("DART does not support early stopping; explicitly disable it")
    return result
