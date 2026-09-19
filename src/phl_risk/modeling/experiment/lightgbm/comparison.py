"""Compact, deterministic reference differences with numerical AUC deltas."""

import json
from collections.abc import Mapping

import pandas as pd

from phl_risk.exceptions import ExperimentError


def flatten(value: Mapping, prefix="") -> dict:
    result = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            result.update(flatten(item, path))
        else:
            result[path] = item
    return result


def parameter_changes(reference: dict, current: dict) -> str:
    before, after = flatten(reference), flatten(current)
    missing = object()

    def display(value):
        return (
            "<absent>"
            if value is missing
            else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        )

    changes = []
    for key in sorted(before.keys() | after.keys()):
        old, new = before.get(key, missing), after.get(key, missing)
        if old != new:
            label = key.removeprefix("model.params.")
            changes.append(f"{label}: {display(old)}→{display(new)}")
    return "; ".join(changes) or "-"


def feature_difference(reference, current) -> dict:
    before, after = set(reference), set(current)
    added = tuple(f for f in current if f not in before)
    removed = tuple(f for f in reference if f not in after)
    n, m = len(reference), len(current)
    if not added and not removed:
        text = "-"
    elif n == m:
        text = f"{n}→{m} (+{len(added)}/-{len(removed)})"
    else:
        text = f"{n}→{m} ({m - n:+d}, {(m - n) / n:+.1%})"
    return {"feature_change": text, "added_features": added, "removed_features": removed}


def compare_runs(
    runs, reference, *, include_oot=False, include_test=False, params="changed"
) -> pd.DataFrame:
    if isinstance(params, str):
        if params not in ("changed", "all"):
            raise ExperimentError("params must be changed, all, or a sequence of parameter names")
        requested = (
            []
            if params == "changed"
            else sorted({k for r in runs for k in r.resolved_config["model"]["params"]})
        )
    elif isinstance(params, (list, tuple)) and all(isinstance(k, str) for k in params):
        requested = list(dict.fromkeys(params))
    else:
        raise ExperimentError("params must be changed, all, or a sequence of parameter names")
    validation_names = list(
        dict.fromkeys(r.resolved_config["train"]["validation_partition"] for r in runs)
    ) or ["valid"]
    columns = [
        "run",
        "run_id",
        "created_at",
        "n_features",
        "feature_change",
        "best_iteration",
        "train_auc",
        *[f"{n}_auc" for n in validation_names],
        "auc_gap",
        *[f"delta_{n}_auc" for n in validation_names],
        "delta_auc_gap",
        "param_changes",
    ]
    holdouts = [
        name
        for name, enabled in (("test", include_test), ("oot", include_oot))
        if enabled and name not in validation_names
    ]
    for name in holdouts:
        columns += [f"{name}_auc", f"delta_{name}_auc"]
    reserved = set(columns)
    labels = {k: f"model.params.{k}" if k in reserved else k for k in requested}
    rows = []
    for run in runs:
        values, base = run.metrics, reference.metrics
        row = {
            "run": run.name,
            "run_id": run.run_id,
            "created_at": run.created_at,
            "n_features": run.n_features,
            "best_iteration": run.best_iteration,
            "feature_change": feature_difference(reference.features, run.features)[
                "feature_change"
            ],
            "param_changes": parameter_changes(reference.resolved_config, run.resolved_config),
            "train_auc": values["train_auc"],
            "auc_gap": values["auc_gap"],
        }
        for part in validation_names + holdouts:
            key = f"{part}_auc"
            row[key] = values.get(key, float("nan"))
            row[f"delta_{key}"] = row[key] - base.get(key, float("nan"))
        same_validation = (
            run.resolved_config["train"]["validation_partition"]
            == reference.resolved_config["train"]["validation_partition"]
        )
        row["delta_auc_gap"] = (
            values["auc_gap"] - base["auc_gap"] if same_validation else float("nan")
        )
        for key, label in labels.items():
            row[label] = run.resolved_config["model"]["params"].get(key)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns + list(labels.values()))
