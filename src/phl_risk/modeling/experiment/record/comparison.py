"""Backend-independent Record projections. No training or model interpretation."""

from collections.abc import Sequence

import pandas as pd

from phl_risk.exceptions import ExperimentError

from .._utils import shanghai_time

_COLUMNS = [
    "run",
    "run_id",
    "name",
    "status",
    "created_at",
    "started_at",
    "family",
    "backend",
    "n_features",
]


def _identity(record):
    return {
        "run": record.to_dict().get("run", record.run_id),
        "run_id": record.run_id,
        "name": record.name,
        "status": record.status,
        "created_at": shanghai_time(record.created_at),
        "started_at": shanghai_time(record.started_at),
        "family": record.model["family"],
        "backend": record.model["backend"],
        "n_features": record.n_features,
    }


def index_records(records, *, status=None) -> pd.DataFrame:
    if status is not None and status not in ("running", "completed", "failed"):
        raise ExperimentError("Unknown Run status")
    return pd.DataFrame(
        [_identity(r) for r in records if status is None or r.status == status], columns=_COLUMNS
    )


def _names(value, default, label, *, available=None, allow_all=False):
    if value is None:
        return list(default)
    if isinstance(value, str):
        if allow_all and value == "all":
            return list(available if available is not None else default)
        raise ExperimentError(
            f"{label} must be a sequence of names" + (" or all" if allow_all else "")
        )
    if not isinstance(value, Sequence) or any(not isinstance(n, str) or not n for n in value):
        raise ExperimentError(f"{label} must be a sequence of names")
    if len(set(value)) != len(value):
        raise ExperimentError(f"Duplicate {label}")
    if available is not None and set(value) - set(available):
        raise ExperimentError(f"Unknown {label}: {sorted(set(value) - set(available))}")
    return list(value)


def _flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and not {"callable", "source_sha256"} <= item.keys():
            result.update(_flatten(item, path))
        else:
            result[path] = item
    return result


def _lookup(facts, path):
    for key in path.split("."):
        if not isinstance(facts, dict) or key not in facts:
            return None
        facts = facts[key]
    return facts


def _parameter(values, key):
    if key in values:
        return values[key]
    matches = [path for path in values if path.endswith("." + key)]
    if len(matches) > 1:
        raise ExperimentError(f"Ambiguous parameter {key!r}; use a full config path")
    return values[matches[0]] if matches else None


def _display_name(value):
    """Show only recorded names; never infer a task from metrics or model files."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [_display_name(item) for item in value]
    if isinstance(value, dict) and isinstance(value.get("callable"), str):
        return value["callable"].rsplit(".", 1)[-1].rsplit(":", 1)[-1]
    return "未记录"


def _task_labels(record):
    config = record.configuration["resolved"]
    labels = {}
    for key in ("objective", "metric", "feval"):
        paths = (
            (f"train.{key}", key, f"params.{key}", f"model.params.{key}")
            if key == "feval"
            else (f"params.{key}", f"model.params.{key}", key)
        )
        value = next(
            (_lookup(config, path) for path in paths if _lookup(config, path) is not None), None
        )
        # Explicit backend-neutral identities take precedence over legacy config paths.
        task = record.to_dict().get("task", {})
        labels[key] = _display_name(task[key] if key in task else value)
    return labels


def compare_records(
    records,
    *,
    metrics=None,
    partitions=None,
    params=None,
    features=False,
    metadata=None,
    fields=None,
    default_partitions=None,
) -> pd.DataFrame:
    records = [r for r in records if r.status == "completed"]
    all_parts = list(dict.fromkeys(p for r in records for p in r.metrics))
    defaults = list(
        dict.fromkeys(p for r in records for p in r.input.get("comparison_partitions", r.metrics))
    )
    if default_partitions is not None:
        defaults = [p for p in default_partitions if p in all_parts]
    parts = _names(partitions, defaults, "partitions", available=all_parts, allow_all=True)
    all_metrics = sorted({m for r in records for v in r.metrics.values() for m in v})
    scores = _names(metrics, all_metrics, "metrics", available=all_metrics)
    task_labels = [_task_labels(record) for record in records]
    show_feval = any(labels["feval"] != "未记录" for labels in task_labels)
    configs = [_flatten(r.configuration["resolved"]) for r in records]
    default_params = sorted({key for config in configs for key in config})
    selected_params = _names(params, default_params, "params", allow_all=True)
    if params == "all":
        selected_params = sorted({key for config in configs for key in config})
    selected_meta = _names(metadata, [], "metadata")
    allowed_fields = set(_COLUMNS) | {
        k
        for r in records
        for k, value in r.result.items()
        if k != "metrics" and isinstance(value, (str, int, float, bool))
    }
    selected_fields = _names(fields, [], "fields", available=allowed_fields)
    selected_fields = [field for field in selected_fields if field != "name"]
    if type(features) is not bool:
        raise ExperimentError("features must be boolean")
    metric_columns = [f"{part}_{metric}" for part in parts for metric in scores]
    columns = [
        "run",
        "name",
        "created_at",
        "objective",
        "metric",
        *(["feval"] if show_feval else []),
        *metric_columns,
        "params",
        *selected_fields,
        *(["features"] if features else []),
        *selected_meta,
    ]
    if len(set(columns)) != len(columns):
        raise ExperimentError("Comparison column collision; select fewer fields")
    for key in selected_meta:
        if records and not any(key in _flatten(r.to_dict()) for r in records):
            raise ExperimentError(f"Unknown metadata: {key}")
    rows = []
    for record, config, labels in zip(records, configs, task_labels, strict=True):
        identity = _identity(record)
        row = {"run": identity["run"], "name": record.name, "created_at": identity["created_at"]}
        row.update({key: labels[key] for key in ("objective", "metric")})
        if show_feval:
            row["feval"] = labels["feval"]
        for part in parts:
            for metric in scores:
                row[f"{part}_{metric}"] = record.metrics.get(part, {}).get(metric, float("nan"))
        row["params"] = {key: _parameter(config, key) for key in selected_params}
        row.update({key: identity.get(key, record.result.get(key)) for key in selected_fields})
        if features:
            row["features"] = record.features
        row.update({key: _lookup(record.to_dict(), key) for key in selected_meta})
        rows.append(row)
    frame = pd.DataFrame(rows, columns=columns)
    for col in metric_columns:
        frame[col] = frame[col].astype(float)
    return frame


def compare_parameters(left, right, *, params=None, only_changed=True):
    if type(only_changed) is not bool:
        raise ExperimentError("only_changed must be boolean")
    first, second = (
        {
            **_flatten(record.configuration["resolved"]),
            **_flatten(record.configuration.get("python_hooks", {}), "python_hooks"),
        }
        for record in (left, right)
    )
    keys = sorted(first.keys() | second.keys())
    if params is not None:
        requested = _names(params, [], "params")
        chosen = set()
        for name in requested:
            matches = [name] if name in keys else [key for key in keys if key.endswith("." + name)]
            if len(matches) != 1:
                raise ExperimentError(f"Unknown or ambiguous parameter {name!r}; use full path")
            chosen.add(matches[0])
        keys = sorted(chosen)
    labels = [_identity(record)["run"] for record in (left, right)]
    if labels[0] == labels[1] or set(labels) & {
        "parameter",
        "change",
        "left_present",
        "right_present",
    }:
        labels = ["left", "right"]
    rows = []
    for key in keys:
        lp, rp = key in first, key in second
        lv, rv = first.get(key), second.get(key)
        equal = lp == rp and type(lv) is type(rv) and lv == rv
        if only_changed and equal:
            continue
        rows.append(
            {
                "parameter": key,
                labels[0]: lv,
                labels[1]: rv,
                "change": "unchanged"
                if equal
                else "added"
                if not lp
                else "removed"
                if not rp
                else "modified",
                "left_present": lp,
                "right_present": rp,
            }
        )
    # object values preserve integers/None; presence columns disambiguate absent vs explicit null.
    return pd.DataFrame(
        rows,
        columns=["parameter", *labels, "change", "left_present", "right_present"],
        dtype=object,
    )
