"""Resolve declaration contracts against attached tabular data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from phl_risk.exceptions import ExperimentError
from phl_risk.modeling import DataPlan, ModelPlan
from phl_risk.modeling.plan import ColumnSplitter, HashSplitter

from .split import assign_partitions


@dataclass(frozen=True)
class RuntimeData:
    partitions: dict[str, pd.DataFrame]
    target: str
    weight: str | None
    categorical: tuple[str, ...]
    schema: dict


def select_features(plan: DataPlan, features=None) -> tuple[str, ...]:
    values = plan.features.all if features is None else features
    if isinstance(values, str):
        raise ExperimentError("features must be a sequence of feature names")
    values = tuple(values)
    if not values or len(set(values)) != len(values):
        raise ExperimentError("features must be non-empty and cannot contain duplicates")
    unknown = set(values) - set(plan.features.all)
    if unknown:
        raise ExperimentError(f"Unknown features: {sorted(unknown)}")
    return tuple(f for f in plan.features.all if f in values)


def resolve_data(data: pd.DataFrame, model_plan: ModelPlan, data_plan: DataPlan) -> RuntimeData:
    data_plan.validate_against(model_plan)
    strategy = model_plan.strategy
    if (
        model_plan.goal.family != "binary_classification"
        or strategy.name != "LightGBM"
        or strategy.family != "tree"
        or strategy.execution_family != "tree"
        or model_plan.objective not in ("binary_logloss", "weighted_binary_logloss")
    ):
        raise ExperimentError("V1 requires BinaryClassification + LightGBM tree execution")
    if not isinstance(data, pd.DataFrame) or data.columns.has_duplicates:
        raise ExperimentError("data must be a DataFrame with unique columns")
    roles = data_plan.roles.bindings
    target = roles["target"][0]
    weight = roles.get("weight", (None,))[0]
    features = select_features(data_plan)
    unsafe = [f for f in features if any(c.isspace() or c in '"[]{}:,' or ord(c) < 32 for c in f)]
    if unsafe:
        raise ExperimentError(f"Unsupported LightGBM feature name(s): {unsafe}; rename explicitly")
    role_columns = {c for columns in roles.values() for c in columns}
    splitter = data_plan.split.splitter
    split_columns = (
        {splitter.column}
        if isinstance(splitter, ColumnSplitter)
        else set(splitter.key)
        if isinstance(splitter, HashSplitter)
        else set()
    )
    overlap = set(features) & (role_columns | split_columns)
    if overlap:
        raise ExperimentError(f"Features overlap role/split columns: {sorted(overlap)}")
    required = set(features) | role_columns | split_columns
    missing = required - set(data.columns)
    if missing:
        raise ExperimentError(f"Missing required columns: {sorted(missing)}")
    y = data[target]
    if y.isna().any() or not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ExperimentError("binary target must be non-missing and contain both 0 and 1 only")
    if weight:
        try:
            w = data[weight].to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ExperimentError("weight must be numeric") from exc
        if not np.isfinite(w).all() or (w < 0).any():
            raise ExperimentError("weight must be finite and non-negative")
    assignment = assign_partitions(data, data_plan.split)
    partitions = {
        name: data.iloc[np.flatnonzero(assignment.to_numpy() == name)].copy()
        for name in data_plan.split.partitions.definitions
    }
    if partitions["train"].empty:
        raise ExperimentError("train partition is empty")
    categories = {}
    for col in data_plan.features.categorical:
        # Vocabulary learned only on train; unseen evaluation labels become missing.
        dtype = pd.CategoricalDtype(pd.unique(partitions["train"][col].dropna()), ordered=False)
        categories[col] = [str(v) for v in dtype.categories]
        for part in partitions.values():
            part[col] = part[col].where(part[col].isin(dtype.categories)).astype(dtype)
    for col in data_plan.features.numerical:
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ExperimentError(f"Numerical feature {col!r} must have numeric dtype")
    schema = {
        "dtypes": {str(c): str(data[c].dtype) for c in required},
        "categorical_levels": categories,
        "n_rows": len(data),
    }
    return RuntimeData(partitions, target, weight, data_plan.features.categorical, schema)


def validate_partitions(runtime: RuntimeData, validation: str) -> None:
    if validation not in runtime.partitions:
        raise ExperimentError(
            f"Unknown validation partition {validation!r}; available partitions: "
            f"{list(runtime.partitions)}"
        )
    for name, part in runtime.partitions.items():
        if part.empty:
            raise ExperimentError(f"{name} partition is empty; AUC cannot be computed")
        effective = part if runtime.weight is None else part.loc[part[runtime.weight] > 0]
        if effective[runtime.target].nunique() != 2:
            raise ExperimentError(
                f"AUC cannot be computed for {name}: requires both 0 and 1 "
                "with positive effective weight"
            )
