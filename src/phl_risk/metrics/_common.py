"""Framework input policies; numerical algorithms live in mature libraries."""

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from phl_risk.exceptions import InvalidMetricError

InvalidPolicy = Literal["nan", "warn", "raise"]


def validate_policy(on_invalid: InvalidPolicy) -> None:
    if on_invalid not in ("nan", "warn", "raise"):
        raise ValueError("on_invalid must be 'nan', 'warn', or 'raise'")


def invalid(message: str, on_invalid: InvalidPolicy) -> float:
    validate_policy(on_invalid)
    if on_invalid == "raise":
        raise InvalidMetricError(message)
    if on_invalid == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    return float("nan")


def vector(values: ArrayLike, name: str) -> NDArray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def weights(sample_weight: ArrayLike | None, size: int) -> NDArray[np.float64]:
    if sample_weight is None:
        return np.ones(size, dtype=float)
    weight = vector(sample_weight, "sample_weight").astype(float)
    if len(weight) != size:
        raise ValueError("sample_weight and target must have equal lengths")
    if not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("sample_weight must contain finite non-negative values")
    return weight


def prepare_binary(
    y_true: ArrayLike, y_score: ArrayLike, sample_weight: ArrayLike | None = None
) -> tuple[NDArray[np.int8], NDArray[np.float64], NDArray[np.float64] | None]:
    """Apply our missing/class/weight policies before calling sklearn APIs."""
    target = vector(y_true, "y_true")
    raw_score = vector(y_score, "y_score")
    if len(target) != len(raw_score):
        raise ValueError("y_true and y_score must have equal lengths")
    if raw_score.dtype.kind in "biuf":
        score = raw_score.astype(float, copy=False)
    else:
        score = pd.to_numeric(pd.Series(raw_score), errors="raise").to_numpy(
            dtype=float, na_value=np.nan
        )
    weight = None if sample_weight is None else weights(sample_weight, len(target))
    valid = ~pd.isna(target) & np.isfinite(score)
    if weight is not None:
        valid &= weight > 0
    target, score = target[valid], score[valid]
    if not np.isin(target, [0, 1]).all():
        raise InvalidMetricError("AUC/KS require binary target values 0 and 1")
    if weight is not None:
        weight = weight[valid]
        if len(weight):
            weight = weight / weight.max()
        effective = weight > 0
    else:
        effective = np.ones(len(target), dtype=bool)
    if not np.any((target == 1) & effective) or not np.any((target == 0) & effective):
        raise InvalidMetricError("AUC/KS require positive effective weight in both classes")
    # sklearn detects ties by subtracting sorted scores. For a finite range
    # wider than float64 can represent, use NumPy's exact ordered codes instead.
    # This preserves every ordering/tie, unlike rescaling which can underflow
    # small distinct scores; ordinary inputs incur no additional sort.
    if score.min() < 0 and score.max() > np.finfo(float).max + score.min():
        _, score = np.unique(score, return_inverse=True)
        score = score.astype(float)
    # Nullable pandas input can become an object array; sklearn expects a
    # concrete binary dtype after our explicit 0/1 validation.
    return target.astype(np.int8), score, weight
