"""Vectorized PSI on aligned proportions; the sole numerical PSI implementation."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._policy import InvalidPolicy, invalid, validate_policy


def validate_epsilon(epsilon: float) -> None:
    if not np.isscalar(epsilon) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")


def psi_from_proportions(
    reference_pct: ArrayLike,
    current_pct: ArrayLike,
    *,
    epsilon: float = 1e-8,
    on_invalid: InvalidPolicy = "nan",
) -> float | NDArray[np.float64]:
    """Reduce the final bin axis of aligned 1D/2D probability arrays.

    Each row must be finite, nonnegative and sum to one (numerical tolerance).
    Clip every probability to epsilon, then renormalize each row.
    Shape/configuration errors always raise;
    invalid distributions follow on_invalid independently for each row.
    """
    validate_policy(on_invalid)
    validate_epsilon(epsilon)
    ref = np.asarray(reference_pct, dtype=np.float64)
    cur = np.asarray(current_pct, dtype=np.float64)
    if ref.ndim not in (1, 2) or ref.shape != cur.shape or ref.shape[-1] == 0:
        raise ValueError("PSI requires aligned 1D or 2D arrays with a nonempty bin axis")
    with np.errstate(over="ignore", invalid="ignore"):
        ref_sum, cur_sum = ref.sum(axis=-1), cur.sum(axis=-1)
    valid = (
        np.isfinite(ref).all(axis=-1)
        & np.isfinite(cur).all(axis=-1)
        & (ref >= 0).all(axis=-1)
        & (cur >= 0).all(axis=-1)
        & np.isclose(ref_sum, 1)
        & np.isclose(cur_sum, 1)
    )
    if not np.all(valid):
        invalid("PSI requires finite nonnegative proportions summing to one", on_invalid)
    # Invalid rows are masked to avoid extra floating-point warnings.
    p = np.maximum(np.where(np.expand_dims(valid, -1), ref, 0), epsilon)
    q = np.maximum(np.where(np.expand_dims(valid, -1), cur, 0), epsilon)
    # Scaling before normalization also makes very large epsilon safe.
    p /= p.max(axis=-1, keepdims=True)
    q /= q.max(axis=-1, keepdims=True)
    p /= p.sum(axis=-1, keepdims=True)
    q /= q.sum(axis=-1, keepdims=True)
    value = np.sum((q - p) * (np.log(q) - np.log(p)), axis=-1)
    value = np.where(valid, value, np.nan)
    return float(value) if ref.ndim == 1 else value
