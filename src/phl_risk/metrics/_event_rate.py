import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ._common import InvalidPolicy, invalid, validate_policy, vector, weights


def event_rate(
    target: ArrayLike,
    event_value: object = 1,
    sample_weight: ArrayLike | None = None,
    *,
    on_invalid: InvalidPolicy = "nan",
) -> float:
    """Event weight divided by non-missing target weight."""
    validate_policy(on_invalid)
    if not pd.api.types.is_scalar(event_value) or pd.isna(event_value):
        raise ValueError("event_value must be a non-missing scalar")
    values = vector(target, "target")
    weight = weights(sample_weight, len(values))
    valid = ~pd.isna(values)
    values, weight = values[valid], weight[valid]
    if len(weight) and weight.max() > 0:
        weight = weight / weight.max()
    denominator = weight.sum()
    if denominator == 0:
        return invalid("EventRate has no positive valid target weight", on_invalid)
    return float(np.sum(weight[values == event_value]) / denominator)
