import numpy as np
from numpy.typing import ArrayLike
from sklearn import metrics as sklearn_metrics

from phl_risk.exceptions import InvalidMetricError

from ._common import InvalidPolicy, invalid, prepare_binary, validate_policy


def ks_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    sample_weight: ArrayLike | None = None,
    *,
    on_invalid: InvalidPolicy = "nan",
) -> float:
    """Maximum absolute separation from sklearn's weighted binary ROC curve."""
    validate_policy(on_invalid)
    try:
        target, score, weight = prepare_binary(y_true, y_score, sample_weight)
    except InvalidMetricError as exc:
        return invalid(str(exc), on_invalid)
    fpr, tpr, _ = sklearn_metrics.roc_curve(
        target, score, pos_label=1, sample_weight=weight, drop_intermediate=False
    )
    return float(np.max(np.abs(tpr - fpr)))
