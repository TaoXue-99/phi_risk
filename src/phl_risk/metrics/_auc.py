from numpy.typing import ArrayLike
from sklearn import metrics as sklearn_metrics

from phl_risk.exceptions import InvalidMetricError

from ._common import InvalidPolicy, invalid, prepare_binary, validate_policy


def auc_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    sample_weight: ArrayLike | None = None,
    *,
    on_invalid: InvalidPolicy = "nan",
) -> float:
    """Binary sklearn ROC AUC with framework missing and invalid-group policies."""
    validate_policy(on_invalid)
    try:
        target, score, weight = prepare_binary(y_true, y_score, sample_weight)
    except InvalidMetricError as exc:
        return invalid(str(exc), on_invalid)
    return float(sklearn_metrics.roc_auc_score(target, score, sample_weight=weight))
