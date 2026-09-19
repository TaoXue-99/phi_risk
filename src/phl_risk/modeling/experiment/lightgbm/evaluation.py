"""Shared weighted AUC and native gain/split importance at the saved iteration."""

import numpy as np
import pandas as pd

from phl_risk.exceptions import ExperimentError
from phl_risk.metrics import auc_score


def evaluate(fit, runtime, features, validation, *, num_threads=None) -> dict:
    metrics = {}
    for name, data in runtime.partitions.items():
        predictions = fit.booster.predict(
            data[list(features)], num_iteration=fit.best_iteration, num_threads=num_threads
        )
        if not np.isfinite(predictions).all():
            raise ExperimentError(f"AUC cannot be computed for {name}: non-finite predictions")
        try:
            metrics[f"{name}_auc"] = auc_score(
                data[runtime.target],
                predictions,
                sample_weight=None if runtime.weight is None else data[runtime.weight],
                on_invalid="raise",
            )
        except ValueError as exc:
            raise ExperimentError(f"AUC cannot be computed for {name}: {exc}") from exc
    metrics["auc_gap"] = metrics["train_auc"] - metrics[f"{validation}_auc"]
    return metrics


def feature_importance(fit, features) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "feature": features,
            "importance_gain": fit.booster.feature_importance("gain", iteration=fit.best_iteration),
            "importance_split": fit.booster.feature_importance(
                "split", iteration=fit.best_iteration
            ),
        }
    )
    for kind in ("gain", "split"):
        frame[f"{kind}_rank"] = (
            frame[f"importance_{kind}"].rank(method="min", ascending=False).astype(int)
        )
    return frame.sort_values("importance_gain", ascending=False, kind="stable").reset_index(
        drop=True
    )
