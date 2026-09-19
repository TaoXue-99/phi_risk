"""Always record both learning curves; stop only against validation."""

from ._utils import require


def build_callbacks(train: dict, history: dict) -> list:
    lgb = require("lightgbm", "lightgbm")
    callbacks = [lgb.record_evaluation(history)]
    early, log = train["early_stopping"], train["log_evaluation"]
    if early["enabled"]:
        callbacks.append(
            lgb.early_stopping(
                stopping_rounds=early["stopping_rounds"],
                first_metric_only=early["first_metric_only"],
                min_delta=early["min_delta"],
                verbose=log["enabled"],
            )
        )
    if log["enabled"]:
        callbacks.append(lgb.log_evaluation(period=log["period"]))
    return callbacks
