"""User-owned starting configuration; no backend import or training policy."""

import json

from phl_risk.exceptions import ExperimentError

from .._snapshot import parameter_snapshot


def baseline_template(objective, metric):
    if not (callable(objective) or isinstance(objective, str) and objective.strip()):
        raise ExperimentError("LightGBM initialization requires objective and metric explicitly")
    if not (
        isinstance(metric, str)
        and metric.strip()
        or isinstance(metric, (list, tuple))
        and metric
        and all(isinstance(item, str) and item.strip() for item in metric)
    ):
        raise ExperimentError(
            "metric must be a non-empty string or list of strings; use 'None' to disable"
        )
    objective_text = json.dumps(parameter_snapshot(objective), ensure_ascii=False)
    metric_text = json.dumps(metric, ensure_ascii=False)
    return f"""# Editable starting values, not a guarantee of model quality.
# A callable objective is an identity record: bind the real function in Python.
# train values are passed explicitly to native lgb.train / callbacks in Python.
params:
  objective: {objective_text}
  metric: {metric_text}
  boosting_type: gbdt
  learning_rate: 0.05
  num_leaves: 15
  max_depth: 4
  min_data_in_leaf: 300
  min_sum_hessian_in_leaf: 0.001
  feature_fraction: 0.8
  bagging_fraction: 0.8
  bagging_freq: 1
  lambda_l1: 1.0
  lambda_l2: 10.0
  min_gain_to_split: 0.0
  max_bin: 255
  seed: 2026
  feature_fraction_seed: 2026
  bagging_seed: 2026
  data_random_seed: 2026
  num_threads: 4
  device_type: cpu
  deterministic: true
  force_col_wise: true
  verbosity: -1
train:
  num_boost_round: 1000
  early_stopping:
    stopping_rounds: 50
    first_metric_only: true
    min_delta: 0.0
    verbose: false
  log_evaluation:
    period: 100
"""
