from copy import deepcopy

import pandas as pd
import pytest

from phl_risk.exceptions import ExperimentError


def test_compare_deltas_features_parameters_reference(exp, config):
    baseline = exp.run(name="baseline", config=config)
    changed = deepcopy(config)
    changed["model"]["params"]["max_depth"] = 2
    candidate = exp.run(name="depth2", config=changed, features=["x0", "x2", "x4"])
    frame = exp.compare()
    assert isinstance(frame, pd.DataFrame)
    row = frame.set_index("run").loc["depth2"]
    assert row.feature_change == "10→3 (-7, -70.0%)"
    assert row.param_changes == "max_depth: 3→2"
    assert row.delta_test_auc == pytest.approx(
        candidate.metrics["test_auc"] - baseline.metrics["test_auc"]
    )
    assert row.delta_auc_gap == pytest.approx(
        candidate.metrics["auc_gap"] - baseline.metrics["auc_gap"]
    )
    assert "oot_auc" not in frame
    assert "oot_auc" in exp.compare(include_oot=True)
    assert "delta_oot_auc" in exp.compare(include_oot=True)
    assert pd.api.types.is_float_dtype(frame.test_auc)
    assert "max_depth" in exp.compare(params="all")
    assert list(exp.compare(params=["max_depth", "num_leaves"]).columns[-2:]) == [
        "max_depth",
        "num_leaves",
    ]
    exp.set_reference(candidate.run_id)
    assert exp.compare().set_index("run").loc["depth2"].delta_test_auc == 0
    assert (
        exp.compare(reference="baseline").set_index("run").loc["depth2"].delta_test_auc
        == row.delta_test_auc
    )


def test_same_count_changed_membership_and_train_control(exp, config):
    exp.run(name="a", config=config, features=["x0", "x1"])
    changed = deepcopy(config)
    changed["train"]["num_boost_round"] = 20
    exp.run(name="b", config=changed, features=["x0", "x2"])
    row = exp.compare().set_index("run").loc["b"]
    assert row.feature_change == "2→2 (+1/-1)"
    assert row.param_changes == "train.num_boost_round: 30→20"


def test_empty_comparison_and_reference_errors(exp, config):
    assert exp.compare().empty
    exp.run(name="same", config=config)
    exp.run(name="same", config=config)
    with pytest.raises(ExperimentError, match="ambiguous"):
        exp.compare(reference="same")
    with pytest.raises(ExperimentError):
        exp.compare(params="invalid")
