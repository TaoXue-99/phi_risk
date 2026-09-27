import json

import pytest

from phl_risk.exceptions import ExperimentError, RunError
from phl_risk.modeling.experiment import Experiment
from phl_risk.modeling.experiment.adapters.hydra import compose_config


def custom_objective(predictions, data):
    return predictions - data.get_label(), predictions * 0 + 1


def test_initialization_requires_explicit_task_and_preserves_existing(tmp_path):
    exp = Experiment(root=tmp_path)
    with pytest.raises(ExperimentError, match="objective.*metric"):
        exp.initialize(method="lgb")
    assert not (tmp_path / "lgb").exists()
    yaml = pytest.importorskip("yaml")
    space = exp.initialize(method="lgb", objective="binary", metric=["auc", "binary_logloss"])
    path = space.config_dir / "baseline.yaml"
    cfg = yaml.safe_load(path.read_text())
    assert cfg["params"]["metric"] == ["auc", "binary_logloss"]
    assert cfg["params"]["objective"] == "binary"
    assert cfg["train"]["num_boost_round"] > 0
    assert cfg["train"]["early_stopping"]["first_metric_only"] is True
    content = path.read_text()
    exp.initialize(method="lgb")
    assert path.read_text() == content
    with pytest.raises(ExperimentError, match="existing|Existing"):
        exp.initialize(method="lgb", objective="regression", metric="l2")
    assert path.read_text() == content


def test_callable_initialization_and_config_snapshot(tmp_path):
    yaml = pytest.importorskip("yaml")
    space = Experiment(root=tmp_path).initialize(
        method="lgb", objective=custom_objective, metric="None"
    )
    cfg = yaml.safe_load((space.config_dir / "baseline.yaml").read_text())
    assert cfg["params"]["objective"]["callable"].endswith("custom_objective")
    cfg["params"]["objective"] = custom_objective
    with space.start_run(config=cfg) as attempt:
        cfg["params"]["max_depth"] = 99
        attempt.log_metrics({"valid": {"custom": 0.2}})
    snapshot = yaml.safe_load((attempt.path / "config.yaml").read_text())
    assert snapshot["params"]["max_depth"] != 99
    assert snapshot["params"]["objective"]["callable"].endswith("custom_objective")
    assert attempt.record.configuration["resolved"] == snapshot
    assert space.compare().iloc[0]["params"]["params.max_depth"] == snapshot["params"]["max_depth"]


def test_hydra_overrides_and_external_yaml_recorded(tmp_path):
    yaml = pytest.importorskip("yaml")
    pytest.importorskip("hydra")
    space = Experiment(root=tmp_path).initialize(method="lgb", objective="binary", metric="auc")
    composed = compose_config(
        config_dir=space.config_dir,
        config_name="baseline",
        overrides=[
            "params.max_depth=2",
            "params.learning_rate=0.02",
            "params.metric=[auc,binary_logloss]",
        ],
    )
    with space.start_run(config=composed) as first:
        first.log_metrics({"valid": {"auc": 0.7}})
    assert yaml.safe_load((first.path / "overrides.yaml").read_text()) == list(composed.overrides)
    assert (
        yaml.safe_load((first.path / "config.source.yaml").read_text())["params"]["max_depth"] == 4
    )
    source = space.config_dir / "alternative.yaml"
    source.write_text("params:\n  max_depth: 5\n  learning_rate: 0.02\n")
    with space.start_run(config=source) as second:
        source.write_text("changed: true")
    assert second.record.configuration["resolved"]["params"]["max_depth"] == 5
    diff = space.compare_params(first.run_id, second.run_id, params=["max_depth"])
    assert diff.parameter.tolist() == ["params.max_depth"]
    assert json.loads((first.path / "run.json").read_text())["configuration"]["overrides"] == list(
        composed.overrides
    )
    with pytest.raises(RunError, match="params.*config|config.*params"):
        space.start_run(config={}, params={})


@pytest.mark.parametrize(
    "objective,metric",
    [(None, "auc"), ("binary", None), (12, "auc"), ("binary", []), ("binary", ["auc", 3])],
)
def test_invalid_task_does_not_create_method(tmp_path, objective, metric):
    with pytest.raises(ExperimentError):
        Experiment(root=tmp_path).initialize(method="lgb", objective=objective, metric=metric)
    assert not (tmp_path / "lgb").exists()


def test_config_cannot_drift_via_log_params(tmp_path):
    pytest.importorskip("yaml")
    space = Experiment(root=tmp_path).initialize(method="generic")
    with space.start_run(config={"depth": 2}) as attempt:
        with pytest.raises(RunError, match="config"):
            attempt.log_params({"new": 1})
    assert attempt.record.configuration["resolved"] == {"depth": 2}


def test_native_training_from_generated_config(tmp_path):
    yaml = pytest.importorskip("yaml")
    lgb = pytest.importorskip("lightgbm")
    import numpy as np

    from phl_risk.modeling.experiment.adapters.lightgbm import (
        enable_lightgbm_compatibility,
        load_lightgbm,
        record_lightgbm,
    )

    if tuple(int(v) for v in lgb.__version__.split(".")[:2]) < (4, 6):
        enable_lightgbm_compatibility()
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(600, 10))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    space = Experiment(root=tmp_path).initialize(
        method="lgb", objective="binary", metric=["auc", "binary_logloss"]
    )
    cfg = yaml.safe_load((space.config_dir / "baseline.yaml").read_text())
    cfg["params"].update(min_data_in_leaf=10, num_threads=1)
    cfg["train"]["num_boost_round"] = 8
    for objective in ["binary", custom_objective]:
        cfg["params"]["objective"] = objective
        with space.start_run(config=cfg) as attempt:
            ds = lgb.Dataset(x[:400], label=y[:400])
            dv = ds.create_valid(x[400:], label=y[400:])
            history = {}
            model = lgb.train(
                cfg["params"],
                ds,
                num_boost_round=cfg["train"]["num_boost_round"],
                valid_sets=[dv],
                valid_names=["valid"],
                callbacks=[
                    lgb.record_evaluation(history),
                    lgb.early_stopping(**cfg["train"]["early_stopping"]),
                    lgb.log_evaluation(**cfg["train"]["log_evaluation"]),
                ],
            )
            assert set(history["valid"]) == {"auc", "binary_logloss"}
            attempt.log_metrics({"valid": {"auc": history["valid"]["auc"][-1]}})
            record_lightgbm(attempt, model, num_iteration=model.current_iteration())
        np.testing.assert_allclose(
            model.predict(x[400:]), load_lightgbm(attempt.record).predict(x[400:])
        )
