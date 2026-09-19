import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from phl_risk.exceptions import ExperimentError, RunError
from phl_risk.metrics import auc_score
from phl_risk.modeling import DataPlan
from phl_risk.modeling.experiment.lightgbm import LightGBMExperiment, LightGBMRun
from phl_risk.modeling.plan import FeatureSpec


def test_native_run_artifacts_and_weighted_prediction(exp, config, inputs):
    run = exp.run(name="baseline", config=config)
    assert run.status == "completed" and run.n_features == 10
    assert run.model.__class__.__name__ == "Booster"
    assert run.best_iteration > 0
    assert set(run.eval_history) == {"train", "test"}
    assert len(run.eval_history["train"]["auc"]) >= run.best_iteration
    for part in ["train", "test", "oot"]:
        data = inputs["data"].query("part == @part")
        prediction = run.model.predict(data[list(run.features)], num_iteration=run.best_iteration)
        assert run.metrics[f"{part}_auc"] == pytest.approx(auc_score(data.y, prediction, data.w))
    assert run.metrics["auc_gap"] == pytest.approx(
        run.metrics["train_auc"] - run.metrics["test_auc"]
    )
    assert set(run.feature_importance.columns) == {
        "feature",
        "importance_gain",
        "importance_split",
        "gain_rank",
        "split_rank",
    }
    for file in [
        "run.json",
        "config.yaml",
        "overrides.yaml",
        "features.json",
        "metrics.json",
        "eval_history.json",
        "feature_importance.csv",
        "model.txt",
    ]:
        assert (run.path / file).is_file()
    assert not (run.path / "predictions.csv").exists()
    loaded = LightGBMRun.load(run.path)
    assert loaded.metrics == run.metrics and loaded.features == run.features
    np.testing.assert_allclose(
        loaded.model.predict(inputs["data"][list(run.features)]),
        run.model.predict(inputs["data"][list(run.features)]),
    )


def test_immutable_duplicate_names_and_reload(exp, config, inputs, tmp_path):
    first = exp.run(name="baseline", config=config)
    before = {p.name: p.read_bytes() for p in first.path.iterdir()}
    second = exp.run(name="baseline", config=config, features=["x3", "x1"])
    assert second.features == ("x1", "x3") and first.run_id != second.run_id
    assert before == {p.name: p.read_bytes() for p in first.path.iterdir()}
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        first.name = "changed"
    metrics = first.metrics
    metrics["test_auc"] = 42
    assert first.metrics["test_auc"] != 42
    reloaded = LightGBMExperiment(name="demo", root=tmp_path, **inputs)
    assert len(reloaded.runs) == 2
    assert reloaded.get_run(first.run_id).metrics == first.metrics
    with pytest.raises(ExperimentError, match="ambiguous"):
        reloaded.get_run("baseline")


def test_reload_contract_rejected(exp, inputs, tmp_path):
    changed = dict(inputs)
    changed["data_plan"] = DataPlan(
        inputs["data_plan"].roles, FeatureSpec(["x1", "x0"]), inputs["data_plan"].split
    )
    with pytest.raises(ExperimentError, match="contract"):
        LightGBMExperiment(name="demo", root=tmp_path, **changed)


def test_failed_training_has_failed_manifest(exp, config):
    config["model"]["params"]["num_leaves"] = 1
    with pytest.raises(RunError):
        exp.run(name="broken", config=config)
    manifests = list(exp.path.glob("runs/*/run.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "failed" and manifest["error_type"]
    assert exp.runs == ()


def test_corrupt_artifact(exp, config):
    run = exp.run(name="baseline", config=config)
    (run.path / "metrics.json").write_text("{}")
    with pytest.raises(RunError, match="artifact"):
        LightGBMRun.load(run.path)


def test_validation_absent_and_empty(exp, config, inputs, tmp_path):
    config["train"]["validation_partition"] = "validation"
    with pytest.raises(ExperimentError, match="available partitions"):
        exp.run(name="invalid", config=config)
    inputs["data"] = inputs["data"].query("part != 'test'")
    empty = LightGBMExperiment(name="empty", root=tmp_path, **inputs)
    config["train"]["validation_partition"] = "test"
    with pytest.raises(ExperimentError, match="test partition is empty"):
        empty.run(name="invalid", config=config)
