import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from phl_risk.exceptions import ExperimentError, OptionalDependencyError, RunError
from phl_risk.modeling import DataPlan, ModelPlan
from phl_risk.modeling.experiment.lightgbm import LightGBMExperiment, LightGBMRun
from phl_risk.modeling.experiment.lightgbm.config import resolve_config
from phl_risk.modeling.experiment.lightgbm.dataset import LGBDatasetBuilder
from phl_risk.modeling.goal import Regression
from phl_risk.modeling.plan import (
    FeatureSpec,
    HashSplitter,
    PartitionSpec,
    RoleSpec,
    SplitSpec,
)
from phl_risk.modeling.strategy import LightGBM


def test_aliases_do_not_silently_override_explicit_seed():
    cfg = resolve_config(
        {"model": {"params": {"random_state": 42, "boosting": "rf", "bagging_freq": 1}}}
    )
    assert cfg["model"]["params"]["seed"] == 42
    assert cfg["model"]["params"]["boosting_type"] == "rf"
    assert "random_state" not in cfg["model"]["params"]
    with pytest.raises(ExperimentError, match="conflict"):
        resolve_config({"model": {"params": {"seed": 2026, "random_state": 42}}})


@pytest.mark.parametrize(
    "params",
    [
        {"task_type": "predict"},
        {"is_predict_raw_score": True},
        {"label": "x1"},
        {"weight": "x0"},
        {"query": "x0"},
    ],
)
def test_control_aliases_are_rejected(params):
    with pytest.raises(ExperimentError):
        resolve_config({"model": {"params": params}})


def test_categorical_builder_schema_weight_and_fresh_datasets(tmp_path, inputs, config):
    pytest.importorskip("lightgbm")
    data = inputs["data"].copy()
    data["cat"] = np.where(data["x0"] > 0, "high", "low")
    data.loc[data.part == "oot", "cat"] = "unseen"
    inputs["data"] = data
    inputs["data_plan"] = DataPlan(
        inputs["data_plan"].roles, FeatureSpec(["x1", "x0"], ["cat"]), inputs["data_plan"].split
    )
    exp = LightGBMExperiment(name="categories", root=tmp_path, **inputs)
    runtime = exp._runtime
    assert runtime.partitions["oot"]["cat"].isna().all()
    first = LGBDatasetBuilder().build(runtime, ("x1", "cat"), "test")
    second = LGBDatasetBuilder().build(runtime, ("cat",), "test")
    assert first.train is not second.train
    assert first.train.feature_name == ["x1", "cat"]
    assert first.train.categorical_feature == ["cat"]
    np.testing.assert_allclose(first.train.weight, data.w[:360])
    run = exp.run(name="cat", config=config)
    assert run.metadata["categorical_features"] == ["cat"]
    assert tuple(run.model.feature_name()) == ("x1", "x0", "cat")
    selected = exp.select_features(base_run=run.run_id, candidate_counts=[2], step=1)
    assert selected.runs[0].n_features == 2


def test_no_early_stopping_uses_full_model(exp, config):
    config["train"]["early_stopping"]["enabled"] = False
    run = exp.run(name="full", config=config)
    assert run.best_iteration == run.model.current_iteration()
    assert run.best_iteration == 30
    assert len(run.eval_history["test"]["auc"]) == 30


def test_reference_persists_and_reordered_hash_contract_rejected(exp, config, inputs, tmp_path):
    exp.run(name="first", config=config)
    second = exp.run(name="second", config=config, features=["x0", "x1"])
    exp.set_reference(second.run_id)
    restored = LightGBMExperiment(name="demo", root=tmp_path, **inputs)
    assert restored.compare().set_index("run").loc["second", "feature_change"] == "-"
    inputs["data"] = inputs["data"].assign(id=np.arange(600))
    plan = inputs["data_plan"]
    inputs["data_plan"] = DataPlan(
        plan.roles,
        plan.features,
        SplitSpec(HashSplitter("id"), PartitionSpec(train=0.6, test=0.2, oot=0.2)),
    )
    LightGBMExperiment(name="hash", root=tmp_path, **inputs)
    inputs["data_plan"] = DataPlan(
        plan.roles,
        plan.features,
        SplitSpec(HashSplitter("id"), PartitionSpec(train=0.6, oot=0.2, test=0.2)),
    )
    with pytest.raises(ExperimentError, match="contract"):
        LightGBMExperiment(name="hash", root=tmp_path, **inputs)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_target",
        "single_test_class",
        "negative_weight",
        "duplicate_columns",
        "role_feature",
        "regression",
    ],
)
def test_clear_runtime_errors(tmp_path, inputs, config, mutation):
    data = inputs["data"].copy()
    plan = inputs["data_plan"]
    if mutation == "missing_target":
        data = data.drop(columns="y")
    elif mutation == "single_test_class":
        data.loc[data.part == "test", "y"] = 0
    elif mutation == "negative_weight":
        data.loc[0, "w"] = -1
    elif mutation == "duplicate_columns":
        data.columns = ["x0"] * len(data.columns)
    elif mutation == "role_feature":
        inputs["data_plan"] = DataPlan(plan.roles, FeatureSpec(["y"]), plan.split)
    else:
        inputs["model_plan"] = ModelPlan(Regression(), LightGBM())
    inputs["data"] = data
    with pytest.raises(ExperimentError):
        experiment = LightGBMExperiment(name="invalid", root=tmp_path, **inputs)
        experiment.run(name="invalid", config=config)


def test_concurrent_allocations_are_unique(exp):
    def allocate(_):
        return exp._store.allocate("duplicate", {})

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(allocate, range(20)))
    assert len(set(paths)) == 20
    assert all(json.loads((path / "run.json").read_text())["status"] == "running" for path in paths)
    assert exp.compare().empty


def test_config_yaml_cannot_diverge_from_manifest(exp, config):
    run = exp.run(name="baseline", config=config)
    path = run.path / "run.json"
    manifest = json.loads(path.read_text())
    manifest["resolved_config"]["model"]["params"]["max_depth"] = 99
    path.write_text(json.dumps(manifest))
    with pytest.raises(RunError, match="artifact"):
        LightGBMRun.load(run.path)


def test_missing_backend_via_public_run(exp, config, monkeypatch):
    import importlib

    original = importlib.import_module

    def blocked(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError("missing test backend")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", blocked)
    with pytest.raises(OptionalDependencyError, match="phl-risk"):
        exp.run(name="missing", config=config)
    assert not list(exp.path.glob("runs/*"))


def test_hash_determinism_across_process_seeds():
    import os

    script = """
import pandas as pd
from phl_risk.modeling.plan import HashSplitter, SplitSpec, PartitionSpec
from phl_risk.modeling.experiment.lightgbm.split import assign_partitions
s=SplitSpec(HashSplitter(['id','group']),PartitionSpec(train=.6,test=.4))
print(assign_partitions(pd.DataFrame({'id':range(100),'group':['a:b']*100}),s).tolist())
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": seed}
        )
        for seed in ["1", "99"]
    ]
    assert outputs[0] == outputs[1]


def test_unweighted_native_run(tmp_path, inputs, config):
    pytest.importorskip("lightgbm")
    plan = inputs["data_plan"]
    inputs["data_plan"] = DataPlan(RoleSpec(target="y"), plan.features, plan.split)
    experiment = LightGBMExperiment(name="unweighted", root=tmp_path, **inputs)
    run = experiment.run(name="baseline", config=config)
    from phl_risk.metrics import auc_score

    test = inputs["data"].query("part == 'test'")
    predictions = run.model.predict(test[list(run.features)], num_iteration=run.best_iteration)
    assert run.metrics["test_auc"] == pytest.approx(auc_score(test.y, predictions))
    assert run.metadata["weight"] is None


def test_train_target_missing_contract_is_modeling_error(tmp_path, inputs):
    from phl_risk.exceptions import ModelingError

    plan = inputs["data_plan"]
    inputs["data_plan"] = DataPlan(RoleSpec(), plan.features, plan.split)
    with pytest.raises(ModelingError, match="target"):
        LightGBMExperiment(name="missing_role", root=tmp_path, **inputs)


@pytest.mark.parametrize("feature_name", ["x a", "x\ta", "x\na", "x:a", "x[a]"])
def test_backend_feature_name_cannot_change_saved_schema(tmp_path, inputs, feature_name):
    plan = inputs["data_plan"]
    inputs["data"] = inputs["data"].rename(columns={"x0": feature_name})
    inputs["data_plan"] = DataPlan(plan.roles, FeatureSpec([feature_name]), plan.split)
    with pytest.raises(ExperimentError, match="feature name"):
        LightGBMExperiment(name="unsafe_name", root=tmp_path, **inputs)
