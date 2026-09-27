import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from phl_risk.exceptions import RunError
from phl_risk.modeling.experiment import Experiment, Run
from phl_risk.modeling.experiment.adapters.lightgbm import load_lightgbm, record_lightgbm


@pytest.mark.parametrize("objective", ["binary", "regression", "regression_l1", "multiclass"])
def test_native_tasks_record_and_reload(native, data, space, objective):
    features = list(data.columns[:10])
    y = data.y if objective == "binary" else data.x0 * 2 + data.x1
    params = dict(objective=objective, metric="None", num_threads=1, verbosity=-1, seed=2026)
    if objective == "multiclass":
        params["num_class"] = 3
        y = np.arange(len(data)) % 3
    with space.start_run(params=params, comparison_partitions=["train", "valid"]) as attempt:
        dataset = native.Dataset(data[features], label=y, weight=data.w)
        model = native.train(params, dataset, num_boost_round=8)
        attempt.log_params({"num_boost_round": 8})
        attempt.log_metrics(
            {
                "train": {"user_metric": 0.5},
                "valid": {"user_metric": 0.4},
                "test": {"user_metric": 0.3},
            }
        )
        record_lightgbm(attempt, model, num_iteration=model.current_iteration())
    run = attempt.record
    assert run.run == "lgb_run_01" and run.config["objective"] == objective
    assert "test_user_metric" not in space.compare()
    actual = load_lightgbm(Run.load(run.path)).predict(data[features], num_threads=1)
    np.testing.assert_allclose(actual, model.predict(data[features], num_threads=1))
    assert run.features == tuple(features)
    pd.testing.assert_frame_equal(
        space.compare(), Experiment(root=space.path.parent).open_method("lgb").compare()
    )


def test_custom_objective_callbacks_and_history_remain_native(native, data, space):
    def objective(raw, dataset):
        p = expit(raw)
        w = dataset.get_weight()
        return (p - dataset.get_label()) * w, p * (1 - p) * w

    def feval(raw, dataset):
        return (
            "mse",
            float(
                np.average((expit(raw) - dataset.get_label()) ** 2, weights=dataset.get_weight())
            ),
            False,
        )

    params = dict(objective=objective, metric="None", num_threads=1, verbosity=-1)
    history = {}
    with space.start_run(params=params, metadata={"code_revision": "native-test-v1"}) as attempt:
        train = native.Dataset(
            data.iloc[:400, :10], label=data.y.iloc[:400], weight=data.w.iloc[:400]
        )
        valid = train.create_valid(
            data.iloc[400:, :10], label=data.y.iloc[400:], weight=data.w.iloc[400:]
        )
        model = native.train(
            params,
            train,
            num_boost_round=20,
            valid_sets=[train, valid],
            valid_names=["train", "valid"],
            feval=feval,
            callbacks=[
                native.record_evaluation(history),
                native.early_stopping(5, verbose=False),
                native.reset_parameter(learning_rate=lambda i: 0.1 / (1 + i / 10)),
            ],
        )
        attempt.log_json("history", history)
        record_lightgbm(
            attempt, model, num_iteration=model.best_iteration or model.current_iteration()
        )
    assert "callable" in attempt.record.config["objective"]
    assert attempt.record.read_json("history")["valid"]["mse"]
    assert (
        load_lightgbm(attempt.record).current_iteration()
        == attempt.record.result["saved_iteration"]
    )


def test_snapshot_survives_native_continued_training(native, data, space):
    params = dict(objective="binary", metric="auc", verbosity=-1, num_threads=1)
    train = native.Dataset(data.iloc[:, :10], label=data.y, free_raw_data=False)
    model = native.train(params, train, num_boost_round=3, keep_training_booster=True)
    with space.start_run(params=params) as attempt:
        record_lightgbm(attempt, model, num_iteration=3)
        model.update()
    assert model.current_iteration() == 4
    assert load_lightgbm(attempt.record).current_iteration() == 3
    continued = native.train(
        params, train, num_boost_round=2, init_model=load_lightgbm(attempt.record)
    )
    assert continued.current_iteration() == 5


def test_native_categorical_and_selected_iteration(native, data, space):
    data["cat"] = pd.Categorical(np.where(data.x0 > 0, "a", "b"))
    features = ["cat", "x2"]
    model = native.train(
        dict(objective="binary", num_threads=1, verbosity=-1),
        native.Dataset(data[features], label=data.y, categorical_feature=["cat"]),
        num_boost_round=4,
    )
    with space.start_run() as attempt:
        record_lightgbm(attempt, model, num_iteration=2)
    assert load_lightgbm(attempt.record).current_iteration() == 2
    assert attempt.record.features == tuple(features)


def test_corrupt_model_and_feature_metadata_fail(native, data, space):
    model = native.train(
        dict(objective="binary", num_threads=1, verbosity=-1),
        native.Dataset(data.iloc[:, :10], label=data.y),
        num_boost_round=2,
    )
    with space.start_run() as attempt:
        record_lightgbm(attempt, model, num_iteration=2)
    record = attempt.record
    manifest = record.path / "run.json"
    facts = json.loads(manifest.read_text())
    facts["input"]["features"][0] = "wrong"
    manifest.write_text(json.dumps(facts))
    with pytest.raises(RunError, match="feature"):
        load_lightgbm(Run.load(record.path))
    (record.path / "model.txt").write_text("corrupt")
    with pytest.raises(RunError):
        Run.load(record.path)


def test_optional_dependency_error(monkeypatch):
    import importlib

    from phl_risk.exceptions import OptionalDependencyError
    from phl_risk.modeling.experiment.adapters.lightgbm import enable_lightgbm_compatibility

    original = importlib.import_module

    def missing(name, *a, **k):
        if name == "lightgbm":
            raise ImportError("test missing backend")
        return original(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(OptionalDependencyError):
        enable_lightgbm_compatibility()
