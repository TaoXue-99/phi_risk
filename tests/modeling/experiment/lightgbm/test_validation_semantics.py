import pytest

from phl_risk.exceptions import ExperimentError
from phl_risk.modeling import DataPlan
from phl_risk.modeling.experiment.lightgbm import LightGBMExperiment
from phl_risk.modeling.experiment.lightgbm.config import resolve_config
from phl_risk.modeling.plan import ColumnSplitter, PartitionSpec, SplitSpec


def test_default_validation_is_valid():
    assert resolve_config({})["train"]["validation_partition"] == "valid"


def test_holdouts_hidden_and_tolerance_uses_validation(tmp_path, inputs, config):
    pytest.importorskip("lightgbm")
    inputs["data"]["part"] = ["train"] * 360 + ["valid"] * 80 + ["test"] * 80 + ["oot"] * 80
    plan = inputs["data_plan"]
    inputs["data_plan"] = DataPlan(
        plan.roles,
        plan.features,
        SplitSpec(
            ColumnSplitter("part"),
            PartitionSpec(train="train", valid="valid", test="test", oot="oot"),
        ),
    )
    config["train"].pop("validation_partition", None)
    exp = LightGBMExperiment(name="valid_demo", root=tmp_path, **inputs)
    assert "valid_auc" in exp.compare().columns
    baseline = exp.run(name="baseline", config=config)
    assert set(baseline.eval_history) == {"train", "valid"}
    assert baseline.metrics["auc_gap"] == pytest.approx(
        baseline.metrics["train_auc"] - baseline.metrics["valid_auc"]
    )
    assert {"test_auc", "oot_auc"} <= baseline.metrics.keys()
    frame = exp.compare()
    assert {"valid_auc", "delta_valid_auc"} <= set(frame)
    assert not {"test_auc", "oot_auc"}.intersection(frame)
    assert "test_auc" in exp.compare(include_test=True)
    assert "oot_auc" not in exp.compare(include_test=True)
    assert "oot_auc" in exp.compare(include_oot=True)
    selection = exp.select_features(base_run=baseline.run_id, candidate_counts=[6, 4], step=0.5)
    assert selection.within_tolerance(tolerance=1).n_features == 4
    with pytest.raises(ExperimentError, match="validation"):
        selection.within_tolerance(metric="test_auc")


def test_missing_valid_does_not_silently_use_test(exp, config):
    config["train"].pop("validation_partition", None)
    with pytest.raises(ExperimentError, match="valid.*available partitions"):
        exp.run(name="missing_valid", config=config)


def test_explicit_legacy_test_validation_remains_supported(exp, config):
    config["train"]["validation_partition"] = "test"
    run = exp.run(name="legacy", config=config)
    assert set(run.eval_history) == {"train", "test"}
    assert "test_auc" in exp.compare()
    assert list(exp.compare(include_test=True).columns).count("test_auc") == 1
