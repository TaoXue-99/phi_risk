import numpy as np
import pandas as pd
import pytest

from phl_risk.exceptions import ExperimentError
from phl_risk.modeling import DataPlan, ModelPlan
from phl_risk.modeling.experiment.lightgbm.config import resolve_config
from phl_risk.modeling.experiment.lightgbm.runtime import resolve_data, select_features
from phl_risk.modeling.experiment.lightgbm.split import assign_partitions
from phl_risk.modeling.goal import BinaryClassification
from phl_risk.modeling.plan import (
    ColumnSplitter,
    FeatureSpec,
    HashSplitter,
    PartitionSpec,
    RandomSplitter,
    RoleSpec,
    SplitSpec,
)
from phl_risk.modeling.strategy import LightGBM


def plans(splitter=None, partitions=None):
    return ModelPlan(BinaryClassification(), LightGBM()), DataPlan(
        RoleSpec(target="y", weight="w"),
        FeatureSpec(["b", "a"], ["cat"]),
        SplitSpec(
            splitter or ColumnSplitter("part"),
            partitions or PartitionSpec(train="tr", test="te", oot="ot"),
        ),
    )


def frame():
    return pd.DataFrame(
        dict(
            b=[1, 2, 3, 4, 5, 6],
            a=[6, 5, 4, 3, 2, 1],
            cat=["a", "b"] * 3,
            y=[0, 1] * 3,
            w=[1.0, 2.0] * 3,
            part=["tr", "tr", "te", "te", "ot", "ot"],
        )
    )


def test_runtime_schema_and_column_assignment():
    mp, dp = plans()
    runtime = resolve_data(frame(), mp, dp)
    assert {k: len(v) for k, v in runtime.partitions.items()} == {"train": 2, "test": 2, "oot": 2}
    assert runtime.target == "y" and runtime.weight == "w"
    assert select_features(dp, ["cat", "b"]) == ("b", "cat")
    assert str(runtime.partitions["train"]["cat"].dtype) == "category"


@pytest.mark.parametrize("values", [[0, 2] * 3, [1] * 6, [0, 1, 0, 1, 0, None]])
def test_binary_rejected(values):
    mp, dp = plans()
    data = frame()
    data["y"] = values
    with pytest.raises(ExperimentError, match="0.*1|binary"):
        resolve_data(data, mp, dp)


@pytest.mark.parametrize("features", [["a", "a"], ["y"], ["missing"], []])
def test_feature_subset_rejected(features):
    with pytest.raises(ExperimentError):
        select_features(plans()[1], features)


def test_hash_group_stability_and_order():
    data = pd.DataFrame({"id": np.repeat(np.arange(500), 2)})
    spec = SplitSpec(HashSplitter("id"), PartitionSpec(train=0.6, test=0.2, oot=0.2))
    first = assign_partitions(data, spec)
    shuffled = assign_partitions(data.sample(frac=1, random_state=7), spec)
    pd.testing.assert_series_equal(first, shuffled.sort_index())
    assert pd.DataFrame({"id": data.id, "part": first}).groupby("id").part.nunique().max() == 1
    assert set(first) == {"train", "test", "oot"}
    data.loc[0, "id"] = np.nan
    with pytest.raises(ExperimentError, match="missing"):
        assign_partitions(data, spec)


def test_unsupported_split():
    with pytest.raises(ExperimentError, match="RandomSplitter.*V1"):
        assign_partitions(frame(), SplitSpec(RandomSplitter(), PartitionSpec(train=0.7, test=0.3)))


def test_config_complete_and_detached():
    original = {"model": {"params": {"max_depth": 2}}}
    cfg = resolve_config(original)
    assert cfg["model"]["params"]["objective"] == "binary"
    assert cfg["model"]["params"]["metric"] == "auc"
    assert cfg["train"]["validation_partition"] == "valid"
    cfg["model"]["params"]["max_depth"] = 9
    assert original["model"]["params"]["max_depth"] == 2


@pytest.mark.parametrize(
    "cfg",
    [
        {"model": {"params": {"objective": "regression"}}},
        {"model": {"params": {"metric": "rmse"}}},
        {"model": {"params": {"num_iterations": 5}}},
        {"train": {"validation_partition": "train"}},
        {"train": {"num_boost_round": 0}},
        {"train": {"typo": 4}},
    ],
)
def test_invalid_config(cfg):
    with pytest.raises(ExperimentError):
        resolve_config(cfg)
