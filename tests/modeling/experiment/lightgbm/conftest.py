import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from phl_risk.modeling import DataPlan, ModelPlan
from phl_risk.modeling.experiment.lightgbm import LightGBMExperiment
from phl_risk.modeling.goal import BinaryClassification
from phl_risk.modeling.plan import ColumnSplitter, FeatureSpec, PartitionSpec, RoleSpec, SplitSpec
from phl_risk.modeling.strategy import LightGBM


@pytest.fixture
def inputs():
    x, y = make_classification(n_samples=600, n_features=10, n_informative=5, random_state=2026)
    data = pd.DataFrame(x, columns=[f"x{i}" for i in range(10)])
    data["y"] = y
    data["w"] = np.linspace(0.5, 2, 600)
    data["part"] = np.repeat(["train", "test", "oot"], [360, 120, 120])
    mp = ModelPlan(BinaryClassification(), LightGBM())
    dp = DataPlan(
        RoleSpec(target="y", weight="w"),
        FeatureSpec(list(data.columns[:10])),
        SplitSpec(ColumnSplitter("part"), PartitionSpec(train="train", test="test", oot="oot")),
    )
    return dict(data=data, model_plan=mp, data_plan=dp)


@pytest.fixture
def config():
    return {
        "model": {
            "params": {"num_threads": 1, "num_leaves": 7, "max_depth": 3, "min_data_in_leaf": 10}
        },
        "train": {
            "num_boost_round": 30,
            "validation_partition": "test",  # Explicit legacy validation naming.
            "early_stopping": {"stopping_rounds": 5},
            "log_evaluation": {"enabled": False},
        },
    }


@pytest.fixture
def exp(tmp_path, inputs):
    pytest.importorskip("lightgbm")
    return LightGBMExperiment(name="demo", root=tmp_path, **inputs)
