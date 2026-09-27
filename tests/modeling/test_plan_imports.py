"""Public entry points must share classes and keep declarations backend-free."""

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "entrypoint,canonical,names",
    [
        ("phl_risk.modeling", "phl_risk.modeling.plan.model_plan", ["ModelPlan"]),
        ("phl_risk.modeling", "phl_risk.modeling.plan.data_plan", ["DataPlan"]),
        (
            "phl_risk.modeling.plan.model_plan",
            "phl_risk.modeling.plan.model_plan.goal",
            ["ModelingGoal", "BinaryClassification", "Regression", "CausalEffect", "Survival"],
        ),
        (
            "phl_risk.modeling.plan.model_plan",
            "phl_risk.modeling.plan.model_plan.strategy",
            ["ModelStrategy", "LightGBM", "MLP"],
        ),
        (
            "phl_risk.modeling.plan",
            "phl_risk.modeling.plan.model_plan",
            ["ModelPlan", "ObjectiveOptions", "RoleRequirements"],
        ),
        (
            "phl_risk.modeling.plan",
            "phl_risk.modeling.plan.data_plan",
            [
                "DataPlan",
                "RoleSpec",
                "FeatureSpec",
                "SplitSpec",
                "PartitionSpec",
                "BaseSplitter",
                "HashSplitter",
                "RandomSplitter",
                "ColumnSplitter",
                "TimeSplitter",
            ],
        ),
    ],
)
def test_public_entry_points_share_class_identity(entrypoint, canonical, names):
    public, implementation = importlib.import_module(entrypoint), importlib.import_module(canonical)
    for name in names:
        assert getattr(public, name) is getattr(implementation, name)
        assert getattr(implementation, name).__module__.startswith("phl_risk.modeling.plan.")


@pytest.mark.parametrize(
    "first",
    [
        "phl_risk.modeling.plan.model_plan.goal.classification",
        "phl_risk.modeling.plan.model_plan.strategy.tree",
        "phl_risk.modeling.plan.data_plan.split.splitter",
        "phl_risk.modeling.plan.model_plan.goal",
        "phl_risk.modeling.plan.model_plan.strategy",
    ],
)
def test_import_orders_without_site_packages(first):
    code = f"""
import importlib
import sys
sys.path.insert(0, "src")
importlib.import_module({first!r})
from phl_risk.modeling.plan.model_plan import ModelPlan, BinaryClassification, LightGBM
from phl_risk.modeling.plan.data_plan import (
    DataPlan, RoleSpec, FeatureSpec, SplitSpec, PartitionSpec, HashSplitter,
)
model = ModelPlan(BinaryClassification(), LightGBM())
data = DataPlan(RoleSpec(target="y"), FeatureSpec(["x"]),
                SplitSpec(HashSplitter("id"), PartitionSpec(train=1)))
data.validate_against(model)
for name in sys.modules:
    assert not name.startswith(("pandas", "numpy", "sklearn", "lightgbm", "torch",
                                "phl_risk.modeling.experiment")), name
"""
    subprocess.run([sys.executable, "-S", "-c", code], check=True)


@pytest.mark.parametrize("name", ["goal", "strategy"])
def test_top_level_declaration_aliases_removed(name):
    from pathlib import Path

    assert not Path("src/phl_risk/modeling", name).exists()
    assert importlib.util.find_spec(f"phl_risk.modeling.{name}") is None
