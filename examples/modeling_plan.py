"""Declare model and data contracts without data, backend imports or training."""

import json

from phl_risk.modeling.goal import BinaryClassification
from phl_risk.modeling.plan import (
    ColumnSplitter,
    DataPlan,
    FeatureSpec,
    HashSplitter,
    ModelPlan,
    PartitionSpec,
    RoleSpec,
    SplitSpec,
)
from phl_risk.modeling.strategy import LightGBM


def main() -> None:
    model_plan = ModelPlan(BinaryClassification(), LightGBM())
    data_plan = DataPlan(
        roles=RoleSpec(
            target="1m_30",
            sample_key=["user_id", "dt"],
            time="dt",
            weight="sample_weight",
        ),
        features=FeatureSpec(
            numerical=["age", "income", "beh_score"],
            categorical=["education", "channel"],
        ),
        split=SplitSpec(
            splitter=HashSplitter(key=["user_id"], random_state=2026),
            partitions=PartitionSpec(train=0.7, test=0.3),
        ),
    )
    model_plan.validate()
    data_plan.validate()
    data_plan.validate_against(model_plan)
    assert model_plan.execution_family == "tree"
    assert model_plan.objective_options.selected == "binary_logloss"
    assert data_plan.roles.bindings["sample_key"] == ("user_id", "dt")
    assert data_plan.split.splitter.key == ("user_id",)
    json.dumps({"model": model_plan.to_dict(), "data": data_plan.to_dict()}, allow_nan=False)
    existing_partition = SplitSpec(
        splitter=ColumnSplitter(column="dtype"),
        partitions=PartitionSpec(train="train", test="test", oot="oot"),
    )
    existing_partition.validate()
    print(model_plan.describe())
    print(data_plan.describe())


if __name__ == "__main__":
    main()
