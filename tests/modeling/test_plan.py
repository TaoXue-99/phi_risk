"""Contract tests for backend-free, immutable modeling declarations."""

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, dataclass
from typing import ClassVar

import pytest

from phl_risk.exceptions import CompatibilityError, ModelingError, PhlRiskError, PlanError
from phl_risk.modeling import DataPlan, ModelPlan
from phl_risk.modeling.goal import (
    BinaryClassification,
    CausalEffect,
    ModelingGoal,
    Regression,
    Survival,
)
from phl_risk.modeling.plan import (
    BaseSplitter,
    ColumnSplitter,
    FeatureSpec,
    HashSplitter,
    ObjectiveOptions,
    PartitionSpec,
    RandomSplitter,
    RoleRequirements,
    RoleSpec,
    SplitSpec,
    TimeSplitter,
)
from phl_risk.modeling.strategy import MLP, LightGBM, ModelStrategy


def data(roles=None, features=None, splitter=None, partitions=None):
    return DataPlan(
        roles=roles if roles is not None else RoleSpec(target="bad"),
        features=features if features is not None else FeatureSpec(numerical=["age"]),
        split=SplitSpec(
            splitter=splitter if splitter is not None else HashSplitter(["user_id"]),
            partitions=partitions if partitions is not None else PartitionSpec(train=0.7, test=0.3),
        ),
    )


@dataclass(frozen=True)
class SemanticTestStrategy(ModelStrategy):
    """Test-only route proves that resolver supports new semantics without class dispatch."""

    name: ClassVar[str] = "SemanticTestStrategy"
    family: ClassVar[str] = "test"
    execution_family: ClassVar[str] = "test"
    data_family: ClassVar[str] = "tabular"
    supported_goal_families: ClassVar[frozenset[str]] = frozenset({"causal", "survival"})
    supported_objective_families: ClassVar[frozenset[str]] = frozenset({"causal_effect", "cox"})


@pytest.mark.parametrize("strategy,family", [(LightGBM(), "tree"), (MLP(), "deep_learning")])
def test_model_resolution(strategy, family):
    plan = ModelPlan(BinaryClassification(), strategy)
    plan.validate()
    assert plan.execution_family == family
    assert plan.data_family == "tabular"
    assert plan.objective_options == ObjectiveOptions(
        ("binary_logloss", "weighted_binary_logloss"), "binary_logloss", "binary_logloss"
    )
    assert plan.objective == plan.objective_options.selected
    assert plan.role_requirements.required == frozenset({"target"})
    assert plan.role_requirements.optional == frozenset({"weight", "sample_key", "time"})


def test_weighted_objective_changes_roles():
    plan = ModelPlan(BinaryClassification(), LightGBM(), objective="weighted_binary_logloss")
    assert plan.objective_options.default == "binary_logloss"
    assert plan.objective_options.selected == "weighted_binary_logloss"
    assert plan.role_requirements.required == frozenset({"target", "weight"})
    with pytest.raises(CompatibilityError, match="Missing required roles: weight"):
        data().validate_against(plan)
    data(RoleSpec(target="bad", weight="w")).validate_against(plan)


@pytest.mark.parametrize("objective", ["mse", "", "binary", 42, [], False])
def test_invalid_objective_context(objective):
    with pytest.raises(CompatibilityError, match="BinaryClassification.*LightGBM.*Available"):
        ModelPlan(BinaryClassification(), LightGBM(), objective)


@pytest.mark.parametrize("goal", [CausalEffect(), Survival()])
def test_unsupported_goal(goal):
    with pytest.raises(CompatibilityError, match="Incompatible Goal.*LightGBM"):
        ModelPlan(goal, LightGBM())


@pytest.mark.parametrize("objective", [None, "mse", "mae"])
def test_regression(objective):
    plan = ModelPlan(Regression(), LightGBM(), objective)
    assert plan.objective_options.selected == (objective or "mse")
    data().validate_against(plan)


@pytest.mark.parametrize(
    "goal,bindings",
    [
        (CausalEffect(), {"outcome": "bad", "treatment": "policy"}),
        (Survival(), {"duration": "days", "event": "default"}),
    ],
)
def test_extension_semantics_required_roles(goal, bindings):
    plan = ModelPlan(goal, SemanticTestStrategy())
    data(RoleSpec(**bindings)).validate_against(plan)
    assert "weight" not in plan.role_requirements.optional
    for role in bindings:
        incomplete = {k: v for k, v in bindings.items() if k != role}
        with pytest.raises(CompatibilityError, match=f"Missing required roles: {role}"):
            data(RoleSpec(**incomplete)).validate_against(plan)
    with pytest.raises(CompatibilityError, match="Unexpected roles: weight"):
        data(RoleSpec(**bindings, weight="w")).validate_against(plan)


def test_role_errors_aggregated():
    with pytest.raises(
        CompatibilityError, match="Missing required roles: target; Unexpected roles: treatment"
    ):
        data(RoleSpec(treatment="strategy")).validate_against(
            ModelPlan(BinaryClassification(), LightGBM())
        )


@pytest.mark.parametrize(
    "binding,expected",
    [
        ("user_id", ("user_id",)),
        (["user_id"], ("user_id",)),
        (["user_id", "dt"], ("user_id", "dt")),
    ],
)
def test_sample_keys(binding, expected):
    assert RoleSpec(sample_key=binding).bindings["sample_key"] == expected


def test_role_overlap_and_split_key_independence():
    plan = data(RoleSpec(target="bad", sample_key=["user_id", "dt"], time="dt"))
    plan.validate_against(ModelPlan(BinaryClassification(), LightGBM()))
    assert plan.split.splitter.key != plan.roles.bindings["sample_key"]
    assert RoleSpec(outcome="bad", treatment="policy").to_dict() == {
        "outcome": ["bad"],
        "treatment": ["policy"],
    }
    assert RoleSpec(duration="days", event="flag").bindings["duration"] == ("days",)
    # Split keys need not even be sample_key components.
    data(RoleSpec(target="bad", sample_key="row_id")).validate()


@pytest.mark.parametrize(
    "roles",
    [
        {"target": ""},
        {"target": "  "},
        {"target": None},
        {"sample_key": []},
        {"sample_key": ["id", "id"]},
        {"time": ["t1", "t2"]},
        {"target": ["a", "b"]},
        {"sample_key": {"id"}},
        {"": "column"},
        {"target": [123]},
    ],
)
def test_bad_roles(roles):
    with pytest.raises(PlanError):
        RoleSpec(**roles)


def test_dynamic_role_binding():
    assert RoleSpec(custom_role="custom_column").bindings["custom_role"] == ("custom_column",)
    with pytest.raises(CompatibilityError, match="custom_role"):
        data(RoleSpec(target="bad", custom_role="c")).validate_against(
            ModelPlan(BinaryClassification(), LightGBM())
        )


def test_features():
    assert FeatureSpec().all == ()
    features = FeatureSpec(["age", "income"], ["channel"])
    assert features.all == ("age", "income", "channel")


@pytest.mark.parametrize(
    "numerical,categorical",
    [
        (["a", "a"], []),
        ([], ["a", "a"]),
        (["a"], ["a"]),
        ([""], []),
        ([], [None]),
        ({"a"}, []),
        ([1], []),
        (None, []),
    ],
)
def test_bad_features(numerical, categorical):
    with pytest.raises(PlanError):
        FeatureSpec(numerical, categorical)


@pytest.mark.parametrize(
    "partitions",
    [
        {"train": 0.7, "test": 0.3},
        {"train": 0.6, "valid": 0.2, "test": 0.2},
        {"train": 0.5, "calibration": 0.2, "holdout": 0.3},
        {"train": 1},
        {"train": 0.7, "test": 0.3000000001},
    ],
)
@pytest.mark.parametrize("splitter", [HashSplitter("id"), RandomSplitter(7), TimeSplitter("dt")])
def test_ratio_modes(partitions, splitter):
    split = SplitSpec(splitter, PartitionSpec(**partitions))
    split.validate()
    assert split.to_dict()["partition_mode"] == "ratios"


@pytest.mark.parametrize(
    "partitions",
    [
        {"train": 0.6, "test": 0.3},
        {"train": -0.1, "test": 1.1},
        {"test": 1},
        {"train": 0, "test": 1},
        {"train": float("nan")},
        {"train": float("inf")},
        {"train": True},
        {"train": "train", "test": "test"},
        {"train": None},
        {"train": 0.5, "": 0.5},
        {"train": [0.7]},
    ],
)
def test_bad_partitions(partitions):
    with pytest.raises(PlanError):
        SplitSpec(HashSplitter("id"), PartitionSpec(**partitions))


@pytest.mark.parametrize(
    "partitions",
    [
        {"train": "train", "test": "test", "oot": "oot"},
        {"train": 10, "test": 20},
        {"train": False, "test": True},
        {"train": 0.1, "test": 0.2},
    ],
)
def test_column_mode(partitions):
    split = SplitSpec(ColumnSplitter("dtype"), PartitionSpec(**partitions))
    assert split.to_dict()["partition_mode"] == "values"
    assert split.partitions.to_dict() == partitions


@pytest.mark.parametrize("values", [("same", "same"), (1, True), (1, 1.0)])
def test_column_values_ambiguous(values):
    with pytest.raises(PlanError, match="distinct"):
        SplitSpec(ColumnSplitter("dtype"), PartitionSpec(train=values[0], test=values[1]))


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (HashSplitter, {"key": []}),
        (HashSplitter, {"key": ["id", "id"]}),
        (HashSplitter, {"key": "id", "random_state": True}),
        (RandomSplitter, {"random_state": -1}),
        (RandomSplitter, {"random_state": 1.5}),
        (RandomSplitter, {"stratify": ""}),
        (ColumnSplitter, {"column": ""}),
        (TimeSplitter, {"time": ""}),
    ],
)
def test_invalid_splitters(factory, kwargs):
    with pytest.raises(PlanError):
        factory(**kwargs)


def test_random_stratify_and_time_order():
    assert RandomSplitter(1, "label").to_dict()["stratify"] == "label"
    split = SplitSpec(TimeSplitter("dt"), PartitionSpec(train=0.6, oot=0.1, test=0.3))
    assert list(split.partitions.definitions) == ["train", "oot", "test"]


def test_immutability_and_defensive_copy():
    names = ["id", "dt"]
    features = ["age"]
    role = RoleSpec(sample_key=names)
    feature = FeatureSpec(features)
    splitter = HashSplitter(names)
    names.append("bad")
    features.append("bad")
    assert role.bindings["sample_key"] == splitter.key == ("id", "dt")
    assert feature.numerical == ("age",)
    for mapping in (role.bindings, PartitionSpec(train=1).definitions):
        with pytest.raises(TypeError):
            mapping["new"] = "bad"
    for instance, attr in [
        (role, "bindings"),
        (feature, "numerical"),
        (splitter, "key"),
        (data(), "roles"),
        (ModelPlan(BinaryClassification(), LightGBM()), "objective"),
        (BinaryClassification(), "name"),
        (LightGBM(), "execution_family"),
    ]:
        with pytest.raises(FrozenInstanceError):
            setattr(instance, attr, "changed")


def test_serialization_and_description(capsys):
    plans = [ModelPlan(BinaryClassification(), LightGBM()), data()]
    for plan in plans:
        original = plan.to_dict()
        assert json.loads(json.dumps(original, allow_nan=False)) == original
        assert type(plan).__name__ in plan.describe()
        assert plan.to_dict() == original
        if isinstance(plan, DataPlan):
            original["roles"]["target"].append("mutated")
            assert plan.roles.bindings["target"] == ("bad",)
    assert capsys.readouterr().out == ""


def test_capability_filtering_and_no_objectives():
    @dataclass(frozen=True)
    class Unweighted(LightGBM):
        supports_weight: ClassVar[bool] = False

    plan = ModelPlan(BinaryClassification(), Unweighted())
    assert plan.objective_options.available == ("binary_logloss",)
    assert "weight" not in plan.role_requirements.optional

    @dataclass(frozen=True)
    class NoObjectives(LightGBM):
        supported_objective_families: ClassVar[frozenset[str]] = frozenset()

    with pytest.raises(CompatibilityError, match="No compatible objectives"):
        ModelPlan(BinaryClassification(), NoObjectives())


def test_data_capabilities():
    with pytest.raises(CompatibilityError, match="categorical"):
        data(features=FeatureSpec(categorical=["channel"])).validate_against(
            ModelPlan(BinaryClassification(), MLP())
        )

    @dataclass(frozen=True)
    class OtherFamily(LightGBM):
        data_family: ClassVar[str] = "sequence"

    with pytest.raises(CompatibilityError, match="Data family"):
        data().validate_against(ModelPlan(BinaryClassification(), OtherFamily()))


@pytest.mark.parametrize(
    "factory,args",
    [
        (ModelPlan, (None, LightGBM())),
        (ModelPlan, (BinaryClassification(), None)),
        (DataPlan, ({}, FeatureSpec(), None)),
        (DataPlan, (RoleSpec(), [], None)),
        (DataPlan, (RoleSpec(), FeatureSpec(), None)),
        (SplitSpec, (None, PartitionSpec(train=1))),
        (SplitSpec, (RandomSplitter(), {})),
        (ObjectiveOptions, (("a",), "b", "a")),
        (RoleRequirements, ({"target"}, {"target"})),
        (RoleRequirements, ("target", ())),
    ],
)
def test_invalid_contract_objects(factory, args):
    with pytest.raises(PlanError):
        factory(*args)


def test_exception_hierarchy_and_no_training_arguments():
    assert issubclass(CompatibilityError, (PlanError, ModelingError, PhlRiskError))
    assert issubclass(PlanError, ValueError)
    with pytest.raises(TypeError):
        LightGBM(max_depth=3)
    with pytest.raises(TypeError):
        BinaryClassification(target="label")
    for base in (ModelingGoal, ModelStrategy, BaseSplitter):
        with pytest.raises(TypeError):
            base()


def test_standard_library_only_and_no_execution_api():
    script = """
import sys
sys.path.insert(0, "src")
from phl_risk.modeling import ModelPlan, DataPlan
from phl_risk.modeling.goal import BinaryClassification
from phl_risk.modeling.strategy import LightGBM, MLP
from phl_risk.modeling.plan import *
ModelPlan(BinaryClassification(), LightGBM()).validate()
for cls in (ModelPlan, DataPlan, LightGBM, MLP, HashSplitter, RandomSplitter,
            ColumnSplitter, TimeSplitter, RoleSpec, FeatureSpec, SplitSpec):
    for method in ("fit", "predict", "transform", "split", "train", "tune", "apply"):
        # DataPlan's split is a declaration field, never a callable.
        assert not callable(getattr(cls, method, None)), (cls, method)
for module in ("pandas", "numpy", "sklearn", "lightgbm", "torch", "optuna", "pydantic"):
    assert module not in sys.modules, module
"""
    subprocess.run([sys.executable, "-S", "-c", script], check=True)


def test_determinism_across_hash_seeds():
    script = """
import json
from phl_risk.modeling import ModelPlan
from phl_risk.modeling.goal import BinaryClassification
from phl_risk.modeling.strategy import LightGBM
print(json.dumps(ModelPlan(BinaryClassification(), LightGBM()).to_dict()))
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": seed}
        )
        for seed in ("1", "123")
    ]
    assert outputs[0] == outputs[1]


def test_custom_objective_role_requirements():
    @dataclass(frozen=True)
    class CustomGoal(BinaryClassification):
        objective_families: ClassVar[tuple[str, ...]] = ("group_loss",)
        objective_role_requirements: ClassVar[tuple[tuple[str, frozenset[str]], ...]] = (
            ("group_loss", frozenset({"group"})),
        )

    @dataclass(frozen=True)
    class CustomRoute(LightGBM):
        supported_objective_families: ClassVar[frozenset[str]] = frozenset({"group_loss"})

    plan = ModelPlan(CustomGoal(), CustomRoute())
    assert plan.role_requirements.required == frozenset({"target", "group"})
    data(RoleSpec(target="bad", group="group_id")).validate_against(plan)
    with pytest.raises(CompatibilityError, match="Missing required roles: group"):
        data().validate_against(plan)


@pytest.mark.parametrize(
    "declaration",
    [
        (("missing", frozenset({"weight"})),),
        (("binary_logloss", frozenset()), ("binary_logloss", frozenset())),
    ],
)
def test_invalid_objective_role_declaration(declaration):
    @dataclass(frozen=True)
    class InvalidGoal(BinaryClassification):
        objective_role_requirements: ClassVar[tuple[tuple[str, frozenset[str]], ...]] = declaration

    with pytest.raises(PlanError):
        ModelPlan(InvalidGoal(), LightGBM())


def test_required_weight_without_capability():
    @dataclass(frozen=True)
    class WeightedGoal(BinaryClassification):
        required_roles: ClassVar[frozenset[str]] = frozenset({"target", "weight"})
        optional_roles: ClassVar[frozenset[str]] = frozenset()

    @dataclass(frozen=True)
    class NoWeight(LightGBM):
        supports_weight: ClassVar[bool] = False

    with pytest.raises(CompatibilityError, match="requires weight"):
        ModelPlan(WeightedGoal(), NoWeight())


def test_capability_drift_is_detected(monkeypatch):
    plan = ModelPlan(BinaryClassification(), LightGBM())
    monkeypatch.setattr(LightGBM, "execution_family", "changed")
    with pytest.raises(PlanError, match="capabilities changed"):
        plan.validate()


def test_duplicate_partition_keyword_rejected():
    with pytest.raises(TypeError):
        PartitionSpec(train=0.5, **{"train": 0.5})


def test_model_example_and_documentation_code():
    import runpy
    from pathlib import Path

    namespace = runpy.run_path("examples/modeling_plan.py")
    namespace["main"]()
    document = Path("docs/modeling_plan.md").read_text()
    context = {}
    for block in document.split("```python\n")[1:]:
        exec(block.split("```", 1)[0], context)


def test_partition_equality_preserves_semantic_order():
    first = PartitionSpec(train=0.6, test=0.2, oot=0.2)
    reordered = PartitionSpec(train=0.6, oot=0.2, test=0.2)
    assert first == PartitionSpec(train=0.6, test=0.2, oot=0.2)
    assert first != reordered
    assert SplitSpec(TimeSplitter("dt"), first) != SplitSpec(TimeSplitter("dt"), reordered)
