"""MLP route declaration; no backend dependency."""

from dataclasses import dataclass
from typing import ClassVar

from ._base import ModelStrategy


@dataclass(frozen=True)
class MLP(ModelStrategy):
    """Choose the MLP route without constructing or configuring a model."""

    name: ClassVar[str] = "MLP"
    family: ClassVar[str] = "deep_learning"
    execution_family: ClassVar[str] = "deep_learning"
    data_family: ClassVar[str] = "tabular"
    supported_goal_families: ClassVar[frozenset[str]] = frozenset(
        {"binary_classification", "regression"}
    )
    supported_objective_families: ClassVar[frozenset[str]] = frozenset(
        {"binary_logloss", "weighted_binary_logloss", "mse", "mae"}
    )
    supports_weight: ClassVar[bool] = True
    supports_categorical: ClassVar[bool] = False
