"""LightGBM route declaration; no backend dependency."""

from dataclasses import dataclass
from typing import ClassVar

from ._base import ModelStrategy


@dataclass(frozen=True)
class LightGBM(ModelStrategy):
    """Choose the LightGBM route without constructing or configuring a model."""

    name: ClassVar[str] = "LightGBM"
    family: ClassVar[str] = "tree"
    execution_family: ClassVar[str] = "tree"
    data_family: ClassVar[str] = "tabular"
    supported_goal_families: ClassVar[frozenset[str]] = frozenset(
        {"binary_classification", "regression"}
    )
    supported_objective_families: ClassVar[frozenset[str]] = frozenset(
        {"binary_logloss", "weighted_binary_logloss", "mse", "mae"}
    )
    supports_weight: ClassVar[bool] = True
    supports_categorical: ClassVar[bool] = True
