"""Regression learning semantics."""

from dataclasses import dataclass
from typing import ClassVar

from ._base import ModelingGoal


@dataclass(frozen=True)
class Regression(ModelingGoal):
    """Declare regression semantics without binding real data columns."""

    name: ClassVar[str] = "Regression"
    family: ClassVar[str] = "regression"
    required_roles: ClassVar[frozenset[str]] = frozenset(("target",))
    objective_families: ClassVar[tuple[str, ...]] = ("mse", "mae")
