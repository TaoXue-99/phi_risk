"""BinaryClassification learning semantics."""

from dataclasses import dataclass
from typing import ClassVar

from ._base import ModelingGoal


@dataclass(frozen=True)
class BinaryClassification(ModelingGoal):
    """Declare binary_classification semantics without binding real data columns."""

    name: ClassVar[str] = "BinaryClassification"
    family: ClassVar[str] = "binary_classification"
    required_roles: ClassVar[frozenset[str]] = frozenset(("target",))
    objective_families: ClassVar[tuple[str, ...]] = ("binary_logloss", "weighted_binary_logloss")
    objective_role_requirements: ClassVar[tuple[tuple[str, frozenset[str]], ...]] = (
        ("weighted_binary_logloss", frozenset({"weight"})),
    )
