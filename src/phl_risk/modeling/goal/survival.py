"""Survival learning semantics."""

from dataclasses import dataclass
from typing import ClassVar

from ._base import ModelingGoal


@dataclass(frozen=True)
class Survival(ModelingGoal):
    """Declare survival semantics without binding real data columns."""

    name: ClassVar[str] = "Survival"
    family: ClassVar[str] = "survival"
    required_roles: ClassVar[frozenset[str]] = frozenset(("duration", "event"))
    objective_families: ClassVar[tuple[str, ...]] = ("cox",)
