"""Learning semantics, independent of real column names."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from .._utils import JSONValue


@dataclass(frozen=True)
class ModelingGoal(ABC):
    """Declare learning semantics; subclass with immutable role/objective capabilities.

    Objective families are ordered semantic identifiers, not backend parameter names.
    Their order determines the preferred objective after capability intersection.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable learning goal name."""

    @property
    @abstractmethod
    def family(self) -> str:
        """Learning semantics used to match strategy capabilities."""

    @property
    @abstractmethod
    def required_roles(self) -> frozenset[str]:
        """Semantic roles every data declaration must bind for this goal."""

    optional_roles: ClassVar[frozenset[str]] = frozenset({"weight", "sample_key", "time"})
    objective_families: ClassVar[tuple[str, ...]] = ()
    objective_role_requirements: ClassVar[tuple[tuple[str, frozenset[str]], ...]] = ()

    def to_dict(self) -> dict[str, JSONValue]:
        """Return column-free, JSON-compatible learning semantics."""
        return {
            "name": self.name,
            "family": self.family,
            "required_roles": sorted(self.required_roles),
            "optional_roles": sorted(self.optional_roles),
            "objective_families": list(self.objective_families),
            "objective_role_requirements": {
                objective: sorted(roles) for objective, roles in self.objective_role_requirements
            },
        }
