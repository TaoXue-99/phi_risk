"""Technical-route capabilities, without model instances or training parameters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from .._utils import JSONValue


@dataclass(frozen=True)
class ModelStrategy(ABC):
    """Declare a route's immutable capabilities for ModelPlan resolution.

    Extensions override class declarations; no backend imports or fitted state belong here.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable technical route name."""

    @property
    @abstractmethod
    def family(self) -> str:
        """Technical family of the chosen route."""

    @property
    @abstractmethod
    def execution_family(self) -> str:
        """Family to be consumed by a future execution resolver."""

    @property
    @abstractmethod
    def data_family(self) -> str:
        """Required structural data family, such as tabular."""

    supported_goal_families: ClassVar[frozenset[str]] = frozenset()
    supported_objective_families: ClassVar[frozenset[str]] = frozenset()
    supports_weight: ClassVar[bool] = False
    supports_categorical: ClassVar[bool] = False
    supports_custom_objective: ClassVar[bool] = False

    def to_dict(self) -> dict[str, JSONValue]:
        """Return capabilities without importing the eventual execution backend."""
        return {
            "name": self.name,
            "family": self.family,
            "execution_family": self.execution_family,
            "data_family": self.data_family,
            "supported_goal_families": sorted(self.supported_goal_families),
            "supported_objective_families": sorted(self.supported_objective_families),
            "supports_weight": self.supports_weight,
            "supports_categorical": self.supports_categorical,
            "supports_custom_objective": self.supports_custom_objective,
        }
