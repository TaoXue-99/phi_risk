"""Immutable model plan assembled from resolved learning and route declarations."""

from dataclasses import dataclass, field

from phl_risk.exceptions import PlanError
from phl_risk.modeling._utils import JSONValue, describe

from .goal import ModelingGoal
from .objective import ObjectiveOptions
from .requirements import RoleRequirements
from .resolver import resolve_model_contract
from .strategy import ModelStrategy


@dataclass(frozen=True)
class ModelPlan:
    """Resolve learning semantics and route capabilities into an immutable contract."""

    goal: ModelingGoal
    strategy: ModelStrategy
    objective: str | None = None
    objective_options: ObjectiveOptions = field(init=False)
    role_requirements: RoleRequirements = field(init=False)
    execution_family: str = field(init=False)
    data_family: str = field(init=False)

    def __post_init__(self) -> None:
        options, roles = resolve_model_contract(self.goal, self.strategy, self.objective)
        object.__setattr__(self, "objective", options.selected)
        object.__setattr__(self, "objective_options", options)
        object.__setattr__(self, "role_requirements", roles)
        object.__setattr__(self, "execution_family", self.strategy.execution_family)
        object.__setattr__(self, "data_family", self.strategy.data_family)

    def validate(self) -> None:
        """Check the resolved declaration without loading data or a backend."""
        options, roles = resolve_model_contract(self.goal, self.strategy, self.objective)
        if (
            options != self.objective_options
            or roles != self.role_requirements
            or self.execution_family != self.strategy.execution_family
            or self.data_family != self.strategy.data_family
        ):
            raise PlanError("Goal or Strategy capabilities changed after ModelPlan resolution")

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a detached JSON-compatible model contract."""
        return {
            "goal": self.goal.to_dict(),
            "strategy": self.strategy.to_dict(),
            "execution_family": self.execution_family,
            "data_family": self.data_family,
            "objective_options": self.objective_options.to_dict(),
            "role_requirements": self.role_requirements.to_dict(),
        }

    def describe(self) -> str:
        """Return a readable model contract; never print it."""
        return describe("ModelPlan", self.to_dict())
