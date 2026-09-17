"""Resolve a goal and strategy into a model contract, never execution."""

from dataclasses import dataclass, field

from phl_risk.exceptions import CompatibilityError, PlanError

from .._utils import JSONValue, describe, name
from ..goal import ModelingGoal
from ..strategy import ModelStrategy
from ._objective import ObjectiveOptions
from ._requirements import RoleRequirements


def _resolve(
    goal: ModelingGoal, strategy: ModelStrategy, objective: str | None
) -> tuple[ObjectiveOptions, RoleRequirements]:
    if not isinstance(goal, ModelingGoal) or not isinstance(strategy, ModelStrategy):
        raise PlanError("ModelPlan requires a ModelingGoal and a ModelStrategy")
    context = f"{goal.name} + {strategy.name}"
    for label in ("execution_family", "data_family", "family", "name"):
        name(getattr(strategy, label), f"{context} strategy {label}")
    if goal.family not in strategy.supported_goal_families:
        raise CompatibilityError(
            f"Incompatible Goal + Strategy: {context}; "
            f"supported goal families: {sorted(strategy.supported_goal_families)}"
        )
    role_contract = RoleRequirements(goal.required_roles, goal.optional_roles)
    objective_roles = dict(goal.objective_role_requirements)
    if len(objective_roles) != len(goal.objective_role_requirements):
        raise PlanError(f"Duplicate objective role requirements for {context}")
    for objective_name, roles in objective_roles.items():
        if objective_name not in goal.objective_families:
            raise PlanError(
                f"Role requirements for unknown objective {objective_name!r}: {context}"
            )
        objective_roles[objective_name] = RoleRequirements(roles, frozenset()).required
    available = tuple(
        o
        for o in goal.objective_families
        if o in strategy.supported_objective_families
        and ("weight" not in objective_roles.get(o, ()) or strategy.supports_weight)
    )
    if not available:
        raise CompatibilityError(f"No compatible objectives for {context}")
    selected = available[0] if objective is None else objective
    if selected not in available:
        raise CompatibilityError(
            f"Objective {selected!r} is incompatible with {context}. "
            f"Available objectives: {', '.join(available)}"
        )
    required = role_contract.required | objective_roles.get(selected, frozenset())
    optional = role_contract.optional - required
    if not strategy.supports_weight:
        if "weight" in required:
            raise CompatibilityError(f"{context} requires weight but strategy does not support it")
        optional -= {"weight"}
    return ObjectiveOptions(available, available[0], selected), RoleRequirements(required, optional)


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
        options, roles = _resolve(self.goal, self.strategy, self.objective)
        object.__setattr__(self, "objective", options.selected)
        object.__setattr__(self, "objective_options", options)
        object.__setattr__(self, "role_requirements", roles)
        object.__setattr__(self, "execution_family", self.strategy.execution_family)
        object.__setattr__(self, "data_family", self.strategy.data_family)

    def validate(self) -> None:
        """Check the resolved declaration without loading data or a backend."""
        options, roles = _resolve(self.goal, self.strategy, self.objective)
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
