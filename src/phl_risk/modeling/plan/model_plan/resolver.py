"""Combine learning semantics and route capabilities into a model contract."""

from phl_risk.exceptions import CompatibilityError, PlanError
from phl_risk.modeling._utils import name

from .goal import ModelingGoal
from .objective import ObjectiveOptions
from .requirements import RoleRequirements
from .strategy import ModelStrategy


def resolve_model_contract(
    goal: ModelingGoal, strategy: ModelStrategy, objective: str | None
) -> tuple[ObjectiveOptions, RoleRequirements]:
    """Resolve objectives and semantic roles without constructing a model or reading data."""
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
