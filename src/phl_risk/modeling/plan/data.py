"""Tabular data organization constrained by a separately resolved ModelPlan."""

from dataclasses import dataclass
from typing import ClassVar

from phl_risk.exceptions import CompatibilityError, PlanError

from .._utils import JSONValue, describe
from .feature import FeatureSpec
from .model import ModelPlan
from .role import RoleSpec
from .split import SplitSpec


@dataclass(frozen=True)
class DataPlan:
    """Declare semantic bindings, tabular features and future partition assignment."""

    roles: RoleSpec
    features: FeatureSpec
    split: SplitSpec
    data_family: ClassVar[str] = "tabular"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check declaration structure only; no columns or actual data are inspected."""
        for key, expected in (("roles", RoleSpec), ("features", FeatureSpec), ("split", SplitSpec)):
            if not isinstance(getattr(self, key), expected):
                raise PlanError(f"DataPlan {key} must be a {expected.__name__}")
        self.split.validate()

    def validate_against(self, model_plan: ModelPlan) -> None:
        """Check required/accepted roles, data family and feature capabilities.

        All semantic mismatches are reported together. Split keys and sample keys
        remain independent; physical column existence is a future runtime concern.
        """
        self.validate()
        if not isinstance(model_plan, ModelPlan):
            raise PlanError("validate_against requires a ModelPlan")
        model_plan.validate()
        requirements = model_plan.role_requirements
        provided = set(self.roles.bindings)
        errors = []
        missing = requirements.required - provided
        unexpected = provided - (requirements.required | requirements.optional)
        if missing:
            errors.append(f"Missing required roles: {', '.join(sorted(missing))}")
        if unexpected:
            errors.append(f"Unexpected roles: {', '.join(sorted(unexpected))}")
        if model_plan.data_family != self.data_family:
            errors.append(
                f"Data family {self.data_family!r} does not satisfy {model_plan.data_family!r}"
            )
        if self.features.categorical and not model_plan.strategy.supports_categorical:
            errors.append(f"{model_plan.strategy.name} does not support categorical features")
        if errors:
            raise CompatibilityError(
                f"DataPlan incompatible with {model_plan.goal.name} + "
                f"{model_plan.strategy.name}: " + "; ".join(errors)
            )

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a detached JSON-compatible data organization contract."""
        return {
            "data_family": self.data_family,
            "roles": self.roles.to_dict(),
            "features": self.features.to_dict(),
            "split": self.split.to_dict(),
        }

    def describe(self) -> str:
        """Return a readable data declaration without printing."""
        return describe("DataPlan", self.to_dict())
