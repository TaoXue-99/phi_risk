"""Dynamic semantic-role to column bindings."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from phl_risk.exceptions import PlanError

from .._utils import Columns, JSONValue, columns, name


@dataclass(frozen=True, init=False)
class RoleSpec:
    """Bind arbitrary semantic roles; only sample_key accepts multiple columns.

    Columns may occur in different roles. Duplicates within a binding are errors.
    Goal-specific role acceptance is checked by DataPlan.validate_against.
    """

    bindings: Mapping[str, tuple[str, ...]]

    def __init__(self, **roles: Columns) -> None:
        normalized = {}
        for role, value in sorted(roles.items()):
            name(role, "role name")
            binding = columns(value, f"role {role!r}")
            if role != "sample_key" and len(binding) != 1:
                raise PlanError(f"Role {role!r} requires exactly one column")
            normalized[role] = binding
        object.__setattr__(self, "bindings", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, JSONValue]:
        """Return detached role bindings as JSON lists."""
        return {role: list(binding) for role, binding in self.bindings.items()}
