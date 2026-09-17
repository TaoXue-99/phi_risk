"""Resolved semantic-role contract, independent of column bindings."""

from dataclasses import dataclass

from phl_risk.exceptions import PlanError

from .._utils import JSONValue, name


@dataclass(frozen=True)
class RoleRequirements:
    """Required and optional semantic roles accepted by a resolved model plan."""

    required: frozenset[str]
    optional: frozenset[str]

    def __post_init__(self) -> None:
        for key in ("required", "optional"):
            values = getattr(self, key)
            if isinstance(values, str):
                raise PlanError(f"{key} roles must be a collection, not a string")
            object.__setattr__(self, key, frozenset(name(v, f"{key} role") for v in values))
        if self.required & self.optional:
            raise PlanError("Required and optional roles must not overlap")

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize roles in stable order."""
        return {"required": sorted(self.required), "optional": sorted(self.optional)}
