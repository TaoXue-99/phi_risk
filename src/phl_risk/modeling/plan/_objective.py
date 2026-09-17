"""Resolved objective selection (semantic names, not backend aliases)."""

from dataclasses import dataclass

from phl_risk.exceptions import PlanError

from .._utils import JSONValue, columns


@dataclass(frozen=True)
class ObjectiveOptions:
    """Available, preferred and selected objectives for one goal/strategy pair."""

    available: tuple[str, ...]
    default: str
    selected: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "available", columns(self.available, "available objectives"))
        if self.default not in self.available or self.selected not in self.available:
            raise PlanError("Default and selected objectives must belong to available objectives")

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize the resolved objective choice."""
        return {
            "available": list(self.available),
            "default": self.default,
            "selected": self.selected,
        }
