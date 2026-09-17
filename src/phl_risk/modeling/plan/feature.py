"""Tabular feature declarations without transformation or dtype inference."""

from dataclasses import dataclass

from phl_risk.exceptions import PlanError

from .._utils import Columns, JSONValue, columns


@dataclass(frozen=True, init=False)
class FeatureSpec:
    """Declare disjoint, ordered numerical and categorical columns."""

    numerical: tuple[str, ...]
    categorical: tuple[str, ...]

    def __init__(self, numerical: Columns = (), categorical: Columns = ()) -> None:
        object.__setattr__(self, "numerical", columns(numerical, "numerical", allow_empty=True))
        object.__setattr__(
            self, "categorical", columns(categorical, "categorical", allow_empty=True)
        )
        overlap = set(self.numerical) & set(self.categorical)
        if overlap:
            raise PlanError(f"Numerical and categorical features overlap: {sorted(overlap)}")

    @property
    def all(self) -> tuple[str, ...]:
        """All features in numerical-then-categorical declaration order."""
        return self.numerical + self.categorical

    def to_dict(self) -> dict[str, JSONValue]:
        """Return detached feature lists."""
        return {"numerical": list(self.numerical), "categorical": list(self.categorical)}
