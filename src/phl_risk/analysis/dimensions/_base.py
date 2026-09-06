from abc import ABC, abstractmethod

from phl_risk.analysis._context import AnalysisContext
from phl_risk.exceptions import DimensionError


def validate_name(name: str) -> None:
    if not isinstance(name, str) or not name or name in ("metric", "value"):
        raise DimensionError(f"Invalid dimension name {name!r}; metric/value are reserved")


class BaseDimension(ABC):
    """Analytical dimension; row-level data representations belong to implementations."""

    def fit(
        self, data: object, y: object = None, context: AnalysisContext | None = None
    ) -> "BaseDimension":
        return self

    @abstractmethod
    def transform(self, data: object, context: AnalysisContext | None = None) -> object:
        """Produce one value per input row, preserving order and index."""

    @property
    @abstractmethod
    def output_name(self) -> str:
        """Stable name of this dimension."""

    @abstractmethod
    def required_columns(self) -> tuple[str, ...]:
        """Raw column dependencies."""

    @property
    def requires_fit(self) -> bool:
        return False

    @property
    def is_fitted(self) -> bool:
        return not self.requires_fit

    def metadata(self) -> dict[str, object]:
        return {"name": self.output_name, "fitted": self.is_fitted}
