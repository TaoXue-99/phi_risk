from abc import ABC, abstractmethod

from phl_risk.analysis._context import AnalysisContext
from phl_risk.analysis._nodes import MeasureSpec
from phl_risk.exceptions import MeasureError


def validate_field(value: str | None, label: str, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        raise MeasureError(f"{label} must be a non-empty column name")
    return value


def measure_name(explicit: str | None, default: str) -> str:
    value = default if explicit is None else explicit
    if not isinstance(value, str) or not value:
        raise MeasureError("Measure name must be a non-empty string")
    return value


class BaseMeasure(ABC):
    """Declarative metric compiled to a backend-neutral calculation specification."""

    mode = "single"

    @abstractmethod
    def compile(self, context: AnalysisContext | None = None) -> MeasureSpec:
        """Resolve dependencies and a stable unique output name."""

    def required_columns(self, context: AnalysisContext | None = None) -> tuple[str, ...]:
        return self.compile(context).required_columns()


class SingleSampleMeasure(BaseMeasure):
    """Explicit single-sample base; legacy BaseMeasure subclasses remain supported."""


class ComparativeMeasure(BaseMeasure):
    """A measure compiled to a ComparativeNode for compute_comparison only."""

    mode = "comparative"
