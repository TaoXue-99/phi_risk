from dataclasses import dataclass

from phl_risk.analysis._context import AnalysisContext
from phl_risk.analysis._nodes import MeasureSpec, RatioNode

from ._base import BaseMeasure, measure_name


@dataclass(frozen=True)
class Ratio(BaseMeasure):
    """Divide two named measures, not raw columns. Zero denominator follows on_invalid."""

    numerator: str
    denominator: str
    name: str | None = None

    def __post_init__(self):
        measure_name(self.numerator, "")
        measure_name(self.denominator, "")

    def compile(self, context: AnalysisContext | None = None) -> MeasureSpec:
        return MeasureSpec(
            measure_name(self.name, f"ratio__{self.numerator}__{self.denominator}"),
            RatioNode(self.numerator, self.denominator),
        )
