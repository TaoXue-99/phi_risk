from dataclasses import dataclass
from typing import Literal

from phl_risk.analysis._context import AnalysisContext
from phl_risk.analysis._nodes import AggregateNode, MeasureSpec
from phl_risk.exceptions import MeasureError

from ._base import SingleSampleMeasure, measure_name, validate_field


@dataclass(frozen=True)
class Sum(SingleSampleMeasure):
    """Unweighted sum of a finite numeric column; missing values propagate by default.

    missing='zero' explicitly treats missing input quantities as zero.
    Empty cells have sum zero. AnalysisContext.weight does not multiply counts.
    """

    column: str
    name: str | None = None
    missing: Literal["propagate", "zero"] = "propagate"

    def __post_init__(self):
        validate_field(self.column, "column")
        if self.missing not in ("propagate", "zero"):
            raise MeasureError("Sum missing must be 'propagate' or 'zero'")

    def compile(self, context: AnalysisContext | None = None) -> MeasureSpec:
        return MeasureSpec(
            measure_name(self.name, f"sum__{self.column}"),
            AggregateNode("sum", self.column, missing=self.missing),
            0.0,
        )
