from dataclasses import dataclass
from typing import Literal

from phl_risk.analysis._context import AnalysisContext
from phl_risk.analysis._nodes import AggregateNode, DerivedMetricNode, MeasureSpec
from phl_risk.exceptions import MeasureError

from ._base import SingleSampleMeasure, measure_name


@dataclass(frozen=True)
class Share(SingleSampleMeasure):
    """Cell row count / analyzed row count, after filters and dimension deletion."""

    denominator: Literal["all"] = "all"
    name: str | None = None

    def __post_init__(self) -> None:
        if self.denominator != "all":
            raise MeasureError("V0.1 Share supports only denominator='all'")

    def compile(self, context: AnalysisContext | None = None) -> MeasureSpec:
        node = DerivedMetricNode(AggregateNode("row_count"), "all")
        return MeasureSpec(measure_name(self.name, "share"), node, 0.0)
