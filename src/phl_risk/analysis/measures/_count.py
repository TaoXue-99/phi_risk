from dataclasses import dataclass

from phl_risk.analysis._context import AnalysisContext
from phl_risk.analysis._nodes import AggregateNode, MeasureSpec

from ._base import BaseMeasure, measure_name


@dataclass(frozen=True)
class Count(BaseMeasure):
    """Unweighted row count, including rows with missing measure inputs."""

    name: str | None = None

    def compile(self, context: AnalysisContext | None = None) -> MeasureSpec:
        return MeasureSpec(measure_name(self.name, "count"), AggregateNode("row_count"), 0.0)
