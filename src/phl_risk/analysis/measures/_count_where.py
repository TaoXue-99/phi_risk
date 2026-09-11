from dataclasses import dataclass

from phl_risk.analysis._context import AnalysisContext
from phl_risk.analysis._expressions import Predicate
from phl_risk.analysis._nodes import AggregateNode, MeasureSpec
from phl_risk.exceptions import MeasureError

from ._base import SingleSampleMeasure, measure_name


@dataclass(frozen=True)
class CountWhere(SingleSampleMeasure):
    """Count rows satisfying a predicate; unknown predicate values do not match."""

    condition: Predicate
    name: str | None = None

    def __post_init__(self):
        if not isinstance(self.condition, Predicate):
            raise MeasureError("CountWhere requires a Col predicate")

    def compile(self, context: AnalysisContext | None = None) -> MeasureSpec:
        return MeasureSpec(
            measure_name(self.name, f"count_where__{self.condition!r}"),
            AggregateNode("count_where", condition=self.condition),
            0.0,
        )
