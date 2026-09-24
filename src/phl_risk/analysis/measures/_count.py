from dataclasses import dataclass, field

from phl_risk.analysis._context import AnalysisContext
from phl_risk.analysis._expressions import Predicate
from phl_risk.analysis._nodes import AggregateNode, MeasureSpec
from phl_risk.exceptions import MeasureError

from ._base import SingleSampleMeasure, measure_name


@dataclass(frozen=True)
class Count(SingleSampleMeasure):
    """Count rows matching where; unknown predicates do not match. No deduplication."""

    name: str | None = None
    where: Predicate | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.where is not None and not isinstance(self.where, Predicate):
            raise MeasureError("where must be a Col predicate or None")

    def compile(self, context: AnalysisContext | None = None) -> MeasureSpec:
        return MeasureSpec(
            measure_name(self.name, "count"),
            AggregateNode("row_count", condition=self.where),
            0.0,
        )
