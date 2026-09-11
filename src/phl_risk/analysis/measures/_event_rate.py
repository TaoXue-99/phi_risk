from dataclasses import dataclass
from typing import Hashable

import pandas as pd

from phl_risk.analysis._context import AnalysisContext
from phl_risk.analysis._nodes import AggregateNode, DerivedMetricNode, MeasureSpec
from phl_risk.exceptions import MeasureError

from ._base import SingleSampleMeasure, measure_name, validate_field


@dataclass(frozen=True)
class EventRate(SingleSampleMeasure):
    """Event count / non-missing target count; optionally weighted."""

    target: str | None = None
    event_value: Hashable = 1
    weight: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not pd.api.types.is_scalar(self.event_value) or pd.isna(self.event_value):
            raise MeasureError("event_value must be a non-missing scalar")
        try:
            hash(self.event_value)
        except TypeError as exc:
            raise MeasureError("event_value must be hashable") from exc
        validate_field(self.target, "target", required=False)
        validate_field(self.weight, "weight", required=False)

    def compile(self, context: AnalysisContext | None = None) -> MeasureSpec:
        context = context or AnalysisContext()
        target = validate_field(
            self.target if self.target is not None else context.target, "target"
        )
        weight = self.weight if self.weight is not None else context.weight
        numerator = AggregateNode("event_count", target, self.event_value, weight)
        denominator = AggregateNode("valid_count", target, weight=weight)
        return MeasureSpec(
            measure_name(self.name, f"event_rate__{target}__{self.event_value}"),
            DerivedMetricNode(numerator, denominator),
        )
