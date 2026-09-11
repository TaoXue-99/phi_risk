from dataclasses import dataclass

from phl_risk.analysis._context import AnalysisContext
from phl_risk.analysis._nodes import GroupMetricNode, MeasureSpec

from ._base import SingleSampleMeasure, measure_name, validate_field


@dataclass(frozen=True)
class KS(SingleSampleMeasure):
    """Maximum absolute separation of binary score distributions."""

    score: str
    target: str | None = None
    weight: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        validate_field(self.score, "score")
        validate_field(self.target, "target", required=False)
        validate_field(self.weight, "weight", required=False)

    def compile(self, context: AnalysisContext | None = None) -> MeasureSpec:
        context = context or AnalysisContext()
        target = validate_field(
            self.target if self.target is not None else context.target, "target"
        )
        weight = self.weight if self.weight is not None else context.weight
        return MeasureSpec(
            measure_name(self.name, f"ks__{self.score}"),
            GroupMetricNode("ks", target, self.score, weight),
        )
