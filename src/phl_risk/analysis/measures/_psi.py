"""PSI configuration and its batched distribution calculation."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd

from phl_risk.analysis._comparison_types import (
    ComparativeCalculation,
    ComparisonGroups,
    ComparisonOutput,
)
from phl_risk.analysis._context import AnalysisContext
from phl_risk.analysis._distribution import distribution_codes, profile
from phl_risk.analysis._nodes import ComparativeNode, MeasureSpec
from phl_risk.analysis.transforms import BaseTransformer, QuantileBinner
from phl_risk.exceptions import EngineError, MeasureError
from phl_risk.metrics._policy import invalid
from phl_risk.metrics._psi import psi_from_proportions, validate_epsilon

from ._base import ComparativeMeasure, measure_name, validate_field


@dataclass(frozen=True)
class PSI(ComparativeMeasure):
    """Compare a field using reference-fitted bins (None means categorical).

    Missing values form a separate bucket by default. Each call fits a private
    binner on all eligible reference rows, not on each group. Counts are unweighted.
    """

    field: str
    binner: BaseTransformer | None = dataclass_field(default_factory=lambda: QuantileBinner(10))
    epsilon: float = 1e-8
    name: str | None = None
    missing: Literal["bucket", "drop"] = "bucket"
    min_samples: int = 1

    def __post_init__(self) -> None:
        validate_field(self.field, "field")
        measure_name(self.name, f"psi__{self.field}")
        validate_epsilon(self.epsilon)
        if self.binner is not None and not isinstance(self.binner, BaseTransformer):
            raise MeasureError("binner must be BaseTransformer or None for categorical PSI")
        if self.missing not in ("bucket", "drop"):
            raise MeasureError("PSI missing must be 'bucket' or 'drop'")
        if (
            isinstance(self.min_samples, bool)
            or not isinstance(self.min_samples, Integral)
            or self.min_samples < 1
        ):
            raise MeasureError("min_samples must be a positive integer")

    def compile(self, context: AnalysisContext | None = None) -> MeasureSpec:
        if context is not None and context.weight is not None:
            raise MeasureError("PSI currently uses unweighted counts; omit context.weight")
        return MeasureSpec(
            measure_name(self.name, f"psi__{self.field}"), ComparativeNode(_PSICalculation(self))
        )


@dataclass(frozen=True)
class _PSICalculation(ComparativeCalculation):
    measure: PSI

    def required_columns(self) -> tuple[str, ...]:
        return (self.measure.field,)

    def evaluate(
        self,
        reference: object,
        current: object,
        groups: ComparisonGroups,
        *,
        backend: str,
        on_invalid: str,
    ) -> ComparisonOutput:
        if (
            backend != "pandas"
            or not isinstance(reference, pd.DataFrame)
            or not isinstance(current, pd.DataFrame)
        ):
            raise EngineError("PSI distribution construction currently supports pandas")
        config = self.measure
        ref, cur = reference[config.field], current[config.field]
        output = np.full(groups.size, np.nan)
        # An empty reference cannot define numeric boundaries. For QuantileBinner
        # an all-missing/nonfinite reference likewise has no learnable definition.
        unavailable = len(ref) == 0
        if isinstance(config.binner, QuantileBinner):
            values = config.binner._series(ref).to_numpy(dtype=float, na_value=np.nan)
            unavailable |= not np.isfinite(values).any()
        if config.binner is not None and unavailable:
            if groups.size:
                invalid(f"PSI {config.field}: reference cannot define numeric bins", on_invalid)
            return ComparisonOutput(output, {"unavailable_reference": True})
        a, b, bins, metadata = distribution_codes(ref, cur, config.binner, config.missing)
        if not bins:
            if groups.size:
                invalid(f"PSI {config.field}: empty distributions", on_invalid)
            return ComparisonOutput(output, metadata)
        reference_profile = profile(groups.reference_codes, a, groups.size, bins)
        current_profile = profile(groups.current_codes, b, groups.size, bins)
        valid = (reference_profile.totals >= config.min_samples) & (
            current_profile.totals >= config.min_samples
        )
        if (~valid).any():
            invalid(
                f"PSI {config.field}: {int((~valid).sum())} groups below "
                f"min_samples={config.min_samples} on one or both sides",
                on_invalid,
            )
        output[valid] = psi_from_proportions(
            reference_profile.proportions[valid],
            current_profile.proportions[valid],
            epsilon=config.epsilon,
            on_invalid=on_invalid,
        )
        metadata.update(
            epsilon=config.epsilon,
            smoothing="clip_then_normalize",
            missing=config.missing,
            min_samples=config.min_samples,
            invalid_groups=int((~valid).sum()),
        )
        return ComparisonOutput(output, metadata)
