from dataclasses import dataclass

import pandas as pd

from phl_risk.analysis._context import AnalysisContext
from phl_risk.analysis.transforms import BaseTransformer
from phl_risk.exceptions import DimensionError, NotFittedError

from ._base import BaseDimension, validate_name
from ._column import column_series
from ._stratified import StratifiedTransform


@dataclass(frozen=True)
class BinDimension(BaseDimension):
    """Learn boundaries from fit_field (default column), then bin column.

    fit_field names a raw reference column, not another dimension. Matching
    boundaries require identical learning data and transformer configuration.
    Optional fit_by learns independent transformers per raw-column stratum;
    compute reuses those fitted definitions and rejects unseen strata.
    """

    column: str
    transformer: BaseTransformer
    name: str | None = None
    fit_field: str | None = None
    fit_by: tuple[str, ...] | list[str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column:
            raise DimensionError("column must be a non-empty string")
        if self.fit_field is not None and (
            not isinstance(self.fit_field, str) or not self.fit_field
        ):
            raise DimensionError("fit_field must be a non-empty string or None")
        if self.fit_by is not None:
            if isinstance(self.fit_by, (str, bytes)):
                raise DimensionError("fit_by must be a non-empty sequence of raw column names")
            try:
                fields = tuple(self.fit_by)
            except TypeError as exc:
                raise DimensionError("fit_by must be a sequence of raw column names") from exc
            if (
                not fields
                or any(not isinstance(f, str) or not f for f in fields)
                or len(set(fields)) != len(fields)
            ):
                raise DimensionError("fit_by must contain distinct non-empty raw column names")
            object.__setattr__(self, "fit_by", fields)
        validate_name(self.output_name)
        if not isinstance(self.transformer, BaseTransformer):
            raise DimensionError("transformer must implement BaseTransformer")

    @property
    def output_name(self) -> str:
        return f"{self.column}_bin" if self.name is None else self.name

    @property
    def requires_fit(self) -> bool:
        return True

    @property
    def is_fitted(self) -> bool:
        if self.fit_by is not None:
            return hasattr(self, "_stratified")
        return self.transformer.is_fitted

    def required_columns(self) -> tuple[str, ...]:
        # Compute dependencies only; the learning column is needed during fit.
        return tuple(dict.fromkeys((self.column, *(self.fit_by or ()))))

    def fit(
        self, data: pd.DataFrame, y: object = None, context: AnalysisContext | None = None
    ) -> "BinDimension":
        if self.fit_by is not None:
            fitted = StratifiedTransform(
                data, self.fit_by, self.fit_field or self.column, self.transformer, y
            )
            object.__setattr__(self, "_stratified", fitted)
        else:
            self.transformer.fit(column_series(data, self.fit_field or self.column), y)
        return self

    def transform(self, data: pd.DataFrame, context: AnalysisContext | None = None) -> pd.Series:
        if not self.is_fitted:
            raise NotFittedError(f"Dimension {self.output_name!r} requires fit(reference)")
        source = column_series(data, self.column)
        result = (
            self._stratified.transform(data, self.fit_by, self.column)
            if self.fit_by is not None
            else self.transformer.transform(source)
        )
        if not isinstance(result, pd.Series) or not result.index.equals(source.index):
            raise DimensionError(f"Transformer for {self.output_name!r} must preserve Series index")
        return result.rename(self.output_name)

    def metadata(self) -> dict[str, object]:
        grouped = {}
        if self.fit_by is not None:
            grouped = {
                "fit_by": self.fit_by,
                "groups": self._stratified.metadata() if self.is_fitted else (),
            }
        return {
            **super().metadata(),
            **grouped,
            "column": self.column,
            "fit_field": self.fit_field or self.column,
            "transform": self.transformer.metadata(),
        }
