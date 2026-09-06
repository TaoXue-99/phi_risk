from dataclasses import dataclass

import pandas as pd

from phl_risk.analysis._context import AnalysisContext
from phl_risk.analysis.transforms import BaseTransformer
from phl_risk.exceptions import DimensionError, NotFittedError

from ._base import BaseDimension, validate_name
from ._column import column_series


@dataclass(frozen=True)
class BinDimension(BaseDimension):
    column: str
    transformer: BaseTransformer
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column:
            raise DimensionError("column must be a non-empty string")
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
        return self.transformer.is_fitted

    def required_columns(self) -> tuple[str, ...]:
        return (self.column,)

    def fit(
        self, data: pd.DataFrame, y: object = None, context: AnalysisContext | None = None
    ) -> "BinDimension":
        self.transformer.fit(column_series(data, self.column), y)
        return self

    def transform(self, data: pd.DataFrame, context: AnalysisContext | None = None) -> pd.Series:
        if not self.is_fitted:
            raise NotFittedError(f"Dimension {self.output_name!r} requires fit(reference)")
        source = column_series(data, self.column)
        result = self.transformer.transform(source)
        if not isinstance(result, pd.Series) or not result.index.equals(source.index):
            raise DimensionError(f"Transformer for {self.output_name!r} must preserve Series index")
        return result.rename(self.output_name)

    def metadata(self) -> dict[str, object]:
        return {**super().metadata(), "transform": self.transformer.metadata()}
