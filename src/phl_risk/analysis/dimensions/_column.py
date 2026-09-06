from dataclasses import dataclass

import pandas as pd

from phl_risk.analysis._context import AnalysisContext
from phl_risk.exceptions import DimensionError

from ._base import BaseDimension, validate_name


def column_series(data: pd.DataFrame, column: str) -> pd.Series:
    if not isinstance(data, pd.DataFrame):
        raise DimensionError("ColumnDimension/BinDimension require a pandas DataFrame")
    if column not in data.columns:
        raise DimensionError(f"Missing dimension source column {column!r}")
    series = data[column]
    if not isinstance(series, pd.Series):
        raise DimensionError(f"Duplicate source column {column!r}")
    return series


@dataclass(frozen=True)
class ColumnDimension(BaseDimension):
    column: str
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column:
            raise DimensionError("column must be a non-empty string")
        validate_name(self.output_name)

    @property
    def output_name(self) -> str:
        return self.column if self.name is None else self.name

    def required_columns(self) -> tuple[str, ...]:
        return (self.column,)

    def transform(self, data: pd.DataFrame, context: AnalysisContext | None = None) -> pd.Series:
        return column_series(data, self.column).rename(self.output_name)
