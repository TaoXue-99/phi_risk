"""Fixed numeric rules. All operations preserve nulls and require real numeric input."""

from numbers import Real

import numpy as np
import pandas as pd

from phl_risk._data import columns_of
from phl_risk.exceptions import DataPrepError

from .._base import StatelessPrepStep


def _numeric_column(X, column):
    columns_of(X, [column], DataPrepError)
    dtype = X[column].dtype
    if (
        not pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
    ):
        raise DataPrepError(f"{column!r} must be real numeric; use ToNumeric first")
    return X[column]


def _bound(value, name):
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
    ):
        raise DataPrepError(f"{name} must be a finite real number or None")


def _output_name(X, column, output_column):
    name = column if output_column is None else output_column
    if not isinstance(name, str) or not name:
        raise DataPrepError("output_column must be a nonempty string")
    if name != column and name in X.columns:
        raise DataPrepError(f"Output column {name!r} already exists")
    return name


class Clip(StatelessPrepStep):
    """Clip selected columns to fixed bounds; None leaves that side unbounded."""

    def __init__(self, columns, lower=None, upper=None):
        self.columns = columns
        self.lower = lower
        self.upper = upper

    def _transform(self, X):
        _bound(self.lower, "lower")
        _bound(self.upper, "upper")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise DataPrepError("lower must not exceed upper")
        for column in columns_of(X, self.columns, DataPrepError):
            series = _numeric_column(X, column)
            # Preserve exact integer values unless fractional bounds require promotion.
            if pd.api.types.is_integer_dtype(series.dtype) and any(
                b is not None and b != int(b) for b in (self.lower, self.upper)
            ):
                series = series.astype(float)
            X[column] = series.clip(self.lower, self.upper)
        return X

    def _audit_details(self, X, output):
        return {
            c: dict(
                lower=self.lower,
                upper=self.upper,
                clipped_count=int((X[c].notna() & X[c].ne(output[c])).sum()),
            )
            for c in columns_of(X, self.columns, DataPrepError)
        }


class LogTransform(StatelessPrepStep):
    """Natural logarithm of positive values; nulls propagate, nonpositive values raise.

    output_column=None replaces the source; a new name appends a column.
    No implicit shift, clipping, or reference-dependent rule is applied.
    """

    def __init__(self, column, output_column=None):
        self.column = column
        self.output_column = output_column

    def _transform(self, X):
        series = _numeric_column(X, self.column)
        name = _output_name(X, self.column, self.output_column)
        if series.le(0).any():
            raise DataPrepError(
                f"LogTransform requires positive non-null values in {self.column!r}"
            )
        X[name] = np.log(series.astype(float))
        return X

    def audit_columns(self, X, output):
        name = self.column if self.output_column is None else self.output_column
        return (self.column,), (name,)

    def _audit_details(self, X, output):
        return dict(function="log", transformed_count=int(X[self.column].notna().sum()))


class LogitTransform(StatelessPrepStep):
    """Clip to [eps, 1-eps], then compute log(x)-log1p(-x); nulls propagate.

    Clipping includes values outside [0, 1] and infinities; audit records their count.
    """

    def __init__(self, column, output_column=None, eps=1e-6):
        self.column = column
        self.output_column = output_column
        self.eps = eps

    def _transform(self, X):
        _bound(self.eps, "eps")
        if self.eps is None or not 0 < self.eps < 0.5 or 1 - self.eps == 1:
            raise DataPrepError("eps must lie in (0, 0.5) with 1-eps representably below 1")
        series = _numeric_column(X, self.column)
        name = _output_name(X, self.column, self.output_column)
        clipped = series.astype(float).clip(self.eps, 1 - self.eps)
        X[name] = np.log(clipped) - np.log1p(-clipped)
        return X

    def audit_columns(self, X, output):
        name = self.column if self.output_column is None else self.output_column
        return (self.column,), (name,)

    def _audit_details(self, X, output):
        source = X[self.column]
        clipped = source.lt(self.eps) | source.gt(1 - self.eps)
        return dict(
            function="logit",
            eps=self.eps,
            clipped_count=int(clipped.sum()),
            transformed_count=int(source.notna().sum()),
        )
