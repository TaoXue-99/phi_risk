import numpy as np
import pandas as pd

from phl_risk.exceptions import DataQualityError

from .._base import BaseQualityCheck
from .._status import CheckStatus
from .._utils import examples_limit, result, selected, threshold


class NumericConvertibleCheck(BaseQualityCheck):
    requires_fit = False

    def __init__(self, columns, min_success_rate=0.99, max_examples=10):
        self.columns = columns
        self.min_success_rate = min_success_rate
        self.max_examples = max_examples

    def _validate(self, X):
        threshold(self.min_success_rate, "min_success_rate", 1)
        examples_limit(self.max_examples)
        columns, details, statuses = selected(X, self.columns), {}, []
        for c in columns:
            source = X[c].notna()
            converted = pd.to_numeric(X[c], errors="coerce")
            failed = source & converted.isna()
            count, nfailed = int(source.sum()), int(failed.sum())
            rate = (count - nfailed) / count if count else float("nan")
            details[c] = dict(
                source_non_null=count,
                success_count=count - nfailed,
                failed_count=nfailed,
                success_rate=rate,
                failed_examples=X.loc[failed, c].head(self.max_examples).tolist(),
            )
            statuses.append(
                (CheckStatus.FAIL if rate < self.min_success_rate else CheckStatus.PASS)
                if count
                else CheckStatus.SKIP
            )
        return result(self, columns, details, statuses)


class FiniteCheck(BaseQualityCheck):
    requires_fit = False

    def __init__(self, columns, allow_nan=True):
        self.columns = columns
        self.allow_nan = allow_nan

    def _validate(self, X):
        columns, details, statuses = selected(X, self.columns), {}, []
        for c in columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                raise DataQualityError(f"FiniteCheck requires numeric column {c!r}")
            values = X[c].to_numpy(dtype=float, na_value=np.nan)
            nan, inf = int(np.isnan(values).sum()), int(np.isinf(values).sum())
            details[c] = dict(
                nan_count=nan,
                positive_inf=int(np.isposinf(values).sum()),
                negative_inf=int(np.isneginf(values).sum()),
            )
            statuses.append(
                (CheckStatus.FAIL if inf or (nan and not self.allow_nan) else CheckStatus.PASS)
                if len(X)
                else CheckStatus.SKIP
            )
        return result(self, columns, details, statuses)


class RangeCheck(BaseQualityCheck):
    requires_fit = False

    def __init__(self, column, min_value=None, max_value=None, inclusive="both"):
        self.column = column
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive

    def _validate(self, X):
        selected(X, [self.column])
        if self.inclusive not in ("both", "left", "right", "neither"):
            raise DataQualityError("inclusive must be both/left/right/neither")
        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise DataQualityError("min_value must not exceed max_value")
        s = X[self.column].dropna()
        invalid = pd.Series(False, index=s.index)
        if self.min_value is not None:
            invalid |= (
                s.lt(self.min_value) if self.inclusive in ("both", "left") else s.le(self.min_value)
            )
        if self.max_value is not None:
            invalid |= (
                s.gt(self.max_value)
                if self.inclusive in ("both", "right")
                else s.ge(self.max_value)
            )
        return result(
            self,
            [self.column],
            dict(out_of_range_count=int(invalid.sum())),
            [
                (CheckStatus.FAIL if invalid.any() else CheckStatus.PASS)
                if len(s)
                else CheckStatus.SKIP
            ],
        )
