import pandas as pd

from phl_risk._data import columns_of
from phl_risk.exceptions import DataPrepError

from .._base import StatelessPrepStep


class ToNumeric(StatelessPrepStep):
    def __init__(self, columns, errors="raise"):
        self.columns = columns
        self.errors = errors

    def _convert(self, series):
        return pd.to_numeric(series, errors=self.errors)

    def _transform(self, X):
        if self.errors not in ("raise", "coerce"):
            raise DataPrepError("errors must be 'raise' or 'coerce'")
        columns = columns_of(X, self.columns, DataPrepError)
        for c in columns:
            try:
                X[c] = self._convert(X[c])
            except (ValueError, TypeError) as exc:
                raise DataPrepError(
                    f"{type(self).__name__} failed for column {c!r}: {exc}"
                ) from exc
        return X

    def _audit_details(self, X, output):
        details = {}
        for c in self.columns:
            source, failed = X[c].notna(), X[c].notna() & output[c].isna()
            count, failures = int(source.sum()), int(failed.sum())
            details[c] = dict(
                source_non_null=count,
                success_count=count - failures,
                failed_count=failures,
                failure_count=failures,
                success_rate=(count - failures) / count if count else None,
                new_missing_count=failures,
                new_null=failures,
                failed_examples=X.loc[failed, c].head(10).tolist(),
                dtype_before=str(X[c].dtype),
                dtype_after=str(output[c].dtype),
            )
        return details


class ToDatetime(ToNumeric):
    def __init__(self, columns, errors="raise", format=None, utc=False):
        self.columns = columns
        self.errors = errors
        self.format = format
        self.utc = utc

    def _convert(self, series):
        return pd.to_datetime(series, errors=self.errors, format=self.format, utc=self.utc)

    def _audit_details(self, X, output):
        details = super()._audit_details(X, output)
        for values in details.values():
            values["new_nat_count"] = values["new_missing_count"]
        return details
