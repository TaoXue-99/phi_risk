from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from phl_risk._data import frame, freeze
from phl_risk.exceptions import DataQualityError


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    count: int
    missing_count: int
    missing_rate: float
    nunique: int
    zero_count: int | None
    zero_rate: float | None
    min: object
    max: object
    mean: float | None
    std: float | None
    quantiles: Mapping
    top_values: tuple

    def __post_init__(self):
        for key in ("quantiles", "top_values"):
            object.__setattr__(self, key, freeze(getattr(self, key)))


@dataclass(frozen=True)
class DataProfile:
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    profiles: Mapping[str, ColumnProfile]

    def __post_init__(self):
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "profiles", freeze(self.profiles))

    @classmethod
    def from_frame(cls, X, max_top_values=20):
        frame(X, DataQualityError)
        if not isinstance(max_top_values, int) or max_top_values < 0:
            raise DataQualityError("max_top_values must be a nonnegative integer")
        profiles = {}
        for name in X:
            s = X[name]
            numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_complex_dtype(s)
            missing = int(s.isna().sum())
            zero = int(s.eq(0).sum()) if numeric else None
            # Quantiles and moments describe finite observations; null rates retain all rows.
            finite = s[np.isfinite(s)].astype(float) if numeric else None
            quantiles = (
                {
                    f"q{int(q * 100):02d}": float(v)
                    for q, v in finite.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).items()
                }
                if numeric and len(finite)
                else {}
            )
            profiles[name] = ColumnProfile(
                name,
                str(s.dtype),
                int(s.count()),
                missing,
                missing / len(X) if len(X) else float("nan"),
                int(s.nunique()),
                zero,
                zero / len(X) if numeric and len(X) else None,
                s.min() if numeric and s.count() else None,
                s.max() if numeric and s.count() else None,
                float(finite.mean()) if numeric and len(finite) else None,
                float(finite.std()) if numeric and len(finite) > 1 else None,
                quantiles,
                tuple((v, int(n)) for v, n in s.value_counts().head(max_top_values).items())
                if not numeric
                else (),
            )
        return cls(len(X), len(X.columns), tuple(X.columns), profiles)
