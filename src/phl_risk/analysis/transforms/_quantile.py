from dataclasses import dataclass
from numbers import Integral
from typing import Hashable, Literal, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from phl_risk.exceptions import NotFittedError, TransformError

from .._intervals import format_intervals
from ._base import BaseTransformer


@dataclass(frozen=True)
class QuantileBinner(BaseTransformer[ArrayLike, pd.Series]):
    """Reference quantiles with ordered labels and extended outer boundaries.

    Finite reference values determine quantiles. Duplicate edges are dropped by
    default; a constant reference yields one bin. Current +/-inf and outliers
    enter the extreme bins. Missing values remain missing. Intervals are right
    closed; include_lowest controls whether -inf belongs to the first bin.
    Precision controls decimal places in metadata and layout (default six),
    increasing when needed to distinguish edges. It never affects assignment.
    """

    n_bins: int = 5
    duplicates: Literal["drop", "raise"] = "drop"
    labels: Sequence[Hashable] | None = None
    include_lowest: bool = True
    precision: int = 6

    def __post_init__(self) -> None:
        if isinstance(self.n_bins, bool) or not isinstance(self.n_bins, Integral):
            raise TransformError("n_bins must be a positive integer")
        if self.n_bins < 1:
            raise TransformError("n_bins must be a positive integer")
        if self.duplicates not in ("drop", "raise"):
            raise TransformError("duplicates must be 'drop' or 'raise'")
        if not isinstance(self.include_lowest, bool):
            raise TransformError("include_lowest must be bool")
        if not isinstance(self.precision, Integral) or self.precision < 0:
            raise TransformError("precision must be a non-negative integer")
        if self.labels is not None:
            if isinstance(self.labels, (str, bytes)):
                raise TransformError("labels must be a sequence of distinct scalar labels")
            labels = tuple(self.labels)
            try:
                unique = len(set(labels)) == len(labels)
            except TypeError as exc:
                raise TransformError("labels must be hashable") from exc
            if not unique or any(not pd.api.types.is_scalar(x) or pd.isna(x) for x in labels):
                raise TransformError("labels must be distinct non-missing scalars")
            object.__setattr__(self, "labels", labels)

    @staticmethod
    def _series(X: ArrayLike) -> pd.Series:
        try:
            series = X if isinstance(X, pd.Series) else pd.Series(X)
            return pd.to_numeric(series, errors="raise")
        except (ValueError, TypeError) as exc:
            raise TransformError("QuantileBinner requires one-dimensional numeric input") from exc

    def fit(self, X: ArrayLike, y: object = None) -> "QuantileBinner":
        values = self._series(X).to_numpy(dtype=float, na_value=np.nan)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise TransformError("QuantileBinner reference has no finite observations")
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        scale = np.max(np.abs(finite))
        # Interpolation can overflow for opposite extreme finite endpoints.
        if scale > np.finfo(float).max / 2:
            raw = np.quantile(finite / scale, quantiles) * scale
        else:
            raw = np.quantile(finite, quantiles)
        edges = np.unique(raw)
        if len(edges) < len(raw) and self.duplicates == "raise":
            raise TransformError("Duplicate reference quantile edges; use duplicates='drop'")
        effective = max(1, len(edges) - 1)
        edges = np.r_[-np.inf, edges[1:-1], np.inf]
        labels = (
            tuple(f"B{i + 1}" for i in range(effective)) if self.labels is None else self.labels
        )
        if len(labels) != effective:
            raise TransformError(f"labels has {len(labels)} entries; learned {effective} bins")
        # Commit only after validation. Tuples make fitted snapshots safe to copy.
        object.__setattr__(self, "_edges", tuple(float(x) for x in edges))
        object.__setattr__(self, "labels_", tuple(labels))
        object.__setattr__(self, "n_bins_", effective)
        object.__setattr__(self, "reference_range_", (float(finite.min()), float(finite.max())))
        return self

    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "_edges")

    @property
    def bin_edges_(self) -> NDArray[np.float64]:
        if not self.is_fitted:
            raise NotFittedError("QuantileBinner requires fit(reference) before transform(current)")
        return np.asarray(self._edges, dtype=float)

    def transform(self, X: ArrayLike) -> pd.Series:
        edges = self.bin_edges_
        series = self._series(X)
        values = series.to_numpy(dtype=float, na_value=np.nan)
        # searchsorted implements right-closed intervals without rounding edges.
        codes = np.searchsorted(edges[1:-1], values, side="left")
        codes[np.isnan(values)] = -1
        if not self.include_lowest:
            codes[np.isneginf(values)] = -1
        return pd.Series(
            pd.Categorical.from_codes(codes, categories=self.labels_, ordered=True),
            index=series.index,
            name=series.name,
        )

    def metadata(self) -> dict[str, object]:
        result = super().metadata()
        if self.is_fitted:
            result.update(
                bin_edges=tuple(self._edges),
                labels=self.labels_,
                n_bins=self.n_bins_,
                reference_range=self.reference_range_,
                include_lowest=self.include_lowest,
                intervals=format_intervals(
                    self._edges, precision=self.precision, include_lowest=self.include_lowest
                ),
            )
        return result
