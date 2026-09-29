"""Shared numeric interval assignment; no learning strategy or Cube dependency."""

from numbers import Integral
from typing import Hashable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from phl_risk.exceptions import NotFittedError, TransformError

from .._intervals import format_intervals
from ._base import BaseTransformer


class _NumericBinner(BaseTransformer[ArrayLike, pd.Series]):
    labels: Sequence[Hashable] | None
    include_lowest: bool
    precision: int
    labels_: tuple[Hashable, ...]
    n_bins_: int
    _edges: tuple[float, ...]
    reference_range_: tuple[float, float]

    @staticmethod
    def _validate_n_bins(n_bins: int) -> None:
        if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or n_bins < 1:
            raise TransformError("n_bins must be a positive integer")

    @classmethod
    def _finite_reference(cls, X: ArrayLike) -> NDArray[np.float64]:
        values = cls._series(X).to_numpy(dtype=float, na_value=np.nan)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise TransformError(f"{cls.__name__} reference has no finite observations")
        return finite

    def _validate_options(self) -> None:
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

    @classmethod
    def _series(cls, X: ArrayLike) -> pd.Series:
        try:
            series = X if isinstance(X, pd.Series) else pd.Series(X)
            return pd.to_numeric(series, errors="raise")
        except (ValueError, TypeError) as exc:
            raise TransformError(f"{cls.__name__} requires one-dimensional numeric input") from exc

    def _commit_edges(
        self,
        edges: Sequence[float] | NDArray[np.float64],
        reference_range: tuple[float, float] | None = None,
    ) -> None:
        effective = len(edges) - 1
        labels = (
            tuple(f"B{i + 1}" for i in range(effective)) if self.labels is None else self.labels
        )
        if len(labels) != effective:
            raise TransformError(f"labels has {len(labels)} entries; learned {effective} bins")
        object.__setattr__(self, "_edges", tuple(float(x) for x in edges))
        object.__setattr__(self, "labels_", tuple(labels))
        object.__setattr__(self, "n_bins_", effective)
        if reference_range is not None:
            object.__setattr__(self, "reference_range_", reference_range)

    @property
    def is_fitted(self) -> bool:
        return hasattr(self, "_edges")

    @property
    def bin_edges_(self) -> NDArray[np.float64]:
        if not self.is_fitted:
            raise NotFittedError(
                f"{type(self).__name__} requires fit(reference) before transform(current)"
            )
        return np.asarray(self._edges, dtype=float)

    def transform(self, X: ArrayLike) -> pd.Series:
        edges = self.bin_edges_
        series = self._series(X)
        values = series.to_numpy(dtype=float, na_value=np.nan)
        # searchsorted implements right-closed intervals without rounding edges.
        codes = np.searchsorted(edges[1:-1], values, side="left")
        codes[np.isnan(values)] = -1
        codes[(values < edges[0]) | (values > edges[-1])] = -1
        if not self.include_lowest:
            codes[values == edges[0]] = -1
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
                include_lowest=self.include_lowest,
                intervals=format_intervals(
                    self._edges, precision=self.precision, include_lowest=self.include_lowest
                ),
            )
        if hasattr(self, "reference_range_"):
            result["reference_range"] = self.reference_range_
        return result
