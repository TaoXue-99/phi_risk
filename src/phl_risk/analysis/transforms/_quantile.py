from dataclasses import dataclass
from typing import Hashable, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike

from phl_risk.exceptions import TransformError

from ._numeric import _NumericBinner


@dataclass(frozen=True)
class QuantileBinner(_NumericBinner):
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
        self._validate_n_bins(self.n_bins)
        if self.duplicates not in ("drop", "raise"):
            raise TransformError("duplicates must be 'drop' or 'raise'")
        self._validate_options()

    def fit(self, X: ArrayLike, y: object = None) -> "QuantileBinner":
        finite = self._finite_reference(X)
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
        edges = np.r_[-np.inf, edges[1:-1], np.inf]
        self._commit_edges(edges, (float(finite.min()), float(finite.max())))
        return self
