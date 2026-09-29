"""Reference-fitted equal-width cut points with open outer tails."""

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from ._numeric import _NumericBinner


@dataclass(frozen=True)
class EqualWidthBinner(_NumericBinner):
    """Split the finite reference range into n_bins equal-width intervals.

    Outer edges extend to +/-inf, matching QuantileBinner's outlier handling.
    Thus interior cut points are equally spaced, but outer intervals are open
    tails. Constant references produce one bin; unrepresentable duplicate
    float64 cut points are merged. NaN stays missing. Assignment is right closed.
    """

    n_bins: int = 5
    labels: Sequence[Hashable] | None = None
    include_lowest: bool = True
    precision: int = 6

    def __post_init__(self) -> None:
        self._validate_n_bins(self.n_bins)
        self._validate_options()

    def fit(self, X: ArrayLike, y: object = None) -> "EqualWidthBinner":
        finite = self._finite_reference(X)
        low, high = float(finite.min()), float(finite.max())
        # Scale before interpolation to avoid overflow in high - low.
        scale = max(abs(low), abs(high))
        raw = (
            np.linspace(low / scale, high / scale, self.n_bins + 1) * scale
            if scale > np.finfo(float).max / 2
            else np.linspace(low, high, self.n_bins + 1)
        )
        edges = np.unique(raw)
        self._commit_edges(np.r_[-np.inf, edges[1:-1], np.inf], (low, high))
        return self
