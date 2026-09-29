"""Explicit, immutable numeric interval definitions."""

from dataclasses import dataclass
from numbers import Real
from typing import Hashable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from phl_risk.exceptions import TransformError

from ._numeric import _NumericBinner


@dataclass(frozen=True)
class FixedBinner(_NumericBinner):
    """Use complete, strictly increasing edges without learning from data.

    Ready for transform immediately. fit validates input but never changes edges.
    Intervals are right closed; include_lowest includes the first edge. Values
    outside the supplied range become missing; provide +/-inf for full coverage.
    """

    edges: Sequence[float]
    labels: Sequence[Hashable] | None = None
    include_lowest: bool = True
    precision: int = 6

    def __post_init__(self) -> None:
        self._validate_options()
        if isinstance(self.edges, (str, bytes)):
            raise TransformError("edges must be a one-dimensional sequence of real boundaries")
        try:
            edges = tuple(self.edges)
            if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in edges):
                raise ValueError("non-real edge")
            values = np.asarray(edges, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TransformError(
                "edges must be a one-dimensional sequence of real boundaries"
            ) from exc
        if len(values) < 2 or np.isnan(values).any() or not np.all(values[1:] > values[:-1]):
            raise TransformError(
                "edges must contain at least two strictly increasing, non-NaN boundaries"
            )
        object.__setattr__(self, "edges", tuple(float(x) for x in values))
        self._commit_edges(values)

    def fit(self, X: ArrayLike, y: object = None) -> "FixedBinner":
        self._series(X)
        return self
