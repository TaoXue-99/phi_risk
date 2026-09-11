"""Pandas field encoding and NumPy batched histograms, independent of PSI."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from phl_risk.exceptions import EngineError

from .transforms import BaseTransformer, QuantileBinner

# Bound each dense group x bucket allocation (80 MB for int64).
MAX_PROFILE_CELLS = 10_000_000


@dataclass(frozen=True)
class DistributionProfile:
    counts: NDArray[np.int64]
    proportions: NDArray[np.float64]
    totals: NDArray[np.int64]


def profile(
    group_codes: NDArray, bin_codes: NDArray, groups: int, bins: int
) -> DistributionProfile:
    if groups * bins > MAX_PROFILE_CELLS:
        raise EngineError("Distribution profile exceeds 10,000,000 cells; reduce groups/categories")
    valid = bin_codes >= 0
    combined = group_codes[valid] * bins + bin_codes[valid]
    counts = np.bincount(combined, minlength=groups * bins).reshape(groups, bins)
    totals = counts.sum(axis=-1)
    proportions = np.zeros(counts.shape, dtype=float)
    np.divide(counts, totals[:, None], out=proportions, where=totals[:, None] > 0)
    return DistributionProfile(counts, proportions, totals)


def category_domain(reference: pd.Series, current: pd.Series) -> tuple:
    """Reference order then current-only values; include declared unused categories."""
    values = []
    for series in (reference, current):
        domain = (
            series.cat.categories
            if isinstance(series.dtype, pd.CategoricalDtype)
            else pd.unique(series.dropna())
        )
        values.extend(domain)
    try:
        return tuple(dict.fromkeys(values))
    except TypeError as exc:
        raise EngineError("Categories must be hashable scalars") from exc


def distribution_codes(
    reference: pd.Series,
    current: pd.Series,
    binner: BaseTransformer | None,
    missing: str,
) -> tuple[NDArray, NDArray, int, dict[str, object]]:
    """Fit a private transformer once on reference; encode both populations."""
    metadata: dict[str, object] = {}
    if binner is not None:
        transformer = deepcopy(binner)
        # The built-in binner is verified non-mutating. Isolate external
        # implementations while avoiding three large per-field copies here.
        trusted = type(transformer) is QuantileBinner
        transformer.fit(reference if trusted else reference.copy(deep=True))
        ref = transformer.transform(reference if trusted else reference.copy(deep=True))
        cur = transformer.transform(current if trusted else current.copy(deep=True))
        for result, source in ((ref, reference), (cur, current)):
            if not isinstance(result, pd.Series) or not result.index.equals(source.index):
                raise EngineError("Distribution binner must return an index-preserving Series")
            if not isinstance(result.dtype, pd.CategoricalDtype):
                raise EngineError("Distribution binner must return categorical bins")
        if not ref.cat.categories.equals(cur.cat.categories):
            raise EngineError("Distribution binner must use the same domain for both populations")
        domain = tuple(ref.cat.categories)
        metadata["transform"] = transformer.metadata()
    else:
        ref, cur = reference, current
        domain = category_domain(ref, cur)
    if binner is not None:
        a = ref.cat.codes.to_numpy(dtype=np.intp, copy=True)
        b = cur.cat.codes.to_numpy(dtype=np.intp, copy=True)
    else:
        a = pd.Categorical(ref, categories=list(domain)).codes.astype(np.intp)
        b = pd.Categorical(cur, categories=list(domain)).codes.astype(np.intp)
    bins = len(domain)
    # An integer code prevents collision with real strings such as '__MISSING__'.
    if missing == "bucket" and ((a < 0).any() or (b < 0).any()):
        a[a < 0] = bins
        b[b < 0] = bins
        metadata["missing_bucket"] = bins
        bins += 1
    metadata["labels"] = domain
    metadata["n_bins"] = bins
    return a, b, bins, metadata
