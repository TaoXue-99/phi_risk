from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from phl_risk.exceptions import DataQualityError
from phl_risk.metrics._psi import psi_score

from .._base import BaseQualityCheck
from .._status import CheckStatus
from .._utils import limits, result, selected, severity


class DriftMetric(Protocol):
    def fit(self, reference: pd.Series, bins: int) -> object: ...
    def score(self, current: pd.Series, state: object) -> tuple[float, dict]: ...


@dataclass(frozen=True)
class PSIState:
    numeric: bool
    domain: tuple
    counts: tuple


class PSIMetric:
    def _counts(self, series, numeric, domain):
        if numeric:
            try:
                values = pd.to_numeric(series, errors="raise").to_numpy(
                    dtype=float, na_value=np.nan
                )
            except (ValueError, TypeError) as exc:
                raise DataQualityError(
                    "Numeric PSI requires numeric-convertible current data"
                ) from exc
            codes = np.searchsorted(domain[1:-1], values, side="left")
            codes[np.isnan(values)] = len(domain) - 1
            return np.bincount(codes, minlength=len(domain))
        # Two separate buckets for unseen categories and true missing values.
        codes = pd.Index(domain).get_indexer(series)
        codes[codes == -1] = len(domain)
        codes[series.isna()] = len(domain) + 1
        return np.bincount(codes, minlength=len(domain) + 2)

    def fit(self, reference, bins):
        if not len(reference):
            raise DataQualityError("PSI reference must not be empty")
        numeric = pd.api.types.is_numeric_dtype(reference)
        if numeric:
            values = reference.to_numpy(dtype=float, na_value=np.nan)
            finite = values[np.isfinite(values)]
            inner = (
                np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1]))
                if len(finite)
                else np.array([])
            )
            if len(finite) and finite.min() == finite.max():
                inner = np.array([np.nextafter(finite[0], -np.inf), finite[0]])
            domain = tuple(np.r_[-np.inf, inner, np.inf])
        else:
            domain = tuple(reference.dropna().unique())
        return PSIState(numeric, domain, tuple(self._counts(reference, numeric, domain)))

    def score(self, current, state):
        counts = self._counts(current, state.numeric, state.domain)
        return psi_score(state.counts, counts), dict(
            reference_counts=state.counts, current_counts=tuple(counts), bins=state.domain
        )


DRIFT_METRICS: dict[str, DriftMetric] = {"psi": PSIMetric()}


class DistributionDriftCheck(BaseQualityCheck):
    def __init__(self, columns, method="psi", bins=10, warn_threshold=0.10, fail_threshold=0.25):
        self.columns = columns
        self.method = method
        self.bins = bins
        self.warn_threshold = warn_threshold
        self.fail_threshold = fail_threshold

    def _fit(self, X, y=None):
        if self.method not in DRIFT_METRICS:
            raise DataQualityError(f"Unknown drift method {self.method!r}")
        if isinstance(self.bins, bool) or not isinstance(self.bins, int) or self.bins < 2:
            raise DataQualityError("bins must be an integer >= 2")
        limits(self.warn_threshold, self.fail_threshold)
        from copy import deepcopy

        self.metric_ = deepcopy(DRIFT_METRICS[self.method])
        self.columns_ = selected(X, self.columns)
        self.reference_states_ = {c: self.metric_.fit(X[c], self.bins) for c in self.columns_}

    def _validate(self, X):
        selected(X, self.columns_)
        details, statuses = {}, []
        for c in self.columns_:
            if not len(X):
                details[c] = dict(value=None, reason="empty current population")
                statuses.append(CheckStatus.SKIP)
                continue
            score, extra = self.metric_.score(X[c], self.reference_states_[c])
            details[c] = dict(value=score, method=self.method, **extra)
            statuses.append(severity(score, self.warn_threshold, self.fail_threshold))
        return result(self, self.columns_, details, statuses)
