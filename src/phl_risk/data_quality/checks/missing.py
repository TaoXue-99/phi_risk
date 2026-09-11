from .._base import BaseQualityCheck
from .._status import CheckStatus
from .._utils import limits, result, selected, severity, threshold


class MissingRateCheck(BaseQualityCheck):
    def __init__(self, columns=None, max_rate=None, warn_delta=None, fail_delta=None):
        self.columns = columns
        self.max_rate = max_rate
        self.warn_delta = warn_delta
        self.fail_delta = fail_delta

    def _fit(self, X, y=None):
        threshold(self.max_rate, "max_rate", 1)
        limits(self.warn_delta, self.fail_delta, 1)
        self.columns_ = selected(X, self.columns)
        self.reference_rates_ = {c: float(X[c].isna().mean()) for c in self.columns_}

    def _validate(self, X):
        selected(X, self.columns_)
        details, statuses = {}, []
        for c in self.columns_:
            rate, ref = float(X[c].isna().mean()), self.reference_rates_[c]
            delta = rate - ref
            details[c] = dict(reference_rate=ref, current_rate=rate, delta=delta)
            statuses.extend(
                [
                    severity(rate, fail=self.max_rate),
                    severity(delta, self.warn_delta, self.fail_delta),
                ]
                if len(X)
                else [CheckStatus.SKIP]
            )
        return result(self, self.columns_, details, statuses)


class MissingLikeCheck(BaseQualityCheck):
    requires_fit = False

    def __init__(
        self, columns, values=None, strip_strings=True, case_insensitive=True, max_rate=None
    ):
        self.columns = columns
        self.values = values
        self.strip_strings = strip_strings
        self.case_insensitive = case_insensitive
        self.max_rate = max_rate

    def _normalize(self, value):
        if isinstance(value, str):
            value = value.strip() if self.strip_strings else value
            value = value.casefold() if self.case_insensitive else value
        return value

    def _validate(self, X):
        threshold(self.max_rate, "max_rate", 1)
        columns = selected(X, self.columns)
        values = ("", "null", "none", "nan", "n/a") if self.values is None else self.values
        values = [self._normalize(v) for v in values]
        details, statuses = {}, []
        for c in columns:
            s = X[c]
            mask = s.notna() & s.map(self._normalize).isin(values)
            rate = float(mask.mean())
            details[c] = dict(missing_like_count=int(mask.sum()), missing_like_rate=rate)
            statuses.append(
                severity(rate, fail=0 if self.max_rate is None else self.max_rate)
                if len(X)
                else CheckStatus.SKIP
            )
        return result(self, columns, details, statuses)
