from .._base import BaseQualityCheck
from .._status import CheckStatus
from .._utils import limits, relative, result, selected, severity, threshold


class CategorySetCheck(BaseQualityCheck):
    def __init__(self, column, values=None, check_new=True, check_missing=True):
        self.column = column
        self.values = values
        self.check_new = check_new
        self.check_missing = check_missing

    def _fit(self, X, y=None):
        selected(X, [self.column])
        self.categories_ = tuple(
            X[self.column].dropna().unique() if self.values is None else dict.fromkeys(self.values)
        )

    def _validate(self, X):
        selected(X, [self.column])
        current = tuple(X[self.column].dropna().unique())
        new = tuple(v for v in current if v not in self.categories_)
        missing = tuple(v for v in self.categories_ if v not in current)
        failed = (self.check_new and new) or (self.check_missing and missing)
        return result(
            self,
            [self.column],
            dict(new_categories=new, missing_categories=missing),
            [CheckStatus.FAIL if failed else CheckStatus.PASS],
        )


class CardinalityCheck(BaseQualityCheck):
    def __init__(self, columns, warn_change=None, fail_change=None):
        self.columns = columns
        self.warn_change = warn_change
        self.fail_change = fail_change

    def _fit(self, X, y=None):
        limits(self.warn_change, self.fail_change)
        self.columns_ = selected(X, self.columns)
        self.reference_counts_ = {c: int(X[c].nunique()) for c in self.columns_}

    def _validate(self, X):
        selected(X, self.columns_)
        details, statuses = {}, []
        for c in self.columns_:
            count, ref = int(X[c].nunique()), self.reference_counts_[c]
            delta = relative(count, ref)
            details[c] = dict(reference=ref, current=count, relative_change=delta)
            statuses.append(severity(abs(delta), self.warn_change, self.fail_change))
        return result(self, self.columns_, details, statuses)


class ConstantCheck(BaseQualityCheck):
    requires_fit = False

    def __init__(self, columns=None, max_dominant_rate=1.0):
        self.columns = columns
        self.max_dominant_rate = max_dominant_rate

    def _validate(self, X):
        threshold(self.max_dominant_rate, "max_dominant_rate", 1)
        columns, details, statuses = selected(X, self.columns), {}, []
        for c in columns:
            counts = X[c].value_counts(dropna=False)
            rate = float(counts.iloc[0] / len(X)) if len(X) else float("nan")
            details[c] = dict(dominant_rate=rate, constant=len(counts) == 1)
            statuses.append(
                (CheckStatus.FAIL if rate >= self.max_dominant_rate else CheckStatus.PASS)
                if len(X)
                else CheckStatus.SKIP
            )
        return result(self, columns, details, statuses)
