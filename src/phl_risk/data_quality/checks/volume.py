from .._base import BaseQualityCheck
from .._status import CheckStatus
from .._utils import limits, relative, result, severity, threshold


class RowCountCheck(BaseQualityCheck):
    def __init__(self, min_rows=None, warn_relative_change=None, fail_relative_change=None):
        self.min_rows = min_rows
        self.warn_relative_change = warn_relative_change
        self.fail_relative_change = fail_relative_change

    def _fit(self, X, y=None):
        threshold(self.min_rows, "min_rows")
        limits(self.warn_relative_change, self.fail_relative_change)
        self.reference_rows_ = len(X)

    def _validate(self, X):
        delta = relative(len(X), self.reference_rows_)
        statuses = [severity(abs(delta), self.warn_relative_change, self.fail_relative_change)]
        if self.min_rows is not None and len(X) < self.min_rows:
            statuses.append(CheckStatus.FAIL)
        return result(
            self,
            (),
            dict(
                reference=self.reference_rows_,
                current=len(X),
                absolute_delta=len(X) - self.reference_rows_,
                relative_delta=delta,
            ),
            statuses,
        )
