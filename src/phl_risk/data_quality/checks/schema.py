from phl_risk.exceptions import DataQualityError

from .._base import BaseQualityCheck
from .._status import CheckStatus
from .._utils import result, selected


class SchemaCheck(BaseQualityCheck):
    allows_duplicate_columns = True

    def __init__(self, columns=None, dtypes=None, allow_extra=False, check_order=True):
        self.columns = columns
        self.dtypes = dtypes
        self.allow_extra = allow_extra
        self.check_order = check_order

    def _fit(self, X, y=None):
        self.columns_ = selected(X, self.columns)
        self.dtypes_ = {c: str(X[c].dtype) for c in self.columns_}
        if self.dtypes is not None:
            if set(self.dtypes) - set(self.columns_):
                raise DataQualityError("dtypes contains columns outside the schema")
            self.dtypes_.update({c: str(t) for c, t in self.dtypes.items()})

    def _validate(self, X):
        missing = tuple(c for c in self.columns_ if c not in X)
        extra = tuple(c for c in X if c not in self.columns_)
        duplicate = tuple(X.columns[X.columns.duplicated()].unique())
        changes = {
            c: {"reference": self.dtypes_[c], "current": str(X[c].dtype)}
            for c in self.columns_
            if c in X and c not in duplicate and str(X[c].dtype) != self.dtypes_[c]
        }
        order = tuple(c for c in X if c in self.columns_) != tuple(
            c for c in self.columns_ if c in X
        )
        details = dict(
            missing_columns=missing,
            extra_columns=extra,
            dtype_changes=changes,
            order_changed=order,
            duplicate_columns=duplicate,
        )
        failed = missing or duplicate or changes or (extra and not self.allow_extra)
        failed = failed or (order and self.check_order)
        return result(
            self, self.columns_, details, [CheckStatus.FAIL if failed else CheckStatus.PASS]
        )
