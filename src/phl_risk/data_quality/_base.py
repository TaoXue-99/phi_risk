from abc import ABC, abstractmethod

from sklearn.base import clone

from phl_risk._data import DataEstimator, check_fitted, commit_fitted, frame
from phl_risk.exceptions import DataQualityError

from ._result import CheckResult


class BaseQualityCheck(DataEstimator, ABC):
    requires_fit = True
    allows_duplicate_columns = False

    def _fit(self, X, y=None):
        """Stateless checks can inherit this no-op hook."""

    @abstractmethod
    def _validate(self, X):
        """Return one CheckResult; never update reference state."""

    def fit(self, X, y=None):
        frame(X, DataQualityError)
        candidate = clone(self)
        candidate.feature_names_in_ = tuple(X.columns)
        candidate.n_features_in_ = len(X.columns)
        candidate._fit(X.copy(deep=True), y)
        return commit_fitted(self, candidate)

    def validate(self, X):
        frame(X, DataQualityError, unique=not self.allows_duplicate_columns)
        if self.requires_fit:
            check_fitted(self)
        result = self._validate(X.copy(deep=True))
        if not isinstance(result, CheckResult):
            raise DataQualityError("_validate must return CheckResult")
        return result

    def fit_validate(self, X, y=None):
        return self.fit(X, y).validate(X)
