from phl_risk._data import (
    DataEstimator,
    check_fitted,
    commit_fitted,
    frame,
    named_estimators,
)
from phl_risk.exceptions import DataQualityError

from ._base import BaseQualityCheck
from ._profile import DataProfile
from ._result import QualityReport


class DataQuality(DataEstimator):
    def __init__(self, checks):
        self.checks = checks

    def fit(self, X, y=None):
        frame(X, DataQualityError)
        candidate = type(self)(self.checks)
        checks = named_estimators(self.checks, BaseQualityCheck, DataQualityError)
        candidate.reference_profile_ = DataProfile.from_frame(X)
        for _, check in checks:
            check.fit(X, y)
        candidate.checks_ = tuple(check for _, check in checks)
        candidate.check_names_ = tuple(name for name, _ in checks)
        candidate.feature_names_in_ = tuple(X.columns)
        candidate.n_features_in_ = len(X.columns)
        return commit_fitted(self, candidate)

    def validate(self, X):
        check_fitted(self)
        frame(X, DataQualityError, unique=False)
        return QualityReport(tuple(check.validate(X) for check in self.checks_))

    def fit_validate(self, X, y=None):
        return self.fit(X, y).validate(X)
