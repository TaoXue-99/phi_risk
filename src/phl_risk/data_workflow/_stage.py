from abc import ABC, abstractmethod

from sklearn.base import clone

from phl_risk._data import DataEstimator, check_fitted
from phl_risk.data_prep import DataPrep
from phl_risk.data_quality import DataQuality
from phl_risk.exceptions import DataWorkflowError

from ._policy import enforce, failure_policy
from ._result import StageResult


class BaseWorkflowStage(DataEstimator, ABC):
    @abstractmethod
    def fit_apply(self, X, y=None, sample_weight=None): ...

    @abstractmethod
    def apply(self, X): ...

    @abstractmethod
    def run(self, X): ...

    def _check_artifact(self):
        """Optional stage persistence hook."""


class QualityStage(BaseWorkflowStage):
    def __init__(self, name, quality, on_fail="raise"):
        self.name = name
        self.quality = quality
        self.on_fail = on_fail

    def fit_apply(self, X, y=None, sample_weight=None):
        failure_policy(self.on_fail)
        if not isinstance(self.quality, DataQuality):
            raise DataWorkflowError("quality must be DataQuality")
        fitted = clone(self.quality).fit(X)
        report = fitted.validate(X)
        enforce(report, self.on_fail, self.name)
        self.quality_ = fitted
        self._is_fitted_ = True
        return X.copy(deep=True)

    def apply(self, X):
        check_fitted(self)
        report = self.quality_.validate(X)
        enforce(report, self.on_fail, self.name)
        return X.copy(deep=True)

    def run(self, X):
        check_fitted(self)
        report = self.quality_.validate(X)
        enforce(report, self.on_fail, self.name)
        return StageResult(self.name, "quality", X, quality_report=report)


class PrepStage(BaseWorkflowStage):
    def __init__(self, name, prep):
        self.name = name
        self.prep = prep

    def fit_apply(self, X, y=None, sample_weight=None):
        if not isinstance(self.prep, DataPrep):
            raise DataWorkflowError("prep must be DataPrep")
        fitted = clone(self.prep)
        output = fitted.fit_transform(X, y, sample_weight)
        self.prep_ = fitted
        self._is_fitted_ = True
        return output

    def apply(self, X):
        check_fitted(self)
        return self.prep_.transform(X)

    def run(self, X):
        check_fitted(self)
        result = self.prep_.run(X)
        return StageResult(self.name, "prep", result.data, prep_audits=result.audits)

    def _check_artifact(self):
        self.prep_._check_artifact()
