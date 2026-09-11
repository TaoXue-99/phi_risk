from sklearn.base import clone

from phl_risk import __version__
from phl_risk._data import DataEstimator, aligned, check_fitted, commit_fitted, frame
from phl_risk.exceptions import DataWorkflowError

from ._result import WorkflowResult
from ._stage import BaseWorkflowStage


class DataWorkflow(DataEstimator):
    def __init__(self, stages):
        self.stages = stages

    def fit(self, X, y=None, sample_weight=None):
        frame(X, DataWorkflowError)
        y = aligned(y, X, "y", DataWorkflowError)
        weights = aligned(sample_weight, X, "sample_weight", DataWorkflowError)
        names, stages = set(), []
        for stage in self.stages:
            if not isinstance(stage, BaseWorkflowStage):
                raise DataWorkflowError("stages must contain BaseWorkflowStage instances")
            if not isinstance(stage.name, str) or not stage.name or stage.name in names:
                raise DataWorkflowError("Stage names must be nonempty and unique")
            names.add(stage.name)
            stages.append(clone(stage))
        current = X.copy(deep=True)
        for stage in stages:
            current = stage.fit_apply(current, y, weights)
            self._check_rows(X, current)
        candidate = type(self)(self.stages)
        candidate.stages_ = tuple(stages)
        candidate.feature_names_in_ = tuple(X.columns)
        candidate.feature_names_out_ = tuple(current.columns)
        candidate.n_features_in_ = len(X.columns)
        return commit_fitted(self, candidate)

    @staticmethod
    def _check_rows(X, current):
        frame(current, DataWorkflowError)
        if len(current) != len(X) or not current.index.equals(X.index):
            raise DataWorkflowError("Workflow stages must preserve rows and index")

    def transform(self, X):
        check_fitted(self)
        frame(X, DataWorkflowError, unique=False)
        current = X.copy(deep=True)
        for stage in self.stages_:
            current = stage.apply(current)
            self._check_rows(X, current)
        return current

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit(X, y, sample_weight).transform(X)

    def run(self, X):
        check_fitted(self)
        frame(X, DataWorkflowError, unique=False)
        current, results = X.copy(deep=True), []
        for stage in self.stages_:
            result = stage.run(current)
            current = result.data
            self._check_rows(X, current)
            results.append(result)
        return WorkflowResult(
            current,
            results,
            dict(
                phl_risk_version=__version__,
                input_rows=len(X),
                input_columns=tuple(X.columns),
                output_rows=len(current),
                output_columns=tuple(current.columns),
            ),
        )

    def _check_artifact(self):
        for stage in self.stages_:
            stage._check_artifact()
