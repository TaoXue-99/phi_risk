from dataclasses import replace

import numpy as np
from sklearn.base import TransformerMixin

from phl_risk._data import (
    DataEstimator,
    aligned,
    check_fitted,
    commit_fitted,
    frame,
    named_estimators,
)
from phl_risk.exceptions import DataPrepError

from ._base import BasePrepStep, FittedPrepStep
from ._result import PrepResult


class DataPrep(TransformerMixin, DataEstimator, auto_wrap_output_keys=None):
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None, sample_weight=None):
        frame(X, DataPrepError)
        y = aligned(y, X, "y", DataPrepError)
        weight = aligned(sample_weight, X, "sample_weight", DataPrepError)
        candidate = type(self)(self.steps)
        candidate.steps_ = named_estimators(self.steps, BasePrepStep, DataPrepError)
        current = X.copy(deep=True)
        schemas = []
        for _, step in candidate.steps_:
            if isinstance(step, FittedPrepStep):
                current = step.fit_transform(current, y, weight)
            else:
                current = step.transform(current)
            schemas.append(tuple(current.columns))
        candidate.step_feature_names_out_ = tuple(schemas)
        candidate.feature_names_in_ = tuple(X.columns)
        candidate.n_features_in_ = len(X.columns)
        candidate.feature_names_out_ = tuple(current.columns)
        return commit_fitted(self, candidate)

    def _input(self, X):
        check_fitted(self)
        frame(X, DataPrepError)
        if not hasattr(self, "step_feature_names_out_"):
            raise DataPrepError("Legacy DataPrep plan: refit to record per-step output schemas")
        if tuple(X.columns) != self.feature_names_in_:
            raise DataPrepError("Input columns/order differ from fit reference")
        return X.copy(deep=True)

    @staticmethod
    def _check_output(X, schema, name):
        if tuple(X.columns) != schema:
            raise DataPrepError(f"Step {name!r} output columns/order differ from fit reference")

    def transform(self, X):
        current = self._input(X)
        for (name, step), schema in zip(self.steps_, self.step_feature_names_out_, strict=True):
            current = step.transform(current)
            self._check_output(current, schema, name)
        return current

    def run(self, X):
        current, audits = self._input(X), []
        for (name, step), schema in zip(self.steps_, self.step_feature_names_out_, strict=True):
            result = step.run(current)
            current = result.data
            self._check_output(current, schema, name)
            audits.append(replace(result.audit, step=name))
        return PrepResult(current, audits)

    def fit_transform(self, X, y=None, sample_weight=None, **fit_params):
        if fit_params:
            raise DataPrepError(f"Unsupported fit parameters: {tuple(fit_params)}")
        return self.fit(X, y, sample_weight).transform(X)

    def get_feature_names_out(self, input_features=None):
        check_fitted(self)
        if input_features is not None and tuple(input_features) != self.feature_names_in_:
            raise DataPrepError("input_features differ from fit reference")
        return np.asarray(self.feature_names_out_, dtype=object)

    def _check_artifact(self):
        for _, step in self.steps_:
            step._check_artifact()
