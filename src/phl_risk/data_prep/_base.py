from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import TransformerMixin, clone

from phl_risk._data import (
    DataEstimator,
    aligned,
    check_fitted,
    columns_of,
    commit_fitted,
    frame,
)
from phl_risk.exceptions import DataPrepError

from ._audit import PrepAudit
from ._result import StepResult
from ._snapshot import PrepSnapshot


class BasePrepStep(DataEstimator, ABC):
    """Common row-preserving transformation and audit protocol; no fit lifecycle.

    Plugins implement _transform. Override audit_columns/_audit_details for
    generated features or step-specific audit information.
    """

    kind = None

    @abstractmethod
    def _transform(self, X):
        """Return a DataFrame, preserving row count, order and index."""

    def _checked_transform(self, X):
        frame(X, DataPrepError)
        output = self._transform(X.copy(deep=True))
        frame(output, DataPrepError)
        if len(output) != len(X) or not output.index.equals(X.index):
            raise DataPrepError("Prep steps must preserve row count, order, and index")
        return output

    def transform(self, X):
        return self._checked_transform(X)

    def audit_columns(self, X, output):
        columns = getattr(self, "columns", None)
        if columns is None and hasattr(self, "column"):
            columns = [self.column]
        inputs = columns_of(X, columns, DataPrepError)
        outputs = inputs
        return inputs, tuple(c for c in outputs if c in output)

    def _audit_details(self, X, output):
        return {}

    def run(self, X):
        output = self.transform(X)
        inputs, outputs = self.audit_columns(X, output)
        audit = PrepAudit(
            type(self).__name__,
            inputs,
            outputs,
            PrepSnapshot.from_frame(X, inputs),
            PrepSnapshot.from_frame(output, outputs),
            self._audit_details(X, output),
            kind=self.kind,
        )
        return StepResult(output, audit)

    def _check_artifact(self):
        """Adapter-specific persistence checks; no-op for ordinary steps."""


class StatelessPrepStep(BasePrepStep):
    """Configuration + transform only; no fit method or learned attributes.

    DataPrep clones configuration and owns the frozen plan/schema. Standalone
    instances can transform different schemas when their selected columns exist.
    """

    kind = "stateless"

    def fit_transform(self, X, y=None, sample_weight=None):
        """Convenience alias; y and sample_weight do not affect fixed rules."""
        return self.transform(X)


class FittedPrepStep(TransformerMixin, BasePrepStep, auto_wrap_output_keys=None):
    """Learn on a candidate and publish state only after a complete successful fit."""

    kind = "fitted"

    @abstractmethod
    def _fit(self, X, y=None, sample_weight=None):
        """Learn reference state in trailing-underscore attributes."""

    def fit(self, X, y=None, sample_weight=None):
        frame(X, DataPrepError)
        y = aligned(y, X, "y", DataPrepError)
        weight = aligned(sample_weight, X, "sample_weight", DataPrepError)
        candidate = clone(self)
        candidate.feature_names_in_ = tuple(X.columns)
        candidate.n_features_in_ = len(X.columns)
        candidate._fit(X.copy(deep=True), y, weight)
        output = candidate._checked_transform(X)
        candidate.feature_names_out_ = tuple(output.columns)
        return commit_fitted(self, candidate)

    def transform(self, X):
        check_fitted(self)
        frame(X, DataPrepError)
        if tuple(X.columns) != self.feature_names_in_:
            raise DataPrepError("Input columns/order differ from fit reference")
        output = self._checked_transform(X)
        if tuple(output.columns) != self.feature_names_out_:
            raise DataPrepError("Transformer output schema changed after fit")
        return output

    def fit_transform(self, X, y=None, sample_weight=None, **fit_params):
        if fit_params:
            raise DataPrepError(f"Unsupported fit parameters: {tuple(fit_params)}")
        return self.fit(X, y, sample_weight).transform(X)

    def get_feature_names_out(self, input_features=None):
        check_fitted(self)
        if input_features is not None and tuple(input_features) != self.feature_names_in_:
            raise DataPrepError("input_features differ from fit reference")
        return np.asarray(self.feature_names_out_, dtype=object)

    def audit_columns(self, X, output):
        columns = getattr(self, "columns_", getattr(self, "columns", None))
        if columns is None and hasattr(self, "column"):
            columns = [self.column]
        inputs = columns_of(X, columns, DataPrepError)
        outputs = getattr(self, "output_columns_", inputs)
        return inputs, tuple(c for c in outputs if c in output)
