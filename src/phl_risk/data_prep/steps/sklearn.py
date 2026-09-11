"""Public sklearn adapter: clone, names, sparse output and explicit weight routing."""

import inspect

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone

from phl_risk._data import columns_of
from phl_risk.exceptions import DataPrepError

from .._base import BasePrepStep


def merge_output(X, values, inputs, names, output):
    if output not in ("replace", "append"):
        raise DataPrepError("output must be replace/append")
    names = tuple(names)
    if len(set(names)) != len(names) or any(not isinstance(n, str) for n in names):
        raise DataPrepError("Output column names must be unique strings")
    untouched = tuple(c for c in X if c not in inputs) if output == "replace" else tuple(X.columns)
    if set(names) & set(untouched):
        raise DataPrepError("Generated output names collide with preserved columns")
    if isinstance(values, pd.DataFrame):
        if not values.index.equals(X.index):
            raise DataPrepError("Backend changed the row index/order")
        transformed = values.copy()
        transformed.columns = list(names)
    elif sparse.issparse(values):
        transformed = pd.DataFrame.sparse.from_spmatrix(values, index=X.index, columns=list(names))
    else:
        array = np.asarray(values)
        if array.ndim != 2 or array.shape != (len(X), len(names)):
            raise DataPrepError("Backend output shape disagrees with row count or feature names")
        transformed = pd.DataFrame(array, index=X.index, columns=list(names))
    if output == "replace" and names == tuple(inputs):
        result = X.copy()
        for c in names:
            result[c] = transformed[c]
        return result
    pieces = []
    inserted = False
    for c in X:
        if output == "replace" and c in inputs:
            if not inserted:
                pieces.append(transformed)
                inserted = True
        else:
            pieces.append(X[[c]])
    if output == "append":
        pieces.append(transformed)
    return pd.concat(pieces, axis=1) if pieces else transformed


class SklearnStep(BasePrepStep):
    def __init__(self, transformer, columns, output="replace", output_columns=None):
        self.transformer = transformer
        self.columns = columns
        self.output = output
        self.output_columns = output_columns

    def _make_transformer(self):
        return clone(self.transformer)

    def _fit(self, X, y=None, sample_weight=None):
        self.columns_ = columns_of(X, self.columns, DataPrepError)
        self.transformer_ = self._make_transformer()
        # Sparse encoders retain sparsity; pandas set_output rejects their sparse result.
        params = self.transformer_.get_params(deep=False)
        if (
            hasattr(self.transformer_, "set_output")
            and not params.get("sparse_output", False)
            and params.get("encode") != "onehot"
        ):
            self.transformer_.set_output(transform="pandas")
        fit_kwargs = {}
        if sample_weight is not None:
            signature = inspect.signature(self.transformer_.fit)
            if "sample_weight" not in signature.parameters:
                raise DataPrepError(
                    f"{type(self.transformer_).__name__} does not accept sample_weight; "
                    "use an explicitly weighted adapter or omit weights"
                )
            fit_kwargs["sample_weight"] = sample_weight
        self.transformer_.fit(X[list(self.columns_)].copy(), y, **fit_kwargs)
        probe = self.transformer_.transform(X[list(self.columns_)].copy())
        width = probe.shape[1]
        if self.output_columns is not None:
            names = tuple(self.output_columns)
        elif width == len(self.columns_) and self.output == "replace":
            names = self.columns_
        elif hasattr(self.transformer_, "get_feature_names_out"):
            names = tuple(self.transformer_.get_feature_names_out(self.columns_))
        elif isinstance(probe, pd.DataFrame):
            names = tuple(probe.columns)
        else:
            raise DataPrepError("Backend must expose output names or supply output_columns")
        if len(names) != width:
            raise DataPrepError("output_columns length differs from backend output")
        self.output_columns_ = names

    def _transform(self, X):
        values = self.transformer_.transform(X[list(self.columns_)].copy())
        return merge_output(X, values, self.columns_, self.output_columns_, self.output)

    def _audit_details(self, X, output):
        return dict(
            backend="sklearn",
            transformer=type(self.transformer_).__name__,
            input_features=self.columns_,
            output_features=self.output_columns_,
        )
