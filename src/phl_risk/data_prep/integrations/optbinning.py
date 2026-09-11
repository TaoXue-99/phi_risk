"""OptBinning stays behind this adapter and is imported only when fitting."""

from copy import deepcopy
from importlib.metadata import version

import numpy as np
import pandas as pd

from phl_risk._data import check_fitted, columns_of
from phl_risk.exceptions import ArtifactError, DataPrepError, OptionalDependencyError

from .._base import BasePrepStep
from ..steps.sklearn import merge_output


def _require_optbinning():
    try:
        from optbinning import BinningProcess
    except ImportError as exc:
        raise OptionalDependencyError(
            "OptBinning support requires `pip install phl-risk[binning]`."
        ) from exc
    return BinningProcess


class OptBinningStep(BasePrepStep):
    def __init__(
        self,
        columns,
        categorical_columns=None,
        metric="woe",
        special_codes=None,
        max_n_prebins=20,
        min_prebin_size=0.05,
        min_n_bins=None,
        max_n_bins=None,
        min_bin_size=None,
        max_bin_size=None,
        max_pvalue=None,
        max_pvalue_policy="consecutive",
        selection_criteria=None,
        binning_fit_params=None,
        binning_transform_params=None,
        metric_special=0,
        metric_missing=0,
        n_jobs=None,
        output="replace",
        binning_process_kwargs=None,
    ):
        self.columns = columns
        self.categorical_columns = categorical_columns
        self.metric = metric
        self.special_codes = special_codes
        self.max_n_prebins = max_n_prebins
        self.min_prebin_size = min_prebin_size
        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size
        self.max_pvalue = max_pvalue
        self.max_pvalue_policy = max_pvalue_policy
        self.selection_criteria = selection_criteria
        self.binning_fit_params = binning_fit_params
        self.binning_transform_params = binning_transform_params
        self.metric_special = metric_special
        self.metric_missing = metric_missing
        self.n_jobs = n_jobs
        self.output = output
        self.binning_process_kwargs = binning_process_kwargs

    def _fit(self, X, y=None, sample_weight=None):
        if y is None:
            raise DataPrepError("OptBinningStep.fit requires a binary target y")
        if not np.isin(np.asarray(y), [0, 1]).all() or len(np.unique(y)) != 2:
            raise DataPrepError("OptBinningStep requires both binary target classes 0 and 1")
        if self.metric not in ("woe", "event_rate", "indices", "bins"):
            raise DataPrepError("metric must be woe/event_rate/indices/bins")
        self.columns_ = columns_of(X, self.columns, DataPrepError)
        if self.categorical_columns is not None and set(self.categorical_columns) - set(
            self.columns_
        ):
            raise DataPrepError("categorical_columns must be a subset of columns")
        for params in (self.binning_fit_params, self.binning_transform_params):
            if params is not None and set(params) - set(self.columns_):
                raise DataPrepError("Per-variable parameters contain unknown columns")
        keys = (
            "max_n_prebins",
            "min_prebin_size",
            "min_n_bins",
            "max_n_bins",
            "min_bin_size",
            "max_bin_size",
            "max_pvalue",
            "max_pvalue_policy",
            "selection_criteria",
            "special_codes",
            "binning_fit_params",
            "binning_transform_params",
            "n_jobs",
        )
        kwargs = deepcopy(self.binning_process_kwargs or {})
        kwargs.update({key: deepcopy(getattr(self, key)) for key in keys})
        kwargs.update(
            variable_names=list(self.columns_),
            categorical_variables=deepcopy(self.categorical_columns),
        )
        if sample_weight is not None:
            for name, params in (self.binning_fit_params or {}).items():
                if params.get("prebinning_method", "cart") != "cart":
                    raise DataPrepError(
                        f"OptBinning weights require cart prebinning; {name!r} uses "
                        f"{params['prebinning_method']!r}"
                    )
        self.binning_process_ = _require_optbinning()(**kwargs)
        self.binning_process_.fit(X[list(self.columns_)].copy(), y, sample_weight=sample_weight)
        self.selected_columns_ = tuple(self.binning_process_.get_support(names=True))
        self.output_columns_ = (
            self.selected_columns_
            if self.output == "replace"
            else tuple(f"{c}__{self.metric}" for c in self.selected_columns_)
        )
        self.backend_version_ = version("optbinning")

    def _transform(self, X):
        if self.selected_columns_:
            values = self.binning_process_.transform(
                X[list(self.columns_)].copy(),
                metric=self.metric,
                metric_special=self.metric_special,
                metric_missing=self.metric_missing,
            )
        else:
            values = pd.DataFrame(index=X.index)
        return merge_output(X, values, self.columns_, self.output_columns_, self.output)

    def summary(self):
        check_fitted(self)
        return self.binning_process_.summary().copy(deep=True)

    def get_binned_variable(self, column):
        check_fitted(self)
        # Return the native class with its full API without exposing mutable fitted state.
        return deepcopy(self.binning_process_.get_binned_variable(column))

    def binning_table(self, column):
        return self.get_binned_variable(column).binning_table.build().copy(deep=True)

    def _audit_details(self, X, output):
        details = dict(
            backend="optbinning",
            backend_version=self.backend_version_,
            metric=self.metric,
            selected_columns=self.selected_columns_,
            output_columns=self.output_columns_,
        )
        for c in self.columns_:
            optb = self.binning_process_.get_binned_variable(c)
            codes = optb.special_codes
            if isinstance(codes, dict):
                codes = [v for values in codes.values() for v in values]
            special = X[c].isin([] if codes is None else codes)
            info = dict(
                input_missing=int(X[c].isna().sum()),
                special_count=int(special.sum()),
                solver_status=optb.status,
            )
            if c in self.selected_columns_:
                index = self.selected_columns_.index(c)
                out = output[self.output_columns_[index]]
                indices = optb.transform(
                    X[c], metric="indices", metric_missing="empirical", metric_special="empirical"
                )
                info.update(
                    output_missing=int(out.isna().sum()),
                    bin_counts=pd.Series(indices).value_counts().to_dict(),
                )
                if pd.api.types.is_numeric_dtype(out):
                    info.update(output_min=out.min(), output_max=out.max())
            details[c] = info
        return details

    def _check_artifact(self):
        check_fitted(self)
        for c in self.columns_:
            if self.binning_process_.get_binned_variable(c).solver == "mip":
                raise ArtifactError(
                    "OptBinning upstream does not recommend saving BinningProcess fitted "
                    'with solver="mip"; refit using solver="cp" before saving.'
                )
