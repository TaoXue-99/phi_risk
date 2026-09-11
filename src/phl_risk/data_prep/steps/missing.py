import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .sklearn import SklearnStep


class MissingImputer(SklearnStep):
    def __init__(
        self,
        columns,
        strategy="median",
        fill_value=None,
        missing_values=np.nan,
        add_indicator=False,
        keep_empty_features=True,
        output="replace",
        output_columns=None,
    ):
        self.columns = columns
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values
        self.add_indicator = add_indicator
        self.keep_empty_features = keep_empty_features
        self.output = output
        self.output_columns = output_columns

    def _make_transformer(self):
        return SimpleImputer(
            strategy=self.strategy,
            fill_value=self.fill_value,
            missing_values=self.missing_values,
            add_indicator=self.add_indicator,
            keep_empty_features=self.keep_empty_features,
        )

    def _fit(self, X, y=None, sample_weight=None):
        # SimpleImputer has no weighted fit. Imputation is explicitly unweighted.
        super()._fit(X, y, None)
        self.imputer_ = self.transformer_

    def _audit_details(self, X, output):
        details = super()._audit_details(X, output)
        details["statistics_"] = self.imputer_.statistics_
        details["weight_policy"] = "unweighted (SimpleImputer)"
        native_names = tuple(self.imputer_.get_feature_names_out(self.columns_))
        for c in self.columns_:
            source = X[c]
            mask = (
                source.isna()
                if pd.isna(self.missing_values)
                else source.eq(self.missing_values).fillna(False)
            )
            target = self.output_columns_[native_names.index(c)] if c in native_names else None
            after = int(output[target].isna().sum()) if target is not None else None
            count = int((mask & output[target].notna()).sum()) if target is not None else 0
            details[c] = dict(
                missing_before=int(mask.sum()),
                missing_after=after,
                imputed_count=count,
                imputed_rate=count / len(X) if len(X) else None,
                output_column=target,
            )
        return details
