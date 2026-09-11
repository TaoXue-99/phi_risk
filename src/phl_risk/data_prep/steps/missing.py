import numpy as np
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
        for c in self.columns_:
            source = X[c]
            if (
                self.missing_values is None
                or isinstance(self.missing_values, float)
                and np.isnan(self.missing_values)
            ):
                mask = source.isna()
            else:
                mask = source.eq(self.missing_values).fillna(False)
            after = int(output[c].isna().sum()) if c in output else None
            details[c] = dict(
                missing_before=int(mask.sum()),
                missing_after=after,
                imputed_count=int((mask & output[c].notna()).sum()) if c in output else 0,
                imputed_rate=float(mask.mean()),
            )
        return details
