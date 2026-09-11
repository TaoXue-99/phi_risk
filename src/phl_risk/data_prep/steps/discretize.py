from sklearn.preprocessing import KBinsDiscretizer

from .sklearn import SklearnStep


class KBinsStep(SklearnStep):
    def __init__(
        self,
        columns,
        n_bins=10,
        strategy="quantile",
        encode="ordinal",
        quantile_method="averaged_inverted_cdf",
        subsample=200000,
        random_state=0,
        output="replace",
        output_columns=None,
    ):
        self.columns = columns
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode
        self.quantile_method = quantile_method
        self.subsample = subsample
        self.random_state = random_state
        self.output = output
        self.output_columns = output_columns

    def _make_transformer(self):
        return KBinsDiscretizer(
            n_bins=self.n_bins,
            strategy=self.strategy,
            encode=self.encode,
            quantile_method=self.quantile_method,
            subsample=self.subsample,
            random_state=self.random_state,
        )
