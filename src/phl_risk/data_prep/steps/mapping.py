from phl_risk._data import columns_of
from phl_risk.exceptions import DataPrepError

from .._base import StatelessPrepStep


class ValueMapper(StatelessPrepStep):
    def __init__(self, column, mapping, handle_unknown="error", unknown_value=None):
        self.column = column
        self.mapping = mapping
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value

    def _transform(self, X):
        columns_of(X, [self.column], DataPrepError)
        if self.handle_unknown not in ("error", "keep", "value"):
            raise DataPrepError("handle_unknown must be error/keep/value")
        mapping = self.mapping
        s = X[self.column]
        known = s.isin(mapping)
        unknown = s.notna() & ~known
        if self.handle_unknown == "error" and unknown.any():
            raise DataPrepError(
                f"Unknown mapping values in {self.column!r}: {s[unknown].head(10).tolist()}"
            )
        out = s.map(mapping).astype(object)
        if self.handle_unknown == "keep":
            out.loc[unknown] = s.loc[unknown].astype(object)
        elif self.handle_unknown == "value":
            out.loc[unknown] = self.unknown_value
        X[self.column] = out.infer_objects()
        return X

    def _audit_details(self, X, output):
        s = X[self.column]
        known = s.isin(self.mapping)
        unknown = s.notna() & ~known
        return dict(
            mapped_count=int(known.sum()),
            unknown_count=int(unknown.sum()),
            mapping_coverage=float(known.mean()),
            unknown_rate=float(unknown.mean()),
            unknown_examples=s[unknown].head(10).tolist(),
        )
