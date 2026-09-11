from dataclasses import dataclass

from phl_risk._data import freeze


@dataclass(frozen=True)
class PrepSnapshot:
    row_count: int
    columns: tuple
    dtypes: object
    missing_count: object
    missing_rate: object
    nunique: object

    def __post_init__(self):
        for key in ("columns", "dtypes", "missing_count", "missing_rate", "nunique"):
            object.__setattr__(self, key, freeze(getattr(self, key)))

    @classmethod
    def from_frame(cls, X, columns):
        columns = tuple(columns)
        return cls(
            len(X),
            columns,
            {c: str(X[c].dtype) for c in columns},
            {c: int(X[c].isna().sum()) for c in columns},
            {c: float(X[c].isna().mean()) for c in columns},
            {c: int(X[c].nunique()) for c in columns},
        )
