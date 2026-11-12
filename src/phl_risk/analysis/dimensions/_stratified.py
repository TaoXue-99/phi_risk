"""Independent transformer state for raw-column strata; no Cube orchestration."""

from copy import deepcopy

import numpy as np
import pandas as pd

from phl_risk.analysis._axis import is_missing
from phl_risk.exceptions import DimensionError, TransformError

from ._column import ColumnDimension, column_series


def validate_stratification(dimensions):
    raw_axes = {
        d.column for d in dimensions if isinstance(d, ColumnDimension) and d.output_name == d.column
    }
    for dimension in dimensions:
        if set(getattr(dimension, "fit_by", None) or ()) - raw_axes:
            raise DimensionError("fit_by fields must also be unrenamed ColumnDimensions in Cube")


def stratum_key(values: tuple) -> tuple:
    return tuple(None if is_missing(value) else value for value in values)


def group_positions(data: pd.DataFrame, fields: tuple[str, ...]):
    # Build positional groups: duplicate input index labels must never align rows.
    keys = pd.DataFrame({name: column_series(data, name).to_numpy() for name in fields})
    for key, positions in keys.groupby(
        list(fields), sort=False, observed=True, dropna=False
    ).indices.items():
        if len(fields) == 1:
            key = (key,)
        yield stratum_key(key), positions


def prepare_strata(data, dimensions, *, keep_missing: bool, return_positions: bool = False):
    """Apply dimension missing policy before attempting per-stratum lookup/fit."""
    fields = tuple(dict.fromkeys(f for d in dimensions for f in (getattr(d, "fit_by", None) or ())))
    if not return_positions and (not fields or keep_missing):
        return data
    positions = np.arange(len(data))
    if fields and not keep_missing:
        keep = np.ones(len(data), dtype=bool)
        for field in fields:
            keep &= column_series(data, field).notna().to_numpy()
        positions = positions[keep]
        data = data.iloc[positions]
    return (data, positions) if return_positions else data


class StratifiedTransform:
    def __init__(self, data, fields, fit_field, transformer, y=None):
        self.groups = {}
        for key, positions in group_positions(data, fields):
            fitted = deepcopy(transformer)
            # Supervised transformers receive the same row subset as X.
            target = y
            if y is not None:
                target = y.iloc[positions] if isinstance(y, pd.Series) else np.asarray(y)[positions]
            try:
                fitted.fit(column_series(data, fit_field).iloc[positions], target)
            except TransformError as exc:
                raise TransformError(f"Stratum {key!r}: {exc}") from exc
            self.groups[key] = fitted

    def transform(self, data, fields, column):
        source = column_series(data, column)
        output = np.empty(len(source), dtype=object)
        output[:] = None
        labels = []
        for fitted in self.groups.values():
            for label in fitted.metadata().get("labels", ()):
                if label not in labels:
                    labels.append(label)
        for key, positions in group_positions(data, fields):
            if key not in self.groups:
                raise DimensionError(
                    f"Unseen fit_by stratum {key!r}; fit on a reference containing it"
                )
            subset = source.iloc[positions]
            values = self.groups[key].transform(subset)
            if not isinstance(values, pd.Series) or not values.index.equals(subset.index):
                raise DimensionError("Stratified transformer must preserve Series index")
            output[positions] = values.to_numpy()
            for label in values.dropna().unique():
                if label not in labels:
                    labels.append(label)
        return pd.Series(
            pd.Categorical(output, categories=labels, ordered=True), index=source.index
        )

    def metadata(self):
        return tuple(
            {"key": key, "transform": deepcopy(fitted.metadata())}
            for key, fitted in self.groups.items()
        )
