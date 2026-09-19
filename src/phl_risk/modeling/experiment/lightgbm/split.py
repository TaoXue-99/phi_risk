"""Execute existing split declarations with stable SHA256 group assignment."""

import hashlib
import json
import math

import numpy as np
import pandas as pd

from phl_risk.exceptions import ExperimentError
from phl_risk.modeling.plan import ColumnSplitter, HashSplitter, SplitSpec


def assign_partitions(data: pd.DataFrame, split: SplitSpec) -> pd.Series:
    splitter = split.splitter
    definitions = split.partitions.definitions
    if isinstance(splitter, ColumnSplitter):
        if splitter.column not in data:
            raise ExperimentError(f"Missing split column: {splitter.column}")
        result = data[splitter.column].map({value: key for key, value in definitions.items()})
        if result.isna().any():
            raise ExperimentError("ColumnSplitter has missing or undeclared partition values")
        return result.rename("partition")
    if not isinstance(splitter, HashSplitter):
        raise ExperimentError(
            f"{type(splitter).__name__} is declared in DataPlan, but LightGBM Experiment V1 "
            "has not implemented its runtime execution"
        )
    if any(key not in data for key in splitter.key):
        raise ExperimentError(f"Missing hash key columns: {splitter.key}")
    keys = data.loc[:, list(splitter.key)]
    if keys.isna().any().any():
        raise ExperimentError("HashSplitter key contains missing values")

    def encode(row):
        # Explicit type tags and structured tuples avoid separator/type collisions.
        parts = []
        for value in row:
            if isinstance(value, (bool, np.bool_)):
                parts.append(["bool", bool(value)])
            elif isinstance(value, (int, np.integer)):
                parts.append(["number", str(int(value))])
            elif isinstance(value, (float, np.floating)) and math.isfinite(value):
                parts.append(
                    ["number", str(int(value)) if value.is_integer() else repr(float(value))]
                )
            elif isinstance(value, str):
                parts.append(["str", value])
            else:
                raise ExperimentError("Hash keys must be finite numbers, booleans or strings")
        raw = json.dumps([splitter.random_state, parts], ensure_ascii=False, separators=(",", ":"))
        # Top 53 bits give an exactly representable number strictly below 1.
        return (int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest(), "big") >> 203) / 2**53

    values = np.array([encode(row) for row in keys.itertuples(index=False, name=None)])
    boundaries = np.cumsum(list(definitions.values()))
    boundaries[-1] = 1.0
    assigned = np.searchsorted(boundaries, values, side="right")
    return pd.Series(
        np.array(list(definitions), dtype=object)[assigned], index=data.index, name="partition"
    )
