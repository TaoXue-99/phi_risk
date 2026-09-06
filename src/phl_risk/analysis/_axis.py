from dataclasses import dataclass
from typing import Hashable, Literal

import numpy as np
import pandas as pd

from phl_risk.exceptions import LayoutError


def is_missing(value: object) -> bool:
    return value is None or (pd.api.types.is_scalar(value) and bool(pd.isna(value)))


@dataclass(frozen=True)
class AxisSpec:
    """Ordered logical axis domain. None is the canonical missing coordinate."""

    name: str
    role: Literal["dimension", "metric"]
    values: tuple[Hashable | None, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise LayoutError("Axis name must be a non-empty string")
        if self.role not in ("dimension", "metric"):
            raise LayoutError("Axis role must be dimension or metric")
        normalized = tuple(None if is_missing(x) else x for x in self.values)
        try:
            if len(set(normalized)) != len(normalized):
                raise LayoutError(f"Axis {self.name!r} has duplicate coordinates")
        except TypeError as exc:
            raise LayoutError(f"Axis {self.name!r} coordinates must be hashable") from exc
        object.__setattr__(self, "values", normalized)

    def codes(self, values: pd.Series) -> np.ndarray:
        nonmissing = [x for x in self.values if x is not None]
        codes = pd.Categorical(values, categories=nonmissing).codes.astype(np.intp)
        # Category positions skip None, which may be anywhere for external results.
        positions = np.array([i for i, x in enumerate(self.values) if x is not None])
        valid = codes >= 0
        codes[valid] = positions[codes[valid]]
        if None in self.values:
            codes[values.isna().to_numpy()] = self.values.index(None)
        if (codes < 0).any():
            raise LayoutError(f"Canonical data contains coordinates outside axis {self.name!r}")
        return codes
