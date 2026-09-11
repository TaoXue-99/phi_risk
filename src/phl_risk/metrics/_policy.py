"""Numerical invalid-result policy, independent of pandas and analysis."""

import warnings
from typing import Literal

from phl_risk.exceptions import InvalidMetricError

InvalidPolicy = Literal["nan", "warn", "raise"]


def validate_policy(on_invalid: InvalidPolicy) -> None:
    if on_invalid not in ("nan", "warn", "raise"):
        raise ValueError("on_invalid must be 'nan', 'warn', or 'raise'")


def invalid(message: str, on_invalid: InvalidPolicy) -> float:
    validate_policy(on_invalid)
    if on_invalid == "raise":
        raise InvalidMetricError(message)
    if on_invalid == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    return float("nan")
