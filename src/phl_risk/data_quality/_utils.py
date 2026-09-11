import math

from phl_risk._data import columns_of
from phl_risk.exceptions import DataQualityError

from ._result import CheckResult
from ._status import CheckStatus, worst


def selected(X, columns):
    return columns_of(X, columns, DataQualityError)


def threshold(value, name, upper=None):
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (upper is not None and value > upper)
    ):
        raise DataQualityError(f"{name} must be finite and in [0, {upper or 'infinity'}]")


def limits(warn, fail, upper=None):
    threshold(warn, "warn threshold", upper)
    threshold(fail, "fail threshold", upper)
    if warn is not None and fail is not None and warn > fail:
        raise DataQualityError("warn threshold must not exceed fail threshold")


def severity(value, warn=None, fail=None):
    if fail is not None and value > fail:
        return CheckStatus.FAIL
    if warn is not None and value > warn:
        return CheckStatus.WARN
    return CheckStatus.PASS


def relative(current, reference):
    return (current - reference) / reference if reference else (0.0 if not current else math.inf)


def result(obj, columns, details, statuses, expected=None):
    status = worst(statuses)
    return CheckResult(
        type(obj).__name__,
        status,
        tuple(columns),
        f"{type(obj).__name__}: {status.value}",
        details,
        expected,
        details,
    )


def examples_limit(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataQualityError("max_examples must be a nonnegative integer")
