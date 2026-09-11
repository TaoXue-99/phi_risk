import warnings
from enum import Enum

from phl_risk.exceptions import DataWorkflowError


class FailurePolicy(str, Enum):
    RAISE = "raise"
    WARN = "warn"
    CONTINUE = "continue"


def failure_policy(value):
    try:
        return FailurePolicy(value)
    except ValueError as exc:
        raise DataWorkflowError("on_fail must be raise/warn/continue") from exc


def enforce(report, policy, name):
    policy = failure_policy(policy)
    if not report.failed:
        return
    message = f"Quality stage {name!r} failed ({len(report.failures)} checks)"
    if policy == FailurePolicy.RAISE:
        error = DataWorkflowError(message)
        error.report = report
        error.stage = name
        raise error
    if policy == FailurePolicy.WARN:
        warnings.warn(message, UserWarning, stacklevel=3)
