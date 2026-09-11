from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from phl_risk._data import freeze, thaw

from ._status import CheckStatus, worst


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    columns: tuple[str, ...]
    message: str
    observed: object = None
    expected: object = None
    details: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "status", CheckStatus(self.status))
        for key in ("columns", "observed", "expected", "details"):
            object.__setattr__(self, key, freeze(getattr(self, key)))

    @property
    def passed(self):
        return self.status == CheckStatus.PASS

    @property
    def failed(self):
        return self.status == CheckStatus.FAIL

    def to_dict(self):
        return {
            "check": self.name,
            "status": self.status.value,
            "columns": thaw(self.columns),
            "observed": thaw(self.observed),
            "expected": thaw(self.expected),
            "message": self.message,
            "details": thaw(self.details),
        }


@dataclass(frozen=True)
class QualityReport:
    results: tuple[CheckResult, ...]

    def __post_init__(self):
        object.__setattr__(self, "results", tuple(self.results))

    @property
    def status(self):
        return worst(r.status for r in self.results)

    @property
    def passed(self):
        return self.status == CheckStatus.PASS

    @property
    def failed(self):
        return self.status == CheckStatus.FAIL

    @property
    def failures(self):
        return tuple(r for r in self.results if r.failed)

    @property
    def warnings(self):
        return tuple(r for r in self.results if r.status == CheckStatus.WARN)

    def to_frame(self):
        return pd.DataFrame(
            [r.to_dict() for r in self.results],
            columns=["check", "status", "columns", "observed", "expected", "message"],
        )

    def summary(self):
        return {
            "status": self.status.value,
            "checks": len(self.results),
            **{s.value: sum(r.status == s for r in self.results) for s in CheckStatus},
        }
