from dataclasses import dataclass

import pandas as pd

from phl_risk._data import thaw

from ._audit import PrepAudit


class FrameResult:
    @property
    def data(self):
        return self._data.copy(deep=True)


@dataclass(frozen=True, init=False)
class StepResult(FrameResult):
    _data: pd.DataFrame
    audit: PrepAudit

    def __init__(self, data, audit):
        object.__setattr__(self, "_data", data.copy(deep=True))
        object.__setattr__(self, "audit", audit)


@dataclass(frozen=True, init=False)
class PrepResult(FrameResult):
    _data: pd.DataFrame
    audits: tuple[PrepAudit, ...]

    def __init__(self, data, audits):
        object.__setattr__(self, "_data", data.copy(deep=True))
        object.__setattr__(self, "audits", tuple(audits))

    def audit_frame(self):
        return pd.DataFrame(
            [
                dict(
                    step=a.step,
                    input_columns=a.input_columns,
                    output_columns=a.output_columns,
                    rows_before=a.before.row_count,
                    rows_after=a.after.row_count,
                    details=thaw(a.details),
                )
                for a in self.audits
            ],
            columns=[
                "step",
                "input_columns",
                "output_columns",
                "rows_before",
                "rows_after",
                "details",
            ],
        )

    def summary(self):
        return dict(rows=len(self._data), columns=tuple(self._data.columns), steps=len(self.audits))
