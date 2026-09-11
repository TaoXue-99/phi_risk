from dataclasses import dataclass

import pandas as pd

from phl_risk._data import freeze
from phl_risk.data_prep._result import FrameResult
from phl_risk.data_quality._status import worst


@dataclass(frozen=True, init=False)
class StageResult(FrameResult):
    name: str
    kind: str
    _data: pd.DataFrame
    quality_report: object
    prep_audits: tuple

    def __init__(self, name, kind, data, quality_report=None, prep_audits=()):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "_data", data.copy(deep=True))
        object.__setattr__(self, "quality_report", quality_report)
        object.__setattr__(self, "prep_audits", tuple(prep_audits))


@dataclass(frozen=True, init=False)
class WorkflowResult(FrameResult):
    _data: pd.DataFrame
    stage_results: tuple[StageResult, ...]
    metadata: object

    def __init__(self, data, stage_results, metadata=None):
        object.__setattr__(self, "_data", data.copy(deep=True))
        object.__setattr__(self, "stage_results", tuple(stage_results))
        object.__setattr__(self, "metadata", freeze(metadata or {}))

    @property
    def quality_reports(self):
        return {
            s.name: s.quality_report for s in self.stage_results if s.quality_report is not None
        }

    @property
    def prep_audits(self):
        return {s.name: s.prep_audits for s in self.stage_results if s.kind == "prep"}

    def summary(self):
        return dict(
            rows=len(self._data),
            columns=tuple(self._data.columns),
            stages=len(self.stage_results),
            quality_status=worst(r.status for r in self.quality_reports.values()).value,
            prep_steps=sum(len(a) for a in self.prep_audits.values()),
        )
