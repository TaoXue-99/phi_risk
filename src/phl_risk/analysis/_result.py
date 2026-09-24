from copy import deepcopy
from typing import Mapping, Sequence

import pandas as pd

from phl_risk.exceptions import LayoutError

from ._axis import AxisSpec
from ._diagnostics import Diagnostic, diagnostic_table
from ._layout import TableLayout
from ._nodes import MeasureSpec


class CubeResult:
    """Sparse canonical observations plus complete ordered logical axis domains.

    Public data and metadata access returns defensive copies. Layout does not
    recompute and does not change this object's canonical data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        axes: tuple[AxisSpec, ...],
        measures: tuple[MeasureSpec, ...],
        metadata: Mapping[str, object] | None = None,
        empty_values: Mapping[str, float] | None = None,
        totals: Mapping[tuple[str, ...], "CubeResult"] | None = None,
        *,
        diagnostics: Sequence[Diagnostic] = (),
    ) -> None:
        axes = tuple(axes)
        names = [a.name for a in axes]
        if not axes or axes[-1].name != "metric" or axes[-1].role != "metric":
            raise LayoutError("Canonical result requires a final metric axis")
        if any(a.role != "dimension" for a in axes[:-1]) or len(names) != len(set(names)):
            raise LayoutError("Result dimension axes must have unique names")
        if list(data.columns) != names + ["value"]:
            raise LayoutError(f"Canonical columns must be {names + ['value']}")
        if data.duplicated(names).any():
            raise LayoutError("Duplicate canonical coordinates; layout cannot aggregate")
        if not pd.api.types.is_numeric_dtype(data["value"]):
            raise LayoutError("Canonical values must be numeric")
        if tuple(m.name for m in measures) != axes[-1].values:
            raise LayoutError("Measure specifications and metric axis disagree")
        for axis in axes:
            axis.codes(data[axis.name])
        self._diagnostics = tuple(diagnostics)
        self._data = data.copy(deep=True)
        self._totals = dict(totals or {})
        self._axes = axes
        self._measures = tuple(measures)
        self._metadata = deepcopy(dict(metadata or {}))
        self._empty_values = {m.name: m.empty_value for m in measures}
        if empty_values is not None:
            self._empty_values.update(empty_values)

    def diagnostics(self) -> pd.DataFrame:
        """Explain observed NaNs, indexed by dimensions and metric.

        Returns a fresh table. Layout-only cells are excluded. Unsupported
        calculation paths use reason_not_recorded; empty means no observed NaNs.
        Totals have their own report via result.total(over=...).diagnostics().
        """
        return diagnostic_table(self._data, [a.name for a in self._axes], self._diagnostics)

    @property
    def data_(self) -> pd.DataFrame:
        return self._data.copy(deep=True)

    @property
    def metadata_(self) -> dict[str, object]:
        return deepcopy(self._metadata)

    @property
    def dimensions_(self) -> tuple[AxisSpec, ...]:
        return self._axes[:-1]

    @property
    def measures_(self) -> tuple[MeasureSpec, ...]:
        return self._measures

    @property
    def axes(self) -> tuple[AxisSpec, ...]:
        return self._axes

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(a.values) for a in self._axes)

    @property
    def ndim(self) -> int:
        return len(self._axes)

    def layout(
        self,
        rows: Sequence[str] | None = None,
        columns: Sequence[str] | None = None,
        *,
        max_cells: int = 1_000_000,
        totals: bool | Sequence[str] = False,
        total_label: str = "Total",
        bin_labels: str = "code",
    ) -> pd.DataFrame:
        if rows is None and columns is None:
            rows, columns = [a.name for a in self.dimensions_], ["metric"]
        elif rows is None:
            rows = [a.name for a in self.axes if a.name not in columns]
        elif columns is None:
            columns = [a.name for a in self.axes if a.name not in rows]
        return TableLayout(
            rows,
            columns,
            max_cells=max_cells,
            totals=totals,
            total_label=total_label,
            bin_labels=bin_labels,
        ).render(self)

    def total(self, over: Sequence[str]) -> "CubeResult":
        """Return a precomputed pooled result, collapsing the specified dimensions."""
        if isinstance(over, str):
            raise LayoutError("over must be a sequence of dimension names")
        over = tuple(over)
        dimensions = tuple(a.name for a in self.dimensions_)
        if len(set(over)) != len(over) or set(over) - set(dimensions):
            raise LayoutError(f"Invalid total dimensions {over!r}; available: {dimensions}")
        if not over:
            return self
        retained = tuple(name for name in dimensions if name not in over)
        if retained not in self._totals:
            raise LayoutError("Totals were not computed; use cube.compute(data, totals=True)")
        return self._totals[retained]

    def __repr__(self) -> str:
        return f"CubeResult(axes={tuple(a.name for a in self.axes)!r}, shape={self.shape})"
