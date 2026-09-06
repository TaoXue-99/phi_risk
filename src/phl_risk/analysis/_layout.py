from dataclasses import dataclass
from itertools import combinations
from math import prod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from phl_risk.exceptions import LayoutError

from ._axis import AxisSpec

if TYPE_CHECKING:
    from ._result import CubeResult


def _index(axes: tuple[AxisSpec, ...]) -> pd.Index:
    if not axes:
        return pd.RangeIndex(1)
    levels = [pd.Index(a.values, name=a.name, tupleize_cols=False) for a in axes]
    if len(axes) == 1:
        return levels[0]
    sizes = [len(a.values) for a in axes]
    codes = [
        np.tile(np.repeat(np.arange(size), prod(sizes[i + 1 :])), prod(sizes[:i]))
        for i, size in enumerate(sizes)
    ]
    return pd.MultiIndex(levels=levels, codes=codes, names=[a.name for a in axes])


@dataclass(frozen=True)
class TableLayout:
    """Exact placement of every logical axis, without aggregation or execution."""

    rows: tuple[str, ...]
    columns: tuple[str, ...]
    values: str = "value"
    max_cells: int = 1_000_000
    totals: bool | tuple[str, ...] = False
    total_label: str = "Total"
    bin_labels: str = "code"

    def __post_init__(self) -> None:
        if isinstance(self.rows, str) or isinstance(self.columns, str):
            raise LayoutError("rows and columns must be sequences of axis names")
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "columns", tuple(self.columns))
        if self.values != "value":
            raise LayoutError("V0.1 canonical value field is 'value'")
        if not isinstance(self.max_cells, int) or self.max_cells < 1:
            raise LayoutError("max_cells must be a positive integer")
        if not isinstance(self.totals, bool):
            if isinstance(self.totals, str):
                raise LayoutError("totals must be bool or a sequence of dimension names")
            object.__setattr__(self, "totals", tuple(self.totals))
        if not isinstance(self.total_label, str) or not self.total_label:
            raise LayoutError("total_label must be a non-empty string")
        if self.bin_labels not in ("code", "interval"):
            raise LayoutError("bin_labels must be 'code' or 'interval'")

    def render(self, result: "CubeResult") -> pd.DataFrame:
        names = tuple(a.name for a in result.axes)
        requested = self.rows + self.columns
        if len(requested) != len(set(requested)) or set(requested) != set(names):
            raise LayoutError(f"Place each axis exactly once: {names}; got {requested}")
        dimensions = tuple(a.name for a in result.dimensions_)
        total_axes = (
            dimensions if self.totals is True else (() if self.totals is False else self.totals)
        )
        if len(set(total_axes)) != len(total_axes) or set(total_axes) - set(dimensions):
            raise LayoutError(f"Invalid total axes {total_axes!r}; available: {dimensions}")
        axes = []
        for axis in result.axes:
            if axis.name in total_axes:
                if self.total_label in axis.values:
                    raise LayoutError(
                        f"Total label {self.total_label!r} collides with {axis.name!r}"
                    )
                axes.append(AxisSpec(axis.name, axis.role, axis.values + (self.total_label,)))
            else:
                axes.append(axis)
        shape = tuple(len(a.values) for a in axes)
        size = prod(shape)
        if size > self.max_cells:
            raise LayoutError(
                f"Layout needs {size:,} cells, exceeding max_cells={self.max_cells:,}"
            )
        # Materialize only for presentation. The metric axis is always last in
        # the canonical tensor; broadcast metric-specific empty-cell values.
        empty = np.array([result._empty_values[m] for m in result.axes[-1].values])
        tensor = np.broadcast_to(empty, shape).copy()
        data = result._data
        if len(data):
            codes = tuple(axis.codes(data[axis.name]) for axis in result.axes)
            tensor[codes] = data["value"].to_numpy(dtype=float)
        for count in range(1, len(total_axes) + 1):
            for collapsed in combinations(total_axes, count):
                total = result.total(over=collapsed)
                data = total._data
                if len(data):
                    codes = tuple(
                        np.full(len(data), len(axis.values), dtype=np.intp)
                        if axis.name in collapsed
                        else axis.codes(data[axis.name])
                        for axis in result.axes
                    )
                    tensor[codes] = data["value"].to_numpy(dtype=float)
        order = tuple(names.index(name) for name in requested)
        by_name = {a.name: a for a in self._display_axes(result, axes)}
        row_axes = tuple(by_name[x] for x in self.rows)
        column_axes = tuple(by_name[x] for x in self.columns)
        values = tensor.transpose(order).reshape(
            prod(len(a.values) for a in row_axes), prod(len(a.values) for a in column_axes)
        )
        return pd.DataFrame(values, index=_index(row_axes), columns=_index(column_axes))

    def _display_axes(self, result: "CubeResult", axes: list[AxisSpec]) -> list[AxisSpec]:
        if self.bin_labels == "code":
            return axes
        mappings = {}
        for dimension in result._metadata.get("dimensions", ()):
            transform = dimension.get("transform", {})
            if "bin_edges" not in transform or "labels" not in transform:
                continue
            edges = transform["bin_edges"]
            labels = transform["labels"]
            intervals = [
                f"{'[' if i == 0 and transform.get('include_lowest', True) else '('}"
                f"{left}, {right}]"
                for i, (left, right) in enumerate(zip(edges[:-1], edges[1:]))
            ]
            mappings[dimension["name"]] = dict(zip(labels, intervals))
        return [
            AxisSpec(
                axis.name,
                axis.role,
                tuple(mappings.get(axis.name, {}).get(value, value) for value in axis.values),
            )
            for axis in axes
        ]
