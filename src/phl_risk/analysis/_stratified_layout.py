"""Contextual bin labels: one code can mean a different interval in each stratum."""

from itertools import product

import numpy as np
import pandas as pd

from phl_risk.exceptions import LayoutError


def _index(keys, names):
    if not names:
        return pd.RangeIndex(1)
    if len(names) == 1:
        return pd.Index([key[0] for key in keys], name=names[0], tupleize_cols=False)
    return pd.MultiIndex.from_tuples(keys, names=names)


def render_stratified(layout, result, axes, tensor, total_axes):
    """Keep only valid stratum/bin pairs, then resolve display coordinates.

    When strata and bins are on opposite axes, intervals form a union on the
    bin axis. Inapplicable intersections remain NaN, including Count cells.
    """
    names = tuple(a.name for a in axes)
    positions = {name: i for i, name in enumerate(names)}
    grouped = []
    for dimension in result._metadata.get("dimensions", ()):
        if not dimension.get("fit_by"):
            continue
        groups = {}
        for group in dimension["groups"]:
            transform = group["transform"]
            labels = transform.get("labels")
            intervals = transform.get("intervals")
            if layout.bin_labels == "interval" and (labels is None or intervals is None):
                raise LayoutError("Stratified interval display requires bin labels and intervals")
            groups[group["key"]] = (labels, dict(zip(labels or (), intervals or ())))
        grouped.append(
            (positions[dimension["name"]], tuple(positions[f] for f in dimension["fit_by"]), groups)
        )
    global_axes = layout._display_axes(result, axes)
    row_positions = tuple(positions[name] for name in layout.rows)
    column_positions = tuple(positions[name] for name in layout.columns)
    rows, columns, cells = {}, {}, []
    # Enumerate in requested row/column order, rather than canonical axis order.
    # Otherwise moving metric ahead of bin would still interleave metrics by bin.
    order = row_positions + column_positions
    inverse = tuple(order.index(i) for i in range(len(axes)))
    for requested_codes in product(*(range(len(axes[i].values)) for i in order)):
        codes = tuple(requested_codes[i] for i in inverse)
        coordinate = tuple(a.values[code] for a, code in zip(axes, codes))
        display = [a.values[code] for a, code in zip(global_axes, codes)]
        valid = True
        for bin_position, key_positions, groups in grouped:
            label = coordinate[bin_position]
            if axes[bin_position].name in total_axes and label == layout.total_label:
                continue
            if label is None:
                continue
            group = groups.get(tuple(coordinate[i] for i in key_positions))
            if group is None or (group[0] is not None and label not in group[0]):
                valid = False
                break
            if layout.bin_labels == "interval":
                display[bin_position] = group[1][label]
                if (
                    axes[bin_position].name in total_axes
                    and display[bin_position] == layout.total_label
                ):
                    raise LayoutError("Total label collides with a stratified interval")
        if not valid:
            continue
        row = tuple(display[i] for i in row_positions)
        column = tuple(display[i] for i in column_positions)
        ri = rows.setdefault(row, len(rows))
        ci = columns.setdefault(column, len(columns))
        cells.append((ri, ci, tensor[codes]))
    # Empty current populations still expose independent axes (e.g. metrics).
    bin_names = {axes[p].name for p, _, _ in grouped}
    for domain, requested in ((rows, layout.rows), (columns, layout.columns)):
        if not domain and not bin_names.intersection(requested):
            for key in product(*(global_axes[positions[name]].values for name in requested)):
                domain.setdefault(key, len(domain))
    nrows = len(rows) if layout.rows else 1
    ncolumns = len(columns) if layout.columns else 1
    size = nrows * ncolumns
    if size > layout.max_cells:
        raise LayoutError(f"Layout needs {size:,} cells, exceeding max_cells={layout.max_cells:,}")
    values = np.full((nrows, ncolumns), np.nan)
    for row, column, value in cells:
        values[row, column] = value
    return pd.DataFrame(
        values, index=_index(rows, layout.rows), columns=_index(columns, layout.columns)
    )
