"""Explanatory text only: never infer or change a binning configuration."""

from .dimensions import BinDimension, ColumnDimension
from .transforms import FixedBinner


def binning_scope(dimension) -> str | None:
    if not isinstance(dimension, BinDimension):
        return None
    if isinstance(dimension.transformer, FixedBinner):
        if dimension.fit_by:
            return (
                f"Fixed boundaries (no learning); applied within fit_by={list(dimension.fit_by)!r}"
            )
        return "Fixed boundaries (no learning)"
    if dimension.fit_by:
        return f"Fit scope: within groups; fit_by={list(dimension.fit_by)!r}"
    return "Fit scope: global (one definition learned from the prepared fit reference)"


def global_binning_hint(dimension, dimensions) -> str | None:
    if (
        not isinstance(dimension, BinDimension)
        or dimension.fit_by
        or isinstance(dimension.transformer, FixedBinner)
    ):
        return None
    grouping = [d for d in dimensions if isinstance(d, ColumnDimension)]
    if not grouping:
        return None
    fields = list(dict.fromkeys(d.column for d in grouping if d.output_name == d.column))
    suggestion = (
        f"For within-group binning, set fit_by to the raw grouping fields you need, "
        f"e.g. fit_by={fields!r}."
        if fields
        else "For within-group binning, set fit_by to raw grouping fields and include "
        "those fields as unrenamed ColumnDimensions."
    )
    return (
        f"Bin dimension {dimension.output_name!r} is fitted globally "
        "on the prepared fit reference; "
        "grouping dimensions do not change its fit scope. " + suggestion
    )
