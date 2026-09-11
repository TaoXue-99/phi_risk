"""Pandas comparative execution: one shared encoding, batched calculations."""

from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from phl_risk.analysis._axis import AxisSpec
from phl_risk.analysis._comparison_types import ComparisonGroups
from phl_risk.analysis._distribution import category_domain
from phl_risk.analysis._plan import CubePlan
from phl_risk.analysis._result import CubeResult
from phl_risk.exceptions import DimensionError, EngineError

if TYPE_CHECKING:
    from ._pandas import PandasEngine


def prepare(
    engine: "PandasEngine", plan: CubePlan, data: pd.DataFrame
) -> tuple[pd.DataFrame, list[pd.Series]]:
    engine._validate_data(data)
    missing = set(plan.required_columns()) - set(data.columns)
    if missing:
        raise EngineError(f"Missing comparison columns: {sorted(missing)}")
    frame = engine.prepare_fit(data, plan.filters)
    transformed = []
    keep = np.ones(len(frame), dtype=bool)
    for dimension in plan.dimensions:
        values = dimension.transform(frame, plan.context)
        if not isinstance(values, pd.Series) or not values.index.equals(frame.index):
            raise DimensionError(f"Dimension {dimension.output_name!r} must preserve Series index")
        transformed.append(values)
        if plan.policy.missing.dimension == "drop":
            keep &= values.notna().to_numpy()
    if not keep.all():
        positions = np.flatnonzero(keep)
        frame = frame.iloc[positions]
        transformed = [s.iloc[positions] for s in transformed]
    return frame, transformed


def encode_groups(plan: CubePlan, ref: list[pd.Series], cur: list[pd.Series], nref: int, ncur: int):
    """Encode joint keys once; axis domains retain declared categorical ordering."""
    axes, columns = [], []
    for dimension, a, b in zip(plan.dimensions, ref, cur):
        domain = category_domain(a, b)
        combined = pd.concat([a, b], ignore_index=True)
        codes = pd.Categorical(combined, categories=list(domain)).codes.astype(np.intp)
        if (codes < 0).any():
            codes[codes < 0] = len(domain)
            domain += (None,)
        axes.append(AxisSpec(dimension.output_name, "dimension", domain))
        columns.append(codes)
    if columns:
        codes, keys = pd.factorize(pd.MultiIndex.from_arrays(columns), sort=False)
        coordinates = keys.to_frame(index=False).to_numpy(dtype=np.intp)
        size = len(keys)
    else:
        codes = np.zeros(nref + ncur, dtype=np.intp)
        coordinates = np.empty((1, 0), dtype=np.intp)
        size = 1
    codes.flags.writeable = False
    return ComparisonGroups(codes[:nref], codes[nref:], size), axes, coordinates


def execute_comparison(engine: "PandasEngine", plan: CubePlan, reference, current) -> CubeResult:
    if plan.mode != "comparative":
        raise EngineError("Use execute for single-sample plans")
    if plan.totals:
        raise EngineError("Comparative totals are not yet supported")
    start = perf_counter()
    ref, ref_dimensions = prepare(engine, plan, reference)
    cur, cur_dimensions = prepare(engine, plan, current)
    groups, axes, coordinates = encode_groups(
        plan, ref_dimensions, cur_dimensions, len(ref), len(cur)
    )
    names = tuple(m.name for m in plan.measures)
    dimensions = {}
    for i, axis in enumerate(axes):
        domain = np.empty(len(axis.values), dtype=object)
        domain[:] = axis.values
        dimensions[axis.name] = domain[coordinates[:, i]]
    parts, diagnostics = [], {}
    for measure in plan.measures:
        output = measure.node.calculation.evaluate(
            ref,
            cur,
            groups,
            backend=engine.name,
            on_invalid=plan.policy.on_invalid,
        )
        values = np.asarray(output.values, dtype=float)
        if values.shape != (groups.size,):
            raise EngineError(
                f"Comparative calculation {measure.name!r} must return one value per group"
            )
        parts.append(
            pd.DataFrame(
                {
                    **dimensions,
                    "metric": pd.Categorical(
                        [measure.name] * groups.size, categories=names, ordered=True
                    ),
                    "value": values,
                }
            )
        )
        diagnostics[measure.name] = output.metadata
    canonical = pd.concat(parts, ignore_index=True)
    axes.append(AxisSpec("metric", "metric", names))
    return CubeResult(
        canonical,
        tuple(axes),
        plan.measures,
        {
            "engine": engine.name,
            "mode": "comparative",
            "reference_input_rows": len(reference),
            "current_input_rows": len(current),
            "reference_analyzed_rows": len(ref),
            "current_analyzed_rows": len(cur),
            "comparison_groups": groups.size,
            "group_encoding_passes": 1,
            "dimensions": tuple(d.metadata() for d in plan.dimensions),
            "measures": diagnostics,
            "compute_duration": perf_counter() - start,
        },
    )
