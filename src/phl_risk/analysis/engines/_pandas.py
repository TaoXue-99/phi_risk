"""Vectorized shared reductions plus explicit group-level numerical kernels."""

from dataclasses import replace
from itertools import combinations
from time import perf_counter

import numpy as np
import pandas as pd

from phl_risk.analysis._axis import AxisSpec
from phl_risk.analysis._context import AnalysisContext
from phl_risk.analysis._diagnostics import Diagnostic
from phl_risk.analysis._nodes import AggregateNode, DerivedMetricNode, GroupMetricNode, RatioNode
from phl_risk.analysis._plan import CubePlan, FilterExpression, FilterLike
from phl_risk.analysis._result import CubeResult
from phl_risk.exceptions import DimensionError, EngineError, InvalidMetricError
from phl_risk.metrics import auc_score, ks_score
from phl_risk.metrics._common import invalid, weights

from ._base import BaseCubeEngine


class PandasEngine(BaseCubeEngine):
    name = "pandas"

    def __repr__(self) -> str:
        return "PandasEngine()"

    @staticmethod
    def _validate_data(data: object) -> None:
        if not isinstance(data, pd.DataFrame):
            raise EngineError("PandasEngine requires a pandas DataFrame")
        if not data.columns.is_unique:
            raise EngineError("PandasEngine does not accept duplicate input columns")

    def prepare_fit(self, data: pd.DataFrame, filters: tuple[FilterLike, ...]) -> pd.DataFrame:
        self._validate_data(data)
        current = data
        for predicate in filters:
            # User callbacks are isolated so accidental assignment cannot mutate input.
            argument = current.copy(deep=True)
            if isinstance(predicate, FilterExpression):
                missing = set(predicate.required_columns()) - set(current.columns)
                if missing:
                    raise EngineError(f"Missing filter columns: {sorted(missing)}")
                mask = predicate.evaluate(argument, backend=self.name)
            else:
                mask = predicate(argument)
            if not isinstance(mask, pd.Series):
                mask = pd.Series(mask, index=current.index)
            if not mask.index.equals(current.index) or not pd.api.types.is_bool_dtype(mask):
                raise EngineError("Filter must return an aligned boolean mask")
            current = current.loc[mask.fillna(False)]
        return current

    def execute(
        self, plan: CubePlan, data: pd.DataFrame, context: AnalysisContext | None = None
    ) -> CubeResult:
        if plan.mode != "single":
            raise EngineError("Use execute_comparison for comparative plans")
        start = perf_counter()
        self._validate_data(data)
        if context is not None and context != plan.context:
            raise EngineError(
                "Context differs from compiled plan; recompile with cube.plan(context=)"
            )
        missing = set(plan.required_columns()) - set(data.columns)
        if missing:
            raise EngineError(f"Missing required columns: {sorted(missing)}")
        current = self.prepare_fit(data, plan.filters)
        filtered_rows = len(current)
        transformed = []
        for dimension in plan.dimensions:
            values = dimension.transform(current, plan.context)
            if not isinstance(values, pd.Series) or not values.index.equals(current.index):
                raise DimensionError(
                    f"Dimension {dimension.output_name!r} must preserve Series index"
                )
            transformed.append(values)
        keep = np.ones(len(current), dtype=bool)
        if plan.policy.missing.dimension == "drop":
            for values in transformed:
                keep &= values.notna().to_numpy()
        current = current.iloc[np.flatnonzero(keep)]
        transformed = [s.iloc[np.flatnonzero(keep)] for s in transformed]
        if plan.totals and 2 ** len(plan.dimensions) > 64:
            raise EngineError("Totals support at most 64 grouping sets; reduce dimensions")
        return self._compute_prepared(
            plan,
            current,
            transformed,
            input_rows=len(data),
            filtered_rows=filtered_rows,
            start=start,
        )

    def execute_comparison(
        self,
        plan: CubePlan,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> CubeResult:
        from ._comparison import execute_comparison

        return execute_comparison(self, plan, reference, current)

    def _compute_prepared(
        self,
        plan: CubePlan,
        current: pd.DataFrame,
        transformed: list[pd.Series],
        *,
        input_rows: int,
        filtered_rows: int,
        start: float,
    ) -> CubeResult:
        """Aggregate an already filtered population; totals reuse the same row universe."""
        total = len(current)
        work = pd.DataFrame(index=pd.RangeIndex(total))
        axes = []
        group_columns = []
        for i, (dimension, values) in enumerate(zip(plan.dimensions, transformed)):
            if isinstance(values.dtype, pd.CategoricalDtype):
                domain = tuple(values.cat.categories)
            else:
                domain = tuple(pd.unique(values.dropna()))
            has_missing = values.isna().any()
            codes = pd.Categorical(values, categories=list(domain)).codes.copy()
            if has_missing:
                codes[codes < 0] = len(domain)
                domain += (None,)
            axes.append(AxisSpec(dimension.output_name, "dimension", domain))
            key = f"d{i}"
            group_columns.append(key)
            work[key] = codes

        grouped = (
            work.groupby(group_columns, sort=False, observed=True, dropna=False)
            if group_columns
            else None
        )
        group_ids = (
            grouped.ngroup().to_numpy(dtype=np.intp)
            if grouped is not None
            else np.zeros(total, dtype=np.intp)
        )
        group_count = grouped.ngroups if grouped is not None else 1

        # A single scratch frame merges every native reduction. Internal names
        # never collide with raw fields because raw fields are not copied here.
        count_node = AggregateNode("row_count")
        aggregate_nodes = tuple(dict.fromkeys((count_node,) + plan.aggregates))
        node_columns = {node: f"a{i}" for i, node in enumerate(aggregate_nodes)}
        weight_cache = {}
        normalized_weights = {}
        missing_columns = {}
        for node in (*aggregate_nodes, *plan.group_metrics):
            if node.weight is not None and node.weight not in weight_cache:
                weight_cache[node.weight] = weights(current[node.weight], total)
        condition_masks = {}
        for node, column in node_columns.items():
            selected = None
            if node.condition is not None:
                if node.condition not in condition_masks:
                    condition_masks[node.condition] = (
                        node.condition.evaluate(current, backend=self.name)
                        .fillna(False)
                        .to_numpy(dtype=bool)
                    )
                selected = condition_masks[node.condition]
            if node.operation == "row_count":
                work[column] = (
                    np.ones(total, dtype=np.int64)
                    if selected is None
                    else selected.astype(np.int64)
                )
                continue
            if node.operation == "sum":
                source = current[node.column]
                if not pd.api.types.is_numeric_dtype(source.dtype) or pd.api.types.is_complex_dtype(
                    source.dtype
                ):
                    raise EngineError(f"Sum requires a real numeric column: {node.column!r}")
                values = source.to_numpy(dtype=float, na_value=np.nan)
                if selected is not None:
                    values = np.where(selected, values, 0.0)
                if np.isinf(values).any():
                    raise EngineError(f"Sum does not accept infinite values: {node.column!r}")
                if node.missing == "propagate":
                    missing_columns[node] = f"missing_{column}"
                    work[missing_columns[node]] = np.isnan(values).astype(np.int64)
                work[column] = np.nan_to_num(values, nan=0.0)
                continue
            if node.operation not in ("valid_count", "event_count"):
                raise EngineError(f"Unsupported aggregate operation {node.operation!r}")
            target = current[node.column]
            mask = (
                target.notna() if node.operation == "valid_count" else target.eq(node.event_value)
            )
            values = mask.fillna(False).to_numpy(dtype=float)
            if node.weight is not None:
                cache_key = (node.column, node.weight)
                if cache_key not in normalized_weights:
                    weight = weight_cache[node.weight]
                    valid_weight = np.where(target.notna().to_numpy(), weight, 0.0)
                    scales = np.zeros(group_count, dtype=float)
                    np.maximum.at(scales, group_ids, valid_weight)
                    scale = scales[group_ids]
                    normalized = np.zeros(total, dtype=float)
                    np.divide(valid_weight, scale, out=normalized, where=scale > 0)
                    normalized_weights[cache_key] = normalized
                values *= normalized_weights[cache_key]
            work[column] = values
        columns = list(node_columns.values()) + list(missing_columns.values())
        if group_columns:
            with np.errstate(over="ignore", invalid="ignore"):
                reduced = grouped[columns].sum()
            coordinates = reduced.index.to_frame(index=False).to_numpy(dtype=int)
            group_positions = grouped.indices if plan.group_metrics else {}
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                reduced = pd.DataFrame([work[columns].sum()], columns=columns)
            coordinates = np.empty((1, 0), dtype=int)
            group_positions = {0: np.arange(total)}
        calculated = {node: reduced[column].to_numpy() for node, column in node_columns.items()}
        for node, column in missing_columns.items():
            calculated[node] = np.where(reduced[column].to_numpy() > 0, np.nan, calculated[node])
        for node in aggregate_nodes:
            if node.operation != "sum":
                continue
            values = calculated[node]
            unknown = (
                reduced[missing_columns[node]].to_numpy() > 0
                if node in missing_columns
                else np.zeros(len(reduced), dtype=bool)
            )
            overflow = ~np.isfinite(values) & ~unknown
            if overflow.any():
                invalid(f"Sum({node.column}): non-finite aggregate", plan.policy.on_invalid)
                calculated[node] = np.where(overflow, np.nan, values)
        kernels = {"auc": auc_score, "ks": ks_score}
        metric_arrays = {
            column: current[column].to_numpy()
            for node in plan.group_metrics
            for column in (node.target, node.score)
        }
        for node in plan.group_metrics:
            if node.kernel not in kernels:
                raise EngineError(f"Unsupported group metric kernel {node.kernel!r}")
            output = np.empty(len(reduced), dtype=float)
            for i, key in enumerate(reduced.index):
                positions = group_positions[key]
                try:
                    output[i] = kernels[node.kernel](
                        metric_arrays[node.target][positions],
                        metric_arrays[node.score][positions],
                        None if node.weight is None else weight_cache[node.weight][positions],
                        on_invalid=plan.policy.on_invalid,
                    )
                except InvalidMetricError as exc:
                    raise InvalidMetricError(f"{node.kernel}, group={key!r}: {exc}") from exc
            calculated[node] = output

        metric_names = tuple(m.name for m in plan.measures)
        axes.append(AxisSpec("metric", "metric", metric_names))
        parts = []
        empty_values = {}
        named_values = {}
        for measure in plan.evaluation_measures:
            node = measure.node
            if isinstance(node, DerivedMetricNode):
                numerator = calculated[node.numerator]
                denominator = (
                    np.full(len(reduced), total)
                    if node.denominator == "all"
                    else calculated[node.denominator]
                )
                values = np.full(len(reduced), np.nan)
                valid = denominator > 0
                np.divide(numerator, denominator, out=values, where=valid)
                if (~valid).any():
                    invalid(
                        f"{measure.name}: zero denominator in {(~valid).sum()} groups",
                        plan.policy.on_invalid,
                    )
            elif isinstance(node, RatioNode):
                numerator = named_values[node.numerator]
                denominator = named_values[node.denominator]
                values = np.full(len(reduced), np.nan)
                valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
                with np.errstate(over="ignore", invalid="ignore"):
                    np.divide(numerator, denominator, out=values, where=valid)
                overflow = valid & ~np.isfinite(values)
                if overflow.any():
                    invalid(f"{measure.name}: non-finite ratio", plan.policy.on_invalid)
                    values[overflow] = np.nan
                zero = denominator == 0
                if zero.any():
                    invalid(
                        f"{measure.name}: zero denominator in {zero.sum()} groups",
                        plan.policy.on_invalid,
                    )
            elif isinstance(node, (AggregateNode, GroupMetricNode)):
                values = calculated[node]
            else:
                raise EngineError(f"Unsupported measure node {type(node).__name__}")
            named_values[measure.name] = values
        diagnostics = []
        for measure_index, measure in enumerate(plan.measures):
            node = measure.node
            values = named_values[measure.name]
            for group_index in np.flatnonzero(pd.isna(values)):
                row = measure_index * len(reduced) + int(group_index)
                input_rows = int(calculated[count_node][group_index])
                if isinstance(node, AggregateNode) and node in missing_columns:
                    affected = int(reduced[missing_columns[node]].iloc[group_index])
                    if affected:
                        diagnostics.append(
                            Diagnostic(
                                row,
                                "missing_propagated",
                                f"{node.column}: {affected} missing values; missing='propagate'.",
                                field=node.column,
                                input_rows=input_rows,
                                affected_rows=affected,
                            )
                        )
                elif isinstance(node, RatioNode):
                    for dependency in (node.numerator, node.denominator):
                        if pd.isna(named_values[dependency][group_index]):
                            diagnostics.append(
                                Diagnostic(
                                    row,
                                    "upstream_missing",
                                    f"Dependency {dependency!r} is missing.",
                                    input_rows=input_rows,
                                    dependency=dependency,
                                )
                            )
                    if named_values[node.denominator][group_index] == 0:
                        diagnostics.append(
                            Diagnostic(
                                row,
                                "zero_denominator",
                                "Denominator is zero.",
                                input_rows=input_rows,
                                dependency=node.denominator,
                            )
                        )
            part = {}
            for i, axis in enumerate(axes[:-1]):
                domain = np.empty(len(axis.values), dtype=object)
                domain[:] = axis.values
                part[axis.name] = domain[coordinates[:, i]]
            part["metric"] = pd.Categorical(
                [measure.name] * len(reduced), categories=metric_names, ordered=True
            )
            part["value"] = values
            parts.append(pd.DataFrame(part))
            empty_values[measure.name] = (
                np.nan
                if isinstance(node, DerivedMetricNode) and node.denominator == "all" and not total
                else measure.empty_value
            )
        canonical = pd.concat(parts, ignore_index=True)
        metadata = {
            "engine": self.name,
            "input_rows": input_rows,
            "filtered_rows": filtered_rows,
            "analyzed_rows": total,
            "result_rows": len(canonical),
            "compute_duration": perf_counter() - start,
            "dimensions": tuple(d.metadata() for d in plan.dimensions),
            "native_aggregation_passes": 1 if group_columns else 0,
            "shared_aggregates": len(aggregate_nodes),
        }
        totals = {}
        if plan.totals:
            # Reuse the same filtered rows and fitted dimension values. No refit,
            # callback re-evaluation, or metric averaging is involved.
            for size in range(len(plan.dimensions)):
                for indices in combinations(range(len(plan.dimensions)), size):
                    dimensions = tuple(plan.dimensions[i] for i in indices)
                    subplan = replace(plan, dimensions=dimensions, filters=(), totals=False)
                    totals[tuple(d.output_name for d in dimensions)] = self._compute_prepared(
                        subplan,
                        current,
                        [transformed[i] for i in indices],
                        input_rows=input_rows,
                        filtered_rows=filtered_rows,
                        start=perf_counter(),
                    )
        metadata["totals_computed"] = plan.totals
        metadata["total_grouping_sets"] = len(totals)
        metadata["compute_duration"] = perf_counter() - start
        return CubeResult(
            canonical,
            tuple(axes),
            plan.measures,
            metadata,
            empty_values,
            totals,
            diagnostics=diagnostics,
        )
