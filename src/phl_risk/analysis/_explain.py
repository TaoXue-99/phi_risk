"""Presentation of a logical plan; no analysis is executed here."""

from typing import TYPE_CHECKING, Literal

import pandas as pd

from phl_risk.exceptions import CubeError

from ._nodes import AggregateNode, ComparativeNode, DerivedMetricNode, GroupMetricNode, RatioNode

if TYPE_CHECKING:
    from ._plan import CubePlan


def explain_plan(plan: "CubePlan", engine: str, format: Literal["table", "text"]):
    rows = [
        ("Execution", "Engine", engine),
        ("Execution", "Mode", plan.mode),
        ("Execution", "Grouping", " × ".join(plan.group_columns) or "global"),
    ]
    for dimension in plan.dimensions:
        rows.append(
            (
                "Dimension",
                dimension.output_name,
                f"Source: {', '.join(dimension.required_columns())}",
            )
        )
        if dimension.requires_fit:
            info = dimension.metadata()
            rows.append(("Transform", dimension.output_name, f"Fitted: {dimension.is_fitted}"))
            if "fit_field" in info:
                rows.append(
                    ("Transform", dimension.output_name, f"Fit source: {info['fit_field']}")
                )
            transform = info.get("transform", {})
            if "bin_edges" in transform:
                rows.append(("Bin edges", dimension.output_name, str(transform["bin_edges"])))
    for measure in plan.measures:
        node = measure.node
        if isinstance(node, ComparativeNode):
            description = f"Two-sample comparison; fields={node.required_columns()}"
        elif isinstance(node, GroupMetricNode):
            description = f"{node.kernel.upper()}: target={node.target}, score={node.score}"
            if node.weight is not None:
                description += f", weight={node.weight}"
        elif isinstance(node, DerivedMetricNode):
            description = (
                "Cell rows / all analyzed rows"
                if node.denominator == "all"
                else f"Event / valid target: {node.numerator.column}; "
                f"event={node.numerator.event_value!r}; weight={node.numerator.weight}"
            )
        elif isinstance(node, RatioNode):
            description = f"Aggregated measure ratio: {node.numerator} / {node.denominator}"
        elif isinstance(node, AggregateNode):
            description = node.operation
            if node.operation == "sum":
                description += f": column={node.column}; missing={node.missing}"
        else:
            description = type(node).__name__
        if isinstance(node, AggregateNode) and node.condition is not None:
            description += f"; condition={node.condition!r}"
        rows.append(("Measure", measure.name, description))
    rows.extend(
        [
            ("Execution", "Shared aggregates", str(len(plan.aggregates))),
            ("Execution", "Group metrics", str(len(plan.group_metrics))),
            ("Execution", "Filters", str(len(plan.filters))),
            (
                "Execution",
                "Totals",
                "Unsupported for comparative mode"
                if plan.mode == "comparative"
                else ("Recompute pooled samples" if plan.totals else "Disabled"),
            ),
            ("Policy", "Dimension missing", plan.policy.missing.dimension),
            (
                "Policy",
                "Field missing",
                "Measure-specific"
                if plan.mode == "comparative"
                else "Drop target / score within each metric",
            ),
            ("Policy", "Invalid metric", plan.policy.on_invalid),
        ]
    )
    table = pd.DataFrame(rows, columns=["Section", "Item", "Description"])
    if format == "table":
        return table
    if format == "text":
        return "Cube Analysis Plan\n" + table.to_string(index=False)
    raise CubeError("explain format must be 'table' or 'text'")
