"""Run: uv run python examples/conditional_measures.py."""

import numpy as np
import pandas as pd

from phl_risk.analysis import Col, Count, Cube, Sum

frame = pd.DataFrame(
    {
        "dt": ["d1", "d1", "d1", "d2"],
        "category": ["A", "B", None, "C"],
        "age": [20, 30, 40, 10],
        "amount": [20.0, 30.0, np.nan, 5.0],
    }
)
condition = Col("category").isin(["A", "B"]) & (Col("age") >= 18)
cube = Cube(
    ["dt"],
    [
        Count(name="all_count"),
        Count(where=condition, name="selected_count"),
        Sum("amount", where=condition, name="selected_amount"),
        Sum("amount", where=Col("amount") > 10, name="over_10_amount"),
        Count(where=Col("category").isna(), name="missing_category"),
    ],
)
result = cube.compute(frame, totals=True)
table = result.layout()
assert table.loc["d1"].tolist() == [3, 2, 50, 50, 1]
assert table.loc["d2"].tolist() == [1, 0, 0, 0, 0]
assert result.total(over=["dt"]).layout().iloc[0].tolist() == [4, 2, 50, 50, 1]
assert result.diagnostics().empty
print(result.layout(totals=True))
print(cube.explain(format="text"))
