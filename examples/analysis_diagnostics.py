"""Run: uv run python examples/analysis_diagnostics.py. Framework preview."""

import numpy as np
import pandas as pd

from phl_risk.analysis import Cube, Funnel

frame = pd.DataFrame(
    {"segment": ["A"] * 1000, "before": [1.0] * 1000, "after": [1.0] * 999 + [np.nan]}
)
result = Cube(["segment"], Funnel(["before", "after"]).measures()).compute(frame, totals=True)
print(result.layout())
print(result.diagnostics().to_string())
print(result.total(over=["segment"]).diagnostics().to_string())
assert result.diagnostics().loc[("A", "after数量"), "affected_rows"] == 1
