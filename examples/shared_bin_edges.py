"""Run: uv run python examples/shared_bin_edges.py.

Learn quantiles from reference.score_a and apply the same ranges to both scores.
score_b need not be equally populated. Both scores should use comparable scales.
"""

import numpy as np
import pandas as pd

from phl_risk.analysis import BinDimension, Count, Cube, QuantileBinner, Share

reference = pd.DataFrame({"score_a": np.arange(1, 101)})
current = pd.DataFrame({"score_a": [10, 25, 45, 65, 90], "score_b": [10, 10, 20, 45, 95]})
cube = Cube(
    [
        BinDimension("score_a", QuantileBinner(n_bins=5)),
        BinDimension("score_b", QuantileBinner(n_bins=5), fit_field="score_a"),
    ],
    [Count(), Share()],
)
# Fit only needs score_a; score_b is only required when computing.
cube.fit(reference)
a, b = cube.dimensions_
np.testing.assert_array_equal(a.transformer.bin_edges_, b.transformer.bin_edges_)
assert b.transform(current).tolist() == ["B1", "B1", "B1", "B3", "B5"]

result = cube.compute(current, totals=True)
print(
    result.layout(
        rows=["metric", "score_a_bin"],
        columns=["score_b_bin"],
        totals=True,
        total_label="总计",
        bin_labels="interval",
    )
)
print(cube.explain(format="text"))

# For self-reference analysis, use the exact same configuration and fit_compute.
cube.fit_compute(current)
a, b = cube.dimensions_
np.testing.assert_array_equal(a.transformer.bin_edges_, b.transformer.bin_edges_)
