"""Run both V0.1 acceptance scenarios: python examples/analysis_examples.py."""

import numpy as np
import pandas as pd

from phl_risk.analysis import (
    AUC,
    KS,
    BinDimension,
    Count,
    Cube,
    EventRate,
    QuantileBinner,
    Share,
)


def stratified_example():
    frame = pd.DataFrame(
        {
            "dt": ["2026-01"] * 8 + ["2026-02"] * 8,
            "user_type": (["new"] * 4 + ["old"] * 4) * 2,
            "label": [0, 0, 1, 1] * 4,
            "score": [
                0.1,
                0.4,
                0.35,
                0.8,
                0.1,
                0.2,
                0.8,
                0.9,
                0.8,
                0.9,
                0.1,
                0.2,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
        }
    )
    cube = Cube(["dt", "user_type"], [AUC("score", "label"), KS("score", "label")])
    result = cube.compute(frame, totals=True)
    expected = [[0.75, 0.5], [1, 1], [0, 1], [0.5, 0]]
    np.testing.assert_allclose(result.layout().to_numpy(), expected)
    print(cube.explain(format="text", totals=True))
    print(result.layout())
    print(result.layout(["dt"], ["metric", "user_type"], totals=["dt"]))
    return result


def cross_example():
    train = pd.DataFrame({"score_a": range(100), "score_b": range(100)})
    oot = pd.DataFrame(
        {
            "score_a": np.repeat([5, 25, 45, 65, 85], 5),
            "score_b": np.tile([5, 25, 45, 65, 85], 5),
            "label": [0, 1, 0, 1, 0] * 5,
        }
    )
    cube = Cube(
        [BinDimension("score_a", QuantileBinner(5)), BinDimension("score_b", QuantileBinner(5))],
        [Count(), Share(), EventRate("label", name="event_rate")],
    ).fit(train)
    learned = [dimension.transformer.bin_edges_ for dimension in cube.dimensions_]
    result = cube.compute(oot, totals=True)
    vertical = result.layout(["metric", "score_a_bin"], ["score_b_bin"])
    horizontal = result.layout(["score_a_bin"], ["metric", "score_b_bin"])
    assert result.shape == (5, 5, 3)
    assert vertical.index.tolist() == [
        (metric, f"B{i}") for metric in ("count", "share", "event_rate") for i in range(1, 6)
    ]
    for edges, dimension in zip(learned, cube.dimensions_):
        np.testing.assert_array_equal(edges, dimension.transformer.bin_edges_)
    np.testing.assert_allclose(vertical.loc["share"], 1 / 25)
    print(result)
    print(vertical)
    print(horizontal)
    with_totals = result.layout(
        ["metric", "score_a_bin"], ["score_b_bin"], totals=True, bin_labels="interval"
    )
    assert with_totals.loc[("count", "Total"), "Total"] == 25
    assert with_totals.loc[("event_rate", "Total"), "Total"] == 0.4
    print(with_totals)
    return result


if __name__ == "__main__":
    stratified_example()
    cross_example()
