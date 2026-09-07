"""Run: uv run python examples/funnel_examples.py. All data is synthetic."""

import numpy as np
import pandas as pd

from phl_risk.analysis import (
    BinDimension,
    Col,
    CountWhere,
    Cube,
    Funnel,
    QuantileBinner,
    Stage,
    Transition,
)


def main():
    stages = ["戳额", "有额", "发标", "提现"]
    funnel = Funnel(stages, rates="both")
    # Nested 0/1 stages, one row per user. No user deduplication is implicit.
    rng = np.random.default_rng(7)
    n = 1000
    progress = rng.integers(0, 5, n)
    detail = pd.DataFrame(
        {
            "user_id": np.arange(n),
            "dt": np.where(np.arange(n) % 2, "2026-09-01", "2026-09-02"),
            "score": rng.normal(size=n),
            **{stage: (progress > i).astype(int) for i, stage in enumerate(stages)},
        }
    )
    by_dt = Cube(["dt"], funnel.measures()).compute(detail, totals=True)
    print("按日期：数量、相邻转化和首环节转化")
    print(by_dt.layout(totals=True, total_label="总计"))

    # Same-data fit and compute.
    own = Cube([BinDimension("score", QuantileBinner(5))], funnel.measures())
    print("自身等频分箱")
    print(own.fit_compute(detail, totals=True).layout(totals=True, bin_labels="interval"))

    # Learn boundaries on reference only. Current bins need not be equally populated.
    reference, current = detail.iloc[:600], detail.iloc[600:]
    cube = Cube([BinDimension("score", QuantileBinner(5))], funnel.measures()).fit(reference)
    edges = cube.dimensions_[0].transformer.bin_edges_.copy()
    result = cube.compute(current, totals=True)
    np.testing.assert_array_equal(cube.dimensions_[0].transformer.bin_edges_, edges)
    pooled = result.total(over=["score_bin"]).data_.set_index("metric")["value"]
    assert pooled["戳额数量"] == current["戳额"].sum()
    assert np.isclose(pooled["戳额→提现转化率"], current["提现"].sum() / current["戳额"].sum())
    print("参考集分箱用于新样本")
    print(result.layout(totals=True, total_label="总计", bin_labels="interval"))

    # Preaggregated counts have exactly the same sum-based semantics.
    # Such a table has no individual scores: it cannot reconstruct individual quantiles.
    aggregated = detail.groupby("dt", sort=False)[stages].sum().reset_index()
    summary = Cube(["dt"], funnel.measures()).compute(aggregated, totals=True)
    pd.testing.assert_frame_equal(by_dt.layout(totals=True), summary.layout(totals=True))

    custom = Funnel(
        [
            Stage("触达", CountWhere(Col("戳额") == 1)),
            Stage("最终提现", CountWhere(Col("提现") == 1)),
        ],
        rates=[Transition(before="触达", after="最终提现", name="端到端转化率")],
    )
    print("自定义条件与转化名称")
    print(Cube(["dt"], custom.measures()).compute(detail).layout())


if __name__ == "__main__":
    main()
