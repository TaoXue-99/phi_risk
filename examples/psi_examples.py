"""0.3.0 PSI 使用案例：在仓库根目录运行 uv run python examples/psi_examples.py。

输入均为合成数据；每一步展示一种调用目的并包含结果断言。
整体/分群分析、数值/类别、日期循环都使用同一个比较框架。
"""

import numpy as np
import pandas as pd

from phl_risk.analysis import PSI, ComputePolicy, Cube, QuantileBinner


def main():
    # 1. 准备两侧数据：同样本基础上引入分数位移、缺失率变化和新类别。
    rng = np.random.default_rng(42)
    size = 1000
    reference = pd.DataFrame(
        {
            "segment": np.where(np.arange(size) % 2, "A", "B"),
            "score": rng.normal(size=size),
            "income": rng.lognormal(size=size),
            "channel": rng.choice(["web", "app"], size=size),
        }
    )
    current = reference.copy(deep=True)
    current["score"] += 0.6
    current.loc[:99, "income"] = np.nan
    current.loc[:99, "channel"] = "store"
    # 2. 整体 PSI：无 dimensions，不要求先 fit Cube。
    overall = Cube(measures=[PSI("score")]).compute_comparison(reference, current)
    assert overall.data_.value.iloc[0] > 0
    print("整体 PSI", overall.layout())

    # 3. 按 segment 分层，批量比较数值及类别。类别显式传 binner=None。
    cube = Cube(
        ["segment"],
        [
            PSI("score", QuantileBinner(10)),
            PSI("income"),
            PSI("channel", binner=None),
        ],
        policy=ComputePolicy(on_invalid="raise"),
    )
    same = cube.compute_comparison(reference, reference)
    np.testing.assert_allclose(same.data_.value, 0)
    result = cube.compute_comparison(reference, current)
    assert (result.data_.value > 0).all()
    print(result.layout(rows=["segment"], columns=["metric"]))
    print(cube.explain())

    # 4. 固定 reference 逐日比较：日期选择在外部完成，Cube 只按 segment 匹配。
    current["dt"] = np.where(np.arange(size) < 500, "2026-09-01", "2026-09-02")
    daily = pd.concat(
        [
            cube.compute_comparison(reference, daily_frame).data_.assign(dt=dt)
            for dt, daily_frame in current.groupby("dt", sort=False)
        ],
        ignore_index=True,
    )
    print(daily.pivot(index="dt", columns=["segment", "metric"], values="value"))
    assert len(daily) == 12

    # 5. dt 也是普通维度：两侧日期不重合时保留两侧 key，PSI 为 NaN。
    a = reference.assign(dt="baseline")
    b = current.assign(dt="current")
    unmatched = Cube(["dt"], [PSI("score")]).compute_comparison(a, b)
    assert unmatched.data_.value.isna().all()

    # 6. 100 字段 + 三维：每个 PSI 对应一个字段，共享一次分组编码。
    fields = [f"feature_{i}" for i in range(100)]
    batch_reference = pd.DataFrame(rng.normal(size=(200, 100)), columns=fields)
    for i in range(3):
        batch_reference[f"group_{i}"] = (np.arange(200) // (2**i)) % 2
    batch = Cube([f"group_{i}" for i in range(3)], [PSI(f) for f in fields])
    batch_result = batch.compute_comparison(batch_reference, batch_reference.copy())
    assert batch_result.shape == (2, 2, 2, 100)
    assert (batch_result.data_.value == 0).all()
    assert batch_result.metadata_["group_encoding_passes"] == 1


if __name__ == "__main__":
    main()
