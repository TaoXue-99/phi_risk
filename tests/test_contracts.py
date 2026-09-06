"""Cross-layer contracts that prevent silent regressions during extension."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from phl_risk.analysis import (
    AUC,
    KS,
    BaseDimension,
    BinDimension,
    ColumnDimension,
    ComputePolicy,
    Count,
    Cube,
    EventRate,
    MissingPolicy,
    QuantileBinner,
    Share,
)
from phl_risk.exceptions import DimensionError, EngineError


@pytest.mark.parametrize("seed", range(5))
def test_random_weighted_groups_against_independent_references(seed):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "g": rng.choice(["z", "a", "b"], 300),
            "h": rng.integers(0, 3, 300),
            "y": rng.integers(0, 2, 300),
            "s": rng.integers(0, 5, 300),
            "w": rng.uniform(0.01, 2, 300),
        }
    )
    measures = [Count(), Share(), EventRate("y", weight="w"), AUC("s", "y", weight="w")]
    table = Cube(["g", "h"], measures).compute(frame).layout()
    for key, group in frame.groupby(["g", "h"], sort=False):
        assert table.loc[key, "count"] == len(group)
        assert table.loc[key, "share"] == pytest.approx(len(group) / len(frame))
        assert table.loc[key, "event_rate__y__1"] == pytest.approx(
            group.loc[group.y == 1, "w"].sum() / group.w.sum()
        )
        assert table.loc[key, "auc__s"] == pytest.approx(
            roc_auc_score(group.y, group.s, sample_weight=group.w)
        )


@pytest.mark.parametrize(
    "values",
    [
        pd.Series(["b", None, "a", "b"], dtype="string"),
        pd.Series([2, None, 1, 2], dtype="Int64"),
        pd.Series(pd.to_datetime(["2026-02-01", None, "2026-01-01", "2026-02-01"], utc=True)),
        pd.Series([("b", 1), None, ("a", 2), ("b", 1)]),
    ],
)
def test_axis_types_and_missing(values):
    frame = pd.DataFrame({"g": values})
    result = Cube(
        ["g"], [Count(), Share()], policy=ComputePolicy(MissingPolicy(dimension="keep"))
    ).compute(frame)
    table = result.layout()
    assert result.shape == (3, 2)
    np.testing.assert_allclose(table["count"].to_numpy(), [2, 1, 1])
    assert table["share"].sum() == 1


def test_native_aggregation_does_not_repeat_for_each_measure(monkeypatch):
    original = pd.DataFrame.groupby
    calls = []

    def recording(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "groupby", recording)
    Cube(["g"], [Count(), Share(), EventRate("y"), AUC("s", "y"), KS("s", "y")]).compute(
        pd.DataFrame({"g": ["a"] * 4, "y": [0, 1, 0, 1], "s": [0, 1, 0.3, 0.8]})
    )
    assert len(calls) == 1


def test_custom_dimension_and_invalid_output():
    @dataclass(frozen=True)
    class Parity(BaseDimension):
        @property
        def output_name(self):
            return "parity"

        def required_columns(self):
            return ("x",)

        def transform(self, data, context=None):
            return data.x % 2

    frame = pd.DataFrame({"x": [1, 2, 3]}, index=[9, 8, 7])
    table = Cube([Parity()]).compute(frame).layout()
    assert table.loc[1, "count"] == 2

    class Broken(Parity):
        def transform(self, data, context=None):
            return super().transform(data).reset_index(drop=True)

    with pytest.raises(DimensionError, match="preserve"):
        Cube([Broken()]).compute(frame)


def test_source_output_and_internal_name_collisions():
    frame = pd.DataFrame({"x": [0, 1], "x_bin": [0, 1], "a0": [1, 2], "d0": [3, 4]})
    result = Cube([BinDimension("x", QuantileBinner(2))], [EventRate("x_bin")]).fit_compute(frame)
    assert result.layout().iloc[:, 0].tolist() == [0, 1]
    assert Cube(["a0"]).compute(frame).layout()["count"].tolist() == [1, 1]
    with pytest.raises(DimensionError, match="Duplicate"):
        Cube([ColumnDimension("x", name="band"), BinDimension("x", QuantileBinner(), name="band")])


def test_all_missing_dimension_and_filter_mask():
    frame = pd.DataFrame({"g": [None, None]})
    result = Cube(["g"], policy=ComputePolicy(MissingPolicy(dimension="keep"))).compute(frame)
    assert result.shape == (1, 1)
    assert result.layout().iloc[0, 0] == 2
    result = Cube(filters=[lambda df: pd.Series([True, pd.NA], dtype="boolean")]).compute(frame)
    assert result.layout().iloc[0, 0] == 1
    with pytest.raises(EngineError, match="aligned"):
        Cube(filters=[lambda df: pd.Series([True, True], index=[9, 8])]).compute(frame)


def test_logs_and_pickle_roundtrip(caplog):
    import pickle

    frame = pd.DataFrame({"x": [0, 1, 2, 3]})
    with caplog.at_level("INFO", logger="phl_risk.analysis"):
        cube = Cube([BinDimension("x", QuantileBinner(2))]).fit(frame)
        result = cube.compute(frame)
    assert [r.message for r in caplog.records] == [
        "analysis.cube.fit.start",
        "analysis.cube.fit.end",
        "analysis.cube.compute.start",
        "analysis.cube.compute.end",
    ]
    restored = pickle.loads(pickle.dumps(cube))
    pd.testing.assert_frame_equal(restored.compute(frame).data_, result.data_)
