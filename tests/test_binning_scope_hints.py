"""Explain scope without warning on valid global binning or changing results."""

import logging
import warnings

import pandas as pd
import pytest

from phl_risk.analysis import (
    BinDimension,
    ColumnDimension,
    Cube,
    EqualWidthBinner,
    FixedBinner,
    QuantileBinner,
)
from phl_risk.exceptions import TransformError

LOGGER = "phl_risk.analysis"
EVENT = "analysis.binning.global_scope"


def data():
    return pd.DataFrame(
        {"group": ["A"] * 5 + ["B"] * 5, "channel": ["app", "web"] * 5, "score": range(10)}
    )


def hints(caplog):
    return [r for r in caplog.records if getattr(r, "event", None) == EVENT]


@pytest.mark.parametrize(
    "binner", [QuantileBinner(2), EqualWidthBinner(2), QuantileBinner(2).fit(pd.Series(range(20)))]
)
def test_grouped_global_binner_emits_one_info_and_preserves_results(caplog, binner):
    model = Cube(["group", BinDimension("score", binner)])
    with (
        caplog.at_level(logging.INFO, logger=LOGGER),
        warnings.catch_warnings(record=True) as caught,
    ):
        result = model.fit_compute(data(), totals=True)
        result.layout(totals=True)
        model.compute(data(), totals=True)
    records = hints(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert records[0].dimension == "score_bin"
    message = records[0].getMessage()
    assert "globally" in message
    assert "prepared fit reference" in message
    assert "fit_by=['group']" in message
    assert not caught
    assert result.metadata_["total_grouping_sets"] == 3
    assert result.total(over=["group", "score_bin"]).layout().loc[0, "count"] == 10


@pytest.mark.parametrize(
    "dimensions",
    [
        ["group", BinDimension("score", QuantileBinner(2), fit_by=["group"])],
        ["group", BinDimension("score", FixedBinner([-float("inf"), 5, float("inf")]))],
        [BinDimension("score", QuantileBinner(2))],
        ["group"],
        [],
    ],
)
def test_no_hint_for_unambiguous_or_fixed_configurations(caplog, dimensions):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        Cube(dimensions).fit_compute(data())
    assert not hints(caplog)


def test_mixed_dimensions_only_report_global_and_suggest_valid_raw_fields(caplog):
    model = Cube(
        [
            ColumnDimension("group", name="customer"),
            "channel",
            BinDimension("score", QuantileBinner(2), name="global_bin"),
            BinDimension("score", QuantileBinner(2), fit_by=["channel"], name="local_bin"),
        ]
    )
    with caplog.at_level(logging.INFO, logger=LOGGER):
        model.fit(data())
    assert len(hints(caplog)) == 1
    message = hints(caplog)[0].getMessage()
    assert "fit_by=['channel']" in message
    assert "fit_by=['customer']" not in message
    assert hints(caplog)[0].dimension == "global_bin"


def test_renamed_group_does_not_suggest_invalid_fit_by(caplog):
    model = Cube(
        [ColumnDimension("group", name="customer"), BinDimension("score", QuantileBinner(2))]
    )
    with caplog.at_level(logging.INFO, logger=LOGGER):
        model.fit(data())
    message = hints(caplog)[0].getMessage()
    assert "unrenamed ColumnDimensions" in message
    assert "fit_by=['customer']" not in message
    assert "fit_by=['group']" not in message


def test_multiple_groups_filters_and_failed_fit(caplog):
    model = Cube(
        ["group", "channel", BinDimension("score", QuantileBinner(2))],
        filters=[lambda d: d.score.ge(5)],
    )
    with caplog.at_level(logging.INFO, logger=LOGGER):
        model.fit(data())
    assert "fit_by=['group', 'channel']" in hints(caplog)[0].getMessage()
    assert model.dimensions_[2].transformer.bin_edges_[1] == 7
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=LOGGER), pytest.raises(TransformError):
        model.fit(data().assign(score=float("nan")))
    assert not hints(caplog)


def test_library_does_not_configure_logging_and_explain_does_not_fit(caplog):
    logger = logging.getLogger(LOGGER)
    before = (
        logger.level,
        logger.disabled,
        logger.propagate,
        tuple(logger.handlers),
        logging.getLogger().level,
        tuple(logging.getLogger().handlers),
    )
    model = Cube(["group", BinDimension("score", QuantileBinner(2))])
    model.fit(data())
    after = (
        logger.level,
        logger.disabled,
        logger.propagate,
        tuple(logger.handlers),
        logging.getLogger().level,
        tuple(logging.getLogger().handlers),
    )
    assert before == after
    caplog.clear()
    fresh = Cube(["group", BinDimension("score", QuantileBinner(2), fit_by=["group"])])
    with caplog.at_level(logging.INFO, logger=LOGGER):
        fresh.explain()
    assert not fresh.is_fitted
    assert not hints(caplog)


def test_explain_distinguishes_scopes_and_total_semantics():
    global_cube = Cube(["group", BinDimension("score", QuantileBinner(2))])
    local_cube = Cube(["group", BinDimension("score", QuantileBinner(2), fit_by=["group"])])
    fixed_cube = Cube(["group", BinDimension("score", FixedBinner([0, 5, 10]))])
    assert "Fit scope: global" in global_cube.explain(format="text")
    assert "Fit scope: within groups; fit_by=['group']" in local_cube.explain(format="text")
    assert "Fixed boundaries (no learning)" in fixed_cube.explain(format="text")
    assert "Recompute pooled samples" in global_cube.explain(totals=True).Description.tolist()
    assert "retain fit_by fields: group" in local_cube.explain(totals=True, format="text")


def test_stratified_explain_does_not_present_template_edges_as_shared_boundaries():
    template = QuantileBinner(2).fit(pd.Series(range(100, 110)))
    model = Cube(["group", BinDimension("score", template, fit_by=["group"])]).fit(data())
    table = model.explain()
    assert not table.Section.eq("Bin edges").any()
    assert "Fitted strata: 2" in table.Description.tolist()
