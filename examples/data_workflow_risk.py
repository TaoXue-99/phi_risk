"""End-to-end risk data lifecycle. Run with uv run --extra binning python ..."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from phl_risk.data_prep import DataPrep
from phl_risk.data_prep.integrations import OptBinningStep
from phl_risk.data_prep.steps import MissingImputer, ToNumeric
from phl_risk.data_quality import DataQuality
from phl_risk.data_quality.checks import (
    CategorySetCheck,
    MissingRateCheck,
    NumericConvertibleCheck,
    SchemaCheck,
    UniqueCheck,
)
from phl_risk.data_workflow import DataWorkflow, PrepStage, QualityStage


def make_risk_data(seed=42, size=600):
    rng = np.random.default_rng(seed)
    score = rng.normal(size=size)
    income = rng.lognormal(8, 0.5, size)
    frame = pd.DataFrame(
        dict(
            user_id=np.arange(size),
            dt=pd.Timestamp("2026-01-01"),
            user_type=rng.choice(["android", "ios", "web"], size),
            income=income.round(2).astype(str),
            age=rng.integers(18, 70, size),
            score=score,
            label=(score + rng.normal(size=size) > 0).astype(int),
        )
    )
    frame.loc[:11, "income"] = None
    return frame


def make_workflow():
    raw = DataQuality(
        [
            SchemaCheck(),
            UniqueCheck(["user_id", "dt"]),
            CategorySetCheck("user_type"),
            MissingRateCheck(["income"], warn_delta=0.1, fail_delta=0.2),
            NumericConvertibleCheck(["income"]),
        ]
    )
    prep = DataPrep(
        [
            ToNumeric(["income"], "coerce"),
            MissingImputer(["income"]),
            OptBinningStep(["income", "score"], max_n_prebins=5),
        ]
    )
    prepared = DataQuality([SchemaCheck(), MissingRateCheck(["income", "score"], max_rate=0)])
    return DataWorkflow(
        [
            QualityStage("raw", raw, on_fail="continue"),
            PrepStage("features", prep),
            QualityStage("prepared", prepared),
        ]
    )


def main():
    train = make_risk_data()
    X, y = train.drop(columns="label"), train.label
    oot = make_risk_data(seed=3).drop(columns="label")
    oot.loc[:209, "income"] = None
    oot.loc[210:219, "income"] = "1000 PHP"
    oot["user_type"] = oot.user_type.replace({"web": "harmony"})
    oot["score"] += 4
    workflow = make_workflow().fit(X, y)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "risk.joblib"
        workflow.save(path)
        loaded = DataWorkflow.load(path)
        result = loaded.run(oot)
        pd.testing.assert_frame_equal(workflow.transform(oot), result.data)
    assert result.quality_reports["raw"].failed
    assert result.quality_reports["prepared"].passed
    assert result.prep_audits["features"][0].details["income"]["new_null"] == 10
    print(result.quality_reports["raw"].to_frame())
    print(result.summary())
    return workflow, result


if __name__ == "__main__":
    main()
