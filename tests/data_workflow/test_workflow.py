import joblib
import pandas as pd
import pytest

from phl_risk.data_prep import DataPrep
from phl_risk.data_prep.steps import MissingImputer, ToNumeric
from phl_risk.data_quality import DataQuality
from phl_risk.data_quality.checks import MissingRateCheck, SchemaCheck
from phl_risk.data_workflow import DataWorkflow, PrepStage, QualityStage
from phl_risk.exceptions import DataWorkflowError


def test_sequential_fit_no_fit_on_transform_and_aggregation(monkeypatch):
    X = pd.DataFrame({"x": ["10", None, "30"]})
    original = X.copy()
    workflow = DataWorkflow(
        [
            QualityStage("raw", DataQuality([SchemaCheck()])),
            PrepStage("prep", DataPrep([ToNumeric(["x"], "coerce"), MissingImputer(["x"])])),
            QualityStage(
                "prepared", DataQuality([SchemaCheck(), MissingRateCheck(["x"], max_rate=0)])
            ),
        ]
    ).fit(X)
    profile = workflow.stages_[-1].quality_.reference_profile_.profiles["x"]
    assert profile.missing_count == 0 and profile.mean == 20
    before = joblib.hash(workflow)

    def forbidden(*args, **kwargs):
        raise AssertionError("fit called during transform")

    monkeypatch.setattr(DataPrep, "fit", forbidden)
    monkeypatch.setattr(DataQuality, "fit", forbidden)
    workflow.transform(X)
    result = workflow.run(X)
    assert joblib.hash(workflow) == before
    assert tuple(result.quality_reports) == ("raw", "prepared")
    assert len(result.prep_audits["prep"]) == 2
    pd.testing.assert_frame_equal(original, X)


def test_arbitrary_sequence_and_atomic_refit():
    X = pd.DataFrame({"x": ["10", None, "30"]})
    workflow = DataWorkflow(
        [
            PrepStage("numeric", DataPrep([ToNumeric(["x"], "coerce")])),
            QualityStage("middle", DataQuality([SchemaCheck()])),
            PrepStage("fill", DataPrep([MissingImputer(["x"])])),
        ]
    ).fit(X)
    assert workflow.transform(X).x.tolist() == [10, 20, 30]
    before = joblib.hash(workflow)
    with pytest.raises(Exception):
        workflow.fit(X.rename(columns={"x": "missing"}))
    assert joblib.hash(workflow) == before


@pytest.mark.parametrize("policy", ["raise", "warn", "continue"])
def test_failure_policy(policy):
    X = pd.DataFrame({"x": [1, 2]})
    workflow = DataWorkflow([QualityStage("schema", DataQuality([SchemaCheck()]), policy)]).fit(X)
    current = X.rename(columns={"x": "lost"})
    if policy == "raise":
        with pytest.raises(DataWorkflowError) as exc:
            workflow.run(current)
        assert exc.value.report.failed and exc.value.stage == "schema"
    elif policy == "warn":
        with pytest.warns(UserWarning):
            assert workflow.run(current).quality_reports["schema"].failed
    else:
        assert workflow.run(current).quality_reports["schema"].failed
