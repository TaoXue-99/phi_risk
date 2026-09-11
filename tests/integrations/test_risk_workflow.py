import importlib.util
from pathlib import Path

import joblib
import pandas as pd
import pytest

from phl_risk.data_prep import BasePrepStep, DataPrep
from phl_risk.data_quality import BaseQualityCheck, CheckResult, CheckStatus, DataQuality
from phl_risk.data_workflow import DataWorkflow, PrepStage, QualityStage
from phl_risk.exceptions import DataWorkflowError


def test_full_risk_incidents(tmp_path):
    pytest.importorskip("optbinning")
    spec = importlib.util.spec_from_file_location(
        "risk_example", Path(__file__).parents[2] / "examples/data_workflow_risk.py"
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    train = example.make_risk_data()
    X, y = train.drop(columns="label"), train.label
    workflow = example.make_workflow().fit(X, y)
    original = X.copy(deep=True)
    oot = X.copy(deep=True)
    oot.loc[:209, "income"] = None
    oot.loc[210:219, "income"] = "unknown"
    oot["score"] += 5
    oot["user_type"] = oot.user_type.replace({"web": "harmony"})
    before = joblib.hash(workflow)
    result = workflow.run(oot)
    assert joblib.hash(workflow) == before
    pd.testing.assert_frame_equal(X, original)
    raw = {r.name: r for r in result.quality_reports["raw"].results}
    assert raw["CategorySetCheck"].details["new_categories"] == ("harmony",)
    assert raw["CategorySetCheck"].details["missing_categories"] == ("web",)
    for name in ("MissingRateCheck", "NumericConvertibleCheck", "DistributionDriftCheck"):
        assert raw[name].failed
    assert result.prep_audits["features"][0].details["income"]["new_null"] == 10
    assert result.prep_audits["features"][1].details["income"]["imputed_count"] == 220
    path = tmp_path / "risk.joblib"
    workflow.save(path)
    loaded = DataWorkflow.load(path)
    pd.testing.assert_frame_equal(loaded.transform(oot), result.data)
    # Missing columns are a gate failure before any prep can execute.
    gate = DataWorkflow([QualityStage("raw", DataQuality([example.SchemaCheck()]))]).fit(X)
    with pytest.raises(DataWorkflowError) as exc:
        gate.run(oot.drop(columns=["income", "score"]))
    assert exc.value.report.results[0].details["missing_columns"] == ("income", "score")


class MyCustomCheck(BaseQualityCheck):
    requires_fit = False

    def _validate(self, X):
        return CheckResult(
            "positive",
            CheckStatus.PASS if X.x.gt(0).all() else CheckStatus.FAIL,
            ("x",),
            "positive values",
        )


class MyCustomPrep(BasePrepStep):
    requires_fit = False

    def _transform(self, X):
        return X.assign(x=X.x + 1)


def test_user_plugins_without_framework_changes():
    workflow = DataWorkflow(
        [
            PrepStage("custom", DataPrep([MyCustomPrep()])),
            QualityStage("custom_check", DataQuality([MyCustomCheck()])),
        ]
    )
    output = workflow.fit_transform(pd.DataFrame({"x": [0, 1]}))
    assert output.x.tolist() == [1, 2]


def test_optional_dependency_import_is_lazy():
    import subprocess
    import sys

    script = """
import sys
from importlib.abc import MetaPathFinder
class BlockOptBinning(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "optbinning" or fullname.startswith("optbinning."):
            raise ImportError("blocked for base-install test")
sys.meta_path.insert(0, BlockOptBinning())
import pandas as pd
import phl_risk
from phl_risk.data_prep.integrations import OptBinningStep
from phl_risk.exceptions import OptionalDependencyError
try:
    OptBinningStep(["x"]).fit(pd.DataFrame({"x": [1., 2.]}), [0, 1])
except OptionalDependencyError as exc:
    assert "phl-risk[binning]" in str(exc)
else:
    raise AssertionError("OptionalDependencyError was not raised")
"""
    subprocess.run([sys.executable, "-c", script], check=True)
