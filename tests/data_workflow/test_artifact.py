import joblib
import pandas as pd
import pytest

from phl_risk.data_prep import DataPrep
from phl_risk.data_prep.steps import MissingImputer
from phl_risk.data_quality import DataQuality
from phl_risk.data_quality.checks import SchemaCheck
from phl_risk.data_workflow import DataWorkflow, PrepStage
from phl_risk.exceptions import ArtifactError


@pytest.mark.parametrize("kind", ["quality", "prep", "workflow"])
def test_roundtrip_and_metadata(tmp_path, kind):
    X = pd.DataFrame({"x": [1.0, 2.0, None]})
    candidates = dict(
        quality=DataQuality([SchemaCheck()]),
        prep=DataPrep([MissingImputer(["x"])]),
        workflow=DataWorkflow([PrepStage("fill", DataPrep([MissingImputer(["x"])]))]),
    )
    fitted = candidates[kind].fit(X)
    path = tmp_path / "artifact.joblib"
    before = joblib.hash(fitted)
    fitted.save(path)
    assert joblib.hash(fitted) == before
    envelope = joblib.load(path)
    assert envelope["artifact_version"] == 1
    assert "joblib" in envelope["dependencies"]
    loaded = type(fitted).load(path)
    if kind == "quality":
        assert loaded.validate(X).summary() == fitted.validate(X).summary()
    else:
        pd.testing.assert_frame_equal(loaded.transform(X), fitted.transform(X))
    envelope["dependencies"]["pandas"] = "0.0.1"
    joblib.dump(envelope, path)
    with pytest.warns(UserWarning, match="trained=0.0.1"):
        type(fitted).load(path)
    envelope["artifact_version"] = 100
    joblib.dump(envelope, path)
    with pytest.raises(ArtifactError, match="version"):
        type(fitted).load(path)


def test_wrong_type_corrupt_and_failed_save_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "artifact.joblib"
    quality = DataQuality([SchemaCheck()]).fit(pd.DataFrame({"x": [1]}))
    quality.save(path)
    with pytest.raises(ArtifactError, match="Expected DataPrep"):
        DataPrep.load(path)
    content = path.read_bytes()

    def broken(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(joblib, "dump", broken)
    with pytest.raises(ArtifactError, match="disk failure"):
        quality.save(path)
    assert path.read_bytes() == content
    path.write_bytes(b"not a pickle")
    with pytest.raises(ArtifactError):
        DataQuality.load(path)
