import pytest

from phl_risk.exceptions import RunError
from phl_risk.modeling.experiment.record.run import Run
from phl_risk.modeling.experiment.record.store import ExperimentStore


def test_running_and_failed_records_are_readable_and_detached(tmp_path, record_facts):
    store = ExperimentStore(tmp_path, "records")
    path = store.allocate("attempt", record_facts)
    running = Run.load(path)
    assert running.status == "running"
    assert running.started_at >= running.created_at
    assert running.features == ("x",)
    value = running.record.to_dict()
    value["input"]["features"].append("changed")
    assert running.features == ("x",)
    store.fail(path, ValueError("bad input"))
    failed = Run.load(path)
    assert failed.status == "failed"
    assert failed.record.to_dict()["error"]["type"] == "ValueError"
    assert failed.metrics == {}
    assert failed.artifacts == {}
    with pytest.raises(RunError, match="completed"):
        _ = failed.artifact_path("weights")
    assert running.status == "running"  # snapshots do not change
    assert len(store.list_records()) == 1


def test_legacy_experiment_rejected_without_rewriting(tmp_path):
    import json

    path = tmp_path / "old"
    path.mkdir()
    manifest = path / "experiment.json"
    manifest.write_text(json.dumps({"artifact_version": 1, "name": "old"}))
    original = manifest.read_bytes()
    with pytest.raises(Exception, match="version|0.5"):
        ExperimentStore(tmp_path, "old")
    assert manifest.read_bytes() == original
