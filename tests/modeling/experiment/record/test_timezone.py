from datetime import datetime
from zoneinfo import ZoneInfo

from phl_risk.modeling.experiment._utils import shanghai_now
from phl_risk.modeling.experiment.record.comparison import index_records
from phl_risk.modeling.experiment.record.run import RunRecord
from phl_risk.modeling.experiment.record.store import ExperimentStore


def test_generated_times_and_directory_use_shanghai(tmp_path):
    assert datetime.fromisoformat(shanghai_now()).utcoffset().total_seconds() == 28800
    store = ExperimentStore(
        tmp_path, "lgb", method={"name": "lightgbm", "family": "tree", "prefix": "lgb"}
    )
    facts = {
        "model": {"family": "tree", "backend": "lightgbm", "backend_version": "4"},
        "input": {},
        "configuration": {"supplied": {}, "resolved": {}},
        "environment": {},
    }
    path = store.allocate("test", facts)
    record = store.list_records()[0]
    assert record.created_at.endswith("+08:00")
    assert path.name.endswith(datetime.fromisoformat(record.created_at).strftime("%Y_%m_%d_%H%M%S"))
    store.fail(path, ValueError("example"))
    assert store.list_records()[0].failed_at.endswith("+08:00")


def test_old_utc_records_display_in_shanghai_without_rewriting(tmp_path):
    store = ExperimentStore(tmp_path, "old")
    facts = {
        "model": {"family": "test", "backend": "test", "backend_version": "1"},
        "input": {},
        "configuration": {"supplied": {}, "resolved": {}},
        "environment": {},
    }
    path = store.allocate("old", facts)
    data = store.list_records()[0].to_dict()
    data["created_at"] = data["started_at"] = "2026-09-26T20:30:00+00:00"
    record = RunRecord.from_dict(data)
    original = (path / "run.json").read_bytes()
    view = index_records([record])
    assert view.iloc[0].created_at == "2026-09-27T04:30:00+08:00"
    assert record.created_at == data["created_at"]
    assert (path / "run.json").read_bytes() == original
    assert datetime.fromisoformat(view.iloc[0].created_at).astimezone(
        ZoneInfo("UTC")
    ) == datetime.fromisoformat(data["created_at"])
