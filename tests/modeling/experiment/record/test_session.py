import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from phl_risk.exceptions import RunError
from phl_risk.modeling.experiment import Experiment, Run
from phl_risk.modeling.experiment.record.artifact import atomic_write
from phl_risk.modeling.experiment.record.store import ExperimentStore


@pytest.fixture
def space(tmp_path):
    return Experiment(root=tmp_path / "experiments").initialize(method="example", family="deep")


def test_generic_backend_space_native_code_and_independent_reload(space):
    with space.start_run(params={"epochs": 2}, comparison_partitions=["valid"]) as session:
        session.log_input(features=["text"], partitions={"valid": {"rows": 2}})
        session.log_metrics({"valid": {"loss": 1.5}, "test": {"loss": 1.7}})
        session.log_result(epochs=2)
        session.log_artifact(
            "weights", "weights.bin", lambda p: p.write_bytes(b"native"), format="bytes"
        )
    run = session.record
    assert run.model["family"] == "deep"
    assert run.artifact_path("weights").read_bytes() == b"native"
    assert run.result["epochs"] == 2
    assert Run.load(run.path).metrics == run.metrics
    frame = space.compare(params=["epochs"])
    assert frame.iloc[0].params == {"epochs": 2}
    assert frame.iloc[0].valid_loss == 1.5 and "test_loss" not in frame
    restored = Experiment(root=space.path.parent).open_method("example")
    assert restored.get_run(run.run).features == ("text",)
    values = run.metrics
    values["valid"]["loss"] = -10
    assert run.metrics["valid"]["loss"] == 1.5


def test_failures_interruptions_and_number_not_reused(space):
    with pytest.raises(KeyboardInterrupt):
        with space.start_run() as session:
            session.log_json("partial", {"value": 1})
            raise KeyboardInterrupt("stop")
    failed = session.record
    assert failed.status == "failed" and failed.error["type"] == "KeyboardInterrupt"
    assert not (failed.path / ".pending").exists()
    assert space.compare().empty
    with space.start_run() as second:
        pass
    assert second.record.run == "example_run_02"


def test_snapshot_and_immutability(space, tmp_path):
    source = tmp_path / "config.yaml"
    source.write_text("depth: 3")
    params = {"depth": 3}
    with space.start_run(params=params) as session:
        session.log_file("source", source)
        params["depth"] = 9
        source.write_text("depth: 9")
    run = session.record
    assert run.config == {"depth": 3}
    assert run.artifact_path("source").read_text() == "depth: 3"
    with pytest.raises(RunError):
        session.log_metrics({"valid": {"loss": 2}})
    with pytest.raises(RunError):
        with session:
            pass
    store = ExperimentStore(
        space.path.parent,
        "example",
        method={"name": "example", "family": "deep", "prefix": "example"},
    )
    before = (run.path / "run.json").read_bytes()
    with pytest.raises(RunError, match="immutable"):
        store.complete(run.path, {}, ())
    store.fail(run.path, RuntimeError("late"))
    assert (run.path / "run.json").read_bytes() == before


@pytest.mark.parametrize(
    "filename", ["../bad", "/absolute", "run.json", "run.json/child", ".pending/x", "a\\b"]
)
def test_unsafe_artifact_paths(space, filename):
    with pytest.raises(RunError):
        with space.start_run() as session:
            session.log_artifact("bad", filename, lambda p: p.write_text("bad"), format="txt")
    assert session.record.status == "failed"


def test_duplicate_artifact_and_params_rejected(space):
    with space.start_run(params={"a": 1}) as session:
        session.log_json("one", {"v": 1})
        with pytest.raises(RunError, match="Duplicate"):
            session.log_json("one", {"v": 2})
        with pytest.raises(RunError, match="different"):
            session.log_params({"a": 2})
        session.log_metrics({"valid": {"loss": 1}})
        with pytest.raises(RunError):
            session.log_metrics({"valid": {"loss": 2}})
    assert session.record.read_json("one") == {"v": 1}


def test_serialization_failure_and_atomic_commit_failure(space, monkeypatch):
    def broken(path):
        path.write_text("partial")
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError):
        with space.start_run() as session:
            session.log_artifact("bad", "bad.txt", broken, format="txt")
    assert session.record.status == "failed"
    assert not list(session.path.rglob("*.txt"))
    from phl_risk.modeling.experiment.record import store

    original = store.atomic_write

    def fail(path, writer, **options):
        if path.name == "data.json":
            raise OSError("disk failed")
        return original(path, writer, **options)

    monkeypatch.setattr(store, "atomic_write", fail)
    with pytest.raises(OSError):
        with space.start_run() as second:
            second.log_json("data", {"x": 1})
    assert second.record.status == "failed"


def test_atomic_previous_file_unchanged(tmp_path):
    path = tmp_path / "record.json"
    path.write_text("previous")

    def broken(temporary):
        temporary.write_text("partial")
        raise OSError("disk failed")

    with pytest.raises(OSError):
        atomic_write(path, broken)
    assert path.read_text() == "previous"
    assert len(list(tmp_path.iterdir())) == 1


def test_concurrent_contexts_and_diff(space):
    def execute(i):
        with space.start_run(params={"depth": i}, name="same") as session:
            session.log_metrics({"valid": {"loss": float(i)}})
        return session.record

    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(execute, range(12)))
    assert len({r.run_id for r in runs}) == 12
    diff = space.compare_params(runs[0].run_id, runs[1].run_id)
    assert diff.parameter.tolist() == ["depth"]
    with pytest.raises(Exception, match="ambiguous"):
        space.get_run("same")


def test_initialization_parallel_spaces_and_no_overwrite(tmp_path):
    exp = Experiment(root=tmp_path / "experiments")
    lgb = exp.initialize(method="lgb", objective="binary", metric="auc")
    xgb = exp.initialize(method="xgb", family="tree")
    deep = exp.initialize(method="torch", family="deep")
    assert lgb.path.parent == xgb.path.parent == deep.path.parent
    source = lgb.config_dir / "baseline.yaml"
    source.write_text("user: edited")
    exp.initialize(method="lgb")
    assert source.read_text() == "user: edited"
    source.unlink()
    exp.open_method("lgb")
    assert not source.exists()


def test_corrupt_manifest_detected(space):
    with space.start_run() as session:
        session.log_json("history", {})
    manifest = session.path / "run.json"
    facts = json.loads(manifest.read_text())
    del facts["result"]["metrics"]
    manifest.write_text(json.dumps(facts))
    with pytest.raises(RunError):
        Run.load(session.path)


def test_native_ufunc_identities_do_not_collapse(space):
    from scipy.special import expit, logit

    with space.start_run(params={"transform": expit}) as first:
        pass
    with space.start_run(params={"transform": logit}) as second:
        pass
    assert first.record.config["transform"] != second.record.config["transform"]
    assert "expit" in first.record.config["transform"]["callable"]
    assert space.compare().iloc[0].params["transform"] == first.record.config["transform"]
    assert space.compare_params(first.record.run, second.record.run).parameter.tolist() == [
        "transform"
    ]


def test_space_label_does_not_require_backend_package_name(space):
    with space.start_run() as session:
        session.log_model_info(family="deep", backend="torch", backend_version="2")
        with pytest.raises(RunError, match="differs"):
            session.log_model_info(family="tree", backend="lightgbm", backend_version="4")
    assert session.record.model["backend"] == "torch"


def test_original_training_error_preserved_if_failure_recording_fails(space, monkeypatch):
    original = RuntimeError("native failure")

    def broken(*args):
        raise OSError("status disk failure")

    monkeypatch.setattr(space._session._store, "fail", broken)
    with pytest.raises(RuntimeError) as caught:
        with space.start_run():
            raise original
    assert caught.value is original
    assert "status disk failure" in original.__notes__[0]


def test_model_free_posthoc_records_do_not_invent_training_seconds(space):
    with space.start_run(metadata={"recording_scope": "posthoc"}) as session:
        session.log_metrics({"external": {"score": 1.0}})
    assert "training_seconds" not in session.record.result
    assert session.record.result["recording_seconds"] >= 0
