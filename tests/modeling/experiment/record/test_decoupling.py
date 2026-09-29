"""Backend-free extensions and compatibility at the recording boundaries."""

import ast
import json
from pathlib import Path

import pytest

from phl_risk.exceptions import ExperimentError, RunError
from phl_risk.modeling.experiment import Experiment, Run


def test_partition_iterators_and_duplicates(tmp_path):
    space = Experiment(root=tmp_path).initialize(method="generic")
    with space.start_run(comparison_partitions=iter(["train", "valid"])) as attempt:
        attempt.log_metrics({"train": {"loss": 1.0}, "valid": {"loss": 2.0}})
    assert attempt.record.input["comparison_partitions"] == ["train", "valid"]
    assert "valid_loss" in space.compare()
    with pytest.raises(RunError, match="Duplicate"):
        space.start_run(comparison_partitions=["valid", "valid"])
    with pytest.raises(RunError, match="sequence"):
        space.start_run(comparison_partitions=42)


def test_method_policy_survives_reload_and_run_omission(tmp_path):
    exp = Experiment(root=tmp_path)
    space = exp.initialize(method="generic", comparison_partitions=iter(["train", "valid"]))
    for selection in [None, ["oot"]]:
        with space.start_run(comparison_partitions=selection) as attempt:
            attempt.log_metrics({"train": {"auc": 0.9}, "valid": {"auc": 0.8}, "oot": {"auc": 0.7}})
    restored = exp.open_method("generic")
    assert restored.comparison_partitions == ("train", "valid")
    assert "oot_auc" not in restored.compare()
    assert "oot_auc" in restored.compare(partitions=["oot"])
    assert restored.compare(runs=[]).empty
    assert exp.initialize(method="generic").comparison_partitions == ("train", "valid")
    with pytest.raises(ExperimentError, match="comparison_partitions"):
        exp.initialize(method="generic", comparison_partitions=["oot"])


def test_explicit_deep_task_labels_do_not_require_lgb_config(tmp_path):
    space = Experiment(root=tmp_path).initialize(method="torch", family="deep")
    with space.start_run(
        params={"optimizer": {"lr": 0.001}, "criterion": "user-owned"},
        task={"objective": "CrossEntropyLoss", "metric": ["accuracy", "loss"]},
    ) as attempt:
        attempt.log_metrics({"valid": {"accuracy": 0.8, "loss": 0.3}})
        attempt.log_artifact(
            "checkpoint", "weights.bin", lambda p: p.write_bytes(b"test"), format="test"
        )
    frame = space.compare()
    assert frame.iloc[0].objective == "CrossEntropyLoss"
    assert frame.iloc[0].metric == ["accuracy", "loss"]
    assert "objective" not in attempt.record.config
    assert attempt.record.artifact_path("checkpoint").read_bytes() == b"test"
    with pytest.raises(RunError, match="task"):
        with space.start_run(task={"unsupported": "value"}):
            pass


def test_explicit_labels_override_legacy_config_without_rewriting_it(tmp_path):
    exp = Experiment(root=tmp_path)
    with exp.start_run(
        params={"objective": "binary"}, task={"objective": "display", "metric": None}
    ) as attempt:
        pass
    assert exp.compare().iloc[0].objective == "display"
    assert attempt.record.config["objective"] == "binary"


def test_store_owns_updates_and_prevents_completed_mutations(tmp_path):
    space = Experiment(root=tmp_path).initialize(method="generic")
    with space.start_run() as attempt:
        attempt.log_input(features=["a"])
        with pytest.raises(RunError, match="reserved"):
            space._store.update_facts(attempt.path, status="completed")
    before = (attempt.path / "run.json").read_bytes()
    with pytest.raises(RunError, match="immutable"):
        space._store.update_facts(attempt.path, metadata={"changed": True})
    assert (attempt.path / "run.json").read_bytes() == before


def test_metadata_only_access_and_explicit_artifact_verification(tmp_path, monkeypatch):
    from phl_risk.modeling.experiment.record import run as module

    space = Experiment(root=tmp_path).initialize(method="generic")
    with space.start_run() as attempt:
        attempt.log_json("history", {"loss": [1.0]})
    calls = []
    original = module.verify_artifact

    def counted(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(module, "verify_artifact", counted)
    item = attempt.record
    item = space.get_run(item.run, verify=False)
    assert not calls
    item.verify_artifacts()
    assert len(calls) == 1
    (item.path / "history.json").write_text("corrupt")
    assert Run.load(item.path, verify=False).status == "completed"
    for access in [
        lambda: item.read_json("history"),
        lambda: Run.load(item.path),
        item.verify_artifacts,
    ]:
        with pytest.raises(RunError):
            access()


def test_corrupt_method_manifest_is_domain_error(tmp_path):
    exp = Experiment(root=tmp_path)
    space = exp.initialize(method="generic")
    (space.path / "method.json").write_text(json.dumps({"artifact_version": 3}))
    with pytest.raises(ExperimentError, match="manifest"):
        exp.open_method("generic")


def test_record_and_initialization_do_not_import_adapters_or_session_helpers():
    import phl_risk.modeling.experiment as module

    root = Path(module.__file__).parent
    for path in [
        *root.joinpath("record").glob("*.py"),
        root / "configuration.py",
        root / "_snapshot.py",
        *root.joinpath("initialization").glob("*.py"),
    ]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert "adapters" not in (node.module or ""), path
                if path.parent.name != "record":
                    assert "session" not in (node.module or ""), path
