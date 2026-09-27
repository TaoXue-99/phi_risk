import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from phl_risk.modeling.experiment.record.comparison import compare_parameters
from phl_risk.modeling.experiment.record.run import Run
from phl_risk.modeling.experiment.record.store import ExperimentStore


def test_numbering_across_processes(tmp_path):
    script = """
import sys
from phl_risk.modeling.experiment.record.store import ExperimentStore
spec = {"name": "example", "family": "test", "prefix": "example"}
store = ExperimentStore(sys.argv[1], "example", method=spec)
facts = {"model": {"family": "test", "backend": "example", "backend_version": "1"},
         "input": {}, "configuration": {"supplied": {}, "resolved": {}}, "environment": {}}
for i in range(3):
    path = store.allocate("parallel", facts)
    store.fail(path, RuntimeError("expected"))
"""
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: subprocess.run(
                    [sys.executable, "-c", script, str(tmp_path)],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                ),
                range(4),
            )
        )
    assert len(results) == 4
    store = ExperimentStore(
        tmp_path,
        "example",
        method={"name": "example", "family": "test", "prefix": "example"},
        create=False,
    )
    records = store.list_records()
    assert len(records) == 12
    assert {r.run for r in records} == {f"example_run_{i:02d}" for i in range(1, 13)}


def test_config_diff_missing_null_nested_and_unchanged(tmp_path):
    store = ExperimentStore(tmp_path, "diff")
    facts = {
        "model": {"family": "test", "backend": "example", "backend_version": "1"},
        "input": {},
        "environment": {},
    }
    records = []
    for name, config in [
        ("a", {"nested": {"null": None, "same": [1, 2]}, "old": 3}),
        ("b", {"nested": {"same": [1, 2]}, "new": None}),
    ]:
        path = store.allocate(
            name, {**facts, "configuration": {"supplied": config, "resolved": config}}
        )
        store.complete(path, {"metrics": {}}, ())
        records.append(Run.load(path).record)
    diff = compare_parameters(*records)
    assert diff.parameter.tolist() == ["nested.null", "new", "old"]
    assert diff.change.tolist() == ["removed", "added", "removed"]
    assert diff.left_present.tolist() == [True, False, True]
    assert diff.right_present.tolist() == [False, True, False]
    assert len(compare_parameters(*records, only_changed=False)) == 4
    assert compare_parameters(*records, params=["same"]).empty


def test_initialization_without_optional_backend(tmp_path):
    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split(".")[0] in {"lightgbm", "hydra", "yaml"}:
            raise ImportError(name)
sys.meta_path.insert(0, Block())
from phl_risk.modeling.experiment import Experiment
exp = Experiment(name="without_backend", root=sys.argv[1])
method = exp.initialize(method="lightgbm", objective="binary", metric="auc")
assert (method.config_dir / "baseline.yaml").exists()
assert exp.open_method("lightgbm").path == method.path
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
    )
