import subprocess
import sys


def test_record_and_experiment_import_without_execution_backends():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.startswith(("phl_risk.modeling.experiment.execution", "lightgbm", "torch",
                                "sklearn", "numpy", "pandas", "hydra", "yaml",
                                "phl_risk.modeling.experiment.adapters")):
            raise ImportError(fullname)
sys.meta_path.insert(0, Block())
from phl_risk.modeling.experiment import Experiment, Run, RunRecord
from tempfile import TemporaryDirectory
with TemporaryDirectory() as root:
    exp = Experiment(name="independent", root=root)
    assert exp.path.is_dir()
    space = exp.initialize(method="torch", family="deep", comparison_partitions=["valid"])
    with space.start_run(params={"epochs": 2}, task={"objective": "user_loss"}) as attempt:
        attempt.log_metrics({"valid": {"loss": 0.2}})
        attempt.log_artifact("weights", "checkpoint.bin",
                             lambda p: p.write_bytes(b"test-only"), format="test")
    assert attempt.record.metrics["valid"]["loss"] == 0.2
    assert attempt.record.artifact_path("weights").read_bytes() == b"test-only"
""",
        ],
        check=True,
    )
