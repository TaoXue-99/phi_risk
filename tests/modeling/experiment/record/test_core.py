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
                                "sklearn", "numpy", "pandas", "hydra")):
            raise ImportError(fullname)
sys.meta_path.insert(0, Block())
from phl_risk.modeling.experiment import Experiment, Run, RunRecord
from tempfile import TemporaryDirectory
with TemporaryDirectory() as root:
    exp = Experiment(name="independent", root=root)
    assert exp.path.is_dir()
""",
        ],
        check=True,
    )
