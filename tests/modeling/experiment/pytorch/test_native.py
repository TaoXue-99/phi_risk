import os
import subprocess
import sys
from pathlib import Path

import pytest

from phl_risk.exceptions import RunError
from phl_risk.modeling.experiment import Experiment


def test_esmm_initializes_without_training_backends(tmp_path):
    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'torch', 'torchkeras', 'tensorboard', 'yaml', 'hydra'}:
            raise ImportError(name)
sys.meta_path.insert(0, Block())
from phl_risk.modeling.experiment import Experiment
space = Experiment(root=sys.argv[1]).initialize(method='esmm_mmoe')
p = space.config_dir / 'baseline.yaml'
assert 'ctcvr_weight' in p.read_text()
p.write_text('user-owned: true')
Experiment(root=sys.argv[1]).initialize(method='esmm_mmoe')
assert p.read_text() == 'user-owned: true'
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
    )


def test_baseline_drives_network_and_loss(tmp_path):
    pytest.importorskip("torch")
    yaml = pytest.importorskip("yaml")
    from phl_risk.modeling.models.esmm_mmoe import ESMMLoss, ESMMMoE

    space = Experiment(root=tmp_path).initialize(method="esmm_mmoe")
    cfg = yaml.safe_load((space.config_dir / "baseline.yaml").read_text())
    cfg["model"].update(num_continuous=3, categorical_cardinalities=[4])
    net = ESMMMoE(**cfg["model"])
    loss = ESMMLoss(**cfg["loss"])
    with space.start_run(config=cfg) as run:
        run.log_result(parameters=sum(p.numel() for p in net.parameters()))
    saved = space.get_run(run.path.name)
    assert saved.model["family"] == "deep"
    assert saved.config["loss"]["ctr_weight"] == loss.ctr_weight


def test_pytorch_snapshot_restore_and_integrity(tmp_path):
    torch = pytest.importorskip("torch")
    from phl_risk.modeling.experiment.adapters.pytorch import load_pytorch, record_pytorch
    from phl_risk.modeling.models.esmm_mmoe import ESMMMoE

    space = Experiment(root=tmp_path).initialize(method="esmm_mmoe")
    net = ESMMMoE(num_continuous=2).eval()
    cat, cont = torch.empty(3, 0, dtype=torch.long), torch.randn(3, 2)
    expected = net(cat, cont)
    with space.start_run(config={"model": net.get_config()}) as run:
        record_pytorch(run, net, model_config=net.get_config())
        with torch.no_grad():
            for p in net.parameters():
                p.add_(10)
    saved = space.get_run(run.path.name)
    restored = load_pytorch(saved, ESMMMoE(**saved.read_json("model_config"))).eval()
    torch.testing.assert_close(restored(cat, cont), expected)
    assert saved.model["backend"] == "pytorch"
    assert all(
        t.device.type == "cpu"
        for t in torch.load(saved.artifact_path("model"), weights_only=True).values()
    )
    saved.artifact_path("model").write_bytes(b"corrupted")
    with pytest.raises(RunError):
        load_pytorch(saved, ESMMMoE(num_continuous=2))


def test_generic_pytorch_model_and_wrong_architecture(tmp_path):
    torch = pytest.importorskip("torch")
    from phl_risk.modeling.experiment.adapters.pytorch import load_pytorch, record_pytorch

    space = Experiment(root=tmp_path).initialize(method="custom_torch", family="deep")
    original = torch.nn.Linear(2, 1)
    with space.start_run(params={}) as run:
        record_pytorch(run, original, model_config={"in_features": 2, "out_features": 1})
    saved = space.get_run(run.path.name)
    with pytest.raises(RunError):
        load_pytorch(saved, torch.nn.Linear(3, 1))
    restored = load_pytorch(saved, torch.nn.Linear(2, 1))
    torch.testing.assert_close(restored.weight, original.weight)
