import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")


def guide():
    path = Path("examples/modeling/esmm_mmoe_experiment.py").resolve()
    spec = importlib.util.spec_from_file_location("esmm_guide", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preprocessing_schema_unknown_and_train_only():
    module = guide()
    frame = module.make_data(120)
    state = module.fit_preprocessor(frame.iloc[:80], ["x0", "x1"], ["channel"])
    before = state["continuous_pipeline"].named_steps["scale"].mean_.copy()
    unseen = frame.iloc[80:].copy()
    unseen["channel"] = "never-seen"
    unseen["x0"] = 100000.0
    cat, cont = module.transform_features(unseen, state)
    assert np.all(cat == 0)
    assert np.isfinite(cont).all()
    np.testing.assert_array_equal(before, state["continuous_pipeline"].named_steps["scale"].mean_)
    with pytest.raises(ValueError, match="Missing"):
        module.transform_features(unseen.drop(columns="x0"), state)
    state = module.fit_preprocessor(frame, ["x0"], [])
    assert module.transform_features(frame, state)[0].shape == (120, 0)


@pytest.mark.filterwarnings(
    "ignore:Type google._upb._message.*uses PyType_Spec.*:DeprecationWarning"
)
def test_guide_both_paths_record_restore_tensorboard(tmp_path):
    pytest.importorskip("torchkeras")
    pytest.importorskip("tensorboard")
    module = guide()
    runs, comparison = module.run_guide(tmp_path, rows=180, epochs=2, tensorboard=True)
    assert len(runs) == len(comparison) == 2
    for run in runs:
        assert run.status == "completed"
        assert {
            "model",
            "model_config",
            "preprocessor",
            "history",
            "tensorboard",
        } <= run.artifacts.keys()
        assert run.result["restored_predictions_match"]
        assert run.result["best_epoch"] in (1, 2)
        assert "ctcvr_auc" in run.metrics["test"]
        assert run.artifact_path("tensorboard").stat().st_size > 0


def test_native_example_never_imports_torchkeras(tmp_path):
    script = """
import importlib.util, sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'torchkeras', 'accelerate', 'torchmetrics', 'tensorboard'}:
            raise AssertionError('Native path imported ' + name)
sys.meta_path.insert(0, Block())
spec = importlib.util.spec_from_file_location('guide', sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
runs, comparison = m.run_guide(sys.argv[2], rows=120, epochs=1, backend='native')
assert len(runs) == 1 and runs[0].result['restored_predictions_match']
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path("examples/modeling/esmm_mmoe_experiment.py").resolve()),
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
    )


def large_guide():
    path = Path("examples/modeling/esmm_mmoe_large_scale.py").resolve()
    spec = importlib.util.spec_from_file_location("esmm_large", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_large_data_chunked_schema_and_loader(tmp_path):
    module = large_guide()
    manifest = module.generate_dataset(tmp_path / "data", rows=1000, chunk_size=173, seed=12)
    assert manifest["rows"] == 1000
    assert sum(manifest["partitions"].values()) == 1000
    loaders = module.make_disk_loaders(tmp_path / "data", batch_size=128, seed=12)
    batch = next(iter(loaders["train"]))
    torch = __import__("torch")
    assert batch[0].dtype == torch.long and batch[1].dtype == torch.float32
    assert torch.all(batch[3] <= batch[2])
    assert sum(len(b[2]) for b in loaders["test"]) == manifest["partitions"]["test"]
    # Fitted mean must use the train range, not held-out observations.
    continuous = np.load(tmp_path / "data" / "continuous.npy", mmap_mode="r")
    np.testing.assert_allclose(
        manifest["preprocessing"]["mean"],
        np.nanmean(continuous[: manifest["partitions"]["train"]].astype("float64"), axis=0),
        rtol=1e-5,
        atol=1e-7,
    )
    with pytest.raises(ValueError, match="existing"):
        module.generate_dataset(tmp_path / "data", rows=1001, chunk_size=173, seed=12)


def test_large_guide_small_fixture_end_to_end(tmp_path):
    pytest.importorskip("torchkeras")
    module = large_guide()
    result = module.run_large_guide(
        tmp_path,
        rows=500,
        epochs=1,
        batch_size=128,
        chunk_size=111,
        backend="both",
        tensorboard=False,
    )
    assert len(result["runs"]) == 2
    assert result["dataset"]["rows"] == 500
    for run in result["runs"]:
        assert run.result["training_rows"] == 300
        assert run.result["training_seconds"] > 0
        assert run.result["samples_per_second"] > 0
        assert "ctcvr_auc" in run.metrics["test"]


def test_large_guide_rejects_unsupported_optimizer_in_baseline(tmp_path):
    import yaml

    from phl_risk.modeling.experiment import Experiment

    space = Experiment(root=tmp_path).initialize(
        method="esmm_mmoe", comparison_partitions=["valid", "test"]
    )
    path = space.config_dir / "baseline.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["optimizer"]["name"] = "unsupported"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="optimizer"):
        large_guide().run_large_guide(
            tmp_path, rows=100, epochs=1, backend="native", tensorboard=False
        )


def test_explicit_cuda_fails_before_creating_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        large_guide().run_large_guide(tmp_path / "absent", rows=100, device="cuda")
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("backend", ["native", "torchkeras"])
def test_cuda_large_guide_roundtrip(tmp_path, backend):
    if not torch.cuda.is_available():
        pytest.skip("Requires an NVIDIA CUDA runtime")
    if backend == "torchkeras":
        pytest.importorskip("torchkeras")
    # Accelerate keeps process-global device state; isolate each training backend.
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            str(Path("examples/modeling/esmm_mmoe_large_scale.py").resolve()),
            "--root",
            str(tmp_path),
            "--rows",
            "500",
            "--epochs",
            "1",
            "--backend",
            backend,
            "--device",
            "cuda",
            "--no-tensorboard",
        ],
        check=True,
    )
