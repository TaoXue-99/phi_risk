"""Five-to-ten-million-row ESMM guide using disk-backed, already-batched data.

Default: 5,000,000 synthetic rows; three epochs for each training path.
Data generation and preprocessing are chunked. No giant pandas DataFrame needed.
"""

import argparse
import hashlib
import json
import math
import random
import zipfile
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from phl_risk.modeling.experiment import Experiment
from phl_risk.modeling.experiment.adapters.pytorch import load_pytorch, record_pytorch
from phl_risk.modeling.experiment.configuration import read_yaml_config
from phl_risk.modeling.models.esmm_mmoe import ESMMLoss, ESMMMoE
from phl_risk.modeling.training.esmm import evaluate_esmm, resolve_esmm_device


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_dataset(directory, rows=5_000_000, chunk_size=100_000, seed=2026):
    """Create/reuse immutable raw memmaps; fit statistics on the train range only."""
    directory = Path(directory)
    if rows < 100 or chunk_size < 1:
        raise ValueError("rows must be >= 100 and chunk_size must be positive")
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["rows"], manifest["seed"], manifest["chunk_size"]) != (rows, seed, chunk_size):
            raise ValueError(
                "Requested generation differs from the existing dataset; use a new path"
            )
        for name, digest in manifest["sha256"].items():
            if _sha256(directory / name) != digest:
                raise ValueError(f"Existing dataset checksum mismatch: {name}")
        return manifest
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Use an empty directory; existing incomplete files are not overwritten")
    started = perf_counter()
    train_end, valid_end = int(rows * 0.6), int(rows * 0.8)
    boundaries = {
        "train": (0, train_end),
        "valid": (train_end, valid_end),
        "test": (valid_end, rows),
    }
    continuous = open_memmap(
        directory / "continuous.npy", mode="w+", dtype="float32", shape=(rows, 6)
    )
    categorical = open_memmap(
        directory / "categorical.npy", mode="w+", dtype="int16", shape=(rows, 2)
    )
    labels = open_memmap(directory / "labels.npy", mode="w+", dtype="uint8", shape=(rows, 2))
    scaler = StandardScaler()
    counts = {name: {"ctr": 0, "ctcvr": 0} for name in boundaries}
    rng = np.random.default_rng(seed)
    for start in range(0, rows, chunk_size):
        end = min(start + chunk_size, rows)
        n = end - start
        x = rng.standard_normal((n, 6), dtype=np.float32)
        cat = np.column_stack([rng.integers(1, 4, n), rng.integers(1, 3, n)]).astype("int16")
        p_ctr = 1 / (1 + np.exp(-(0.8 * x[:, 0] - 0.6 * x[:, 1] + 0.5 * (cat[:, 0] == 1))))
        p_cvr = 1 / (1 + np.exp(-(-0.7 + 0.9 * x[:, 2] + 0.6 * x[:, 0] - 0.3 * (cat[:, 1] == 1))))
        y_ctr = (rng.random(n) < p_ctr).astype("uint8")
        y_ctcvr = y_ctr * (rng.random(n) < p_cvr).astype("uint8")
        x[rng.random(n) < 0.03, 0] = np.nan
        cat[rng.random(n) < 0.02, 0] = 0
        continuous[start:end], categorical[start:end] = x, cat
        labels[start:end] = np.column_stack([y_ctr, y_ctcvr])
        if start < train_end:
            scaler.partial_fit(x[: min(end, train_end) - start])
        for name, (left, right) in boundaries.items():
            lo, hi = max(start, left), min(end, right)
            if lo < hi:
                counts[name]["ctr"] += int(y_ctr[lo - start : hi - start].sum())
                counts[name]["ctcvr"] += int(y_ctcvr[lo - start : hi - start].sum())
        if end == rows or end % 1_000_000 == 0:
            print(f"Generated {end:,}/{rows:,} rows", flush=True)
    for array in (continuous, categorical, labels):
        array.flush()
    del continuous, categorical, labels
    manifest = {
        "format_version": 1,
        "rows": rows,
        "seed": seed,
        "chunk_size": chunk_size,
        "partitions": {k: right - left for k, (left, right) in boundaries.items()},
        "boundaries": boundaries,
        "positive_counts": counts,
        "preprocessing": {
            "continuous_features": [f"x{i}" for i in range(6)],
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "imputation": "train_mean",
            "fit_partition": "train",
            "categorical_features": ["channel", "device"],
            "vocabularies": {
                "channel": {"search": 1, "feed": 2, "direct": 3},
                "device": {"mobile": 1, "desktop": 2},
            },
            "categorical_cardinalities": [4, 3],
            "unknown_category_id": 0,
        },
        "sha256": {
            name: _sha256(directory / name)
            for name in ("continuous.npy", "categorical.npy", "labels.npy")
        },
        "generation_seconds": perf_counter() - started,
        "disk_bytes": sum(p.stat().st_size for p in directory.glob("*.npy")),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def transform_batch(categorical, continuous, preprocessing):
    """Apply persisted train statistics, including missing-value handling."""
    x = np.array(continuous, dtype=np.float32, copy=True)
    mean = np.asarray(preprocessing["mean"], dtype=np.float32)
    scale = np.asarray(preprocessing["scale"], dtype=np.float32)
    x = np.where(np.isnan(x), mean, x)
    x = (x - mean) / scale
    cat = np.array(categorical, dtype=np.int64, copy=True)
    return torch.from_numpy(cat), torch.from_numpy(x)


class DiskBatches(IterableDataset):
    """Yield full batches from mmap; shuffle block order with O(number-of-batches) memory.

    Single-process loader by design. Already-batched iteration avoids millions
    of Python __getitem__ calls. A new deterministic block permutation is used
    for each training epoch; row-level random shuffling is not claimed.
    """

    def __init__(self, directory, partition, batch_size=4096, shuffle=False, seed=2026):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.start, self.end = self.manifest["boundaries"][partition]
        self.batch_size, self.shuffle, self.seed, self.epoch = batch_size, shuffle, seed, 0

    def __len__(self):
        return math.ceil((self.end - self.start) / self.batch_size)

    def __iter__(self):
        if get_worker_info() is not None:
            raise ValueError("DiskBatches guide uses num_workers=0 to prevent duplicated batches")
        continuous = np.load(self.directory / "continuous.npy", mmap_mode="r")
        categorical = np.load(self.directory / "categorical.npy", mmap_mode="r")
        labels = np.load(self.directory / "labels.npy", mmap_mode="r")
        blocks = np.arange(len(self))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(blocks)
            self.epoch += 1
        for block in blocks:
            left = self.start + int(block) * self.batch_size
            right = min(left + self.batch_size, self.end)
            cat, cont = transform_batch(
                categorical[left:right], continuous[left:right], self.manifest["preprocessing"]
            )
            target = np.array(labels[left:right], dtype=np.float32, copy=True)
            yield cat, cont, torch.from_numpy(target[:, 0]), torch.from_numpy(target[:, 1])


def make_disk_loaders(directory, batch_size=4096, seed=2026):
    loaders = {
        name: DataLoader(
            DiskBatches(directory, name, batch_size, seed=seed), batch_size=None, num_workers=0
        )
        for name in ("train", "valid", "test")
    }
    loaders["fit"] = DataLoader(
        DiskBatches(directory, "train", batch_size, shuffle=True, seed=seed),
        batch_size=None,
        num_workers=0,
    )
    return loaders


class EpochTiming:
    """torchkeras callback recording elapsed epoch time and TensorBoard metrics."""

    def __init__(self, writer=None):
        self.writer = writer
        self.rows = []

    def on_fit_start(self, model):
        self.start = perf_counter()

    def on_validation_epoch_end(self, model):
        row = {key: values[-1] for key, values in model.history.items()}
        row["epoch_seconds"] = perf_counter() - self.start
        self.rows.append(row)
        if self.writer:
            for key, value in row.items():
                if key != "epoch":
                    self.writer.add_scalar(key, value, int(row["epoch"]))
            self.writer.flush()
        print(
            f"epoch={row['epoch']} seconds={row['epoch_seconds']:.2f} "
            f"val_loss={row['val_loss']:.5f} ctcvr_auc={row.get('val_ctcvr_auc')}",
            flush=True,
        )
        self.start = perf_counter()


def fit_native(net, loss_fn, optimizer, scheduler, loaders, cfg, checkpoint, writer=None):
    device = next(net.parameters()).device
    settings = cfg["train"]
    best, best_epoch, stale = None, None, 0
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        started = perf_counter()
        net.train()
        totals, count = {}, 0
        for cat, cont, y1, y12 in loaders["fit"]:
            cat, cont, y1, y12 = (value.to(device) for value in (cat, cont, y1, y12))
            optimizer.zero_grad()
            parts = loss_fn.components(net(cat, cont), {"ctr": y1, "ctcvr": y12})
            parts["loss"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), settings["max_grad_norm"])
            optimizer.step()
            count += len(y1)
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value.item() * len(y1)
        scheduler.step()
        scores, _ = evaluate_esmm(net, loaders["valid"], loss_fn)
        row = {
            "epoch": epoch,
            "epoch_seconds": perf_counter() - started,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v / count for k, v in totals.items()},
            **{f"val_{k}": v for k, v in scores.items()},
        }
        history.append(row)
        if writer:
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
            writer.flush()
        score = row[settings["monitor"]]
        improved = best is None or (score < best if settings["mode"] == "min" else score > best)
        if improved:
            best, best_epoch, stale = score, epoch, 0
            torch.save(net.state_dict(), checkpoint)
        else:
            stale += 1
        print(
            f"epoch={epoch} seconds={row['epoch_seconds']:.2f} "
            f"val_loss={row['val_loss']:.5f} ctcvr_auc={row.get('val_ctcvr_auc')}",
            flush=True,
        )
        if stale >= settings["patience"]:
            break
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return pd.DataFrame(history), best_epoch


def run_large_guide(
    root,
    rows=5_000_000,
    epochs=3,
    batch_size=4096,
    chunk_size=100_000,
    backend="both",
    tensorboard=True,
    threads=4,
    device="cpu",
):
    if epochs < 1 or backend not in ("both", "native", "torchkeras") or threads < 1:
        raise ValueError("Positive epochs/threads and a supported backend are required")
    device = resolve_esmm_device(device)
    root = Path(root).resolve()
    directory = root / f"dataset_{rows}"
    manifest = generate_dataset(directory, rows=rows, chunk_size=chunk_size)
    print(f"Dataset: {rows:,} rows, {manifest['disk_bytes'] / 1024**2:.1f} MiB", flush=True)
    space = Experiment(root=root).initialize(
        method="esmm_mmoe", comparison_partitions=["valid", "test"]
    )
    cfg = read_yaml_config(space.config_dir / "baseline.yaml")
    cfg["model"].update(
        num_continuous=6,
        categorical_cardinalities=[4, 3],
        share_dim=64,
        base_dim=32,
        expert_units=16,
        num_experts=4,
    )
    cfg["data"].update(
        batch_size=batch_size,
        continuous_features=[f"x{i}" for i in range(6)],
        categorical_features=["channel", "device"],
        rows=rows,
        chunk_size=chunk_size,
        block_shuffle=True,
    )
    cfg["train"].update(epochs=epochs, patience=3, seed=manifest["seed"])
    cfg["scheduler"].update(step_size=2, interval="epoch")
    cfg["data"].update(num_workers=0, drop_last=False)
    cfg["torchkeras"].update(
        cpu=device.type == "cpu", mixed_precision="no", gradient_accumulation_steps=1
    )
    optimizer_options = dict(cfg["optimizer"])
    optimizer_name = optimizer_options.pop("name")
    optimizers = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
    if optimizer_name not in optimizers:
        raise ValueError("This guide supports Adam/SGD; construct another optimizer explicitly")
    if cfg["scheduler"]["name"] != "StepLR":
        raise ValueError("This guide supports StepLR; construct another scheduler explicitly")
    completed = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        for training_backend in ["native", "torchkeras"] if backend == "both" else [backend]:
            random.seed(cfg["train"]["seed"])
            np.random.seed(cfg["train"]["seed"])
            torch.manual_seed(cfg["train"]["seed"])
            actual_cfg = deepcopy(cfg)
            actual_cfg["execution"] = {
                "backend": training_backend,
                "device": str(device),
                "threads": threads,
            }
            loaders = make_disk_loaders(directory, batch_size, cfg["train"]["seed"])
            net = ESMMMoE(**cfg["model"]).to(device)
            loss_fn = ESMMLoss(**cfg["loss"])
            optimizer = optimizers[optimizer_name](net.parameters(), **optimizer_options)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg["scheduler"]["step_size"], gamma=cfg["scheduler"]["gamma"]
            )
            versions = {"torch": version("torch")}
            if training_backend == "torchkeras":
                versions.update({key: version(key) for key in ("torchkeras", "accelerate")})
            if tensorboard:
                versions["tensorboard"] = version("tensorboard")
            with space.start_run(
                config=actual_cfg,
                name=f"{training_backend}_{rows}",
                metadata={
                    "training_framework": training_backend,
                    "versions": versions,
                    "synthetic_data": True,
                    "dataset": str(directory),
                },
                task={"objective": "ESMM weighted BCE", "metric": ["ctr_auc", "ctcvr_auc", "loss"]},
            ) as run:
                run.log_input(
                    features=[f"x{i}" for i in range(6)] + ["channel", "device"],
                    partition_rows=manifest["partitions"],
                    labels=cfg["data"]["labels"],
                )
                run.log_json("dataset_manifest", manifest)
                run.log_json("preprocessor", manifest["preprocessing"])
                writer = None
                tb_dir = run.path / "tensorboard"
                started = perf_counter()
                try:
                    if tensorboard:
                        from torch.utils.tensorboard import SummaryWriter

                        writer = SummaryWriter(str(tb_dir))
                    with TemporaryDirectory(prefix="large-esmm-") as scratch:
                        checkpoint = Path(scratch) / "best.pt"
                        if training_backend == "native":
                            history, best_epoch = fit_native(
                                net, loss_fn, optimizer, scheduler, loaders, cfg, checkpoint, writer
                            )
                        else:
                            from phl_risk.modeling.training.torchkeras import ESMMKerasModel

                            timing = EpochTiming(writer)
                            trainer = ESMMKerasModel(
                                net,
                                loss_fn,
                                optimizer=optimizer,
                                lr_scheduler=scheduler,
                                scheduler_interval="epoch",
                                max_grad_norm=cfg["train"]["max_grad_norm"],
                            )
                            history = trainer.fit(
                                loaders["fit"],
                                loaders["valid"],
                                epochs=epochs,
                                ckpt_path=str(checkpoint),
                                patience=cfg["train"]["patience"],
                                monitor=cfg["train"]["monitor"],
                                mode=cfg["train"]["mode"],
                                callbacks=[timing],
                                **cfg["torchkeras"],
                            )
                            history["epoch_seconds"] = [r["epoch_seconds"] for r in timing.rows]
                            values = history[cfg["train"]["monitor"]].to_numpy()
                            index = (
                                np.argmin(values)
                                if cfg["train"]["mode"] == "min"
                                else np.argmax(values)
                            )
                            best_epoch = int(history.iloc[index]["epoch"])
                finally:
                    if writer:
                        writer.flush()
                        writer.close()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                training_seconds = perf_counter() - started
                # torchkeras returns the best network on CPU; restore evaluation device.
                net.to(device)
                metrics, undefined = {}, {}
                eval_started = perf_counter()
                for partition in ("train", "valid", "test"):
                    metrics[partition], undefined[partition] = evaluate_esmm(
                        net, loaders[partition], loss_fn
                    )
                run.log_metrics(metrics)
                run.log_json("undefined_metrics", undefined)
                run.log_json("history", history.to_dict(orient="list"))
                run.log_result(
                    best_epoch=best_epoch,
                    epochs_run=len(history),
                    training_rows=manifest["partitions"]["train"],
                    training_seconds=training_seconds,
                    evaluation_seconds=perf_counter() - eval_started,
                    samples_per_second=manifest["partitions"]["train"]
                    * len(history)
                    / training_seconds,
                    throughput_definition=(
                        "training rows visited / fit wall seconds including validation"
                    ),
                    selected_weights="best_validation",
                )
                record_pytorch(run, net, model_config=net.get_config())
                if tensorboard:

                    def write_zip(path):
                        with zipfile.ZipFile(
                            path, "w", compression=zipfile.ZIP_DEFLATED
                        ) as archive:
                            for file in tb_dir.rglob("*"):
                                if file.is_file():
                                    archive.write(file, file.relative_to(tb_dir))

                    run.log_artifact("tensorboard", "tensorboard.zip", write_zip, format="zip")
                net.eval()
                probe = next(iter(loaders["test"]))
                with torch.no_grad():
                    expected = {
                        key: value.cpu()
                        for key, value in net(probe[0].to(device), probe[1].to(device)).items()
                    }
            saved = space.get_run(run.path.name)
            restored = load_pytorch(saved, ESMMMoE(**saved.read_json("model_config"))).eval()
            saved_prep = saved.read_json("preprocessor")
            test_start, _ = manifest["boundaries"]["test"]
            test_end = test_start + len(probe[0])
            cat, cont = transform_batch(
                np.load(directory / "categorical.npy", mmap_mode="r")[test_start:test_end],
                np.load(directory / "continuous.npy", mmap_mode="r")[test_start:test_end],
                saved_prep,
            )
            with torch.no_grad():
                torch.testing.assert_close(restored(cat, cont), expected, rtol=1e-4, atol=1e-6)
            completed.append(saved)
            print(f"Completed {saved.run_id}", flush=True)
        comparison = space.compare(
            runs=[run.run_id for run in completed],
            metrics=["ctr_auc", "ctcvr_auc", "loss"],
            partitions=["valid", "test"],
            fields=["training_rows", "training_seconds", "samples_per_second", "best_epoch"],
            params=["model.num_experts", "data.batch_size"],
            metadata=["metadata.training_framework"],
        )
        comparison.to_csv(space.reports_dir / "large_scale_comparison.csv", index=False)
        print(comparison.to_string(index=False), flush=True)
        return {"runs": completed, "comparison": comparison, "dataset": manifest}
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments/esmm_mmoe_large"))
    parser.add_argument("--rows", type=int, default=5_000_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--backend", choices=["both", "native", "torchkeras"], default="both")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    run_large_guide(
        args.root,
        rows=args.rows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        backend=args.backend,
        tensorboard=not args.no_tensorboard,
        threads=args.threads,
        device=args.device,
    )


if __name__ == "__main__":
    main()
