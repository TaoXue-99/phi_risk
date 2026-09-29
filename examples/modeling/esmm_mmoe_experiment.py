"""ESMM + MMoE: native PyTorch / torchkeras with the same Experiment protocol.

Run from the repository root:
  uv run --extra torchkeras --extra tensorboard python examples/modeling/esmm_mmoe_experiment.py
  # Native only: append --backend native --no-tensorboard to the script command.

See esmm_mmoe_complete_guide.md and the companion notebook for the walkthrough.
All data are synthetic; no private data, checkpoints or credentials are required.
"""

import argparse
import hashlib
import random
import zipfile
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from phl_risk.modeling.experiment import Experiment
from phl_risk.modeling.experiment.adapters.pytorch import load_pytorch, record_pytorch
from phl_risk.modeling.experiment.configuration import read_yaml_config
from phl_risk.modeling.models.esmm_mmoe import ESMMLoss, ESMMMoE
from phl_risk.modeling.training.esmm import evaluate_esmm, resolve_esmm_device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_data(rows=2400, seed=2026):
    """Generate an actual conversion chain, then introduce missing features."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 6))
    channel = rng.choice(["search", "feed", "direct"], size=rows)
    device = rng.choice(["mobile", "desktop"], size=rows)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    ctr = sigmoid(0.8 * x[:, 0] - 0.6 * x[:, 1] + 0.5 * (channel == "search"))
    cvr = sigmoid(-0.7 + 0.9 * x[:, 2] + 0.6 * x[:, 0] - 0.3 * (device == "mobile"))
    y_ctr = rng.binomial(1, ctr)
    y_ctcvr = y_ctr * rng.binomial(1, cvr)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(6)])
    frame["channel"], frame["device"] = channel, device
    frame["y_ctr"], frame["y_ctcvr"] = y_ctr, y_ctcvr
    frame.loc[rng.random(rows) < 0.03, "x0"] = np.nan
    frame.loc[rng.random(rows) < 0.02, "channel"] = None
    assert (frame.y_ctcvr <= frame.y_ctr).all()
    return frame


def split_data(frame, seed=2026):
    """Stratify the three valid chain outcomes; split BEFORE fitting preprocessing."""
    chain = frame.y_ctr + frame.y_ctcvr
    train, rest = train_test_split(frame, test_size=0.4, random_state=seed, stratify=chain)
    valid, test = train_test_split(
        rest, test_size=0.5, random_state=seed, stratify=rest.y_ctr + rest.y_ctcvr
    )
    return {"train": train.copy(), "valid": valid.copy(), "test": test.copy()}


def fit_preprocessor(train, continuous, categorical):
    """Example-only preprocessing, fitted solely on the training partition.

    ID 0 represents missing/unseen categories. Artifacts contain a plain dictionary
    and sklearn objects, so loading does not depend on a class defined in __main__.
    """
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    pipeline.fit(train[continuous])
    vocabularies = {
        column: {
            value: i + 1
            for i, value in enumerate(sorted(train[column].dropna().astype(str).unique()))
        }
        for column in categorical
    }
    return {
        "continuous_features": list(continuous),
        "categorical_features": list(categorical),
        "continuous_pipeline": pipeline,
        "vocabularies": vocabularies,
        "unknown_category_id": 0,
    }


def transform_features(frame, state):
    columns = state["continuous_features"] + state["categorical_features"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    continuous = (
        state["continuous_pipeline"]
        .transform(frame[state["continuous_features"]])
        .astype(np.float32)
    )
    categorical = np.empty((len(frame), len(state["categorical_features"])), dtype=np.int64)
    for i, column in enumerate(state["categorical_features"]):
        vocabulary = state["vocabularies"][column]
        categorical[:, i] = [
            vocabulary.get(str(value), 0) if pd.notna(value) else 0 for value in frame[column]
        ]
    return categorical, continuous


def make_loaders(partitions, state, cfg):
    datasets = {}
    labels = cfg["data"]["labels"]
    for name, frame in partitions.items():
        cat, cont = transform_features(frame, state)
        datasets[name] = TensorDataset(
            torch.from_numpy(cat),
            torch.from_numpy(cont),
            torch.tensor(frame[labels["ctr"]].to_numpy(), dtype=torch.float32),
            torch.tensor(frame[labels["ctcvr"]].to_numpy(), dtype=torch.float32),
        )
    loader_args = {
        "batch_size": cfg["data"]["batch_size"],
        "num_workers": cfg["data"]["num_workers"],
    }
    train_loader = DataLoader(
        datasets["train"],
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg["train"]["seed"]),
        drop_last=cfg["data"]["drop_last"],
        **loader_args,
    )
    # Evaluate every row, with no dropout from DataLoader and no shuffle.
    eval_loaders = {
        name: DataLoader(dataset, shuffle=False, **loader_args)
        for name, dataset in datasets.items()
    }
    return train_loader, eval_loaders


def build_optimization(net, cfg):
    # Explicit choices: no eval(), dynamic import or general-purpose object factory.
    options = dict(cfg["optimizer"])
    name = options.pop("name")
    optimizer_class = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}[name]
    optimizer = optimizer_class(net.parameters(), **options)
    options = dict(cfg["scheduler"])
    scheduler_name, interval = options.pop("name"), options.pop("interval")
    if scheduler_name != "StepLR":
        raise ValueError("This example uses StepLR; construct other schedulers explicitly")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **options)
    return optimizer, scheduler, interval


def train_native(
    net,
    loss_fn,
    optimizer,
    scheduler,
    train_loader,
    valid_loader,
    cfg,
    checkpoint,
    writer=None,
    device="cpu",
):
    """Transparent PyTorch loop: edit this function freely for your own training."""
    net.to(device)
    history, best_score, best_epoch, stale = [], None, None, 0
    settings = cfg["train"]
    for epoch in range(1, settings["epochs"] + 1):
        net.train()
        total, count = 0.0, 0
        for step, (x_cat, x_cont, y_ctr, y_ctcvr) in enumerate(train_loader):
            optimizer.zero_grad()
            predictions = net(x_cat.to(device), x_cont.to(device))
            targets = {"ctr": y_ctr.to(device), "ctcvr": y_ctcvr.to(device)}
            components = loss_fn.components(predictions, targets)
            components["loss"].backward()
            if settings["max_grad_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), settings["max_grad_norm"])
            optimizer.step()
            if cfg["scheduler"]["interval"] == "batch":
                scheduler.step()
            n = len(y_ctr)
            total += components["loss"].item() * n
            count += n
            if writer is not None:
                writer.add_scalar(
                    "batch/loss", components["loss"].item(), (epoch - 1) * len(train_loader) + step
                )
        if not count:
            raise ValueError("Training DataLoader is empty")
        if cfg["scheduler"]["interval"] == "epoch":
            scheduler.step()
        validation, _ = evaluate_esmm(net, valid_loader, loss_fn)
        row = {
            "epoch": epoch,
            "train_loss": total / count,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"val_{k}": value for k, value in validation.items()},
        }
        history.append(row)
        if writer is not None:
            for name, value in row.items():
                if name != "epoch":
                    writer.add_scalar(name, value, epoch)
        score = row[settings["monitor"]]
        improved = best_score is None or (
            score < best_score if settings["mode"] == "min" else score > best_score
        )
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(net.state_dict(), checkpoint)
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return pd.DataFrame(history), best_epoch


class TensorBoardHistory:
    """A torchkeras callback borrowing the caller's writer; never closes it."""

    def __init__(self, writer):
        self.writer = writer

    def on_validation_epoch_end(self, model):
        epoch = int(model.history["epoch"][-1])
        for name, values in model.history.items():
            if name != "epoch":
                self.writer.add_scalar(name, values[-1], epoch)
        self.writer.flush()


def train_torchkeras(
    net, loss_fn, optimizer, scheduler, train_loader, valid_loader, cfg, checkpoint, writer=None
):
    from phl_risk.modeling.training.torchkeras import ESMMKerasModel

    trainer = ESMMKerasModel(
        net,
        loss_fn,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        scheduler_interval=cfg["scheduler"]["interval"],
        max_grad_norm=cfg["train"]["max_grad_norm"],
    )
    settings = cfg["train"]
    history = trainer.fit(
        train_data=train_loader,
        val_data=valid_loader,
        epochs=settings["epochs"],
        patience=settings["patience"],
        monitor=settings["monitor"],
        mode=settings["mode"],
        ckpt_path=str(checkpoint),
        callbacks=[TensorBoardHistory(writer)] if writer is not None else [],
        **cfg["torchkeras"],
    )
    scores = history[settings["monitor"]].to_numpy()
    best = np.argmin(scores) if settings["mode"] == "min" else np.argmax(scores)
    # fit() has already loaded the selected checkpoint into trainer.net.
    return history, int(history.iloc[best]["epoch"])


def predict_frame(net, frame, state):
    cat, cont = transform_features(frame, state)
    device = next(net.parameters()).device
    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            predictions = net(torch.from_numpy(cat).to(device), torch.from_numpy(cont).to(device))
        return pd.DataFrame(
            {k: v.cpu().numpy().ravel() for k, v in predictions.items()}, index=frame.index
        )
    finally:
        net.train(was_training)


def archive_tensorboard(run, directory):
    def write(path):
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(directory.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(directory))

    run.log_artifact("tensorboard", "tensorboard.zip", write, format="zip")


def run_one(space, cfg, partitions, state, backend, tensorboard=False, device="cpu"):
    """Same recording contract for either training framework."""
    cfg = deepcopy(cfg)
    device = resolve_esmm_device(device)
    cfg["execution"] = {"backend": backend, "device": str(device)}
    cfg["torchkeras"]["cpu"] = device.type == "cpu"
    seed_everything(cfg["train"]["seed"])
    net = ESMMMoE(**cfg["model"]).to(device)
    loss_fn = ESMMLoss(**cfg["loss"])
    optimizer, scheduler, _ = build_optimization(net, cfg)
    train_loader, loaders = make_loaders(partitions, state, cfg)
    versions = {"torch": version("torch")}
    if backend == "torchkeras":
        versions.update({name: version(name) for name in ("torchkeras", "accelerate")})
    if tensorboard:
        versions["tensorboard"] = version("tensorboard")
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(pd.concat(partitions.values()), index=True).to_numpy().tobytes()
    ).hexdigest()
    with space.start_run(
        name=backend,
        config=cfg,
        metadata={
            "training_framework": backend,
            "versions": versions,
            "synthetic_data": True,
            "data_sha256": fingerprint,
        },
        task={"objective": "ESMM weighted BCE", "metric": ["ctr_auc", "ctcvr_auc", "loss"]},
        comparison_partitions=["valid", "test"],
    ) as run:
        run.log_input(
            features=state["continuous_features"] + state["categorical_features"],
            continuous_features=state["continuous_features"],
            categorical_features=state["categorical_features"],
            labels=cfg["data"]["labels"],
            partition_rows={k: len(v) for k, v in partitions.items()},
        )
        writer = None
        tb_dir = run.path / "tensorboard"
        try:
            if tensorboard:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter(str(tb_dir))
            with TemporaryDirectory(prefix="esmm-checkpoint-") as scratch:
                checkpoint = Path(scratch) / "best.pt"
                if backend == "native":
                    history, best_epoch = train_native(
                        net,
                        loss_fn,
                        optimizer,
                        scheduler,
                        train_loader,
                        loaders["valid"],
                        cfg,
                        checkpoint,
                        writer,
                        device=device,
                    )
                elif backend == "torchkeras":
                    history, best_epoch = train_torchkeras(
                        net,
                        loss_fn,
                        optimizer,
                        scheduler,
                        train_loader,
                        loaders["valid"],
                        cfg,
                        checkpoint,
                        writer,
                    )
                else:
                    raise ValueError("backend must be native or torchkeras")
        finally:
            if writer is not None:
                writer.flush()
                writer.close()
        metrics, undefined = {}, {}
        for name, loader in loaders.items():
            metrics[name], undefined[name] = evaluate_esmm(net, loader, loss_fn)
        run.log_metrics(metrics)
        run.log_json("undefined_metrics", undefined)
        run.log_json("history", history.to_dict(orient="list"))
        run.log_result(
            best_epoch=best_epoch,
            epochs_run=len(history),
            selected_weights="best_validation",
            monitor=cfg["train"]["monitor"],
        )
        record_pytorch(run, net, model_config=net.get_config())
        run.log_artifact(
            "preprocessor",
            "preprocessor.joblib",
            lambda path: joblib.dump(state, path),
            format="joblib",
        )
        # In-run smoke test uses independent objects; completed artifact validation follows below.
        restored = ESMMMoE(**net.get_config())
        restored.load_state_dict({k: v.cpu() for k, v in net.state_dict().items()})
        expected = predict_frame(net, partitions["test"], state)
        actual = predict_frame(restored, partitions["test"], state)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        run.log_result(restored_predictions_match=True)
        if tensorboard:
            archive_tensorboard(run, tb_dir)
    saved = space.get_run(run.path.name)
    # Load the actual checksummed artifacts, including feature processing.
    restored = load_pytorch(saved, ESMMMoE(**saved.read_json("model_config")))
    saved_state = joblib.load(saved.artifact_path("preprocessor"))
    actual = predict_frame(restored, partitions["test"], saved_state)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    return saved


def run_guide(root, rows=2400, epochs=6, backend="both", tensorboard=False, device="cpu"):
    if rows < 100 or epochs < 1:
        raise ValueError("Use at least 100 rows and one epoch for the stratified demonstration")
    if backend not in ("both", "native", "torchkeras"):
        raise ValueError("backend must be both, native or torchkeras")
    resolve_esmm_device(device)
    exp = Experiment(root=root)
    space = exp.initialize(method="esmm_mmoe", comparison_partitions=["valid", "test"])
    cfg = read_yaml_config(space.config_dir / "baseline.yaml")
    cfg = deepcopy(cfg)
    continuous, categorical = [f"x{i}" for i in range(6)], ["channel", "device"]
    partitions = split_data(make_data(rows, cfg["train"]["seed"]), cfg["train"]["seed"])
    state = fit_preprocessor(partitions["train"], continuous, categorical)
    cfg["model"].update(
        num_continuous=len(continuous),
        categorical_cardinalities=[len(state["vocabularies"][c]) + 1 for c in categorical],
        share_dim=24,
        base_dim=12,
        expert_units=8,
    )
    cfg["data"].update(
        continuous_features=continuous, categorical_features=categorical, batch_size=64
    )
    cfg["train"].update(epochs=epochs, patience=3)
    cfg["scheduler"].update(step_size=2)
    backends = ["native", "torchkeras"] if backend == "both" else [backend]
    runs = [run_one(space, cfg, partitions, state, name, tensorboard, device) for name in backends]
    comparison = space.compare(
        runs=[run.run_id for run in runs],
        metrics=["ctr_auc", "ctcvr_auc", "loss"],
        partitions=["valid", "test"],
        params=["optimizer.lr", "model.num_experts"],
        metadata=["metadata.training_framework"],
    )
    print(comparison.to_string(index=False))
    comparison.to_csv(space.reports_dir / "guide_comparison.csv", index=False)
    if len(runs) == 2:
        print(space.compare_params(runs[0].run_id, runs[1].run_id).to_string(index=False))
    return runs, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments/esmm_mmoe_guide"))
    parser.add_argument("--backend", choices=["both", "native", "torchkeras"], default="both")
    parser.add_argument("--rows", type=int, default=2400)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    run_guide(args.root, args.rows, args.epochs, args.backend, not args.no_tensorboard, args.device)


if __name__ == "__main__":
    main()
