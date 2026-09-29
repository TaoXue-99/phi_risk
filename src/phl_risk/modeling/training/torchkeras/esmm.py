"""Local ESMM runners for torchkeras 3.9.9; no global monkey patches."""

import math
from functools import partial

import torch
from torchkeras.kerasmodel import EpochRunner, KerasModel, StepRunner
from tqdm.auto import tqdm

from ..esmm import _ESMMAccumulator


class ESMMStepRunner(StepRunner):
    def __init__(self, *args, scheduler_interval="epoch", max_grad_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1:
            raise ValueError("ESMM runners currently support single-process training only")
        self.scheduler_interval = scheduler_interval
        self.max_grad_norm = max_grad_norm
        self.accumulator = _ESMMAccumulator()

    def __call__(self, batch):
        x_cat, x_cont, y_ctr, y_ctcvr = batch
        targets = {"ctr": y_ctr, "ctcvr": y_ctcvr}
        with self.accelerator.autocast():
            predictions = self.net(x_cat, x_cont)
        # Probability BCE is evaluated outside autocast in float32.
        predictions = {name: value.float() for name, value in predictions.items()}
        components = self.loss_fn.components(predictions, targets)
        if not torch.isfinite(components["loss"]):
            raise ValueError("Nonfinite training loss")
        if self.stage == "train" and self.optimizer is not None:
            self.accelerator.backward(components["loss"])
            if self.accelerator.sync_gradients and self.max_grad_norm is not None:
                self.accelerator.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.lr_scheduler is not None and self.scheduler_interval == "batch":
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
        self.accumulator.update(components, predictions, targets)
        detached = {name: value.detach() for name, value in predictions.items()}
        for metric in self.metrics_dict.values():
            metric.update(detached, targets)
        losses = {
            f"{self.stage}_{name}": value.detach().item() for name, value in components.items()
        }
        metrics = {"lr": self.optimizer.param_groups[0]["lr"]} if self.optimizer else {}
        return losses, metrics


class ESMMEpochRunner(EpochRunner):
    def __call__(self, dataloader):
        runner = self.step_runner
        total = min(getattr(dataloader, "size", len(dataloader)), len(dataloader))
        loop = tqdm(
            enumerate(dataloader, 1),
            total=total,
            disable=self.quiet or not self.accelerator.is_local_main_process,
        )
        for step, batch in loop:
            if step > total:
                break
            with self.accelerator.accumulate(self.net):
                losses, metrics = runner(batch)
            log = {**losses, **metrics}
            loop.set_postfix(**log)
            if hasattr(self, "progress") and self.accelerator.is_local_main_process:
                self.progress.set_postfix(i=step, n=total, **log)
        scores, self.undefined_metrics = runner.accumulator.compute()
        result = {f"{self.stage}_{name}": value for name, value in scores.items()}
        for name, metric in runner.metrics_dict.items():
            value = float(metric.compute().item())
            if not math.isfinite(value):
                raise ValueError(f"Custom metric {name} is not finite")
            result[f"{self.stage}_{name}"] = value
            metric.reset()
        if self.stage == "train" and runner.optimizer is not None:
            if runner.lr_scheduler is not None and runner.scheduler_interval == "epoch":
                runner.lr_scheduler.step()
            result["lr"] = runner.optimizer.param_groups[0]["lr"]
        return result


class ESMMKerasModel(KerasModel):
    """KerasModel using ESMM batches and a loss exposing components().

    Built-in metrics: loss, ctr_loss, ctcvr_loss, ctr_auc, ctcvr_auc (stage-prefixed).
    Single-class AUC is omitted. Select a defined monitor such as val_loss.
    Optional metrics_dict values are stateful modules with update(predictions_dict,
    targets_dict), compute() -> scalar tensor, and reset().
    Scheduler must support parameterless step(); ReduceLROnPlateau belongs in a
    validation callback instead. Standard epoch and batch schedulers are supported.
    """

    EpochRunner = ESMMEpochRunner

    def __init__(
        self,
        net,
        loss_fn,
        metrics_dict=None,
        optimizer=None,
        lr_scheduler=None,
        *,
        scheduler_interval="epoch",
        max_grad_norm=1.0,
    ):
        if scheduler_interval not in ("epoch", "batch"):
            raise ValueError("scheduler_interval must be epoch or batch")
        if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be positive and finite, or None")
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise ValueError("Use a validation callback for ReduceLROnPlateau")
        reserved = {"loss", "ctr_loss", "ctcvr_loss", "ctr_auc", "ctcvr_auc"}
        if reserved.intersection(metrics_dict or {}):
            raise ValueError("Custom metrics must not overwrite built-in ESMM metrics")
        super().__init__(net, loss_fn, metrics_dict or {}, optimizer, lr_scheduler)
        self.StepRunner = partial(
            ESMMStepRunner, scheduler_interval=scheduler_interval, max_grad_norm=max_grad_norm
        )

    def forward(self, x_cat, x_cont):
        return self.net(x_cat, x_cont)

    def load_ckpt(self, ckpt_path=None):
        self.net.load_state_dict(
            torch.load(
                ckpt_path if ckpt_path is not None else self.ckpt_path,
                map_location="cpu",
                weights_only=True,
            )
        )
        self.from_scratch = False
