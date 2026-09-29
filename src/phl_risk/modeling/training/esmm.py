"""Framework-independent ESMM evaluation for native PyTorch and torchkeras."""

import torch
from sklearn.metrics import roc_auc_score


def resolve_esmm_device(device="cpu"):
    """Resolve the examples' explicit CPU/single-CUDA contract without fallback.

    Select a physical GPU using CUDA_VISIBLE_DEVICES before starting Python.
    MPS and distributed training are outside this example contract.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA requested but unavailable; install a CUDA PyTorch build "
            "and check the NVIDIA driver"
        )
    return torch.device(device)


class _ESMMAccumulator:
    """Sample-weighted loss and exact epoch AUC (O(N) prediction storage)."""

    def __init__(self):
        self.count = 0
        self.loss_sums = {}
        self.predictions = {name: [] for name in ("ctr", "ctcvr")}
        self.targets = {name: [] for name in ("ctr", "ctcvr")}

    def update(self, components, predictions, targets):
        n = predictions["ctr"].shape[0]
        self.count += n
        for name, value in components.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite {name}")
            self.loss_sums[name] = self.loss_sums.get(name, 0.0) + value.detach().item() * n
        for name in self.predictions:
            self.predictions[name].append(predictions[name].detach().float().cpu().reshape(-1))
            self.targets[name].append(targets[name].detach().cpu().reshape(-1))

    def compute(self):
        if not self.count:
            raise ValueError("Cannot evaluate an empty DataLoader")
        scores = {k: v / self.count for k, v in self.loss_sums.items()}
        undefined = {}
        for name in self.predictions:
            y = torch.cat(self.targets[name]).numpy()
            p = torch.cat(self.predictions[name]).numpy()
            if len(set(y.tolist())) < 2:
                undefined[f"{name}_auc"] = "Only one target class is present"
            else:
                scores[f"{name}_auc"] = float(roc_auc_score(y, p))
        return scores, undefined


def evaluate_esmm(net, loader, loss_fn):
    """Evaluate the current weights without changing their device or final mode.

    Batch contract: (x_cat, x_cont, y_ctr, y_ctcvr). Returns (finite_metrics,
    undefined_metric_reasons). No CVR score is inferred from whole-space labels.
    Predictions for exact AUC are collected on CPU; loss is weighted by sample count.
    """
    device = next(net.parameters()).device
    was_training = net.training
    net.eval()
    accumulator = _ESMMAccumulator()
    try:
        with torch.no_grad():
            for x_cat, x_cont, y_ctr, y_ctcvr in loader:
                predictions = net(x_cat.to(device), x_cont.to(device))
                targets = {"ctr": y_ctr.to(device), "ctcvr": y_ctcvr.to(device)}
                accumulator.update(loss_fn.components(predictions, targets), predictions, targets)
        return accumulator.compute()
    finally:
        net.train(was_training)
