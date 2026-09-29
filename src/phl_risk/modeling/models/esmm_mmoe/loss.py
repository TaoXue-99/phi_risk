"""ESMM's full-space CTR and CTCVR supervision, usable in any PyTorch loop."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ESMMLoss(nn.Module):
    def __init__(self, ctr_weight=1.0, ctcvr_weight=1.0):
        super().__init__()
        if (
            not all(math.isfinite(w) and w >= 0 for w in (ctr_weight, ctcvr_weight))
            or ctr_weight + ctcvr_weight <= 0
        ):
            raise ValueError("Loss weights must be finite, nonnegative and not both zero")
        self.ctr_weight, self.ctcvr_weight = float(ctr_weight), float(ctcvr_weight)

    def components(self, predictions, targets):
        """Return scalar loss/ctr_loss/ctcvr_loss; accept labels [N] or [N, 1]."""
        labels = {}
        for name in ("ctr", "ctcvr"):
            p, y = predictions[name], targets[name]
            if p.ndim != 2 or p.shape[1] != 1 or p.shape[0] == 0:
                raise ValueError("Predictions must have nonempty shape [N, 1]")
            if y.ndim == 1:
                y = y.unsqueeze(1)
            if y.shape != p.shape or not torch.all((y == 0) | (y == 1)):
                raise ValueError("Labels must be binary with shape [N] or [N, 1]")
            if not torch.isfinite(p).all() or torch.any((p < 0) | (p > 1)):
                raise ValueError("Predictions must be finite probabilities")
            labels[name] = y.to(device=p.device, dtype=p.dtype)
        if labels["ctr"].shape != labels["ctcvr"].shape:
            raise ValueError("Task label shapes differ")
        if torch.any(labels["ctcvr"] > labels["ctr"]):
            raise ValueError("Conversion chain requires y_ctcvr <= y_ctr")
        ctr = F.binary_cross_entropy(predictions["ctr"], labels["ctr"])
        ctcvr = F.binary_cross_entropy(predictions["ctcvr"], labels["ctcvr"])
        return {
            "loss": self.ctr_weight * ctr + self.ctcvr_weight * ctcvr,
            "ctr_loss": ctr,
            "ctcvr_loss": ctcvr,
        }

    def forward(self, predictions, targets):
        return self.components(predictions, targets)["loss"]
