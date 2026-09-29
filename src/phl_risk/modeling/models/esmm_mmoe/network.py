"""ESMM + MMoE extracted from TaoXue-99/ModelMatrix's model/esmm.py.

The shared MLP, expert/gate tensors and task towers preserve the original
architecture. Device placement and business preprocessing belong to callers.
"""

import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


def _initialize_shared(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=1 / math.sqrt(module.in_features))
        nn.init.zeros_(module.bias)


class _SharedNetwork(nn.Module):
    def __init__(
        self,
        num_continuous,
        cardinalities,
        embedding_dim,
        share_dim,
        base_dim,
        dropout,
        batch_norm,
        normalize_shared,
    ):
        super().__init__()
        self.normalize_shared = normalize_shared
        self.embeddings = nn.ModuleList(nn.Embedding(n, embedding_dim) for n in cardinalities)
        cat_dim = len(cardinalities) * embedding_dim
        self.batch_norm = nn.BatchNorm1d(cat_dim) if cat_dim and batch_norm else None
        total = num_continuous + cat_dim
        layers = [
            nn.Linear(total, share_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(share_dim, share_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(share_dim, base_dim),
            nn.ELU(),
        ]
        if batch_norm:
            layers.insert(0, nn.BatchNorm1d(total))
        self.DNN = nn.Sequential(*layers)
        self.DNN.apply(_initialize_shared)

    def forward(self, x_cat, x_cont):
        if self.embeddings:
            embedded = torch.cat([e(x_cat[:, i]) for i, e in enumerate(self.embeddings)], dim=1)
            if self.batch_norm is not None:
                embedded = self.batch_norm(embedded)
            x_cont = torch.cat([embedded, x_cont], dim=1)
        shared = self.DNN(x_cont)
        return F.normalize(shared, dim=1, eps=1e-12) if self.normalize_shared else shared


class _MMoE(nn.Module):
    def __init__(self, input_size, units, num_experts, use_expert_bias, use_gate_bias, activation):
        super().__init__()
        self.expert_kernels = nn.Parameter(torch.randn(input_size, units, num_experts))
        self.gate_kernels = nn.ParameterList(
            [nn.Parameter(torch.randn(input_size, num_experts)) for _ in range(2)]
        )
        self.expert_kernels_bias = nn.Parameter(
            torch.randn(units, num_experts), requires_grad=use_expert_bias
        )
        self.gate_kernels_bias = nn.ParameterList(
            [nn.Parameter(torch.randn(num_experts), requires_grad=use_gate_bias) for _ in range(2)]
        )
        self.use_expert_bias, self.use_gate_bias = use_expert_bias, use_gate_bias
        self.activation = activation

    def forward(self, x):
        experts = torch.einsum("bi,ium->bum", x, self.expert_kernels)
        if self.use_expert_bias:
            experts = experts + self.expert_kernels_bias
        if self.activation == "elu":
            experts = F.elu(experts)
        outputs = []
        for kernel, bias in zip(self.gate_kernels, self.gate_kernels_bias, strict=True):
            gate = x @ kernel
            if self.use_gate_bias:
                gate = gate + bias
            outputs.append((experts * gate.softmax(dim=-1).unsqueeze(1)).sum(dim=-1))
        return outputs


class _Tower(nn.Module):
    def __init__(self, input_dim, base_dim, dropout):
        super().__init__()
        layers = []
        for width in (input_dim, base_dim, base_dim):
            layers.extend([nn.Linear(width, base_dim), nn.ELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(base_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.mlp(x))


class ESMMMoE(nn.Module):
    """Two-stage conversion model returning ctr, cvr and their product ctcvr.

    Inputs: categorical int64 [N, C] and floating continuous [N, D]. An empty
    [N, 0] categorical tensor is supported. Category IDs must be in [0, cardinality).
    Reserve an ID for unknown/missing categories in the external encoder.
    BatchNorm training requires more than one sample per batch.
    """

    def __init__(
        self,
        num_continuous,
        categorical_cardinalities=(),
        embedding_dim=4,
        share_dim=64,
        base_dim=32,
        expert_units=16,
        num_experts=4,
        dropout=0.1,
        batch_norm=False,
        normalize_shared=False,
        use_expert_bias=True,
        use_gate_bias=True,
        expert_activation="elu",
    ):
        super().__init__()
        cardinalities = list(categorical_cardinalities)
        for name, value in {
            "num_continuous": num_continuous,
            "embedding_dim": embedding_dim,
            "share_dim": share_dim,
            "base_dim": base_dim,
            "expert_units": expert_units,
            "num_experts": num_experts,
        }.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if any(type(n) is not int or n <= 0 for n in cardinalities):
            raise ValueError("categorical_cardinalities must contain positive integers")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for flag in (batch_norm, normalize_shared, use_expert_bias, use_gate_bias):
            if type(flag) is not bool:
                raise ValueError("Boolean options require Python bool values")
        if expert_activation not in ("elu", "identity"):
            raise ValueError("expert_activation must be elu or identity")
        self._config = {
            "num_continuous": num_continuous,
            "categorical_cardinalities": cardinalities,
            "embedding_dim": embedding_dim,
            "share_dim": share_dim,
            "base_dim": base_dim,
            "expert_units": expert_units,
            "num_experts": num_experts,
            "dropout": dropout,
            "batch_norm": batch_norm,
            "normalize_shared": normalize_shared,
            "use_expert_bias": use_expert_bias,
            "use_gate_bias": use_gate_bias,
            "expert_activation": expert_activation,
        }
        self.share_network = _SharedNetwork(
            num_continuous,
            cardinalities,
            embedding_dim,
            share_dim,
            base_dim,
            dropout,
            batch_norm,
            normalize_shared,
        )
        self.mmoe = _MMoE(
            base_dim, expert_units, num_experts, use_expert_bias, use_gate_bias, expert_activation
        )
        self.p1_network = _Tower(expert_units, base_dim, dropout)
        self.p2_network = _Tower(expert_units, base_dim, dropout)

    def get_config(self):
        """Detached constructor parameters for reproducible reconstruction."""
        return deepcopy(self._config)

    def forward(self, x_cat, x_cont):
        cards = self._config["categorical_cardinalities"]
        if (
            x_cont.ndim != 2
            or x_cont.shape[1] != self._config["num_continuous"]
            or not x_cont.is_floating_point()
            or x_cont.shape[0] == 0
        ):
            raise ValueError("x_cont must be a nonempty floating [N, num_continuous] tensor")
        if x_cat.shape != (len(x_cont), len(cards)) or x_cat.dtype != torch.long:
            raise ValueError("x_cat must be int64 [N, number of categorical features]")
        if x_cat.device != x_cont.device:
            raise ValueError("Categorical and continuous inputs must be on the same device")
        if not torch.isfinite(x_cont).all():
            raise ValueError("Continuous features must be finite; impute before the model")
        for index, size in enumerate(cards):
            if torch.any((x_cat[:, index] < 0) | (x_cat[:, index] >= size)):
                raise ValueError(f"Category IDs out of range for feature {index}")
        if self.training and self._config["batch_norm"] and len(x_cont) < 2:
            raise ValueError("BatchNorm training requires batch size > 1; adjust the DataLoader")
        task1, task2 = self.mmoe(self.share_network(x_cat, x_cont))
        ctr, cvr = self.p1_network(task1), self.p2_network(task2)
        return {"ctr": ctr, "cvr": cvr, "ctcvr": ctr * cvr}
