"""Low-rank residual adapters for the last decoder MLPs."""

import math

import torch
from torch import nn


class LowRankLinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.down = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    def forward(self, x):
        update = (x.float() @ self.down.T) @ self.up.T
        return self.base(x) + update.to(x.dtype)


def attach_adapters(model, rank, layers):
    adapters = {}
    for index in layers:
        mlp = model.model.layers[index].mlp
        adapter = LowRankLinear(mlp.down_proj, rank)
        mlp.down_proj = adapter
        adapters[str(index)] = adapter
    return adapters


def restore_adapters(model, checkpoint):
    if "adapters" not in checkpoint:
        return
    adapter_states = checkpoint["adapters"]
    adapters = attach_adapters(model, checkpoint["rank"], [int(index) for index in adapter_states])
    with torch.no_grad():
        for index, adapter in adapters.items():
            adapter.down.copy_(adapter_states[index]["down"])
            adapter.up.copy_(adapter_states[index]["up"])
