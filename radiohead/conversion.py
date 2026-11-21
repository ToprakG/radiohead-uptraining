"""Full Qwen conversion and compact trainable-weight checkpoints."""

import torch

from .adaptation import LowRankLinear, attach_adapters
from .attention import RadioAttention


def convert_all(model, solver, rank):
    layers = model.model.layers
    for layer in layers:
        radio = RadioAttention(layer.self_attn, solver, training_length=1024)
        for name in ("q_proj", "v_proj", "o_proj"):
            setattr(radio, name, LowRankLinear(getattr(radio, name), rank))
        layer.self_attn = radio
    attach_adapters(model, rank, range(len(layers)))
    if not all(isinstance(layer.self_attn, RadioAttention) for layer in layers):
        raise RuntimeError("Full Radio Head conversion failed")


def trainable_weights(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def restore_full(model, solver, checkpoint):
    model.requires_grad_(False)
    convert_all(model, solver, checkpoint["rank"])
    parameters = trainable_weights(model)
    weights = checkpoint["weights"]
    if checkpoint["layers"] != len(model.model.layers) or parameters.keys() != weights.keys():
        raise ValueError("Full conversion checkpoint does not match this model")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(weights[name])
