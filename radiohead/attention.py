"""Grouped-query extension of the pinned upstream KM attention."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RadioAttention(nn.Module):
    def __init__(self, teacher, solve_mu, training_length=1024, query_chunk=128):
        super().__init__()
        self.q_proj, self.v_proj, self.o_proj = teacher.q_proj, teacher.v_proj, teacher.o_proj
        self.layer_idx = getattr(teacher, "layer_idx", 0)
        self.head_dim = teacher.head_dim
        self.groups = teacher.num_key_value_groups
        self.heads = self.q_proj.out_features // self.head_dim
        self.solve_mu = solve_mu
        self.query_chunk = query_chunk
        device = self.q_proj.weight.device
        eps = training_length ** -0.5
        coupling = 1.5 * eps / math.sqrt(
            2 / self.head_dim * sum((age - 1.5) ** -2 for age in range(1, training_length))
        )
        self.log_eps = nn.Parameter(torch.full((self.heads,), math.log(eps), device=device))
        self.log_coupling = nn.Parameter(torch.full_like(self.log_eps, math.log(coupling)))
        self.log_gain = nn.Parameter(torch.zeros_like(self.log_eps))
        self.log_selectivity = nn.Parameter(torch.zeros_like(self.log_eps))
        self.logit_floor = nn.Parameter(torch.full_like(self.log_eps, math.log(1e-8)))
        self.dial = nn.Linear(self.q_proj.in_features, self.heads, device=device)
        nn.init.zeros_(self.dial.weight)
        nn.init.constant_(self.dial.bias, 1.5 * eps)

    def project_unit(self, projection, hidden):
        projected = projection(hidden).unflatten(-1, (-1, self.head_dim)).transpose(1, 2)
        return F.normalize(projected.float(), dim=-1)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_value=None, cache_position=None, **kwargs):
        length = hidden_states.shape[1]
        device = hidden_states.device
        if cache_position is None:
            cache_position = torch.arange(length, device=device)
        positions = kwargs.get("position_ids")
        if positions is None:
            positions = cache_position[None, :]
        q = self.project_unit(self.q_proj, hidden_states)
        v = self.project_unit(self.v_proj, hidden_states)
        if past_key_value is not None:
            from transformers.cache_utils import DynamicCache

            if not isinstance(past_key_value, DynamicCache):
                raise ValueError("RadioAttention supports the Qwen2 dynamic cache")
            _, v = past_key_value.update(v, v, self.layer_idx)
        key_order = torch.arange(v.shape[-2], device=device)
        if past_key_value is None:
            key_positions = positions
        else:
            key_positions = key_order[None, :] + (positions[:, :1] - cache_position[0])
        v = v.repeat_interleave(self.groups, dim=1)
        rotated_v = torch.stack((-v[..., 1::2], v[..., 0::2]), dim=-1).flatten(-2)
        dial = self.dial(hidden_states.to(self.dial.weight.dtype)).float().transpose(1, 2)
        coupling = self.log_coupling.float().exp()
        eps = self.log_eps.float().exp()[None, :, None, None]
        floor = self.logit_floor.float().sigmoid()[None, :, None, None]
        selectivity = self.log_selectivity.float().exp()[None, :, None, None]
        k = coupling[None, :, None, None]
        chunks, powers = [], []
        for start in range(0, length, self.query_chunk):
            rows = slice(start, start + self.query_chunk)
            query = q[:, :, rows]
            overlap = (query @ v.transpose(-1, -2)).square()
            overlap = overlap + (query @ rotated_v.transpose(-1, -2)).square()
            gain2 = floor + (1 - floor) * overlap.clamp(1e-12, 1).pow(selectivity)
            age = positions[:, rows, None] - key_positions[:, None, :]
            detuning = dial[:, :, rows, None] - eps * age[:, None]
            valid = cache_position[rows, None] >= key_order[None, :]
            if attention_mask is not None:
                mask = attention_mask[:, :, rows, :key_order.numel()]
                valid = valid & (mask if mask.dtype == torch.bool else mask == 0)
            kg2 = k.square() * gain2 * valid
            detuning2 = detuning.square()
            width = self.solve_mu(kg2, detuning2, 1e-3 * coupling).unsqueeze(-1)
            denominator = width.square() + detuning2
            amplitude = k * gain2.sqrt() * valid / denominator
            chunks.append((amplitude * width) @ v - (amplitude * detuning) @ rotated_v)
            if kwargs.get("output_attentions"):
                powers.append(kg2 / denominator)
        combined = torch.cat(chunks, dim=2) * self.log_gain.float().exp()[None, :, None, None]
        output = combined.transpose(1, 2).flatten(2).to(hidden_states.dtype)
        return self.o_proj(output), torch.cat(powers, dim=2) if powers else None
