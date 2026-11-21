"""Check grouped-query Radio Head against upstream KM with one group per head."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream/radiohead/code/lab")]

from layers import Cfg, KM, solve_mu
from radiohead.attention import RadioAttention


def main():
    torch.manual_seed(0)
    steps, width, heads = 24, 32, 2
    upstream = KM(Cfg(max_len=steps, km_same=True, mask_self=False), width, heads)
    adapter = RadioAttention(
        SimpleNamespace(q_proj=upstream.W_q, v_proj=upstream.W_v, o_proj=upstream.W_o,
                        head_dim=width // heads, num_key_value_groups=1),
        solve_mu, training_length=steps, query_chunk=7,
    )
    with torch.no_grad():
        adapter.dial.load_state_dict(upstream.W_alpha.state_dict())
        adapter.log_eps.copy_(upstream.log_eps)
        adapter.log_coupling.copy_(upstream.log_K)
        adapter.log_gain.copy_(upstream.log_gain)
        adapter.log_selectivity.copy_(upstream.log_n)
        adapter.logit_floor.copy_(upstream.raw_rho)

    x = torch.rand(2, steps, width, requires_grad=True)
    valid = torch.ones(2, steps, dtype=torch.bool)
    valid[1, -2:] = False
    expected, _, diagnostics = upstream(x, key_valid=valid)
    actual, power = adapter(x, attention_mask=valid[:, None, None, :].expand(-1, 1, steps, -1),
                            output_attentions=True)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(power.sum(-1), diagnostics["wsum"], rtol=1e-4, atol=1e-5)
    expected_gradient, = torch.autograd.grad(expected.square().mean(), x)
    actual_gradient, = torch.autograd.grad(actual.square().mean(), x)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-3, atol=1e-6)
    print("Upstream parity passed for outputs, attention power, and input gradients.")


if __name__ == "__main__":
    main()
