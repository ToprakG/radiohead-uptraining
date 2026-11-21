"""Adapt a stock Qwen or a distilled Radio Head hybrid with a matched data stream."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream/radiohead/code/lab")]

from eval_retrieval import make_prompt
from layers import solve_mu
from radiohead.adaptation import attach_adapters
from radiohead.attention import RadioAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--distilled", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", default="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()
    if args.length < 1024 or args.rank < 1 or args.steps < 1:
        parser.error("Requires length >= 1024 and positive rank and steps")
    torch.manual_seed(0)
    rng = random.Random(100_000)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    text = tokenizer.encode(args.text.read_text(encoding="utf-8"), add_special_tokens=False)
    if len(text) < args.length:
        parser.error("Training corpus is shorter than one training example")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, attn_implementation="eager",
        torch_dtype=torch.float32 if args.device == "cpu" else torch.bfloat16,
    ).to(args.device).train().requires_grad_(False)
    checkpoint = {"model": args.model, "revision": args.revision, "rank": args.rank}
    radio = None
    if args.distilled:
        checkpoint.update(torch.load(args.distilled, map_location="cpu", weights_only=True))
        if (checkpoint["model"], checkpoint["revision"]) != (args.model, args.revision):
            parser.error("Distilled checkpoint must match the model and revision")
        layer = model.model.layers[checkpoint["layer"]]
        radio = RadioAttention(layer.self_attn, solve_mu, training_length=checkpoint["training_length"])
        radio.load_state_dict(checkpoint["state_dict"])
        # Keep the pretrained projection dtype for inference, but optimize fp32 master weights.
        radio.requires_grad_(True)

        for projection in (radio.q_proj, radio.v_proj, radio.o_proj):
            projection.requires_grad_(False)
        layer.self_attn = radio
    adapters = attach_adapters(model, args.rank, range(max(0, len(model.model.layers) - 4), len(model.model.layers)))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    for step in range(args.steps):
        if step % 10 == 0:
            seed = rng.randrange(100_000, 2**31)
            prompt, answer = make_prompt(tokenizer, args.length, args.length // 2, seed, step % 20 == 0)
            tokens = prompt + tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
            labels = [-100] * len(prompt) + tokens[len(prompt):]
        else:
            start = rng.randrange(len(text) - args.length + 1)
            tokens = labels = text[start:start + args.length]
        inputs = torch.tensor([tokens], device=args.device)
        targets = torch.tensor([labels], device=args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=inputs, labels=targets, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        print(json.dumps({"step": step + 1, "tokens": len(tokens), "loss": loss.item()}), flush=True)

    if radio is not None:
        checkpoint["state_dict"] = radio.state_dict()
    checkpoint["adapters"] = {
        index: {"down": adapter.down.detach().cpu(), "up": adapter.up.detach().cpu()}
        for index, adapter in adapters.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)


if __name__ == "__main__":
    main()
