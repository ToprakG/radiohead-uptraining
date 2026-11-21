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
from radiohead.adaptation import attach_adapters, restore_adapters
from radiohead.attention import RadioAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--distilled", type=Path)
    start.add_argument("--resume", type=Path, help="Warm-start from an adapted checkpoint")
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", default="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--retrieval-every", type=int, default=10)
    parser.add_argument("--absent-every", type=int,
                        help="Frequency of absent cases among retrieval examples (default: 2, resume: 5)")
    parser.add_argument("--confusable-training", action="store_true",
                        help="Draw background from the training text and use similar record IDs")
    args = parser.parse_args()
    if args.absent_every is None:
        args.absent_every = 5 if args.resume else 2
    if args.length < 1024 or min(args.rank, args.steps, args.retrieval_every, args.absent_every) < 1:
        parser.error("Requires length >= 1024 and positive rank, steps, and retrieval interval")
    torch.manual_seed(0)
    rng = random.Random(args.seed)
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
    initial = args.resume or args.distilled
    if initial:
        checkpoint.update(torch.load(initial, map_location="cpu", weights_only=True))
        if (checkpoint["model"], checkpoint["revision"]) != (args.model, args.revision):
            parser.error("Starting checkpoint must match the model and revision")
    if "layer" in checkpoint:
        layer = model.model.layers[checkpoint["layer"]]
        radio = RadioAttention(layer.self_attn, solve_mu, training_length=checkpoint["training_length"])
        radio.load_state_dict(checkpoint["state_dict"])
        # Keep the pretrained projection dtype for inference, but optimize fp32 master weights.
        radio.requires_grad_(True)

        for projection in (radio.q_proj, radio.v_proj, radio.o_proj):
            projection.requires_grad_(False)
        layer.self_attn = radio
    if args.resume:
        if checkpoint.get("rank") != args.rank or "adapters" not in checkpoint:
            parser.error("Resume checkpoint must have adapters of the specified rank")
        restore_adapters(model, checkpoint)
        adapters = {
            index: model.model.layers[int(index)].mlp.down_proj for index in checkpoint["adapters"]
        }
    else:
        adapters = attach_adapters(
            model, args.rank, range(max(0, len(model.model.layers) - 4), len(model.model.layers))
        )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    for step in range(args.steps):
        if step % args.retrieval_every == 0:
            seed = rng.randrange(100_000, 2**31)
            absent = (step // args.retrieval_every) % args.absent_every == 0
            background = text if args.confusable_training else None
            distance = rng.randrange(args.length * 2 // 5, args.length * 3 // 5)
            prompt, answer = make_prompt(tokenizer, args.length, distance, seed, absent, background)
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
