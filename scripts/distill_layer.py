"""Fit one Radio Head layer to a frozen Qwen2 attention teacher."""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream/radiohead/code/lab")]

from layers import solve_mu
from radiohead.attention import RadioAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, type=Path, help="UTF-8 training corpus")
    parser.add_argument("--validation", required=True, type=Path, help="Separate held-out UTF-8 corpus")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", default="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)

    def blocks(path):
        ids = tokenizer.encode(path.read_text(encoding="utf-8"), add_special_tokens=False)
        if len(ids) < args.length:
            parser.error(f"{path} must contain at least {args.length} tokens")
        return torch.tensor(ids[:len(ids) // args.length * args.length]).view(-1, args.length)

    training, validation = blocks(args.text), blocks(args.validation)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, attn_implementation="eager",
        torch_dtype=torch.float32 if args.device == "cpu" else torch.bfloat16,
    ).to(args.device).eval().requires_grad_(False)
    layer = model.model.layers[args.layer]
    teacher = layer.self_attn
    student = RadioAttention(copy.deepcopy(teacher).float(), solve_mu, training_length=args.length)
    student.requires_grad_(True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
    captured = {}

    def capture(module, positional, keywords, output):
        hidden = keywords["hidden_states"] if "hidden_states" in keywords else positional[0]
        captured.update(hidden=hidden.float(), mask=keywords.get("attention_mask"), target=output[0].float())

    handle = teacher.register_forward_hook(capture, with_kwargs=True)

    def loss_on(block):
        with torch.no_grad():
            model.model(block[None].to(args.device), use_cache=False)
        prediction, _ = student(captured["hidden"], attention_mask=captured["mask"])
        loss = torch.nn.functional.mse_loss(prediction, captured["target"])
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite distillation loss. Teacher finite: {torch.isfinite(captured['target']).all().item()}, "
                f"student finite: {torch.isfinite(prediction).all().item()}"
            )
        return loss

    def measure(label):
        with torch.no_grad():
            losses = [loss_on(block).item() for block in validation[:4]]
        print(json.dumps({"phase": label, "validation_mse": sum(losses) / len(losses)}), flush=True)

    try:
        measure("initial")
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_on(training[step % len(training)])
            loss.backward()
            optimizer.step()
            print(json.dumps({"step": step + 1, "mse": loss.item()}), flush=True)
        measure("distilled")
    finally:
        handle.remove()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": student.state_dict(), "model": args.model, "revision": args.revision,
                "layer": args.layer, "training_length": args.length}, args.output)


if __name__ == "__main__":
    main()
