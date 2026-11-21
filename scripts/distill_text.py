"""Fit one Radio Head replacement using only ordinary document windows."""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream/radiohead/code/lab")]

from layers import solve_mu
from radiohead.attention import RadioAttention


def save_checkpoint(path, student, optimizer, rng, args, updates):
    temporary = path.with_suffix(".tmp")
    torch.save(
        {"state_dict": student.state_dict(), "optimizer": optimizer.state_dict(),
         "sampler": rng.getstate(), "updates": updates, "model": args.model,
         "revision": args.revision, "layer": 14, "training_length": 256, "arm": args.arm},
        temporary,
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", default="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("short", "mixed"), required=True)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    rng = random.Random(128)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    documents = [
        tokenizer.encode(path.read_text(encoding="utf-8", errors="replace"),
                         add_special_tokens=False, verbose=False)
        for path in sorted(args.train.glob("*.txt"))
    ]
    documents = [document for document in documents if len(document) >= 4096]
    if not documents:
        parser.error("Training requires at least one document with 4096 tokens")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval().requires_grad_(False)
    layer = teacher.model.layers[14]
    student = RadioAttention(copy.deepcopy(layer.self_attn).float(), solve_mu, training_length=256)
    student.requires_grad_(True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
    captured = {}

    def capture(module, positional, keywords, output):
        captured.update(hidden=keywords["hidden_states"].float(),
                        mask=keywords.get("attention_mask"),
                        positions=keywords.get("position_ids"), target=output[0].float())

    handle = layer.self_attn.register_forward_hook(capture, with_kwargs=True)
    if args.arm == "short":
        schedule = [1024] * 80
    else:
        schedule = [1024] * 16 + [2048] * 32 + [4096] * 32
        rng.shuffle(schedule)
    if args.calibrate:
        schedule = [1024, 2048, 4096]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Match the same 4096-token source spans across arms.
    span_rng = random.Random(781)
    try:
        for step, length in enumerate(schedule, 1):
            document = span_rng.choice(documents)
            start = span_rng.randrange(len(document) - 4096 + 1)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            torch.cuda.reset_peak_memory_stats()
            for offset in range(0, 4096, length):
                ids = torch.tensor([document[start + offset:start + offset + length]], device="cuda")
                with torch.no_grad():
                    teacher.model(ids, use_cache=False)
                prediction, _ = student(captured["hidden"], attention_mask=captured["mask"],
                                        position_ids=captured["positions"])
                target = captured["target"]
                loss = (prediction - target).square().mean() / target.square().mean().clamp_min(1e-8)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite transfer loss at update {step}")
                (loss * (length / 4096)).backward()
                losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            print(json.dumps({"arm": args.arm, "step": step, "window": length, "input_tokens": 4096,
                              "relative_mse": sum(losses) / len(losses),
                              "peak_vram_bytes": torch.cuda.max_memory_allocated()}), flush=True)
            if step % 20 == 0 or step == len(schedule):
                save_checkpoint(args.output, student, optimizer, rng, args, step)
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
