"""Uptrain fully converted Qwen on next-token targets from ordinary documents."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream/radiohead/code/lab")]

from layers import solve_mu
from radiohead.conversion import convert_all, restore_full, trainable_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", default="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    rng = random.Random(181)
    model_args = dict(revision=args.revision, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    teacher = AutoModelForCausalLM.from_pretrained(args.model, **model_args).cuda().eval().requires_grad_(False)
    if len(teacher.model.layers) != 28:
        raise ValueError("The full-Qwen experiment requires 28 decoder layers")
    student = AutoModelForCausalLM.from_pretrained(args.model, **model_args).cuda().requires_grad_(False)
    convert_all(student, solve_mu, args.rank)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(trainable_weights(student).values(), lr=3e-4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    documents = [
        tokenizer.encode(path.read_text(encoding="utf-8", errors="replace"),
                         add_special_tokens=False, verbose=False)
        for path in sorted(args.train.glob("*.txt"))
    ]
    documents = [doc for doc in documents if len(doc) > 2048]
    if not documents:
        parser.error("Training needs document windows longer than 2048 tokens")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(step):
        temporary = args.output.with_suffix(".tmp")
        torch.save({"full_radio": True, "layers": len(student.model.layers), "model": args.model,
                    "revision": args.revision, "rank": args.rank, "step": step,
                    "weights": {k: p.detach().cpu() for k, p in trainable_weights(student).items()}},
                   temporary)
        temporary.replace(args.output)

    student.train()
    schedule = [128, 512, 1024, 2048] if args.smoke else [512] * 16 + [1024] * 48 + [2048] * 32
    if not args.smoke:
        rng.shuffle(schedule)
        schedule = (schedule * ((args.steps + len(schedule) - 1) // len(schedule)))[:args.steps]
    for step, length in enumerate(schedule, 1):
        doc = rng.choice(documents)
        offset = rng.randrange(len(doc) - length)
        tokens = torch.tensor([doc[offset:offset + length + 1]], device="cuda")
        positions = torch.tensor(
            sorted(rng.sample(range(length), min(64, length))), device="cuda"
        )
        targets = tokens[:, positions + 1].flatten()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            hidden = teacher.model(tokens[:, :-1], use_cache=False).last_hidden_state
            teacher_logits = teacher.lm_head(hidden[:, positions]).float().flatten(0, 1)
        hidden = student.model(tokens[:, :-1], use_cache=False).last_hidden_state
        logits = student.lm_head(hidden[:, positions]).float().flatten(0, 1)
        ce = F.cross_entropy(logits, targets)
        kl = F.kl_div(logits.log_softmax(-1), teacher_logits.log_softmax(-1),
                      reduction="batchmean", log_target=True)
        (ce + kl).backward()
        torch.nn.utils.clip_grad_norm_(trainable_weights(student).values(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        print(json.dumps({"step": step, "tokens": length, "supervised_positions": len(positions),
                          "ce": ce.item(), "kl": kl.item(),
                          "peak_vram_bytes": torch.cuda.max_memory_allocated()}), flush=True)
        if step % 16 == 0 or step == len(schedule):
            save(step)
    if args.smoke:
        student.eval()
        student.gradient_checkpointing_disable()
        checkpoint = torch.load(args.output, weights_only=True, map_location="cpu")
        del teacher
        reloaded = AutoModelForCausalLM.from_pretrained(args.model, **model_args).cuda().eval()
        restore_full(reloaded, solve_mu, checkpoint)
        with torch.inference_mode():
            ids = tokens[:, :32]
            uncached = student(ids, use_cache=False).logits[:, -1].float()
            restored = reloaded(ids, use_cache=False).logits[:, -1].float()
            torch.testing.assert_close(restored, uncached, atol=0.02, rtol=0.02)
            prefix = reloaded(ids[:, :-1], use_cache=True)
            decode = reloaded(ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True)
            drift = (decode.logits[:, -1].float() - restored).abs()
            print(json.dumps({"bf16_cache_max_logit_drift": drift.max().item(),
                              "bf16_cache_mean_logit_drift": drift.mean().item(),
                              "bf16_top_token_matches": bool(
                                  decode.logits[:, -1].argmax(-1).eq(restored.argmax(-1)).all()
                              )}), flush=True)
            del student
            reloaded.float()
            dense_fp32 = reloaded(ids, use_cache=False).logits[:, -1]
            prefix_fp32 = reloaded(ids[:, :-1], use_cache=True)
            decode_fp32 = reloaded(ids[:, -1:], past_key_values=prefix_fp32.past_key_values,
                                   use_cache=True).logits[:, -1]
            torch.testing.assert_close(decode_fp32, dense_fp32, atol=0.007, rtol=0.005)
        print("28-layer checkpoint reload and FP32 cached decoding parity passed.", flush=True)


if __name__ == "__main__":
    main()
