"""Score generated answers on seeded synthetic multi-key retrieval prompts."""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def make_prompt(tokenizer, length, distance, seed, absent):
    rng = random.Random(seed)
    keys = rng.sample(range(1000, 9999), 16)
    values = rng.sample(range(10000, 99999), len(keys))
    target = keys[0]
    expected = "ABSENT" if absent else str(values[0])
    records = list(zip(keys[1:], values[1:]))
    distractors = [f"Record REF-{key} has code {value}." for key, value in records]
    needle = "" if absent else f"Record REF-{target} has code {values[0]}."
    marker = "ARCHIVE_CONTENT_PLACEHOLDER"
    message = (
        f"Read the archive. Give only the five-digit code for the requested record "
        f"or ABSENT if the record is missing.\n{marker}\n"
        f"What is the code for record REF-{target}?"
    )
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True
    )
    prefix, suffix = formatted.split(marker)

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    prefix_ids, suffix_ids = encode(prefix), encode(suffix)
    filler = encode("\nThe archive contains regional filing notes and administrative correspondence.\n")
    entries = [encode(f"\n{line}\n") for line in distractors]
    needle_ids = encode(f"\n{needle}\n")
    # Build in token space so test length and the distance after the evidence are exact.
    after = distance - len(suffix_ids)
    before = length - len(prefix_ids) - len(needle_ids) - distance
    if min(before, after) < sum(map(len, entries)):
        raise ValueError("Choose a distance that leaves room for distractors on both sides")

    def haystack(size, records):
        tokens = []
        gap = (size - sum(map(len, records))) // (len(records) + 1)
        for record in records:
            tokens.extend((filler * (gap // len(filler) + 1))[:gap])
            tokens.extend(record)
        tail = size - len(tokens)
        tokens.extend((filler * (tail // len(filler) + 1))[:tail])
        return tokens

    ids = prefix_ids + haystack(before, entries[:7]) + needle_ids
    ids += haystack(after, entries[7:]) + suffix_ids
    return ids, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", required=True, help="Model commit SHA")
    parser.add_argument("--checkpoint", "--radio-checkpoint", dest="checkpoint", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--distances", type=int, nargs="+", default=[1024, 4096, 8192, 15360])
    parser.add_argument("--examples", type=int, default=25, help="Examples per distance and condition")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, attn_implementation="eager" if args.checkpoint else None,
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32
    ).to(args.device).eval()
    use_cache = True
    if args.checkpoint:
        root = Path(__file__).resolve().parents[1]
        sys.path[:0] = [str(root), str(root / "upstream/radiohead/code/lab")]
        from layers import solve_mu
        from radiohead.adaptation import restore_adapters
        from radiohead.attention import RadioAttention

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if (checkpoint["model"], checkpoint["revision"]) != (args.model, args.revision):
            parser.error("Adaptation checkpoint must match the model and revision")
        restore_adapters(model, checkpoint)
        if "layer" in checkpoint:
            layer = model.model.layers[checkpoint["layer"]]
            radio = RadioAttention(layer.self_attn, solve_mu, training_length=checkpoint["training_length"])
            radio.load_state_dict(checkpoint["state_dict"])
            layer.self_attn = radio.eval()
            use_cache = False
    with torch.inference_mode():
        for distance in args.distances:
            for absent in (False, True):
                for index in range(args.examples):
                    seed = args.seed + index
                    ids, expected = make_prompt(tokenizer, args.length, distance, seed, absent)
                    inputs = torch.tensor([ids], device=args.device)
                    output = model.generate(
                        inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                        max_new_tokens=32, use_cache=use_cache, pad_token_id=tokenizer.eos_token_id,
                    )
                    response = tokenizer.decode(output[0, len(ids):], skip_special_tokens=True).strip()
                    answers = set(re.findall(r"(?<!\d)\d{5}(?!\d)", response))
                    if re.fullmatch(r"ABSENT[.!]?", response, re.IGNORECASE):
                        prediction = "ABSENT"
                    else:
                        prediction = next(iter(answers)) if len(answers) == 1 else None
                    print(json.dumps({
                        "model": args.model, "revision": args.revision, "seed": seed,
                        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                        "length": len(ids), "distance": distance, "absent": absent,
                        "expected": expected, "response": response, "prediction": prediction,
                        "correct": prediction == expected,
                    }), flush=True)


if __name__ == "__main__":
    main()
