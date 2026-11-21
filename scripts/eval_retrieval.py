"""Score generated answers on seeded synthetic multi-key retrieval prompts."""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def make_prompt(tokenizer, length, distance, seed, absent, background=None):
    rng = random.Random(seed)
    if background is None:
        keys = rng.sample(range(1000, 9999), 16)
    else:
        prefix = rng.randrange(10, 100) * 100
        keys = [prefix + suffix for suffix in rng.sample(range(100), 16)]
    values = rng.sample(range(10000, 99999), len(keys))
    target = keys[0]
    expected = "ABSENT" if absent else str(values[0])
    records = list(zip(keys[1:], values[1:]))
    distractors = [f"Record REF-{key} has code {value}." for key, value in records]
    needle = f"Record REF-{target} has code {values[0]}."
    if absent:
        if background is None:
            needle = ""
        else:
            decoy = rng.choice([key for key in range(prefix, prefix + 100) if key not in keys])
            needle = f"Record REF-{decoy} has code {values[0]}."
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

    background_offset = random.Random(seed ^ 0x52414449).randrange(len(background)) if background else 0

    def filler_tokens(size):
        nonlocal background_offset
        if not background:
            return (filler * (size // len(filler) + 1))[:size]
        end = background_offset + size
        if end <= len(background):
            segment = background[background_offset:end]
        else:
            tail = background[background_offset:]
            remaining = size - len(tail)
            segment = tail + (background * (remaining // len(background) + 1))[:remaining]
        background_offset = end % len(background)
        return segment

    def haystack(size, records):
        tokens = []
        gap = (size - sum(map(len, records))) // (len(records) + 1)
        for record in records:
            tokens.extend(filler_tokens(gap))
            tokens.extend(record)
        tail = size - len(tokens)
        tokens.extend(filler_tokens(tail))
        return tokens

    ids = prefix_ids + haystack(before, entries[:7]) + needle_ids
    ids += haystack(after, entries[7:]) + suffix_ids
    return ids, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", required=True, help="Model commit SHA")
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--checkpoint", "--radio-checkpoint", dest="checkpoint", type=Path)
    models.add_argument("--raw-layer", type=int, help="Evaluate an initialized Radio Head replacement")
    parser.add_argument("--raw-init-length", type=int, default=256)
    parser.add_argument("--language", type=Path, help="Score held-out text loss instead of retrieval")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--distances", type=int, nargs="+", default=[1024, 4096, 8192, 15360])
    parser.add_argument("--examples", type=int, default=25, help="Examples per distance and condition")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--background", type=Path, help="Held-out UTF-8 text for confusable-key evaluation")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    background = None
    if args.background:
        background = tokenizer.encode(args.background.read_text(encoding="utf-8"), add_special_tokens=False)
        if len(background) < args.length:
            parser.error("Background text must contain at least one context window")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16 if args.device != "cpu" else torch.float32
    ).to(args.device).eval()
    if args.checkpoint or args.raw_layer is not None:
        root = Path(__file__).resolve().parents[1]
        sys.path[:0] = [str(root), str(root / "upstream/radiohead/code/lab")]
        from layers import solve_mu
        from radiohead.adaptation import restore_adapters
        from radiohead.attention import RadioAttention

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if (checkpoint["model"], checkpoint["revision"]) != (args.model, args.revision):
                parser.error("Adaptation checkpoint must match the model and revision")
            restore_adapters(model, checkpoint)
        else:
            checkpoint = {"layer": args.raw_layer, "training_length": args.raw_init_length}
        if "layer" in checkpoint:
            layer = model.model.layers[checkpoint["layer"]]
            radio = RadioAttention(layer.self_attn, solve_mu, training_length=checkpoint["training_length"])
            if "state_dict" in checkpoint:
                radio.load_state_dict(checkpoint["state_dict"])
            layer.self_attn = radio.eval()
    if args.language:
        corpus = tokenizer.encode(args.language.read_text(encoding="utf-8"), add_special_tokens=False)
        with torch.inference_mode():
            for index in range(min(args.examples, len(corpus) // args.length)):
                tokens = torch.tensor([corpus[index * args.length:(index + 1) * args.length]], device=args.device)
                hidden = model.model(tokens[:, :-1], use_cache=False).last_hidden_state
                count = min(128, hidden.shape[1])
                logits = model.lm_head(hidden[:, -count:]).float()
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, -count:].flatten())
                print(json.dumps({"length": args.length, "window": index, "scored_tokens": count,
                                  "nll": loss.item()}), flush=True)
        return
    with torch.inference_mode():
        for distance in args.distances:
            for absent in (False, True):
                for index in range(args.examples):
                    seed = args.seed + index
                    ids, expected = make_prompt(tokenizer, args.length, distance, seed, absent, background)
                    inputs = torch.tensor([ids], device=args.device)
                    output = model.generate(
                        inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
                        max_new_tokens=32, use_cache=True, pad_token_id=tokenizer.eos_token_id,
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
                        "raw_layer": args.raw_layer,
                        "length": len(ids), "distance": distance, "absent": absent,
                        "background": str(args.background) if args.background else None,
                        "expected": expected, "response": response, "prediction": prediction,
                        "correct": prediction == expected,
                    }), flush=True)


if __name__ == "__main__":
    main()
