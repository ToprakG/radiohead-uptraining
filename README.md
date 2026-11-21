# Radio Head Uptraining

Can oscillator attention be transplanted into a pretrained 1B-class language model while preserving language quality and retrieving distant evidence?

The experimental target is 90% exact-match retrieval on held-out 16K prompts with multiple plausible keys. Report accuracy by needle-to-query distance and false answers on absent-key prompts.

## Reproduce the upstream experiment

```sh
git submodule update --init
uv venv --python 3.12
uv pip install -r requirements.txt
scripts/reproduce.sh test
scripts/reproduce.sh recall
```

The recall command runs a one-seed comparison using the authors' quickstart settings. Full experiment plans and published per-run results live in the submodule.

## Transplant experiment

1. Port upstream KM and check outputs, gradients, masks, and attention power against the pinned implementation.
2. Measure stock Qwen2.5-1.5B-Instruct. Distill one replacement layer on the frozen teacher's inputs and then adapt the hybrid with LoRA.
3. Compare against stock Qwen and softmax adapted on the same examples and token budget. Evaluate language loss, retrieval by distance, absent-key errors, latency, and peak memory.
4. Evaluate short-only adaptation first. If needed, train a separate checkpoint with a length curriculum and report it separately.

[Original preprint](https://arxiv.org/abs/2611.13146) · [Authors' implementation](https://github.com/mystic-blue/radiohead)

## Check the port and establish the stock baseline

```sh
.venv/bin/python scripts/check_port.py
mkdir -p runs
.venv/bin/python scripts/eval_retrieval.py \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  > runs/qwen-stock-16k.jsonl
```

The initial evaluation uses 15 competing records and deterministic synthetic filler.
Keep its prompt seeds out of adaptation and add varied held-out background text
before drawing broader conclusions.

## Distill one Radio Head layer

Supply two separate UTF-8 text corpora. The pilot fits layer 14 to the frozen
teacher's attention output and records validation MSE before and after training.

```sh
.venv/bin/python scripts/distill_layer.py --text train.txt --validation validation.txt \
  --output runs/radio-layer14.pt
```

Pass `--radio-checkpoint runs/radio-layer14.pt` to the retrieval evaluator to
test the transplant. The hybrid supports the pinned Qwen2 dynamic KV cache.

## Uptrain and evaluate matched models

The pilot updates low-rank adapters in the last four MLP down-projections for
both models, with additional receiver updates in the hybrid. The distilled
Q/V/O weights are frozen during this phase. Both runs draw the same ordinary
text and synthetic retrieval examples.

```sh
.venv/bin/python scripts/uptrain.py --text train.txt --output runs/softmax-adapted.pt
.venv/bin/python scripts/uptrain.py --text train.txt --distilled runs/radio-layer14.pt \
  --output runs/radio-adapted.pt
.venv/bin/python scripts/eval_retrieval.py \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --checkpoint runs/radio-adapted.pt > runs/radio-16k.jsonl
```

The current evaluator is a synthetic pilot. One prompt at a given distance
checks execution. A claim about 90% needs the full held-out evaluation and
uncertainty estimates.
