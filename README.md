# Radio Head Uptraining

Can oscillator attention be transplanted into a pretrained 1B-class language model while preserving language quality and retrieving distant evidence?

The experimental target is 90% exact-match retrieval on held-out 16K prompts with multiple plausible keys. Report accuracy by needle-to-query distance and false answers on absent-key prompts.

## Reproduce the upstream experiment

```sh
git submodule update --init
uv venv --python 3.12
uv pip install -r upstream/radiohead/requirements.txt
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
