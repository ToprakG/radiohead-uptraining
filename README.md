# Radio Head Uptraining

We replaced all 28 attention layers in **Qwen2.5-1.5B-Instruct** with the Radio Head operator and uptrained the converted model on ordinary text. **Uptraining partially recovered language modeling. It recovered zero keys on our small 16K retrieval screen.**

## Results

| Model | 16K present keys recovered ↑ | 16K absent keys answered correctly ↑ | Held-out language loss ↓ |
|---|---:|---:|---:|
| Stock Qwen | 4/4 | 3/4 | 2.99 |
| Raw 28-layer Radio Head | 0/4 | 0/4 | 11.45 |
| Uptrained 28-layer Radio Head | 0/4 | 0/4 | 7.31 |

The full model trained for 96 updates on 122,880 ordinary-text input tokens at 512, 1,024, and 2,048 tokens per example. The objective combined true next-token cross-entropy and KL to frozen Qwen's next-token distribution, scored at 64 sampled positions per window. The original model weights were frozen. We trained the Radio Head receiver parameters and rank-8 adapters on the Q, V, and output projections and MLP down-projections.

The same eight confusable-key prompts were used for every model, with two present and two absent cases at each of two evidence distances, 4,096 and 15,360 tokens. Language loss is mean cross-entropy in nats/token on the last 128 prediction positions of six held-out 4K windows. This is a screening result for one checkpoint and training recipe. Eight synthetic prompts leave substantial uncertainty about retrieval accuracy. [Predictions, loss, and training logs](results/full-conversion/)

In a separate 48-key development experiment, replacing only layer 14 reduced recovery from **44/48 to 7/48**. Mixed-length ordinary-text layer-output distillation recovered **20/48**, and 4K language loss stayed close to stock. These results motivate measuring long retrieval directly even when ordinary-text loss looks healthy. [One-layer results](results/one-layer/)

## Reproduce

Clone with submodules. The CUDA pilot ran on a single RTX 4090 with PyTorch 2.7.1 and Transformers 4.51.3. The upstream implementation is pinned in the submodule.

```sh
git clone --recurse-submodules https://github.com/ToprakG/radiohead-uptraining.git
cd radiohead-uptraining
python -m pip install -r requirements.txt
python scripts/check_port.py
python scripts/prepare_corpus.py runs/corpus
python scripts/uptrain_full.py --smoke --train runs/corpus/train --output runs/smoke.pt
python scripts/uptrain_full.py --train runs/corpus/train --output runs/full.pt --steps 96
```

The [release checkpoint](https://github.com/ToprakG/radiohead-uptraining/releases/tag/v0.1.0) is a trainable-weight delta. The evaluation script loads the pinned public Qwen base and applies the checkpoint.

```bash
revision=989aa7980e4cf806f80c7fef2b1adb7bc71aa306
for model in stock raw trained; do
  opts=()
  if [[ "$model" == raw ]]; then opts=(--raw-full); fi
  if [[ "$model" == trained ]]; then opts=(--checkpoint runs/full.pt); fi
  python scripts/eval_retrieval.py --device cuda --revision "$revision" --length 16384 \
    --distances 4096 15360 --background runs/corpus/background/whitman-leaves.txt \
    --examples 2 --seed 971023 --distance-seed-stride 1000 "${opts[@]}" \
    > "runs/$model-16k.jsonl"
  python scripts/eval_retrieval.py --device cuda --revision "$revision" --length 4096 \
    --language runs/corpus/validation/austen-persuasion.txt --examples 6 "${opts[@]}" \
    > "runs/$model-nll.jsonl"
done
python scripts/summarize_retrieval.py runs/{stock,raw,trained}-16k.jsonl
```

The port passed the upstream attention checks and the full-conversion FP32 cached-versus-uncached decoding smoke check. The BF16 smoke check showed logit drift between those paths with the same top token. The 16K runs report cached BF16 generation.

## References

- [Everything in its right place: attention is a radio head](https://arxiv.org/abs/2611.13146) by Arola-Fernández, Khadraoui, and Arenas. [Upstream code](https://github.com/mystic-blue/radiohead) · [Upstream MIT notice](licenses/upstream-radiohead-MIT.txt)
- [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [Pinned Gutenberg corpus manifest](results/full-conversion/corpus.json)
