# Slowrun Limited Track Baseline

Challenge: fixed 100M tokens FineWeb, 8xH100, 1hr time limit, minimize val loss.
Training runs for 12 epochs (1.2B total tokens seen).
Steps = 1.2B / 524,288 batch = 2,289 optimizer steps.

## Step 1: Prepare data

```bash
python prepapre_data.py
# produces fineweb_data/fineweb_train.pt (100M tokens) and fineweb_val.pt (10M tokens)
# uses GPT-2 tokenizer (vocab 50,257) — same script as slowrun/prepare_data.py
```

## Step 2: Train

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --run=slowrun-baseline \
    --model-tag=slowrun-baseline \
    \
    --depth=20 \
    --aspect-ratio=64 \
    --head-dim=128 \
    --max-seq-len=2048 \
    --window-pattern=SSSL \
    \
    --data=fineweb \
    --fineweb-data-dir=fineweb_data \
    \
    --num-iterations=2289 \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    \
    --matrix-lr=0.01 \
    --scalar-lr=0.025 \
    --embedding-lr=0.0375 \
    --unembedding-lr=0.0005 \
    --weight-decay=1.3 \
    \
    --warmup-steps=0 \
    --warmdown-ratio=0.2 \
    --final-lr-frac=0.0 \
    \
    --eval-every=250 \
    --eval-tokens=10000000 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --max-train-minutes=60
```

LRs above are slowrun's effective values after its internal `lr_multiplier=0.25` is baked in
(slowrun's raw values: matrix=0.04, scalar=0.1, embed=0.15, unembed=0.002).

## Weight decay caveat

nanochat auto-scales `--weight-decay` by `sqrt(B/B_ref) * (D_ref / target_tokens)`.
Since `B=B_ref=524288` the sqrt term is 1, but the `(D_ref/target_tokens)` term shrinks the
WD significantly for large depths (because `target_tokens` is computed from the compute-optimal
data:param ratio for the model, which is much larger than 1.2B for depth=20).

Check the log at startup for: `Scaling weight decay from 1.3 to X`
The effective value X must be ~1.3 to match slowrun's heavy regularization regime.
If it's much lower, increase `--weight-decay` proportionally to compensate,
or skip the auto-scaling entirely by hardcoding `weight_decay_scaled = args.weight_decay`
at line 325 of `scripts/base_train.py`.



---

## NCA pre-pre-training

Two-phase pipeline: NCA pre-pre-training → language pre-training with transferred weights.

### Step 1: NCA pre-pre-training (10M tokens)

Single GPU:
```bash
conda activate ml
python -m scripts.base_pre_pre_train \
    --run=nca-ppt-d20 \
    --model-tag=d20 \
    --depth=20 \
    --aspect-ratio=64 \
    --head-dim=128 \
    --window-pattern=SSSL \
    --total-tokens=10000000 \
    --eval-tokens=1000000 \
    --device-batch-size=16 \
    --total-batch-size=16384 \
    --warmup-steps=61 \
    --warmdown-ratio=0.1 \
    --eval-every=100 \
    --save-every=610 \
    --num-rules=5000
```

Multi-GPU (8xH100):
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_pre_pre_train \
    --run=nca-ppt-d20 \
    --model-tag=d20 \
    --depth=20 \
    --aspect-ratio=64 \
    --head-dim=128 \
    --window-pattern=SSSL \
    --total-tokens=10000000 \
    --eval-tokens=1000000 \
    --device-batch-size=16 \
    --total-batch-size=16384 \
    --warmup-steps=61 \
    --warmdown-ratio=0.1 \
    --eval-every=100 \
    --save-every=610 \
    --num-rules=5000
```

Training budget: ~610 steps (10M / 16384). Checkpoints saved to `~/.cache/nanochat/nca_checkpoints/d20/`.

### Step 2: Evaluate NCA model

```bash
python -m scripts.base_pre_pre_eval \
    --model-tag=d20 \
    --eval=loss,accuracy \
    --eval-tokens=1000000 \
    --num-accuracy-seqs=100 \
    --context-frames=5
```

### Step 3: Language pre-training with transferred NCA weights

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --run=lang-from-nca-d20 \
    --model-tag=lang-from-nca-d20 \
    \
    --depth=20 \
    --aspect-ratio=64 \
    --head-dim=128 \
    --max-seq-len=2048 \
    --window-pattern=SSSL \
    \
    --data=fineweb \
    --fineweb-data-dir=fineweb_data \
    \
    --pre-pre-train-checkpoint=~/.cache/nanochat/nca_checkpoints/d20 \
    \
    --num-iterations=2289 \
    --device-batch-size=4 \
    --total-batch-size=524288 \
    \
    --eval-every=250 \
    --eval-tokens=10000000 \
    --save-every=-1 \
    --max-train-minutes=60
```

### Verifying transfer worked

At startup, `base_train.py` prints a transfer summary like:
```
NCA weight transfer: 142 weights transferred, 3 skipped (vocab-dependent)
Skipped: transformer.wte.weight, lm_head.weight, value_embeds.*.weight
```

The key metric is whether the NCA-initialized model reaches lower perplexity than a scratch
baseline at the same language token budget. Plot both val loss curves side-by-side.

### Quick smoke test (CPU, depth=4)

```bash
# Step 1: short NCA pre-pre-training (~25 steps)
python -m scripts.base_pre_pre_train \
    --depth=4 --device-batch-size=4 \
    --total-batch-size=4096 --total-tokens=100000 \
    --eval-tokens=10000 --eval-every=10 --num-rules=100

# Step 2: eval
python -m scripts.base_pre_pre_eval --model-tag=d4 --eval=loss,accuracy --num-rules=50 --num-accuracy-seqs=10

# Step 3: language pre-train with transfer
python -m scripts.base_train \
    --depth=4 --max-seq-len=512 --device-batch-size=1 \
    --total-batch-size=512 --num-iterations=5 \
    --pre-pre-train-checkpoint=~/.cache/nanochat/nca_checkpoints/d4
```