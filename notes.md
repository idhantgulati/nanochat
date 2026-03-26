# Slowrun Limited Track Baseline

Challenge: fixed 100M tokens FineWeb, 8xH100, 1hr time limit, minimize val loss.
Training runs for 12 epochs (1.2B total tokens seen).
Steps = 1.2B / 524,288 batch = 2,289 optimizer steps.

## Step 1: Prepare data

```bash
python prepare_data.py
# produces fineweb_data/fineweb_train.pt (100M tokens) and fineweb_val.pt (10M tokens)
# uses GPT-2 tokenizer (vocab 50,257) — same script as slowrun/prepare_data.py
```

## Step 2: Train

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --run=slowrun-baseline \
    --model-tag=slowrun-baseline \
    \
    --depth=30 \
    --aspect-ratio=59 \
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

Model: depth=30, aspect-ratio=59 → base_dim=1770, n_embd=ceil(1770/128)*128=1792, n_head=14.
Matches slowrun's n_layer=30, n_embd=1792, n_head=14 (~2.7B parameter model).

LRs are slowrun's effective values after its internal `lr_multiplier=0.25`
(slowrun raw: matrix=0.04, scalar=0.1, embed=0.15, unembed=0.002).
dmodel_lr_scale is bypassed in gpt.py so these values reach the optimizer unchanged.



---
---

# NCA pre-pre-training

Two-phase pipeline: NCA pre-pre-training → language pre-training with transferred weights.

### Step 1: NCA pre-pre-training (10M tokens)

<!-- Single GPU:
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
``` -->

Multi-GPU (8xH100):
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_pre_pre_train \
    --run=nca-ppt-d30 \
    --model-tag=d30 \
    --depth=30 \
    --aspect-ratio=59 \
    --head-dim=128 \
    --window-pattern=SSSL \
    --total-tokens=10000000 \
    --eval-tokens=1000000 \
    --device-batch-size=2 \
    --total-batch-size=16384 \
    --warmup-steps=61 \
    --warmdown-ratio=0.1 \
    --eval-every=100 \
    --save-every=610 \
    --num-rules=5000
```

NOTE: device-batch-size=2 (not 16) is required for 8 GPUs.
world_tokens_per_step = device_batch_size × seq_len × world_size = 2 × 1024 × 8 = 16384 = total_batch_size. grad_accum=1.
Using device-batch-size=16 with total-batch-size=16384 triggers an assertion error (16384 % 131072 ≠ 0).

Training budget: ~610 steps (10M / 16384). Checkpoints saved to `~/.cache/nanochat/nca_checkpoints/d30/`.

### Step 2: Evaluate NCA model

```bash
python -m scripts.base_pre_pre_eval \
    --model-tag=d30 \
    --eval=loss,accuracy \
    --eval-tokens=1000000 \
    --num-accuracy-seqs=100 \
    --context-frames=5
```

### Step 3: Language pre-training with transferred NCA weights

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --run=lang-from-nca-d30 \
    --model-tag=lang-from-nca-d30 \
    \
    --depth=30 \
    --aspect-ratio=59 \
    --head-dim=128 \
    --max-seq-len=2048 \
    --window-pattern=SSSL \
    \
    --pre-pre-train-checkpoint=~/.cache/nanochat/nca_checkpoints/d30 \
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