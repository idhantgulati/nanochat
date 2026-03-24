"""
NCA pre-pre-training evaluation script.

Evaluates a model trained with base_pre_pre_train.py on NCA data.

Supports two evaluation modes (comma-separated via --eval):
  --eval loss      : Cross-entropy loss on held-out NCA sequences (unseen rules)
  --eval accuracy  : Grid-prediction accuracy (autoregressive next-frame generation)

Usage examples:

    # Evaluate the latest NCA checkpoint for model d20
    python -m scripts.base_pre_pre_eval --model-tag d20

    # Evaluate only loss, using 1M tokens
    python -m scripts.base_pre_pre_eval --model-tag d20 --eval loss --eval-tokens 1000000

    # Evaluate on a specific step
    python -m scripts.base_pre_pre_eval --model-tag d20 --step 610
"""

import os
import json
import argparse
import math

import torch
import torch.nn.functional as F

from nanochat.nca import NCATokenizer, NCADataset, nca_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type
from nanochat.checkpoint_manager import load_checkpoint, find_last_step
from nanochat.gpt import GPT, GPTConfig

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="NCA pre-pre-training evaluation")
parser.add_argument("--eval", type=str, default="loss,accuracy",
                    help="comma-separated eval modes: loss, accuracy (default: both)")
parser.add_argument("--model-tag",    type=str, default=None,
                    help="model tag identifying the NCA checkpoint directory")
parser.add_argument("--step",         type=int, default=None,
                    help="checkpoint step to evaluate (default: last)")
parser.add_argument("--checkpoint-dir", type=str, default=None,
                    help="override NCA checkpoint directory")
parser.add_argument("--device-type",  type=str, default="",
                    help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--device-batch-size", type=int, default=16,
                    help="per-device batch size for loss evaluation")
parser.add_argument("--eval-tokens",  type=int, default=1_000_000,
                    help="token budget for loss evaluation (default: 1M)")
# NCA data generation (must match training; defaults are paper values)
parser.add_argument("--grid-size",    type=int,   default=12)
parser.add_argument("--patch-size",   type=int,   default=2)
parser.add_argument("--num-colors",   type=int,   default=10)
parser.add_argument("--nca-temperature", type=float, default=1e-3)
parser.add_argument("--burn-in",      type=int,   default=10)
parser.add_argument("--gzip-filter",  action="store_true", default=True)
parser.add_argument("--no-gzip-filter", dest="gzip_filter", action="store_false")
parser.add_argument("--gzip-low",     type=float, default=0.5)
parser.add_argument("--gzip-high",    type=float, default=1.0)
parser.add_argument("--num-rules",    type=int,   default=1000,
                    help="number of rules for eval data (smaller is fine for evaluation)")
# Accuracy-eval specific
parser.add_argument("--context-frames", type=int, default=5,
                    help="number of context frames to feed before generating the next frame")
parser.add_argument("--num-accuracy-seqs", type=int, default=100,
                    help="number of sequences for accuracy evaluation")
args = parser.parse_args()

# Parse eval modes
eval_modes = {m.strip() for m in args.eval.split(",")}
valid_modes = {"loss", "accuracy"}
invalid = eval_modes - valid_modes
if invalid:
    parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

# -----------------------------------------------------------------------------
# Setup
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# Resolve checkpoint directory
base_dir = get_base_dir()
if args.checkpoint_dir:
    checkpoint_dir = args.checkpoint_dir
else:
    model_tag = args.model_tag or "d20"
    checkpoint_dir = os.path.join(base_dir, "nca_checkpoints", model_tag)

step = args.step if args.step is not None else find_last_step(checkpoint_dir)
print0(f"Loading NCA checkpoint: {checkpoint_dir} (step {step})")

model_data, _, meta = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

# BF16 → float on CPU/MPS
if device.type in {"cpu", "mps"}:
    model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}

model_config_kwargs = meta["model_config"]
if "window_pattern" not in model_config_kwargs:
    model_config_kwargs["window_pattern"] = "L"
model_config = GPTConfig(**model_config_kwargs)

# Load NCA config from checkpoint metadata (or fall back to CLI defaults)
nca_cfg = meta.get("nca_config", {})
grid_size   = nca_cfg.get("grid_size",   args.grid_size)
patch_size  = nca_cfg.get("patch_size",  args.patch_size)
num_colors  = nca_cfg.get("num_colors",  args.num_colors)
temperature = nca_cfg.get("temperature", args.nca_temperature)
gzip_low    = nca_cfg.get("gzip_low",    args.gzip_low)
gzip_high   = nca_cfg.get("gzip_high",   args.gzip_high)

with torch.device("meta"):
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()
model.load_state_dict(model_data, strict=True, assign=True)
model.eval()
del model_data

print0(f"Model: {sum(p.numel() for p in model.parameters()):,} params | "
       f"vocab_size={model_config.vocab_size} | seq_len={model_config.sequence_len}")

# NCA tokenizer (matches training configuration)
tokenizer = NCATokenizer(patch_size=patch_size, num_colors=num_colors)
tokens_per_frame = tokenizer.get_tokens_per_frame(grid_size)
print0(f"NCA config: grid={grid_size}x{grid_size}, patch={patch_size}x{patch_size}, "
       f"colors={num_colors}, tokens/frame={tokens_per_frame}")
print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

nca_kwargs = dict(
    seq_len=model_config.sequence_len, grid_size=grid_size,
    patch_size=patch_size, num_colors=num_colors, temperature=temperature,
    burn_in=args.burn_in, dT=1,
    gzip_filter=args.gzip_filter, gzip_low=gzip_low, gzip_high=gzip_high,
    num_rules=args.num_rules, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size,
)

# =============================================================================
# Loss evaluation
# =============================================================================
if "loss" in eval_modes:
    print0("\n" + "=" * 80)
    print0("NCA Loss Evaluation")
    print0("=" * 80)

    num_eval_seqs = math.ceil(args.eval_tokens / model_config.sequence_len)
    print0(f"Generating {num_eval_seqs} evaluation sequences (unseen rules, seed=777)...")

    eval_dataset = NCADataset(num_sequences=num_eval_seqs, seed=777, **nca_kwargs)
    eval_dataset.generate(verbose=(ddp_rank == 0))

    eval_loader = nca_data_loader(eval_dataset, args.device_batch_size, device,
                                   ddp_rank=ddp_rank, ddp_world_size=ddp_world_size, shuffle=False)
    eval_steps = max(num_eval_seqs // (args.device_batch_size * ddp_world_size), 1)

    total_loss = 0.0
    total_tokens_counted = 0
    with torch.no_grad():
        for i in range(eval_steps):
            x, y = next(eval_loader)
            loss = model(x, y)
            total_loss += loss.item()
            total_tokens_counted += x.numel()

    avg_loss = total_loss / eval_steps
    # Convert to bits per patch token (analogous to BPB but for NCA tokens)
    bits_per_token = avg_loss / math.log(2)
    print0(f"NCA validation loss: {avg_loss:.6f} | bits/token: {bits_per_token:.4f}")

# =============================================================================
# Grid prediction accuracy evaluation
# =============================================================================
if "accuracy" in eval_modes:
    print0("\n" + "=" * 80)
    print0("NCA Grid Prediction Accuracy")
    print0("=" * 80)
    print0(f"Context frames: {args.context_frames} | Predicting: next 1 frame")

    num_acc_seqs = args.num_accuracy_seqs
    print0(f"Generating {num_acc_seqs} test sequences (seed=555)...")

    acc_nca_kwargs = dict(**nca_kwargs)
    acc_nca_kwargs["num_rules"] = max(args.num_rules, num_acc_seqs)
    acc_dataset = NCADataset(num_sequences=num_acc_seqs, seed=555, **acc_nca_kwargs)
    acc_dataset.generate(verbose=(ddp_rank == 0))

    patches_per_frame = (grid_size // patch_size) ** 2
    context_len = args.context_frames * tokens_per_frame

    total_cells = 0
    correct_cells = 0
    exact_frames = 0

    with torch.no_grad():
        for seq_idx in range(num_acc_seqs):
            # Get the full sequence
            x_full, y_full = acc_dataset[seq_idx]  # (seq_len,)

            # Context: first `context_frames` frames
            context = x_full[:context_len].unsqueeze(0).to(device)  # (1, context_len)

            # Ground truth: tokens for the next frame (after the context)
            gt_frame_start = context_len  # index in x_full where next frame starts
            gt_tokens = y_full[gt_frame_start : gt_frame_start + tokens_per_frame]  # (38,)

            # Skip sequences that are too short
            if gt_frame_start + tokens_per_frame > len(y_full):
                continue

            # Autoregressively generate the next frame (tokens_per_frame tokens)
            generated = context.clone()
            pred_tokens = []
            for _ in range(tokens_per_frame):
                logits = model(generated)           # (1, T, vocab_size)
                next_token = logits[0, -1].argmax(-1)  # greedy decoding (temperature=0)
                pred_tokens.append(next_token.item())
                generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

            pred_tokens = torch.tensor(pred_tokens, dtype=torch.long)  # (tokens_per_frame,)

            # Compare predicted vs ground truth (patch tokens only, skip delimiters)
            # Frame structure: [start_token, patch_0, ..., patch_35, end_token]
            pred_patches = pred_tokens[1:-1]   # strip start/end delimiters
            gt_patches   = gt_tokens[1:-1].cpu()

            # Cell-wise accuracy: decode patches and compare
            pred_grid = tokenizer.decode_frame_tokens(pred_patches, grid_size)
            gt_grid   = tokenizer.decode_frame_tokens(gt_patches,   grid_size)

            cell_match = (pred_grid == gt_grid).sum().item()
            total_cells += pred_grid.numel()
            correct_cells += cell_match
            if cell_match == pred_grid.numel():
                exact_frames += 1

    cell_accuracy = correct_cells / total_cells if total_cells > 0 else 0.0
    frame_accuracy = exact_frames / num_acc_seqs if num_acc_seqs > 0 else 0.0
    print0(f"Cell accuracy:  {cell_accuracy:.4f} ({correct_cells}/{total_cells})")
    print0(f"Frame accuracy: {frame_accuracy:.4f} ({exact_frames}/{num_acc_seqs} exact matches)")

compute_cleanup()
