"""
NCA pre-pre-training script.

Trains the nanochat GPT model on synthetic Neural Cellular Automata (NCA) data
before standard language pre-training, following the approach from:

    "Training Language Models via Neural Cellular Automata"
    https://arxiv.org/abs/2603.10055

The model is trained with next-token prediction on tokenised NCA grid trajectories.
After pre-pre-training, the non-embedding weights (all transformer blocks) are
transferred to a standard language pre-training run via:

    python -m scripts.base_train --pre-pre-train-checkpoint <path>

Usage examples:

    # Quick CPU test (small model, tiny token budget)
    python -m scripts.base_pre_pre_train \\
        --depth=4 --max-seq-len=1024 --device-batch-size=4 \\
        --total-batch-size=4096 --total-tokens=100000 \\
        --eval-tokens=10000 --eval-every=10 --num-rules=100

    # Full pre-pre-training run (GPU, paper settings)
    torchrun --nproc_per_node=8 -m scripts.base_pre_pre_train \\
        --run=nca-ppt \\
        --total-tokens=10000000 --eval-tokens=1000000
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import datetime
import argparse
from dataclasses import asdict

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig
from nanochat.nca import NCATokenizer, NCADataset, nca_data_loader
from nanochat.common import (
    compute_init, compute_cleanup, print0, DummyWandb, print_banner,
    get_base_dir, autodetect_device_type, get_peak_flops,
    COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized,
)
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="NCA pre-pre-training")
# Logging
parser.add_argument("--run", type=str, default="dummy",
                    help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="",
                    help="cuda|cpu|mps (empty = autodetect)")
# Model architecture
parser.add_argument("--depth",          type=int,   default=20,    help="Transformer depth")
parser.add_argument("--aspect-ratio",   type=int,   default=64,    help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim",       type=int,   default=128,   help="target attention head dimension")
parser.add_argument("--max-seq-len",    type=int,   default=1024,  help="context length (paper uses 1024)")
parser.add_argument("--window-pattern", type=str,   default="SSSL",
                    help="sliding window attention pattern (L=full, S=quarter context)")
# Training horizon
parser.add_argument("--total-tokens",     type=int, default=10_000_000,
                    help="total NCA training tokens (default: 10M)")
parser.add_argument("--total-batch-size", type=int, default=16384,
                    help="total batch size in tokens per step (paper: 16 seqs x 1024 = 16384)")
parser.add_argument("--tokens-per-epoch", type=int, default=-1,
                    help="tokens per epoch for fixed-dataset multi-epoch training (-1 = disabled)")
parser.add_argument("--num-epochs",       type=int, default=-1,
                    help="number of epochs over the fixed dataset (-1 = disabled)")
# Optimisation
parser.add_argument("--device-batch-size", type=int,   default=16,   help="per-device batch size in sequences")
parser.add_argument("--matrix-lr",         type=float, default=0.02, help="Muon LR for matrix params")
parser.add_argument("--embedding-lr",      type=float, default=0.2,  help="AdamW LR for embeddings")
parser.add_argument("--unembedding-lr",    type=float, default=0.004,help="AdamW LR for lm_head")
parser.add_argument("--scalar-lr",         type=float, default=0.5,  help="AdamW LR for scalars")
parser.add_argument("--weight-decay",      type=float, default=0.0,  help="weight decay (paper: 0)")
parser.add_argument("--warmup-steps",      type=int,   default=61,
                    help="LR warmup steps (~10%% of total; 61 for the 10M token default)")
parser.add_argument("--warmdown-ratio",    type=float, default=0.65,
                    help="fraction of iterations for LR warmdown")
parser.add_argument("--final-lr-frac",     type=float, default=0.05,
                    help="final LR as fraction of peak LR")
parser.add_argument("--resume-from-step",  type=int,   default=-1,
                    help="resume from this step (-1 = disabled)")
# NCA data generation
parser.add_argument("--grid-size",        type=int,   default=12,  help="NCA grid side length (paper: 12)")
parser.add_argument("--patch-size",       type=int,   default=2,   help="patch size for tokenisation (paper: 2)")
parser.add_argument("--num-colors",       type=int,   default=10,  help="cell state alphabet size (paper: 10)")
parser.add_argument("--nca-temperature",  type=float, default=1e-3,
                    help="NCA softmax temperature (paper: 1e-3, near-deterministic)")
parser.add_argument("--burn-in",          type=int,   default=10,
                    help="NCA warm-up steps to discard before recording (paper: 10)")
parser.add_argument("--dt",               type=int,   default=1,
                    help="record every dT-th NCA step (paper: 1)")
parser.add_argument("--gzip-filter",      action="store_true", default=True,
                    help="filter rules by gzip complexity (default: on)")
parser.add_argument("--no-gzip-filter",   dest="gzip_filter", action="store_false",
                    help="disable gzip complexity filtering")
parser.add_argument("--gzip-low",         type=float, default=0.5,
                    help="lower gzip complexity bound (paper: 0.5 for web/math)")
parser.add_argument("--gzip-high",        type=float, default=1.0,
                    help="upper gzip complexity bound")
parser.add_argument("--num-rules",        type=int,   default=5000,
                    help="NCA rule pool size (more = more diversity)")
parser.add_argument("--regenerate-every", type=int,   default=-1,
                    help="regenerate training data with new rules every N steps (-1 = never)")
# Evaluation
parser.add_argument("--eval-tokens",       type=int, default=1_000_000,
                    help="NCA evaluation token budget (default: 1M)")
parser.add_argument("--eval-every",        type=int, default=100,
                    help="evaluate validation loss every N steps (-1 = disable)")
parser.add_argument("--save-every",        type=int, default=-1,
                    help="save checkpoint every N steps (-1 = only at end)")
parser.add_argument("--max-train-minutes", type=float, default=-1,
                    help="stop after N minutes of training time (-1 = disabled)")
parser.add_argument("--max-wall-minutes",  type=float, default=-1,
                    help="stop after N total wall-clock minutes (-1 = disabled)")
# Output
parser.add_argument("--model-tag",      type=str, default=None,
                    help="override model tag for checkpoint directory name")
parser.add_argument("--checkpoint-dir", type=str, default=None,
                    help="override checkpoint directory (default: <base_dir>/nca_checkpoints/<model_tag>)")
args = parser.parse_args()
user_config = vars(args).copy()

# -----------------------------------------------------------------------------
# Distributed / device init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
synchronize    = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops  = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat", name=args.run, config=user_config)

from nanochat.flash_attention import USE_FA3
if USE_FA3:
    print0("Using Flash Attention 3")
elif args.window_pattern != "L":
    print0("!" * 80)
    print0("WARNING: SDPA has no sliding window support. Use --window-pattern L on non-Hopper GPUs.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# NCA tokenizer and vocab size
nca_tokenizer = NCATokenizer(patch_size=args.patch_size, num_colors=args.num_colors)
vocab_size = nca_tokenizer.total_vocab_size  # 10002 for patch=2, colors=10
print0(f"NCA vocab size: {vocab_size} "
       f"({args.num_colors}^{args.patch_size**2} patch tokens + 2 delimiters)")

# -----------------------------------------------------------------------------
# Model initialisation
def build_model_meta(depth):
    base_dim  = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

model = build_model_meta(args.depth)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device)
model.init_weights()

base_dir       = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
checkpoint_dir = args.checkpoint_dir or os.path.join(base_dir, "nca_checkpoints", output_dirname)
print0(f"Checkpoint directory: {checkpoint_dir}")

resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data

# -----------------------------------------------------------------------------
# Compile
orig_model = model
model = torch.compile(model, dynamic=False)

# -----------------------------------------------------------------------------
# Optimizer: Muon+AdamW with weight_decay=0 (paper recommendation)
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    scalar_lr=args.scalar_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=0.0,
)
if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None

# -----------------------------------------------------------------------------
# Training horizon and gradient accumulation
tokens_per_fwdbwd      = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
total_batch_size        = args.total_batch_size

assert total_batch_size % world_tokens_per_fwdbwd == 0, (
    f"total_batch_size ({total_batch_size}) must be divisible by "
    f"device_batch_size x seq_len x world_size = {world_tokens_per_fwdbwd}. "
    f"Adjust --device-batch-size or --total-batch-size."
)
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
if args.tokens_per_epoch > 0 and args.num_epochs > 0:
    # Epoch mode: fixed dataset, looped num_epochs times
    total_tokens   = args.tokens_per_epoch * args.num_epochs
    num_iterations = total_tokens // total_batch_size
    total_tokens   = total_batch_size * num_iterations  # snap to batch boundary
    num_train_seqs = math.ceil(args.tokens_per_epoch / args.max_seq_len)
else:
    # Default mode: total_tokens drives everything (~1 epoch)
    num_iterations = args.total_tokens // total_batch_size
    total_tokens   = total_batch_size * num_iterations
    num_train_seqs = math.ceil(total_tokens / args.max_seq_len)

print0(f"NCA pre-pre-training: {total_tokens:,} tokens over {num_iterations:,} steps")
if args.tokens_per_epoch > 0 and args.num_epochs > 0:
    print0(f"Epoch mode: {num_train_seqs:,} sequences/epoch x {args.num_epochs} epochs")
print0(f"Batch: {args.device_batch_size} seqs x {args.max_seq_len} tokens x "
       f"{ddp_world_size} ranks x {grad_accum_steps} accum = {total_batch_size:,} tokens/step")

# -----------------------------------------------------------------------------
# Generate NCA training and validation data
num_val_seqs   = math.ceil(args.eval_tokens / args.max_seq_len)

nca_kwargs = dict(
    seq_len=args.max_seq_len, grid_size=args.grid_size,
    patch_size=args.patch_size, num_colors=args.num_colors,
    temperature=args.nca_temperature, burn_in=args.burn_in, dT=args.dt,
    gzip_filter=args.gzip_filter, gzip_low=args.gzip_low, gzip_high=args.gzip_high,
    num_rules=args.num_rules, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size,
)

print0(f"Generating training data ({num_train_seqs:,} sequences)...")
train_dataset = NCADataset(num_sequences=num_train_seqs, seed=42, **nca_kwargs)
train_dataset.generate(verbose=master_process)

print0(f"Generating validation data ({num_val_seqs:,} sequences)...")
val_dataset = NCADataset(num_sequences=num_val_seqs, seed=999, **nca_kwargs)
val_dataset.generate(verbose=master_process)

train_loader     = nca_data_loader(train_dataset, args.device_batch_size, device,
                                    ddp_rank=ddp_rank, ddp_world_size=ddp_world_size, shuffle=True)
build_val_loader = lambda: nca_data_loader(val_dataset, args.device_batch_size, device,
                                            ddp_rank=ddp_rank, ddp_world_size=ddp_world_size, shuffle=False)

x, y = next(train_loader)  # kick off first batch

# -----------------------------------------------------------------------------
# LR and Muon momentum schedulers (trapezoidal: warmup → constant → warmdown)
def get_lr_multiplier(it):
    warmup_iters   = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress + (1 - progress) * args.final_lr_frac

def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return 0.85 * (1 - frac) + 0.97 * frac
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    return 0.97

# -----------------------------------------------------------------------------
# Training loop
if not resuming:
    step = 0
    val_loss = None
    min_val_loss = float("inf")
    smooth_train_loss = 0.0
    total_training_time = 0.0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_loss = meta_data.get("val_loss")
    min_val_loss = loop_state["min_val_loss"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

t_loop_start = time.time()
while True:
    train_time_exceeded = args.max_train_minutes > 0 and total_training_time / 60 >= args.max_train_minutes
    wall_time_exceeded  = args.max_wall_minutes  > 0 and (time.time() - t_loop_start) / 60 >= args.max_wall_minutes
    last_step = step == num_iterations or train_time_exceeded or wall_time_exceeded
    flops_so_far = orig_model.estimate_flops() * total_batch_size * step

    # ------------------------------------------------------------------
    # Validation: NCA cross-entropy loss on held-out rules
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        orig_model.eval()
        val_loader = build_val_loader()
        eval_steps = max(args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size), 1)
        total_val_loss = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for _ in range(eval_steps):
                vx, vy = next(val_loader)
                total_val_loss += orig_model(vx, vy)
        total_val_loss /= eval_steps
        if ddp:
            dist.all_reduce(total_val_loss, op=dist.ReduceOp.AVG)
        val_loss = total_val_loss.item()
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        val_bpb = val_loss / (math.log(2) * args.patch_size ** 2)
        print0(f"Step {step:05d} | NCA val loss: {val_loss:.6f} | bpb: {val_bpb:.4f}")
        wandb_run.log({"step": step, "val/loss": val_loss, "val/bpb": val_bpb,
                       "total_training_time": total_training_time})
        orig_model.train()

    # ------------------------------------------------------------------
    # Checkpoint
    if last_step or (step > 0 and step != args.resume_from_step
                     and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir, step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_loss": val_loss,
                "model_config": model_config_kwargs,
                "nca_config": {
                    "grid_size": args.grid_size, "patch_size": args.patch_size,
                    "num_colors": args.num_colors, "temperature": args.nca_temperature,
                    "gzip_low": args.gzip_low, "gzip_high": args.gzip_high,
                },
                "training_phase": "nca_pre_pre_train",
                "user_config": user_config,
                "loop_state": {
                    "min_val_loss": min_val_loss,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # ------------------------------------------------------------------
    # Optionally regenerate training data for diversity
    if args.regenerate_every > 0 and step > 0 and step % args.regenerate_every == 0:
        new_seed = 42 + step
        print0(f"Step {step}: regenerating training data (seed={new_seed})...")
        train_dataset.regenerate(new_seed, verbose=master_process)
        train_loader = nca_data_loader(train_dataset, args.device_batch_size, device,
                                        ddp_rank=ddp_rank, ddp_world_size=ddp_world_size)
        x, y = next(train_loader)

    # ------------------------------------------------------------------
    # Training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y = next(train_loader)  # prefetch next batch while GPU is busy

    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group.get("kind") == "muon":
            group["momentum"] = muon_momentum

    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # ------------------------------------------------------------------
    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    if step > 0:
        total_training_time += dt
    if step > 0:
        eta_secs = (total_training_time / step) * (num_iterations - step)
        eta_str = f" | eta: {eta_secs/60:.1f}m"
    else:
        eta_str = ""
    wall_clock = datetime.datetime.now().strftime("%H:%M:%S")
    wall_time = time.time() - t_loop_start
    train_bpb = debiased / (math.log(2) * args.patch_size ** 2)
    print0(
        f"[{wall_clock}] step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | "
        f"loss: {debiased:.6f} | bpb: {train_bpb:.4f} | lrm: {lrm:.3f} | dt: {dt*1000:.1f}ms | "
        f"tok/sec: {tok_per_sec:,} | train: {total_training_time/60:.1f}m | "
        f"wall: {wall_time/60:.1f}m{eta_str}"
    )
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_time": total_training_time,
            "train/loss": debiased,
            "train/bpb": train_bpb,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
        })

    first_step = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1
    if first_step:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# -----------------------------------------------------------------------------
# Final stats
total_wall_time = time.time() - t_loop_start
print0(f"Peak memory: {get_max_memory() / 1024**2:.1f} MiB")
print0(f"Training time: {total_training_time/60:.1f}m | Wall time: {total_wall_time/60:.1f}m")
if val_loss is not None:
    print0(f"Min validation loss: {min_val_loss:.6f}")
print0(f"Checkpoint: {checkpoint_dir}")

wandb_run.finish()
compute_cleanup()
