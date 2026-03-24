"""
NCA (Neural Cellular Automata) pre-pre-training utilities.

Ports from the JAX reference implementation at:
  nca-pre-pretraining/utils/nca.py
  nca-pre-pretraining/utils/tokenizers.py

Components:
  NCANetwork      - tiny CNN defining the transition rule for one NCA
  init_rule       - initialize NCA network weights from a seed
  step_nca        - single NCA step (one-hot → network → categorical sample)
  rollout         - simulate a full NCA trajectory
  gzip_complexity - compute gzip compression ratio as a complexity proxy
  filter_rule     - check if a rule passes the gzip complexity band filter
  generate_filtered_rules - find N rules that pass the filter
  NCATokenizer    - patch-based tokenization of NCA grid trajectories
  NCADataset      - in-memory PyTorch Dataset of NCA token sequences
  nca_data_loader - infinite DDP-aware data generator
"""

import io
import gzip
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# =============================================================================
# NCA Neural Network
# =============================================================================

class NCANetwork(nn.Module):
    """Tiny CNN defining the transition rule for one NCA.

    Architecture (ports from JAX reference utils/nca.py lines 104-114):
        Conv2d(d_state, 4, 3x3) -- perceive 3x3 neighbourhood (periodic boundaries)
        Conv2d(4, 16, 1x1)      -- pointwise expansion
        ReLU
        Conv2d(16, d_state, 1x1) -- project to per-cell state logits
    """

    def __init__(self, d_state: int = 10):
        super().__init__()
        self.d_state = d_state
        # No built-in padding; we apply circular padding manually before conv1
        self.conv1 = nn.Conv2d(d_state, 4,  kernel_size=3, padding=0, bias=True)
        self.conv2 = nn.Conv2d(4,       16, kernel_size=1, bias=True)
        self.conv3 = nn.Conv2d(16, d_state, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (d_state, H, W)
        x = x.unsqueeze(0)                          # (1, d_state, H, W)
        x = F.pad(x, (1, 1, 1, 1), mode='circular') # periodic/toroidal boundaries
        x = self.conv1(x)                            # (1, 4, H, W)
        x = self.conv2(x)                            # (1, 16, H, W)
        x = F.relu(x)
        x = self.conv3(x)                            # (1, d_state, H, W)
        return x.squeeze(0)                          # (d_state, H, W)


def init_rule(seed: int, d_state: int = 10) -> NCANetwork:
    """Create an NCANetwork with weights randomly initialised from *seed*.

    Each unique seed produces a unique dynamics rule. The rule is defined
    entirely by the network weights, independent of the initial grid state.
    """
    net = NCANetwork(d_state=d_state)
    rng = torch.Generator()
    rng.manual_seed(seed)
    with torch.no_grad():
        for param in net.parameters():
            nn.init.normal_(param, mean=0.0, std=1.0, generator=rng)
    return net


# =============================================================================
# NCA Simulator
# =============================================================================

def init_grid(rng: torch.Generator, grid_size: int, d_state: int) -> torch.Tensor:
    """Uniformly sample a random discrete grid. Returns (H, W) int64 tensor."""
    return torch.randint(0, d_state, (grid_size, grid_size), generator=rng)


@torch.no_grad()
def step_nca(
    state: torch.Tensor,   # (H, W) int64
    net: NCANetwork,
    rng: torch.Generator,
    identity_bias: float = 0.0,
    temperature: float = 1e-3,
) -> torch.Tensor:
    """Single NCA step: one-hot encode → apply network → categorical sample."""
    d_state = net.d_state
    H, W = state.shape

    # One-hot encode: (H, W, d_state) -> (d_state, H, W) channels-first
    state_oh = F.one_hot(state, num_classes=d_state).float().permute(2, 0, 1)

    # Apply transition network to get per-cell logits
    logits = net(state_oh)  # (d_state, H, W)

    # Optional identity bias: adds current state to logits (more stable dynamics)
    if identity_bias != 0.0:
        logits = logits + identity_bias * state_oh

    # Low temperature makes the dynamics nearly deterministic (paper uses τ=1e-3)
    logits = logits / temperature  # (d_state, H, W)

    # Categorical sample: reshape to (H*W, d_state), sample, reshape back
    probs = torch.softmax(logits.reshape(d_state, -1).T, dim=-1)  # (H*W, d_state)
    next_flat = torch.multinomial(probs, 1, replacement=True, generator=rng).squeeze(-1)
    return next_flat.reshape(H, W)


def rollout(
    rule_seed: int,
    init_seed: int,
    num_frames: int,
    grid_size: int = 12,
    d_state: int = 10,
    burn_in: int = 10,
    dT: int = 1,
    temperature: float = 1e-3,
) -> torch.Tensor:
    """Simulate an NCA trajectory and return the recorded frames.

    Args:
        rule_seed:  Seed for NCA network weights (defines the dynamics rule).
        init_seed:  Seed for initial grid state and stochastic dynamics.
        num_frames: Number of frames to record.
        grid_size:  Grid side length (H = W = grid_size).
        d_state:    Discrete cell state alphabet size.
        burn_in:    Warm-up steps to discard before recording.
        dT:         Record every dT-th step (1 = every step).
        temperature: Softmax temperature (low → near-deterministic).

    Returns:
        trajectory: (num_frames, H, W) int64 tensor.
    """
    net = init_rule(rule_seed, d_state)
    net.eval()

    rng = torch.Generator()
    rng.manual_seed(init_seed)

    state = init_grid(rng, grid_size, d_state)

    # Burn-in: run dynamics but discard frames
    for _ in range(burn_in):
        state = step_nca(state, net, rng, temperature=temperature)

    # Record num_frames frames at interval dT
    frames = []
    steps_needed = num_frames * dT
    for step in range(steps_needed):
        state = step_nca(state, net, rng, temperature=temperature)
        if (step + 1) % dT == 0:
            frames.append(state.clone())

    return torch.stack(frames, dim=0)  # (num_frames, H, W)


# =============================================================================
# Gzip Complexity Filtering
# =============================================================================

def gzip_complexity(byte_data: bytes) -> float:
    """Compute gzip compression ratio = compressed_bytes / raw_bytes.

    Interpretation:
        Near 0: highly compressible (trivially simple / repetitive dynamics) -> filtered out
        Near 1: incompressible (maximally chaotic / random dynamics) -> filtered out
        0.5-1.0 band: structured but non-trivial complexity -> retained for training
    """
    if len(byte_data) == 0:
        return 0.0
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9) as f:
        f.write(byte_data)
    return len(buf.getvalue()) / len(byte_data)


def filter_rule(
    rule_seed: int,
    tokenizer: "NCATokenizer",
    grid_size: int = 12,
    d_state: int = 10,
    temperature: float = 1e-3,
    filter_steps: int = 10,
    low: float = 0.5,
    high: float = 1.0,
) -> bool:
    """Return True if the rule's gzip complexity falls within (low, high).

    Generates a short probe trajectory using a canonical init_seed=0 so that
    the same rule always gives the same filter decision.
    """
    traj = rollout(rule_seed, init_seed=0, num_frames=filter_steps,
                   grid_size=grid_size, d_state=d_state,
                   burn_in=0, dT=1, temperature=temperature)
    tokens = tokenizer.encode_trajectory(traj)
    # Only use patch tokens (strip start/end delimiters) for the complexity measure
    patch_mask = tokens < tokenizer.grid_start_token
    patch_tokens = tokens[patch_mask]
    if len(patch_tokens) == 0:
        return False
    byte_data = patch_tokens.numpy().astype('uint16').tobytes()
    ratio = gzip_complexity(byte_data)
    return low < ratio < high


def generate_filtered_rules(
    num_rules: int,
    tokenizer: "NCATokenizer",
    grid_size: int = 12,
    d_state: int = 10,
    temperature: float = 1e-3,
    low: float = 0.5,
    high: float = 1.0,
    base_seed: int = 0,
    verbose: bool = True,
) -> list:
    """Generate a list of rule seeds whose dynamics pass the gzip complexity filter.

    Iterates seeds starting from base_seed until num_rules are accepted.
    """
    accepted = []
    candidate = base_seed
    tested = 0
    max_candidates = num_rules * 30  # safety limit

    while len(accepted) < num_rules and tested < max_candidates:
        if filter_rule(candidate, tokenizer, grid_size, d_state, temperature, low=low, high=high):
            accepted.append(candidate)
        candidate += 1
        tested += 1
        if verbose and tested % 1000 == 0:
            print(f"  Rule filtering: {len(accepted)}/{num_rules} accepted, {tested} tested...")

    if len(accepted) < num_rules:
        print(f"  Warning: only {len(accepted)}/{num_rules} rules found after {tested} candidates. "
              f"Consider adjusting --gzip-low/--gzip-high.")
    elif verbose:
        print(f"  Rule filtering done: {len(accepted)}/{num_rules} accepted from {tested} candidates.")

    return accepted


# =============================================================================
# NCA Tokenizer
# =============================================================================

class NCATokenizer:
    """Tokenises NCA grid trajectories using non-overlapping 2×2 patches.

    Ports from reference utils/tokenizers.py.

    Vocabulary layout:
        0 … (num_colors^patch_size²)-1  : patch tokens (bijective mixed-radix)
        grid_start_token (10000)         : start-of-frame delimiter
        grid_end_token   (10001)         : end-of-frame delimiter

    For the paper's configuration (patch_size=2, num_colors=10):
        10000 patch tokens + 2 delimiters = 10002 total tokens.
    """

    def __init__(self, patch_size: int = 2, num_colors: int = 10):
        self.patch_size = patch_size
        self.num_colors = num_colors
        self.patch_vocab = num_colors ** (patch_size ** 2)   # 10^4 = 10000
        self.grid_start_token = self.patch_vocab              # 10000
        self.grid_end_token   = self.patch_vocab + 1          # 10001
        self.total_vocab_size = self.patch_vocab + 2          # 10002

        # Precompute per-position base multipliers: [10^0, 10^1, 10^2, 10^3]
        self._powers = torch.tensor(
            [num_colors ** i for i in range(patch_size ** 2)],
            dtype=torch.long,
        )

    def get_tokens_per_frame(self, grid_size: int) -> int:
        """Tokens for one grid frame: (H/p)*(W/p) patches + 2 delimiters."""
        patches_per_frame = (grid_size // self.patch_size) ** 2
        return patches_per_frame + 2

    def encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode a single (H, W) int grid into patch tokens of shape (num_patches,)."""
        H, W = grid.shape
        ps = self.patch_size
        assert H % ps == 0 and W % ps == 0, f"Grid {H}×{W} not divisible by patch_size {ps}"

        # Reshape into (num_patches, ps²): row-major patch extraction
        patches = grid.reshape(H // ps, ps, W // ps, ps)
        patches = patches.permute(0, 2, 1, 3).reshape(-1, ps * ps)  # (num_patches, ps²)

        # Mixed-radix encode: token = sum(pixel[i] * num_colors^i)
        tokens = (patches * self._powers.unsqueeze(0)).sum(dim=-1)   # (num_patches,)
        return tokens

    def encode_frame(self, grid: torch.Tensor) -> torch.Tensor:
        """Encode one grid frame with delimiters: [start, patches…, end]."""
        patch_tokens = self.encode_grid(grid)
        start = torch.tensor([self.grid_start_token], dtype=torch.long)
        end   = torch.tensor([self.grid_end_token],   dtype=torch.long)
        return torch.cat([start, patch_tokens, end])

    def encode_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        """Encode a (T, H, W) trajectory into a flat token sequence of length T*tokens_per_frame."""
        frames = [self.encode_frame(traj[t]) for t in range(traj.shape[0])]
        return torch.cat(frames)

    def decode_patch_token(self, token: int) -> torch.Tensor:
        """Decode a single patch token back to (patch_size, patch_size) cell values."""
        ps = self.patch_size
        cells = []
        for _ in range(ps * ps):
            cells.append(token % self.num_colors)
            token //= self.num_colors
        return torch.tensor(cells, dtype=torch.long).reshape(ps, ps)

    def decode_frame_tokens(self, tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
        """Decode a sequence of patch tokens (no delimiters) to a (grid_size, grid_size) grid."""
        ps = self.patch_size
        n = grid_size // ps
        grid = torch.zeros(grid_size, grid_size, dtype=torch.long)
        for idx, token in enumerate(tokens.tolist()):
            r, c = idx // n, idx % n
            grid[r*ps:(r+1)*ps, c*ps:(c+1)*ps] = self.decode_patch_token(token)
        return grid


# =============================================================================
# NCA Dataset
# =============================================================================

class NCADataset(Dataset):
    """In-memory dataset of NCA token sequences for pre-pre-training.

    Each item is a (input_tokens, target_tokens) pair of shape (seq_len,).
    Call generate() once before using as a dataset, or regenerate() to refresh
    with new rules and trajectories for training diversity.
    """

    def __init__(
        self,
        num_sequences: int,
        seq_len: int = 1024,
        grid_size: int = 12,
        patch_size: int = 2,
        num_colors: int = 10,
        temperature: float = 1e-3,
        burn_in: int = 10,
        dT: int = 1,
        gzip_filter: bool = True,
        gzip_low: float = 0.5,
        gzip_high: float = 1.0,
        num_rules: int = 5000,
        seed: int = 42,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ):
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.grid_size = grid_size
        self.temperature = temperature
        self.burn_in = burn_in
        self.dT = dT
        self.gzip_filter = gzip_filter
        self.gzip_low = gzip_low
        self.gzip_high = gzip_high
        self.num_rules = num_rules
        self.seed = seed
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size

        self.tokenizer = NCATokenizer(patch_size=patch_size, num_colors=num_colors)
        self.tokens_per_frame = self.tokenizer.get_tokens_per_frame(grid_size)

        # Number of frames to generate: enough so T*tokens_per_frame >= seq_len+1
        # (the +1 is for the LM input/target shift)
        self.num_frames = math.ceil((seq_len + 1) / self.tokens_per_frame)

        self.data: torch.Tensor | None = None  # set by generate()

    def generate(self, verbose: bool = True) -> None:
        """Pre-generate all NCA sequences and store in memory as (N, seq_len+1)."""
        rank0 = self.ddp_rank == 0
        if verbose and rank0:
            print(f"Generating {self.num_sequences} NCA sequences "
                  f"(seed={self.seed}, rank={self.ddp_rank}/{self.ddp_world_size})...")

        # Use a rank-specific seed so each DDP rank generates different data
        effective_seed = self.seed + self.ddp_rank * 999983  # large prime offset

        # 1. Build pool of filtered (or unfiltered) rules
        if self.gzip_filter:
            if verbose and rank0:
                print(f"  Filtering rules (gzip band: ({self.gzip_low:.2f}, {self.gzip_high:.2f}))...")
            rules = generate_filtered_rules(
                self.num_rules, self.tokenizer,
                self.grid_size, self.tokenizer.num_colors,
                self.temperature, self.gzip_low, self.gzip_high,
                base_seed=effective_seed * 7 + 1,
                verbose=(verbose and rank0),
            )
        else:
            rules = list(range(effective_seed * 7 + 1, effective_seed * 7 + 1 + self.num_rules))

        if len(rules) == 0:
            raise RuntimeError(
                "No rules passed the gzip complexity filter. "
                "Try reducing --gzip-low or disabling --gzip-filter."
            )

        # 2. Generate one trajectory per sequence (round-robin across rule pool)
        all_seqs = []
        for i in range(self.num_sequences):
            rule_seed = rules[i % len(rules)]
            # Unique init state per sequence: combine effective_seed + sequence index
            init_seed = effective_seed + i * 6271  # prime spacing

            traj = rollout(
                rule_seed, init_seed,
                self.num_frames, self.grid_size,
                self.tokenizer.num_colors,
                self.burn_in, self.dT, self.temperature,
            )
            tokens = self.tokenizer.encode_trajectory(traj)  # (num_frames * tokens_per_frame,)

            # Truncate to seq_len+1 for the LM shift
            tokens = tokens[:self.seq_len + 1]
            all_seqs.append(tokens)

        self.data = torch.stack(all_seqs, dim=0)  # (N, seq_len+1)

        if verbose and rank0:
            print(f"  Done. Data shape: {self.data.shape}, "
                  f"memory: {self.data.nbytes / 1024 / 1024:.1f} MB")

    def regenerate(self, new_seed: int, verbose: bool = True) -> None:
        """Refresh the dataset with a new seed for rule and trajectory diversity."""
        self.seed = new_seed
        self.generate(verbose=verbose)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int):
        assert self.data is not None, "Call generate() before accessing NCADataset."
        seq = self.data[idx]  # (seq_len+1,)
        return seq[:-1], seq[1:]  # input (seq_len,), target (seq_len,)


# =============================================================================
# NCA Data Loader
# =============================================================================

def nca_data_loader(
    dataset: NCADataset,
    batch_size: int,
    device: torch.device,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    shuffle: bool = True,
):
    """Infinite DDP-aware generator yielding (inputs, targets) batches on *device*.

    Each DDP rank processes a disjoint shard of the dataset (strided by world_size).
    Shuffles within each rank's shard at the start of every pass through the data.

    Yields:
        x: (batch_size, seq_len) long tensor on *device*
        y: (batch_size, seq_len) long tensor on *device*
    """
    n = len(dataset)
    rank_indices = list(range(ddp_rank, n, ddp_world_size))

    rng = torch.Generator()
    rng.manual_seed(dataset.seed + ddp_rank * 31337)

    while True:
        if shuffle:
            order = torch.randperm(len(rank_indices), generator=rng).tolist()
            shard = [rank_indices[i] for i in order]
        else:
            shard = rank_indices[:]

        for start in range(0, len(shard) - batch_size + 1, batch_size):
            batch_indices = shard[start : start + batch_size]
            xs, ys = zip(*[dataset[i] for i in batch_indices])
            x = torch.stack(xs).to(device)
            y = torch.stack(ys).to(device)
            yield x, y
