"""Self-supervised causal TCN encoder.

The encoder is trained with two objectives:

* Masked reconstruction: spans of windows are replaced by a learned mask token
  and the network reconstructs their features.
* Temporal triplet loss: windows close in time must be closer in embedding
  space than windows far apart.

The network is causal end to end (dilated causal convolutions, no temporal
normalisation), so the same ``forward`` is used for training and for extracting
embeddings: the embedding at position ``t`` only depends on windows ``<= t``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader

from .config import BASE_SEED, EncoderConfig, TCNParams
from .utils import resolve_device

logger = logging.getLogger(__name__)

CLOSE_FRAC = 0.05
FAR_FRAC = 0.30
CLIP_PERCENTILES = (1.0, 99.0)
GRAD_CLIP_NORM = 1.0
WARMUP_EPOCHS = 3
LR_COSINE_FLOOR = 0.05
NORM_EPS = 1e-8
EARLY_STOPPING_MIN_DELTA = 1e-4
VAL_MASK_SEED = 12345
FINAL_SEED_OFFSET = 999
FINAL_EVAL_SEED_OFFSET = 777


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class CausalTCNBlock(nn.Module):
    """Residual block with two weight-normalised dilated causal convolutions.

    Causality is enforced by padding on the left only. There is no
    normalisation over the time axis, which would leak future information.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=dilation))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # Reference TCN initialisation (kept for reproducibility of trained models).
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, L)
        out = self.dropout(self.act(self.conv1(F.pad(x, (self.pad, 0)))))
        out = self.dropout(self.act(self.conv2(F.pad(out, (self.pad, 0)))))
        return self.act(out + x)


class CausalTCNAutoencoder(nn.Module):
    """Input projection -> stack of causal TCN blocks -> reconstruction head."""

    def __init__(self, n_features: int, params: TCNParams, max_dilation: int = 32) -> None:
        super().__init__()
        channels = params.channels
        self.input_proj = nn.Linear(n_features, channels)
        self.mask_token = nn.Parameter(torch.zeros(channels))
        nn.init.normal_(self.mask_token, std=0.02)
        self.tcn = nn.Sequential(*[
            CausalTCNBlock(channels, params.kernel_size, min(2 ** i, max_dilation), params.dropout)
            for i in range(params.num_layers)
        ])
        self.recon_head = nn.Linear(channels, n_features)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(embeddings, reconstruction)``, both ``(B, L, ·)``.

        ``mask`` (``(B, L)`` bool) marks the positions replaced by the mask token.
        """
        tokens = self.input_proj(x)
        if mask is not None:
            tokens = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens)
        emb = self.tcn(tokens.transpose(1, 2)).transpose(1, 2)
        return emb, self.recon_head(emb)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

class SequencePreprocessor:
    """Percentile clipping and robust scaling."""

    def fit(self, sequences: Sequence[np.ndarray]) -> SequencePreprocessor:
        rows = np.concatenate(sequences, axis=0)
        self.lower_, self.upper_ = np.percentile(rows, CLIP_PERCENTILES, axis=0)
        self.scaler_ = RobustScaler().fit(np.clip(rows, self.lower_, self.upper_))
        return self

    def transform(self, sequence: np.ndarray) -> np.ndarray:
        clipped = np.clip(sequence, self.lower_, self.upper_)
        return self.scaler_.transform(clipped).astype(np.float32)


def sequence_variance(sequences: Sequence[np.ndarray]) -> float:
    """Global variance, used to normalise the reconstruction loss."""
    return float(np.var(np.concatenate([s.ravel() for s in sequences])))


# --------------------------------------------------------------------------- #
# Masking, triplets and losses
# --------------------------------------------------------------------------- #

def markov_mask(length: int, mask_ratio: float, mean_span: float, rng: np.random.Generator) -> np.ndarray:
    """Two-state Markov mask with the given masked fraction and mean span length."""
    p_end = 1.0 / mean_span
    p_start = mask_ratio / (1.0 - mask_ratio) * p_end
    mask = np.zeros(length, dtype=bool)
    state = rng.random() < mask_ratio
    for i in range(length):
        mask[i] = state
        if rng.random() < (p_end if state else p_start):
            state = not state
    return mask


def sample_triplets(
    length: int, n_triplets: int, rng: np.random.Generator,
    close_frac: float = CLOSE_FRAC, far_frac: float = FAR_FRAC,
) -> list[tuple[int, int, int]]:
    """Sample ``(anchor, positive, negative)`` indices: positives close, negatives far."""
    if length < 5:
        return []
    close = max(1, int(close_frac * length))
    far = max(close + 1, int(far_frac * length))
    triplets = []
    for _ in range(n_triplets):
        a = int(rng.integers(0, length))
        positives = [i for i in range(max(0, a - close), min(length - 1, a + close) + 1) if i != a]
        if not positives:
            continue
        p = positives[rng.integers(0, len(positives))]
        negatives = [i for i in range(length) if abs(i - a) > far]
        if not negatives:
            continue
        triplets.append((a, p, negatives[rng.integers(0, len(negatives))]))
    return triplets


def compute_losses(
    model: CausalTCNAutoencoder,
    x: torch.Tensor,
    valid: torch.Tensor,
    params: TCNParams,
    var_target: float,
    rng: np.random.Generator,
    fixed_mask: torch.Tensor | None = None,
    fixed_triplets: list[list[tuple[int, int, int]]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised ``(reconstruction_loss, triplet_loss)`` for a padded batch.

    ``valid`` marks real (non-padding) positions. Padding is on the right, so
    causality guarantees it never affects valid positions.
    """
    batch, length, _ = x.shape
    lengths = valid.sum(dim=1).tolist()

    if fixed_mask is None:
        mask_np = np.zeros((batch, length), dtype=bool)
        for i, n in enumerate(lengths):
            mask_np[i, :n] = markov_mask(n, params.mask_ratio, params.mean_span_length, rng)
        fixed_mask = torch.from_numpy(mask_np).to(x.device)
    mask = fixed_mask & valid

    emb, recon = model(x, mask)
    recon_loss = F.mse_loss(recon[mask], x[mask]) if mask.any() else x.new_zeros(())

    idx = [
        (i, a, p, n)
        for i, n_valid in enumerate(lengths)
        for a, p, n in (fixed_triplets[i] if fixed_triplets is not None
                        else sample_triplets(n_valid, params.n_triplets, rng))
    ]
    if idx:
        bi, ai, pi, ni = torch.as_tensor(idx, device=x.device, dtype=torch.long).T
        emb = F.normalize(emb, p=2, dim=-1, eps=1e-8)
        d_ap = torch.norm(emb[bi, ai] - emb[bi, pi], dim=-1)
        d_an = torch.norm(emb[bi, ai] - emb[bi, ni], dim=-1)
        rank_loss = F.relu(d_ap - d_an + params.margin).mean()
    else:
        rank_loss = x.new_zeros(())

    return recon_loss / (var_target + NORM_EPS), rank_loss / (2.0 + params.margin)


def combine_losses(recon: torch.Tensor, rank: torch.Tensor, lambda_loss: float) -> torch.Tensor:
    return lambda_loss * recon + (1 - lambda_loss) * rank


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def _collate(batch: Sequence[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad sequences to a common length; return ``(x, valid_mask)``."""
    max_len = max(len(s) for s in batch)
    x = np.zeros((len(batch), max_len, batch[0].shape[1]), dtype=np.float32)
    valid = np.zeros((len(batch), max_len), dtype=bool)
    for i, seq in enumerate(batch):
        x[i, :len(seq)] = seq
        valid[i, :len(seq)] = True
    return torch.from_numpy(x), torch.from_numpy(valid)


def _lr_factor(epoch: int, total_epochs: int) -> float:
    """Linear warm-up followed by cosine decay to ``LR_COSINE_FLOOR``."""
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = min(1.0, (epoch - WARMUP_EPOCHS) / max(1, total_epochs - WARMUP_EPOCHS))
    return LR_COSINE_FLOOR + (1 - LR_COSINE_FLOOR) * 0.5 * (1 + math.cos(math.pi * progress))


def _setup_training(n_features, params, max_dilation, total_epochs, device):
    model = CausalTCNAutoencoder(n_features, params, max_dilation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: _lr_factor(e, total_epochs))
    return model, optimizer, scheduler


def _make_loader(sequences, batch_size, seed) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(list(sequences), batch_size=batch_size, shuffle=True,
                      collate_fn=_collate, generator=generator)


def _train_epoch(model, loader, optimizer, params, lambda_loss, var_target, rng, device) -> None:
    model.train()
    for xb, valid in loader:
        xb, valid = xb.to(device), valid.to(device)
        optimizer.zero_grad()
        loss = combine_losses(*compute_losses(model, xb, valid, params, var_target, rng), lambda_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()


@torch.no_grad()
def _evaluate(model, x, valid, params, lambda_loss, var_target, mask, triplets):
    model.eval()
    recon, rank = compute_losses(model, x, valid, params, var_target, rng=None,
                                 fixed_mask=mask, fixed_triplets=triplets)
    return combine_losses(recon, rank, lambda_loss).item(), recon.item(), rank.item()


def _fixed_eval_inputs(sequences, params, seed, device):
    """Padded batch with a fixed mask and fixed triplets for a stable validation loss."""
    x, valid = _collate(sequences)
    rng = np.random.default_rng(seed)
    mask = np.zeros(valid.shape, dtype=bool)
    triplets = []
    for i, seq in enumerate(sequences):
        n = len(seq)
        mask[i, :n] = markov_mask(n, params.mask_ratio, params.mean_span_length, rng)
        triplets.append(sample_triplets(n, params.n_triplets, rng))
    return x.to(device), valid.to(device), torch.from_numpy(mask).to(device), triplets


def train_with_early_stopping(
    train_sequences: Sequence[np.ndarray],
    val_sequence: np.ndarray,
    params: TCNParams,
    lambda_loss: float,
    var_target: float,
    *,
    device: torch.device,
    max_epochs: int,
    patience: int,
    seed: int,
    max_dilation: int = 32,
) -> tuple[float, int]:
    """Train on ``train_sequences`` with early stopping on ``val_sequence``.

    Used inside cross-validation. Returns ``(best_val_loss, best_epoch)``.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    loader = _make_loader(train_sequences, min(params.batch_size, len(train_sequences)), seed)
    model, optimizer, scheduler = _setup_training(
        train_sequences[0].shape[1], params, max_dilation, max_epochs, device)
    val_x, val_valid, val_mask, val_triplets = _fixed_eval_inputs(
        [val_sequence], params, VAL_MASK_SEED, device)

    best_loss, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, max_epochs + 1):
        _train_epoch(model, loader, optimizer, params, lambda_loss, var_target, rng, device)
        scheduler.step()
        loss, _, _ = _evaluate(model, val_x, val_valid, params, lambda_loss, var_target,
                               val_mask, val_triplets)
        if loss < best_loss - EARLY_STOPPING_MIN_DELTA:
            best_loss, best_epoch, stale = loss, epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_loss, best_epoch


def train_fixed_epochs(
    sequences: Sequence[np.ndarray],
    params: TCNParams,
    lambda_loss: float,
    var_target: float,
    *,
    n_epochs: int,
    device: torch.device,
    seed: int,
    eval_seed: int,
    max_dilation: int = 32,
) -> tuple[CausalTCNAutoencoder, dict[str, float]]:
    """Train on all sequences for ``n_epochs``; keep the epoch with the lowest loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, optimizer, scheduler = _setup_training(
        sequences[0].shape[1], params, max_dilation, n_epochs, device)
    loader = _make_loader(sequences, min(params.batch_size, len(sequences)), seed)
    eval_x, eval_valid, eval_mask, eval_triplets = _fixed_eval_inputs(
        sequences, params, eval_seed, device)

    best = {"loss": float("inf")}
    best_state = None
    for _ in range(n_epochs):
        _train_epoch(model, loader, optimizer, params, lambda_loss, var_target, rng, device)
        scheduler.step()
        loss, recon, rank = _evaluate(model, eval_x, eval_valid, params, lambda_loss,
                                      var_target, eval_mask, eval_triplets)
        if loss < best["loss"]:
            best = {"loss": loss, "recon_loss": recon, "rank_loss": rank}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best


# --------------------------------------------------------------------------- #
# Public encoder
# --------------------------------------------------------------------------- #

class TCNEncoder:
    """Fitted preprocessing + causal TCN that maps sequences to embeddings."""

    CONFIG_FILE = "encoder.json"
    WEIGHTS_FILE = "weights.pt"
    PREPROCESSOR_FILE = "preprocessor.joblib"

    def __init__(self, config: EncoderConfig, seed: int = BASE_SEED,
                 device: str | torch.device | None = None) -> None:
        self.config = config
        self.seed = seed
        self.device = resolve_device(device)
        self.preprocessor_: SequencePreprocessor | None = None
        self.model_: CausalTCNAutoencoder | None = None
        self.fit_stats_: dict[str, float] = {}

    @property
    def embed_dim(self) -> int:
        return self.config.tcn.channels

    def fit(self, sequences: Sequence[np.ndarray]) -> TCNEncoder:
        """Fit preprocessing and train the network on all ``sequences``."""
        sequences = list(sequences)
        self.preprocessor_ = SequencePreprocessor().fit(sequences)
        transformed = [self.preprocessor_.transform(s) for s in sequences]
        self.model_, self.fit_stats_ = train_fixed_epochs(
            transformed, self.config.tcn, self.config.lambda_loss, sequence_variance(transformed),
            n_epochs=self.config.n_epochs, device=self.device,
            seed=self.seed + FINAL_SEED_OFFSET, eval_seed=self.seed + FINAL_EVAL_SEED_OFFSET,
            max_dilation=self.config.max_dilation,
        )
        logger.info("Encoder trained: %s", self.fit_stats_)
        return self

    @torch.no_grad()
    def embed(self, sequences: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
        """L2-normalised causal embeddings, shape ``(n_windows, embed_dim)`` per dataset."""
        if self.model_ is None:
            raise RuntimeError("Encoder is not fitted.")
        self.model_.eval()
        out = {}
        for key, seq in sequences.items():
            x = torch.from_numpy(self.preprocessor_.transform(seq)).unsqueeze(0).to(self.device)
            emb, _ = self.model_(x)
            out[key] = F.normalize(emb, p=2, dim=-1, eps=1e-8).squeeze(0).cpu().numpy()
        return out

    def save(self, directory: str | Path) -> None:
        if self.model_ is None:
            raise RuntimeError("Encoder is not fitted.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_.state_dict(), directory / self.WEIGHTS_FILE)
        joblib.dump(self.preprocessor_, directory / self.PREPROCESSOR_FILE)
        metadata = {
            "config": asdict(self.config),
            "seed": self.seed,
            "n_features": self.model_.input_proj.in_features,
            "fit_stats": self.fit_stats_,
        }
        (directory / self.CONFIG_FILE).write_text(json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, directory: str | Path, device: str | torch.device | None = None) -> TCNEncoder:
        directory = Path(directory)
        metadata = json.loads((directory / cls.CONFIG_FILE).read_text())
        encoder = cls(EncoderConfig.from_dict(metadata["config"]), metadata["seed"], device)
        encoder.preprocessor_ = joblib.load(directory / cls.PREPROCESSOR_FILE)
        encoder.fit_stats_ = metadata.get("fit_stats", {})
        model = CausalTCNAutoencoder(metadata["n_features"], encoder.config.tcn,
                                     encoder.config.max_dilation).to(encoder.device)
        state = torch.load(directory / cls.WEIGHTS_FILE, map_location=encoder.device, weights_only=True)
        model.load_state_dict(state)
        encoder.model_ = model.eval()
        return encoder