"""Reproducibility and device helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def configure_determinism(seed: int) -> None:
    """Seed every RNG and enable deterministic GPU kernels.

    Must be called before the first CUDA operation. TF32 is enabled because it
    is deterministic; ``cudnn.benchmark`` is disabled because some of the
    kernels it may pick are not.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Return the requested device, defaulting to CUDA when available."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)