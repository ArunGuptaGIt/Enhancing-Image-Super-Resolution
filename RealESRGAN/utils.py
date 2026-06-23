from __future__ import annotations

import random
from contextlib import nullcontext
from typing import ContextManager

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed random generators for reproducible training behavior."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_ctx(device: torch.device, enabled: bool = True) -> ContextManager[object]:
    """Return an autocast context on CUDA, otherwise a no-op context manager."""

    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()
