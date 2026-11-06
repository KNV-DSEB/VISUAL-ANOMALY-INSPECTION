"""Reproducibility and device selection utilities."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str | None = None) -> torch.device:
    """Return requested torch.device or auto-select CUDA if available."""
    if preferred and preferred != "auto":
        if preferred.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
