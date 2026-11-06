"""Registry and factory for the three anomaly detection methods, plus checkpoint helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.utils.io import load_json, save_json

from .autoencoder import AutoencoderMethod, ConvAutoencoder
from .base import BaseAnomalyMethod
from .one_class import OneClassMethod
from .patchcore import PatchCoreMethod, PatchFeatureExtractor, greedy_coreset_subsampling

METHOD_REGISTRY: dict[str, type[BaseAnomalyMethod]] = {
    "autoencoder": AutoencoderMethod,
    "patchcore": PatchCoreMethod,
    "oneclass": OneClassMethod,
}

CONFIG_FILE = "method_config.json"


def build_method(
    name: str,
    config: dict[str, Any],
    device: str | torch.device | None = None,
) -> BaseAnomalyMethod:
    """Instantiate one of the 3 anomaly detection methods by name."""
    key = name.lower().strip()
    if key not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method {name!r}. Valid methods: {sorted(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[key](config=config, device=device)


def save_method(method: BaseAnomalyMethod, ckpt_dir: str | Path) -> Path:
    """Save method artifacts plus the config and validation threshold needed to reload it."""
    out = method.save(ckpt_dir)
    save_json(
        {"method": method.name, "config": method.config, "threshold": method.threshold_},
        Path(out) / CONFIG_FILE,
    )
    return out


def load_method(
    name: str, ckpt_dir: str | Path, device: str | torch.device | None = None
) -> BaseAnomalyMethod:
    """Rebuild a method from the config stored next to its checkpoint and load its weights."""
    ckpt = Path(ckpt_dir)
    meta = load_json(ckpt / CONFIG_FILE)
    if meta["method"] != name.lower().strip():
        raise ValueError(f"Checkpoint {ckpt} belongs to {meta['method']!r}, not {name!r}.")
    method = build_method(name, meta["config"], device=device)
    method.load(ckpt)
    method.threshold_ = meta.get("threshold")
    return method


__all__ = [
    "METHOD_REGISTRY",
    "AutoencoderMethod",
    "BaseAnomalyMethod",
    "ConvAutoencoder",
    "OneClassMethod",
    "PatchCoreMethod",
    "PatchFeatureExtractor",
    "build_method",
    "greedy_coreset_subsampling",
    "load_method",
    "save_method",
]
