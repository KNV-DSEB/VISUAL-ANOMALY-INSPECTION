"""Unified registry and factory for the three anomaly detection methods."""
from __future__ import annotations

from typing import Any

import torch

from .autoencoder import AutoencoderMethod, ConvAutoencoder
from .base import BaseAnomalyMethod
from .one_class import GlobalFeatureExtractor, OneClassMethod
from .patchcore import PatchCoreMethod, PatchFeatureExtractor, greedy_coreset_subsampling

METHOD_REGISTRY: dict[str, type[BaseAnomalyMethod]] = {
    "autoencoder": AutoencoderMethod,
    "patchcore": PatchCoreMethod,
    "one_class": OneClassMethod,
}


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


__all__ = [
    "METHOD_REGISTRY",
    "AutoencoderMethod",
    "BaseAnomalyMethod",
    "ConvAutoencoder",
    "GlobalFeatureExtractor",
    "OneClassMethod",
    "PatchCoreMethod",
    "PatchFeatureExtractor",
    "build_method",
    "greedy_coreset_subsampling",
]
