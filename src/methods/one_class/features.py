"""Image-level features from a frozen ImageNet ResNet for the one-class method.

Each requested stage (layer1..layer4) is global-average-pooled and the vectors are
concatenated, e.g. wide_resnet50_2 with [layer2, layer3] -> 512 + 1024 = 1536 dims.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

STAGES = ("layer1", "layer2", "layer3", "layer4")


class FeatureExtractor(nn.Module):
    def __init__(self, backbone: str = "wide_resnet50_2", layers=("layer2", "layer3"),
                 weights: str | None = "IMAGENET1K_V1"):
        super().__init__()
        layers = list(layers)
        bad = [name for name in layers if name not in STAGES]
        if not layers or bad:
            raise ValueError(f"layers must be a non-empty subset of {STAGES}, got {layers}")
        net = models.get_model(backbone, weights=weights)
        if not all(hasattr(net, s) for s in STAGES):
            raise ValueError(f"backbone {backbone!r} is not a ResNet-style network")
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        last = max(STAGES.index(name) for name in layers)
        self.stages = nn.ModuleDict({s: getattr(net, s) for s in STAGES[: last + 1]})
        self.layers = layers
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # frozen: BatchNorm always uses ImageNet stats
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        pooled = []
        for name, stage in self.stages.items():
            x = stage(x)
            if name in self.layers:
                pooled.append(F.adaptive_avg_pool2d(x, 1).flatten(1))
        return torch.cat(pooled, dim=1)


def extract_features(extractor: FeatureExtractor, dataset: Dataset, device: torch.device,
                     batch_size: int = 32, num_workers: int = 0) -> dict:
    """Features in dataset order: dict(features (N,D) float32, labels, paths, defect_types)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    extractor = extractor.to(device)
    feats, labels, paths, defects = [], [], [], []
    for batch in loader:
        feats.append(extractor(batch["image"].to(device)).float().cpu().numpy())
        labels.append(np.asarray(batch["label"]))
        paths += list(batch["path"])
        defects += list(batch["defect_type"])
    return {
        "features": np.concatenate(feats).astype(np.float32),
        "labels": np.concatenate(labels).astype(int),
        "paths": np.asarray(paths),
        "defect_types": np.asarray(defects),
    }
