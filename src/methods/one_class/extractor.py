"""Multi-scale global feature extraction for One-Class Classification (OC-01)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.backbone import FeatureExtractor


class GlobalFeatureExtractor(nn.Module):
    """Extract pooled image-level representations across multiple CNN layers."""

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3", "layer4"),
        pretrained: bool = True,
        pooling: str = "avg_plus_max",
    ) -> None:
        super().__init__()
        self.extractor = FeatureExtractor(
            backbone_name=backbone_name,
            layers=layers,
            pretrained=pretrained,
        )
        self.pooling = pooling.lower().strip()

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return 2D feature tensor [B, D] for a batch of images."""
        feat_dict = self.extractor(images)
        pooled_parts: list[torch.Tensor] = []

        for layer in self.extractor.layers:
            fmap = feat_dict[layer]
            avg_p = F.adaptive_avg_pool2d(fmap, (1, 1)).flatten(1)
            if self.pooling == "avg":
                pooled_parts.append(avg_p)
            elif self.pooling == "max":
                max_p = F.adaptive_max_pool2d(fmap, (1, 1)).flatten(1)
                pooled_parts.append(max_p)
            else:
                max_p = F.adaptive_max_pool2d(fmap, (1, 1)).flatten(1)
                pooled_parts.append(torch.cat([avg_p, max_p], dim=1))

        features = torch.cat(pooled_parts, dim=1)
        return F.normalize(features, p=2, dim=1)
