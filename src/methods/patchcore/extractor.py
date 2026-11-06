"""Multi-scale local patch feature extraction for PatchCore (PC-01)."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.backbone import FeatureExtractor


class PatchFeatureExtractor(nn.Module):
    """Extract locally aggregated, multi-scale patch embeddings from a CNN backbone."""

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        pretrained: bool = True,
        patch_size: int = 3,
        proj_dim: int | None = 384,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.extractor = FeatureExtractor(
            backbone_name=backbone_name,
            layers=layers,
            pretrained=pretrained,
        )
        self.patch_size = int(patch_size)
        self.proj_dim = int(proj_dim) if proj_dim else None
        self.seed = int(seed)
        self.projection: torch.Tensor | None = None
        self.raw_embed_dim_: int | None = None
        self.embed_dim_: int | None = None

    def _ensure_projection(self, in_dim: int, device: torch.device) -> None:
        self.raw_embed_dim_ = in_dim
        if self.proj_dim is None or self.proj_dim >= in_dim:
            self.embed_dim_ = in_dim
            self.projection = None
            return
        self.embed_dim_ = self.proj_dim
        if self.projection is None or self.projection.shape != (in_dim, self.proj_dim):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.seed)
            proj = torch.randn(in_dim, self.proj_dim, generator=gen, dtype=torch.float32)
            proj = F.normalize(proj, p=2, dim=0)
            self.projection = proj.to(device)
        elif self.projection.device != device:
            self.projection = self.projection.to(device)

    @torch.inference_mode()
    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Return patch embeddings [B, H_p * W_p, D] and spatial grid size (H_p, W_p)."""
        feat_dict = self.extractor(images)
        layer_maps = [feat_dict[layer] for layer in self.extractor.layers]

        # Local neighborhood smoothing (average pooling with stride 1)
        if self.patch_size > 1:
            pad = self.patch_size // 2
            smoothed = [
                F.avg_pool2d(
                    fmap,
                    kernel_size=self.patch_size,
                    stride=1,
                    padding=pad,
                    count_include_pad=False,
                )
                for fmap in layer_maps
            ]
        else:
            smoothed = layer_maps

        target_h, target_w = smoothed[0].shape[-2:]
        aligned = [smoothed[0]]
        for fmap in smoothed[1:]:
            if fmap.shape[-2:] != (target_h, target_w):
                fmap = F.interpolate(
                    fmap, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            aligned.append(fmap)

        concat_map = torch.cat(aligned, dim=1)  # [B, C_total, H_p, W_p]
        b, c, h_p, w_p = concat_map.shape
        patches = concat_map.permute(0, 2, 3, 1).reshape(b, h_p * w_p, c)

        self._ensure_projection(c, patches.device)
        if self.projection is not None:
            patches = torch.matmul(patches, self.projection)

        return patches, (h_p, w_p)
