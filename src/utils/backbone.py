"""Shared CNN feature extractor backbone for PatchCore and One-Class methods."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torchvision import models


class _TinyDummyBackbone(nn.Module):
    """Ultra-lightweight deterministic backbone for fast unit/smoke tests on low-RAM machines."""

    def __init__(self) -> None:
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(12345)
            self.stem = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
            )
            self.layer1 = nn.Sequential(
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def build_backbone(
    name: str = "resnet18", pretrained: bool = True, weights: str = "IMAGENET1K_V1"
) -> nn.Module:
    """Instantiate a torchvision ResNet/WideResNet or a tiny test backbone.

    Pretrained weights are pinned by name (default IMAGENET1K_V1, the weights used by the
    original PatchCore) so results do not change when torchvision changes its defaults.
    """
    name = name.lower().strip()
    if name == "wideresnet50":
        name = "wide_resnet50_2"
    if name == "tiny_test":
        model = _TinyDummyBackbone()
    elif name in {"resnet18", "resnet50", "wide_resnet50_2"}:
        model = models.get_model(name, weights=weights if pretrained else None)
    else:
        raise ValueError(
            f"Unsupported backbone {name!r}. Choose from: resnet18, wide_resnet50_2, resnet50, tiny_test."
        )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


class FeatureExtractor(nn.Module):
    """Extract intermediate feature maps from named layers using forward hooks."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        layers: Sequence[str] = ("layer2", "layer3"),
        pretrained: bool = True,
        weights: str = "IMAGENET1K_V1",
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.layers = tuple(layers)
        self.pretrained = pretrained
        self.weights = weights if pretrained else None
        self.backbone = build_backbone(backbone_name, pretrained=pretrained, weights=weights)
        self._features: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        for layer_name in self.layers:
            module = dict(self.backbone.named_modules()).get(layer_name)
            if module is None:
                raise ValueError(
                    f"Layer {layer_name!r} not found in backbone {backbone_name!r}."
                )
            handle = module.register_forward_hook(self._make_hook(layer_name))
            self._hooks.append(handle)

    def _make_hook(self, name: str):
        def _hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            self._features[name] = output

        return _hook

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._features.clear()
        _ = self.backbone(x)
        return {layer: self._features[layer] for layer in self.layers}
