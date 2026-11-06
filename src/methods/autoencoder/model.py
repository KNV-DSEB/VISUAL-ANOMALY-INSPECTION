"""Convolutional Autoencoder architecture for normal image reconstruction (AE-01)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    """Bilinear upsampling + convolution avoids checkerboard artifacts in anomaly heatmaps."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _ConvBlock(in_ch, out_ch, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class ConvAutoencoder(nn.Module):
    """Compact Convolutional Autoencoder mapping [B, 3, H, W] -> latent -> [B, 3, H, W]."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_channels: int = 128,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.encoder = nn.Sequential(
            _ConvBlock(in_channels, c1, stride=2),          # H/2
            _ConvBlock(c1, c2, stride=2),                   # H/4
            _ConvBlock(c2, c3, stride=2),                   # H/8
            nn.Conv2d(c3, latent_channels, kernel_size=3, stride=2, padding=1, bias=False),  # H/16
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            _UpBlock(latent_channels, c3),                  # H/8
            _UpBlock(c3, c2),                               # H/4
            _UpBlock(c2, c1),                               # H/2
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),  # H
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out
