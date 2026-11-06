"""Visualization utilities for anomaly heatmaps, galleries, and evaluation curves."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize_image(
    image: torch.Tensor | np.ndarray,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> np.ndarray:
    """Convert normalized [3, H, W] tensor/array into uint8 RGB image [H, W, 3]."""
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
    rgb = np.clip(arr * std_arr + mean_arr, 0.0, 1.0)
    return (rgb * 255.0).round().astype(np.uint8)


def normalize_anomaly_map(
    anomaly_map: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Normalize a 2D anomaly map into [0, 1] float32 range."""
    amap = np.asarray(anomaly_map, dtype=np.float32).squeeze()
    lo = float(np.min(amap)) if vmin is None else float(vmin)
    hi = float(np.max(amap)) if vmax is None else float(vmax)
    if hi - lo < 1e-8:
        return np.zeros_like(amap, dtype=np.float32)
    return np.clip((amap - lo) / (hi - lo), 0.0, 1.0)


def overlay_heatmap(
    rgb_image: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.45,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Blend a JET heatmap over an uint8 RGB image [H, W, 3] -> uint8 RGB [H, W, 3]."""
    h, w = rgb_image.shape[:2]
    norm_map = normalize_anomaly_map(anomaly_map, vmin=vmin, vmax=vmax)
    if norm_map.shape != (h, w):
        norm_map = cv2.resize(norm_map, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_u8 = (norm_map * 255.0).round().astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = (1.0 - alpha) * rgb_image.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).round().astype(np.uint8)


def save_localization_gallery(
    samples: list[dict],
    output_path: str | Path,
    title: str = "Localization Comparison Gallery",
) -> Path:
    """Save a comparison figure with rows of (Input, GT Mask, AE Map, PatchCore Map).

    Each dict in `samples` can contain:
      - image_rgb: uint8 [H, W, 3]
      - gt_mask: [H, W] in {0, 1}
      - ae_map / patchcore_map: optional [H, W] float
      - ae_range / patchcore_range: optional (vmin, vmax) so every row of a method shares one color
        scale. Without it each heatmap is min-max normalised on its own, which makes a normal image
        look as hot as a defective one.
      - caption: optional str (category / defect_type / verdict)
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = max(len(samples), 1)
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3.0 * n_rows), squeeze=False)
    col_titles = ["Input Image", "Ground-Truth Mask", "Autoencoder Heatmap", "PatchCore Heatmap"]
    for c, col_title in enumerate(col_titles):
        axes[0, c].set_title(col_title, fontsize=10, fontweight="bold")

    for r, sample in enumerate(samples):
        rgb = sample["image_rgb"]
        gt = np.asarray(sample["gt_mask"]).squeeze()
        caption = sample.get("caption", f"Sample {r + 1}")

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_ylabel(caption, fontsize=9)
        axes[r, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)

        for key, col in (("ae", 2), ("patchcore", 3)):
            if sample.get(f"{key}_map") is not None:
                vmin, vmax = sample.get(f"{key}_range") or (None, None)
                axes[r, col].imshow(overlay_heatmap(rgb, sample[f"{key}_map"], vmin=vmin, vmax=vmax))
            else:
                axes[r, col].text(0.5, 0.5, "N/A", ha="center", va="center")

        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cost_tradeoff_curve(
    cost_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "False-Alarm Rate & Expected Cost vs. Validation Percentile",
) -> Path:
    """Plot FPR, Recall, and Expected Cost across validation percentiles (for I-01)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(cost_df["percentile"], cost_df["fpr"], marker="o", label="False Positive Rate (FPR)")
    ax1.plot(cost_df["percentile"], cost_df["recall"], marker="s", label="Defect Recall (TPR)")
    ax1.plot(cost_df["percentile"], cost_df["f1"], marker="^", label="F1 Score")
    ax1.set_xlabel("Validation Normal Percentile Threshold")
    ax1.set_ylabel("Rate / Score")
    ax1.set_title("Operating Point Trade-off")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    cost_cols = [c for c in cost_df.columns if c.startswith("cost_ratio_")]
    for col in cost_cols:
        ratio_label = col.replace("cost_ratio_", "Cost (C_FN / C_FP = ") + ")"
        ax2.plot(cost_df["percentile"], cost_df[col], marker="o", label=ratio_label)
    ax2.set_xlabel("Validation Normal Percentile Threshold")
    ax2.set_ylabel("Normalized Expected Cost per Image")
    ax2.set_title("Asymmetric False-Alarm vs. Missed-Defect Cost")
    ax2.grid(True, alpha=0.3)
    if cost_cols:
        ax2.legend()

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
