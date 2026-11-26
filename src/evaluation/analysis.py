"""Error analysis, per-defect-type breakdown, and preprocessing crop audit utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .threshold import apply_threshold


def breakdown_by_defect_type(
    pred_df: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Summarize detection rate / false alarm rate per (category, defect_type) on test split."""
    test_df = pred_df[pred_df["split"] == "test"].copy()
    if test_df.empty:
        raise ValueError("No test split rows found for defect breakdown.")

    rows = []
    for (cat, defect), group in test_df.groupby(["category", "defect_type"], sort=True):
        tau = thresholds[str(cat)]
        scores = group["score"].to_numpy(dtype=np.float64)
        preds = apply_threshold(scores, tau)
        is_defect = int(group["label"].iloc[0]) == 1
        n_samples = len(group)
        n_flagged = int((preds == 1).sum())
        rows.append(
            {
                "category": str(cat),
                "defect_type": str(defect),
                "label": int(group["label"].iloc[0]),
                "n_samples": n_samples,
                "n_flagged": n_flagged,
                "detection_or_fpr_rate": float(n_flagged / max(n_samples, 1)),
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "threshold": float(tau),
                "role": "defect_recall" if is_defect else "normal_false_alarm",
            }
        )
    return pd.DataFrame(rows)


def find_error_cases(
    pred_df: pd.DataFrame,
    thresholds: dict[str, float],
    top_k: int = 5,
) -> pd.DataFrame:
    """Extract representative False Positives, Missed Defects (FN), True Positives, and True Negatives."""
    test_df = pred_df[pred_df["split"] == "test"].copy()
    if test_df.empty:
        raise ValueError("No test split rows found for error analysis.")

    test_df["threshold"] = test_df["category"].map(thresholds)
    test_df["pred"] = (test_df["score"] >= test_df["threshold"]).astype(int)
    test_df["margin"] = test_df["score"] - test_df["threshold"]

    def _outcome(row: pd.Series) -> str:
        if row["label"] == 1 and row["pred"] == 1:
            return "TP"
        if row["label"] == 0 and row["pred"] == 1:
            return "FP"
        if row["label"] == 1 and row["pred"] == 0:
            return "FN"
        return "TN"

    test_df["outcome"] = test_df.apply(_outcome, axis=1)

    selected = []
    for cat, group in test_df.groupby("category", sort=True):
        fps = group[group["outcome"] == "FP"].sort_values("score", ascending=False).head(top_k)
        fns = group[group["outcome"] == "FN"].sort_values("score", ascending=True).head(top_k)
        tps = group[group["outcome"] == "TP"].sort_values("score", ascending=False).head(top_k)
        tns = group[group["outcome"] == "TN"].sort_values("score", ascending=True).head(top_k)
        selected.append(pd.concat([fps, fns, tps, tns], ignore_index=True))

    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def audit_mask_crop_loss(
    mask_pil_or_array: Image.Image | np.ndarray,
    resize_size: int = 256,
    crop_size: int = 224,
) -> dict[str, int | float | bool]:
    """Measure defect mask area lost when resizing and center-cropping a ground-truth mask."""
    if isinstance(mask_pil_or_array, np.ndarray):
        arr = mask_pil_or_array
        if arr.ndim == 3:
            arr = arr.squeeze()
        mask_pil = Image.fromarray((arr > 0).astype(np.uint8) * 255, mode="L")
    else:
        mask_pil = mask_pil_or_array.convert("L")

    resized = TF.resize(mask_pil, resize_size, interpolation=InterpolationMode.NEAREST)
    cropped = TF.center_crop(resized, [crop_size, crop_size])

    before_bin = np.asarray(resized, dtype=np.uint8) > 0
    after_bin = np.asarray(cropped, dtype=np.uint8) > 0

    pixels_before = int(before_bin.sum())
    pixels_after = int(after_bin.sum())
    pixels_lost = max(0, pixels_before - pixels_after)
    area_loss_fraction = float(pixels_lost / pixels_before) if pixels_before > 0 else 0.0

    return {
        "orig_width": int(mask_pil.width),
        "orig_height": int(mask_pil.height),
        "resized_width": int(resized.width),
        "resized_height": int(resized.height),
        "pixels_before_crop": pixels_before,
        "pixels_after_crop": pixels_after,
        "pixels_lost": pixels_lost,
        "area_loss_fraction": area_loss_fraction,
        "is_border_affected": bool(pixels_lost > 0),
        "is_completely_lost": bool(pixels_before > 0 and pixels_after == 0),
    }
