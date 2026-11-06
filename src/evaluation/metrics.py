"""Image-level and pixel-level metrics (AUROC, F1, FPR, Pixel AUROC, PRO) for E-01 & L-01."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import label as connected_components
from sklearn.metrics import roc_auc_score

from .schema import validate_predictions
from .threshold import apply_threshold, compute_category_thresholds


def compute_image_metrics(
    labels: np.ndarray | Sequence[int],
    scores: np.ndarray | Sequence[float],
    threshold: float,
) -> dict[str, float]:
    """Compute image-level AUROC and threshold-based metrics (Precision, Recall, F1, FPR)."""
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or y_true.shape != y_score.shape:
        raise ValueError("labels and scores must be non-empty 1D arrays of equal length.")

    if len(np.unique(y_true)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_score))

    y_pred = apply_threshold(y_score, threshold)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)

    return {
        "image_auroc": auroc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "accuracy": float(accuracy),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_pixel_auroc(
    gt_masks: np.ndarray,
    anomaly_maps: np.ndarray,
) -> float:
    """Compute pixel-level AUROC across all pixels in a category's test set."""
    masks = (np.asarray(gt_masks) > 0).astype(np.uint8).reshape(-1)
    maps = np.asarray(anomaly_maps, dtype=np.float32).reshape(-1)
    if masks.shape != maps.shape or masks.size == 0:
        raise ValueError("gt_masks and anomaly_maps must have matching non-empty shapes.")
    if len(np.unique(masks)) < 2:
        return float("nan")
    return float(roc_auc_score(masks, maps))


def compute_pro_score(
    gt_masks: np.ndarray,
    anomaly_maps: np.ndarray,
    integration_limit: float = 0.3,
    num_thresholds: int = 200,
) -> float:
    """Compute the MVTec AD Per-Region Overlap (PRO) score integrated up to FPR = 0.3.

    Each connected defect component in ground-truth masks is weighted equally
    regardless of its pixel area. Uses sorted 1D score arrays + binary search
    so memory footprint stays minimal even on 4 GB RAM machines.
    """
    masks = (np.asarray(gt_masks) > 0).astype(np.uint8)
    maps = np.asarray(anomaly_maps, dtype=np.float32)
    if masks.ndim != 3 or masks.shape != maps.shape:
        raise ValueError(
            f"Expected 3D arrays [N, H, W] of identical shape, got {masks.shape} and {maps.shape}."
        )

    region_scores: list[np.ndarray] = []
    for i in range(masks.shape[0]):
        if not np.any(masks[i]):
            continue
        labeled, n_comp = connected_components(masks[i])
        for comp_id in range(1, n_comp + 1):
            comp_vals = np.sort(maps[i][labeled == comp_id])
            if comp_vals.size > 0:
                region_scores.append(comp_vals)

    neg_scores = np.sort(maps[masks == 0])
    if not region_scores or neg_scores.size == 0:
        return float("nan")

    min_score = float(np.min(maps))
    max_score = float(np.max(maps))
    if max_score - min_score < 1e-12:
        return 0.0

    # Sweep thresholds from highest to lowest so both FPR and PRO increase monotonically
    thresholds = np.linspace(max_score, min_score, num_thresholds, dtype=np.float32)
    n_neg = float(neg_scores.size)
    neg_idx = np.searchsorted(neg_scores, thresholds, side="left")
    fprs = (n_neg - neg_idx) / n_neg

    pro_sum = np.zeros_like(thresholds, dtype=np.float64)
    for r_scores in region_scores:
        r_len = float(r_scores.size)
        r_idx = np.searchsorted(r_scores, thresholds, side="left")
        pro_sum += (r_len - r_idx) / r_len
    pros = pro_sum / len(region_scores)

    if fprs[0] > 0.0:
        fprs = np.concatenate([[0.0], fprs])
        pros = np.concatenate([[0.0], pros])

    within = fprs <= integration_limit
    fpr_clipped = fprs[within]
    pro_clipped = pros[within]

    if fpr_clipped[-1] < integration_limit:
        above_idx = int(np.searchsorted(fprs, integration_limit, side="right"))
        if above_idx < len(fprs):
            x0, x1 = fprs[above_idx - 1], fprs[above_idx]
            y0, y1 = pros[above_idx - 1], pros[above_idx]
            if x1 > x0:
                y_interp = y0 + (y1 - y0) * (integration_limit - x0) / (x1 - x0)
            else:
                y_interp = y1
        else:
            y_interp = pro_clipped[-1]
        fpr_clipped = np.concatenate([fpr_clipped, [integration_limit]])
        pro_clipped = np.concatenate([pro_clipped, [y_interp]])

    area = float(np.trapezoid(pro_clipped, fpr_clipped))
    return float(np.clip(area / integration_limit, 0.0, 1.0))


def compute_pixel_metrics(
    gt_masks: np.ndarray,
    anomaly_maps: np.ndarray,
    compute_pro: bool = True,
    integration_limit: float = 0.3,
) -> dict[str, float]:
    """Compute pixel-level localization metrics (Pixel AUROC and optional PRO)."""
    pixel_auroc = compute_pixel_auroc(gt_masks, anomaly_maps)
    pro = (
        compute_pro_score(gt_masks, anomaly_maps, integration_limit=integration_limit)
        if compute_pro
        else float("nan")
    )
    return {
        "pixel_auroc": pixel_auroc,
        "pixel_pro": pro,
    }


def evaluate_predictions(
    pred_df: pd.DataFrame,
    anomaly_maps: np.ndarray | None = None,
    gt_masks: np.ndarray | None = None,
    threshold_cfg: dict[str, Any] | None = None,
    compute_pro: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run unified per-category + mean evaluation on a prediction bundle.

    - Thresholds are derived per category ONLY from `split == 'val'`.
    - Test metrics are computed per category on `split == 'test'`.
    - Returns `(summary_df, category_thresholds)`.
    """
    validate_predictions(pred_df, anomaly_maps=anomaly_maps, gt_masks=gt_masks)
    cfg = threshold_cfg or {}
    rule = cfg.get("rule", "percentile_of_val_normal_scores")
    percentile = float(cfg.get("percentile", 95.0))
    k_std = float(cfg.get("k_std", 3.0))

    thresholds = compute_category_thresholds(
        pred_df, rule=rule, percentile=percentile, k_std=k_std
    )

    test_df = pred_df[pred_df["split"] == "test"]
    if test_df.empty:
        raise ValueError("Prediction DataFrame contains no 'test' rows to evaluate.")

    rows: list[dict[str, Any]] = []
    for cat in sorted(test_df["category"].unique()):
        if cat not in thresholds:
            raise ValueError(f"Missing validation threshold for category {cat!r}.")
        cat_mask = (pred_df["split"] == "test") & (pred_df["category"] == cat)
        cat_test = pred_df[cat_mask]
        img_metrics = compute_image_metrics(
            cat_test["label"].to_numpy(),
            cat_test["score"].to_numpy(),
            threshold=thresholds[cat],
        )
        row: dict[str, Any] = {"category": cat, **img_metrics}

        if anomaly_maps is not None and gt_masks is not None:
            idx = np.flatnonzero(cat_mask.to_numpy())
            pix_metrics = compute_pixel_metrics(
                gt_masks[idx], anomaly_maps[idx], compute_pro=compute_pro
            )
            row.update(pix_metrics)
        else:
            row["pixel_auroc"] = float("nan")
            row["pixel_pro"] = float("nan")
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    metric_cols = [
        "image_auroc",
        "precision",
        "recall",
        "f1",
        "fpr",
        "accuracy",
        "threshold",
        "pixel_auroc",
        "pixel_pro",
    ]
    mean_row: dict[str, Any] = {"category": "mean"}
    for col in metric_cols:
        if col in summary_df.columns:
            mean_row[col] = float(summary_df[col].mean(skipna=True))
    for count_col in ("tp", "fp", "tn", "fn"):
        if count_col in summary_df.columns:
            mean_row[count_col] = int(summary_df[count_col].sum())

    summary_df = pd.concat([summary_df, pd.DataFrame([mean_row])], ignore_index=True)
    return summary_df, thresholds
