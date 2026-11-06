"""Metric implementations shared by all methods. Higher score == more anomalous."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.integrate import trapezoid
from sklearn.metrics import roc_auc_score


def image_auroc(labels, scores) -> float:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def threshold_metrics(labels, scores, threshold: float) -> dict:
    """Image-level precision/recall/F1. Predicted anomalous if score >= threshold."""
    labels = np.asarray(labels).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def pixel_auroc(anomaly_maps, gt_masks, max_pixels=None, seed: int = 42) -> float:
    """Pixel-level AUROC over all pixels of all test images in a category.

    max_pixels: optional seeded random subsample to bound memory.
    """
    s = np.asarray(anomaly_maps).reshape(-1)
    y = (np.asarray(gt_masks) > 0).reshape(-1).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    if max_pixels is not None and len(s) > max_pixels:
        idx = np.random.default_rng(seed).choice(len(s), max_pixels, replace=False)
        s, y = s[idx], y[idx]
        if len(np.unique(y)) < 2:
            return float("nan")
    return float(roc_auc_score(y, s))


def pro_score(anomaly_maps, gt_masks, fpr_limit: float = 0.3, n_points: int = 200) -> float:
    """Per-Region Overlap AUC for FPR in [0, fpr_limit], normalised to [0, 1].

    FPR is measured over every non-defect pixel of every test image (dataset-level), so false
    alarms next to a defect inside a defective image are counted too.
    Thresholds are taken at evenly spaced FPR levels of the normal-pixel score
    distribution, which keeps it deterministic and memory-light.
    """
    maps = np.asarray(anomaly_maps, dtype=np.float64)
    masks = np.asarray(gt_masks) > 0
    has_defect = masks.reshape(len(masks), -1).any(axis=1)

    normal_scores = np.sort(maps[~masks])  # all non-defect pixels, defective images included
    if normal_scores.size == 0 or not has_defect.any():
        return float("nan")

    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    regions = []
    for m, mask in zip(maps[has_defect], masks[has_defect]):
        lab, n = ndimage.label(mask, structure=structure)
        for k in range(1, n + 1):
            regions.append(np.sort(m[lab == k]))

    fpr_levels = np.linspace(0.0, fpr_limit, n_points)
    thr = np.quantile(normal_scores, 1.0 - fpr_levels)  # fpr=0 -> max normal score

    fprs = 1.0 - np.searchsorted(normal_scores, thr, side="right") / normal_scores.size
    pros = np.zeros_like(thr)
    for r in regions:
        pros += 1.0 - np.searchsorted(r, thr, side="right") / r.size
    pros /= len(regions)

    order = np.argsort(fprs)
    fprs, pros = fprs[order], pros[order]
    keep = fprs <= fpr_limit
    fprs, pros = fprs[keep], pros[keep]
    if fprs[-1] < fpr_limit:  # ties: extend flat to the limit
        fprs = np.append(fprs, fpr_limit)
        pros = np.append(pros, pros[-1])
    return float(trapezoid(pros, fprs) / fpr_limit)
