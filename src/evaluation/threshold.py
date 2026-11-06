"""Validation-only threshold selection and false-alarm cost trade-off analysis."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def compute_validation_threshold(
    val_scores: np.ndarray | Sequence[float],
    rule: str = "percentile_of_val_normal_scores",
    percentile: float = 95.0,
    k_std: float = 3.0,
) -> float:
    """Compute an anomaly score threshold using normal validation scores only.

    Never pass official test set scores or labels to this function.
    """
    scores = np.asarray(val_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("val_scores cannot be empty when computing threshold.")
    if not np.isfinite(scores).all():
        raise ValueError("val_scores contain NaN or infinite values.")

    rule_name = rule.lower().strip()
    if rule_name in {"percentile_of_val_normal_scores", "percentile"}:
        if not (0.0 < percentile <= 100.0):
            raise ValueError(f"percentile must be in (0, 100], got {percentile}")
        return float(np.percentile(scores, percentile))
    if rule_name in {"mean_plus_k_std", "gaussian"}:
        return float(np.mean(scores) + k_std * np.std(scores))
    if rule_name in {"max_val_normal", "max"}:
        return float(np.max(scores))
    raise ValueError(
        f"Unsupported threshold rule {rule!r}. Choose from: "
        "percentile_of_val_normal_scores, mean_plus_k_std, max_val_normal."
    )


def compute_category_thresholds(
    pred_df: pd.DataFrame,
    rule: str = "percentile_of_val_normal_scores",
    percentile: float = 95.0,
    k_std: float = 3.0,
) -> dict[str, float]:
    """Compute per-category thresholds strictly from `split == 'val'` rows."""
    val_df = pred_df[pred_df["split"] == "val"]
    if val_df.empty:
        raise ValueError(
            "Prediction DataFrame contains no 'val' rows; cannot derive validation thresholds."
        )
    if (val_df["label"] != 0).any():
        raise ValueError("Validation split must contain normal samples (label == 0) only.")

    thresholds: dict[str, float] = {}
    for cat, group in val_df.groupby("category", sort=True):
        thresholds[str(cat)] = compute_validation_threshold(
            group["score"].to_numpy(),
            rule=rule,
            percentile=percentile,
            k_std=k_std,
        )
    return thresholds


def apply_threshold(scores: np.ndarray | Sequence[float], threshold: float) -> np.ndarray:
    """Convert continuous anomaly scores into binary predictions {0: normal, 1: defect}."""
    arr = np.asarray(scores, dtype=np.float64)
    return (arr >= float(threshold)).astype(np.int64)


def analyze_false_alarm_cost(
    test_labels: np.ndarray | Sequence[int],
    test_scores: np.ndarray | Sequence[float],
    val_scores: np.ndarray | Sequence[float],
    cost_ratios: Sequence[float] = (1.0, 5.0, 10.0),
    percentiles: Sequence[float] = (80.0, 85.0, 90.0, 92.5, 95.0, 97.5, 99.0, 99.5),
) -> pd.DataFrame:
    """Evaluate operating points derived from validation percentiles on test data (for I-01).

    For each candidate validation percentile p:
      - threshold tau_p is computed ONLY on `val_scores`
      - resulting FP, FN, FPR, Recall, Precision, F1, and normalized expected cost
        C = (1 * FP + r * FN) / N_test are reported for each cost ratio r = C_FN / C_FP.
    """
    y_true = np.asarray(test_labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(test_scores, dtype=np.float64).reshape(-1)
    n_total = max(len(y_true), 1)
    n_neg = max(int((y_true == 0).sum()), 1)
    n_pos = max(int((y_true == 1).sum()), 1)

    rows = []
    for p in percentiles:
        tau = compute_validation_threshold(val_scores, percentile=float(p))
        y_pred = apply_threshold(y_score, tau)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / n_pos
        fpr = fp / n_neg
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        row: dict[str, float | int] = {
            "percentile": float(p),
            "threshold": float(tau),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "fpr": float(fpr),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        for ratio in cost_ratios:
            ratio_key = f"cost_ratio_{int(ratio) if float(ratio).is_integer() else ratio}"
            row[ratio_key] = float((1.0 * fp + float(ratio) * fn) / n_total)
        rows.append(row)

    return pd.DataFrame(rows)
