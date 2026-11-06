"""Error analysis and per-defect-type breakdown utilities for E-03."""
from __future__ import annotations

import numpy as np
import pandas as pd

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
