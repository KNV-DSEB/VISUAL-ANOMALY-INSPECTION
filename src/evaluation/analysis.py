"""Error analysis (E-03) and false-alarm cost analysis (I-01) on saved predictions.

All thresholds come from validation normal scores through `percentile_threshold`.
These tables are descriptive: they show what each operating point would do on the test set.
Do not pick the final operating point by minimising a cost computed on the official test set.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .schema import Predictions
from .thresholds import percentile_threshold

DEFAULT_PERCENTILES = (80.0, 85.0, 90.0, 92.5, 95.0, 97.5, 99.0, 99.5)


def _check_pair(val: Predictions, test: Predictions) -> None:
    if (val.method, val.category) != (test.method, test.category):
        raise ValueError("val and test predictions must be the same method/category")
    if test.split != "test":
        raise ValueError("second argument must be the test split")


def analyze_false_alarm_cost(
    val: Predictions,
    test: Predictions,
    cost_ratios: Sequence[float] = (1.0, 5.0, 10.0),
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> pd.DataFrame:
    """Operating points at several validation percentiles: FP, FN, FPR, recall, precision, F1, cost.

    Normalised cost per test image is (FP + r * FN) / N for each cost ratio r = C_FN / C_FP.
    """
    _check_pair(val, test)
    y, s = test.labels, test.scores
    n_total, n_neg, n_pos = max(len(y), 1), max(int((y == 0).sum()), 1), max(int((y == 1).sum()), 1)
    rows = []
    for p in percentiles:
        tau = percentile_threshold(val, float(p))
        pred = (s >= tau).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / n_pos
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row = {"method": test.method, "category": test.category, "percentile": float(p),
               "threshold": tau, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "fpr": fp / n_neg,
               "precision": precision, "recall": recall, "f1": f1}
        for r in cost_ratios:
            key = f"cost_ratio_{int(r) if float(r).is_integer() else r}"
            row[key] = (fp + float(r) * fn) / n_total
        rows.append(row)
    return pd.DataFrame(rows)


def breakdown_by_defect_type(val: Predictions, test: Predictions, percentile: float = 95.0) -> pd.DataFrame:
    """Flag rate per defect type (recall for defects, false-alarm rate for good) at the val threshold."""
    _check_pair(val, test)
    if test.defect_types is None:
        raise ValueError("test predictions have no defect_types")
    tau = percentile_threshold(val, percentile)
    df = pd.DataFrame({"defect_type": test.defect_types, "score": test.scores, "label": test.labels})
    rows = []
    for defect, g in df.groupby("defect_type", sort=True):
        flagged = int((g["score"] >= tau).sum())
        rows.append({
            "method": test.method, "category": test.category, "defect_type": str(defect),
            "label": int(g["label"].iloc[0]), "n_samples": len(g), "n_flagged": flagged,
            "flag_rate": flagged / len(g),
            "role": "defect_recall" if int(g["label"].iloc[0]) == 1 else "normal_false_alarm",
            "mean_score": float(g["score"].mean()), "threshold": tau,
        })
    return pd.DataFrame(rows)


def find_error_cases(val: Predictions, test: Predictions, percentile: float = 95.0,
                     top_k: int = 5) -> pd.DataFrame:
    """Most confident false positives, missed defects, and for contrast true positives / negatives."""
    _check_pair(val, test)
    tau = percentile_threshold(val, percentile)
    df = pd.DataFrame({
        "method": test.method, "category": test.category, "path": test.paths,
        "defect_type": test.defect_types if test.defect_types is not None else "",
        "label": test.labels, "score": test.scores,
    })
    df["threshold"] = tau
    df["margin"] = df["score"] - tau
    pred = (df["score"] >= tau).astype(int)
    df["outcome"] = np.select(
        [(df.label == 1) & (pred == 1), (df.label == 0) & (pred == 1), (df.label == 1) & (pred == 0)],
        ["TP", "FP", "FN"], default="TN")
    parts = [
        df[df.outcome == "FP"].nlargest(top_k, "score"),
        df[df.outcome == "FN"].nsmallest(top_k, "score"),
        df[df.outcome == "TP"].nlargest(top_k, "score"),
        df[df.outcome == "TN"].nsmallest(top_k, "score"),
    ]
    return pd.concat(parts, ignore_index=True)
