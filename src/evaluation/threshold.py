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


def bootstrap_threshold_stability(
    val_scores: np.ndarray | Sequence[float],
    test_labels: np.ndarray | Sequence[int],
    test_scores: np.ndarray | Sequence[float],
    percentiles: Sequence[float] = (90.0, 95.0),
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Assess validation-threshold stability via bootstrap resampling of normal val scores."""
    v_scores = np.asarray(val_scores, dtype=np.float64).reshape(-1)
    y_true = np.asarray(test_labels, dtype=np.int64).reshape(-1)
    y_score = np.asarray(test_scores, dtype=np.float64).reshape(-1)
    if v_scores.size == 0:
        raise ValueError("val_scores cannot be empty.")
    if y_true.size != y_score.size or y_true.size == 0:
        raise ValueError("test_labels and test_scores must be non-empty and equal length.")

    n_val = int(v_scores.size)
    n_neg = max(int((y_true == 0).sum()), 1)
    n_pos = max(int((y_true == 1).sum()), 1)

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n_val, size=(int(n_bootstrap), n_val))
    boot_val = v_scores[boot_indices]

    rows: list[dict[str, float | int]] = []
    for p in percentiles:
        p_val = float(p)
        tau = compute_validation_threshold(v_scores, percentile=p_val)
        y_pred = apply_threshold(y_score, tau)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / n_pos
        fpr = fp / n_neg
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        boot_taus = np.percentile(boot_val, p_val, axis=1)
        boot_preds = y_score[None, :] >= boot_taus[:, None]
        is_pos = y_true == 1
        is_neg = y_true == 0
        tp_b = (boot_preds & is_pos[None, :]).sum(axis=1).astype(np.float64)
        fp_b = (boot_preds & is_neg[None, :]).sum(axis=1).astype(np.float64)
        rec_b = tp_b / n_pos
        fpr_b = fp_b / n_neg
        denom_p = tp_b + fp_b
        prec_b = np.where(denom_p > 0, tp_b / np.maximum(denom_p, 1.0), 0.0)
        denom_f1 = prec_b + rec_b
        f1_b = np.where(denom_f1 > 0, 2.0 * prec_b * rec_b / np.maximum(denom_f1, 1e-12), 0.0)

        rows.append(
            {
                "percentile": p_val,
                "n_val": n_val,
                "threshold": float(tau),
                "threshold_std": float(np.std(boot_taus, ddof=1)) if n_bootstrap > 1 else 0.0,
                "threshold_ci_low": float(np.percentile(boot_taus, 2.5)),
                "threshold_ci_high": float(np.percentile(boot_taus, 97.5)),
                "recall": float(recall),
                "recall_std": float(np.std(rec_b, ddof=1)) if n_bootstrap > 1 else 0.0,
                "recall_ci_low": float(np.percentile(rec_b, 2.5)),
                "recall_ci_high": float(np.percentile(rec_b, 97.5)),
                "fpr": float(fpr),
                "fpr_std": float(np.std(fpr_b, ddof=1)) if n_bootstrap > 1 else 0.0,
                "fpr_ci_low": float(np.percentile(fpr_b, 2.5)),
                "fpr_ci_high": float(np.percentile(fpr_b, 97.5)),
                "precision": float(precision),
                "f1": float(f1),
                "f1_std": float(np.std(f1_b, ddof=1)) if n_bootstrap > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compare_operating_percentiles(
    pred_df: pd.DataFrame,
    low_p: float = 90.0,
    high_p: float = 95.0,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare two validation percentile operating points side-by-side per category."""
    low_label = f"p{int(low_p) if float(low_p).is_integer() else low_p}"
    high_label = f"p{int(high_p) if float(high_p).is_integer() else high_p}"

    rows: list[dict[str, float | int | str]] = []
    for cat, grp in pred_df.groupby("category", sort=True):
        val_scores = grp[grp["split"] == "val"]["score"].to_numpy()
        test_grp = grp[grp["split"] == "test"]
        boot_df = bootstrap_threshold_stability(
            val_scores=val_scores,
            test_labels=test_grp["label"].to_numpy(),
            test_scores=test_grp["score"].to_numpy(),
            percentiles=(low_p, high_p),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        r_low = boot_df.iloc[0]
        r_high = boot_df.iloc[1]
        rows.append(
            {
                "category": str(cat),
                "n_val": int(r_low["n_val"]),
                f"recall_{low_label}": float(r_low["recall"]),
                f"fpr_{low_label}": float(r_low["fpr"]),
                f"precision_{low_label}": float(r_low["precision"]),
                f"f1_{low_label}": float(r_low["f1"]),
                f"recall_{high_label}": float(r_high["recall"]),
                f"fpr_{high_label}": float(r_high["fpr"]),
                f"precision_{high_label}": float(r_high["precision"]),
                f"f1_{high_label}": float(r_high["f1"]),
                "delta_recall": float(r_low["recall"] - r_high["recall"]),
                "delta_fpr": float(r_low["fpr"] - r_high["fpr"]),
                f"threshold_std_{low_label}": float(r_low["threshold_std"]),
                f"threshold_std_{high_label}": float(r_high["threshold_std"]),
                f"recall_std_{high_label}": float(r_high["recall_std"]),
                f"fpr_std_{high_label}": float(r_high["fpr_std"]),
            }
        )

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        mean_row: dict[str, float | int | str] = {"category": "mean"}
        for col in out_df.columns:
            if col == "category":
                continue
            mean_row[col] = float(out_df[col].mean())
        out_df = pd.concat([out_df, pd.DataFrame([mean_row])], ignore_index=True)
    return out_df
