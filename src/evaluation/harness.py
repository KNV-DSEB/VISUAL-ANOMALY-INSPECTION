"""Common evaluation harness: one implementation for all three methods."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .metrics import image_auroc, pixel_auroc, pro_score, threshold_metrics
from .schema import Predictions
from .thresholds import percentile_threshold

METRIC_COLUMNS = [
    "image_auroc", "precision", "recall", "f1", "false_alarm_rate",
    "pixel_auroc", "pro",
]
ROW_COLUMNS = (
    ["method", "category", "n_test", "n_good", "n_defect", "threshold"]
    + METRIC_COLUMNS + ["tp", "fp", "fn", "tn"]
)


def evaluate_category(val: Predictions, test: Predictions, cfg: dict | None = None) -> dict:
    """Metrics for one (method, category). Threshold is fit on val only."""
    cfg = cfg or {}
    if val.method != test.method or val.category != test.category:
        raise ValueError("val and test predictions must be the same method/category")
    if test.split != "test":
        raise ValueError("second argument must be the test split")

    thr = percentile_threshold(val, cfg.get("threshold_percentile", 95.0))
    tm = threshold_metrics(test.labels, test.scores, thr)
    fa = tm["fp"] / (tm["fp"] + tm["tn"]) if tm["fp"] + tm["tn"] else float("nan")

    px = pro = float("nan")
    if test.has_maps and cfg.get("pixel_metrics", True):
        px = pixel_auroc(test.anomaly_maps, test.gt_masks,
                         max_pixels=cfg.get("pixel_auroc_max_pixels"),
                         seed=cfg.get("seed", 42))
        if cfg.get("compute_pro", True):
            pro = pro_score(test.anomaly_maps, test.gt_masks,
                            fpr_limit=cfg.get("pro_fpr_limit", 0.3))

    return dict(
        method=test.method, category=test.category,
        n_test=len(test.scores), n_good=int((test.labels == 0).sum()),
        n_defect=int((test.labels == 1).sum()), threshold=thr,
        image_auroc=image_auroc(test.labels, test.scores),
        precision=tm["precision"], recall=tm["recall"], f1=tm["f1"],
        false_alarm_rate=fa, pixel_auroc=px, pro=pro,
        tp=tm["tp"], fp=tm["fp"], fn=tm["fn"], tn=tm["tn"],
    )


def overall_row(rows: list[dict]) -> dict:
    """Macro average over categories (every category weighs the same)."""
    out = dict(method=rows[0]["method"], category="overall",
               n_test=sum(r["n_test"] for r in rows),
               n_good=sum(r["n_good"] for r in rows),
               n_defect=sum(r["n_defect"] for r in rows), threshold=float("nan"))
    for c in METRIC_COLUMNS:
        vals = [r[c] for r in rows if not np.isnan(r[c])]
        out[c] = float(np.mean(vals)) if vals else float("nan")
    for c in ("tp", "fp", "fn", "tn"):
        out[c] = sum(r[c] for r in rows)
    return out


def evaluate_method(pred_dir, method: str, categories: list[str], cfg: dict | None = None):
    """Per-category rows + overall row for one method. Missing files raise."""
    rows = []
    for cat in categories:
        val = Predictions.load(pred_dir, method, cat, "val")
        test = Predictions.load(pred_dir, method, cat, "test")
        rows.append(evaluate_category(val, test, cfg))
    return rows + [overall_row(rows)]


def write_table(rows: list[dict], out_csv, out_json=None) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    if out_json:
        Path(out_json).write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
