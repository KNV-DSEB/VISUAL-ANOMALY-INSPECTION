"""Error analysis (E-03): flag rate per defect type and the most confident errors per category.

    python scripts/analyze_errors.py --methods autoencoder patchcore oneclass

Writes results/metrics/defect_breakdown.csv and results/metrics/error_cases.csv. The val threshold
(percentile from configs/protocol.yaml) is applied exactly as in scripts/evaluate.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import load_protocol  # noqa: E402
from src.evaluation import Predictions, breakdown_by_defect_type, find_error_cases  # noqa: E402
from src.utils import ensure_dir  # noqa: E402


def analyze(pred_dir, out_dir, protocol: dict, methods: list[str], top_k: int = 5):
    percentile = float(protocol["threshold"]["percentile"])
    breakdown, errors = [], []
    for method in methods:
        for cat in protocol["dataset"]["categories"]:
            val = Predictions.load(pred_dir, method, cat, "val")
            test = Predictions.load(pred_dir, method, cat, "test")
            breakdown.append(breakdown_by_defect_type(val, test, percentile))
            errors.append(find_error_cases(val, test, percentile, top_k))
    out = ensure_dir(out_dir)
    b, e = pd.concat(breakdown, ignore_index=True), pd.concat(errors, ignore_index=True)
    b.to_csv(out / "defect_breakdown.csv", index=False)
    e.to_csv(out / "error_cases.csv", index=False)
    return b, e


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--methods", nargs="*", default=["autoencoder", "patchcore", "oneclass"])
    ap.add_argument("--pred-dir", default="results/predictions")
    ap.add_argument("--out-dir", default="results/metrics")
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    b, e = analyze(args.pred_dir, args.out_dir, load_protocol(args.protocol), args.methods, args.top_k)
    print(f"{len(b)} defect-type rows, {len(e)} error-case rows -> {args.out_dir}")


if __name__ == "__main__":
    main()
