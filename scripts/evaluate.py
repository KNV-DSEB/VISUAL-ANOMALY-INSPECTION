"""Unified evaluation harness for E-01, E-02, E-03, and L-01."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_protocol
from src.evaluation import (
    breakdown_by_defect_type,
    evaluate_predictions,
    find_error_cases,
    load_prediction_bundle,
)
from src.utils import ensure_dir, resolve_path


def evaluate_runs(
    artifacts_dir: Path,
    results_dir: Path,
    protocol: dict,
    runs: list[str] | None = None,
    compute_pro: bool = True,
) -> dict[str, pd.DataFrame]:
    """Evaluate all available prediction bundles and export comparison & localization tables."""
    pred_root = artifacts_dir / "predictions"
    if not pred_root.exists():
        raise FileNotFoundError(f"Predictions directory not found: {pred_root}")

    if runs:
        run_names = runs
    else:
        run_names = sorted(
            p.name for p in pred_root.iterdir() if (p / "predictions.csv").exists()
        )
    if not run_names:
        raise ValueError(f"No prediction bundles found inside {pred_root}")

    metrics_dir = ensure_dir(results_dir / "metrics")
    threshold_cfg = protocol.get("threshold", {})

    main_rows: list[dict] = []
    loc_tables: list[pd.DataFrame] = []

    for run_name in run_names:
        bundle_dir = pred_root / run_name
        bundle = load_prediction_bundle(bundle_dir)
        meta = bundle.get("metadata", {})
        method_name = meta.get("method", run_name)

        summary_df, thresholds = evaluate_predictions(
            bundle["df"],
            anomaly_maps=bundle["anomaly_maps"],
            gt_masks=bundle["gt_masks"],
            threshold_cfg=threshold_cfg,
            compute_pro=compute_pro,
        )
        summary_df.insert(0, "run_name", run_name)
        summary_df.insert(1, "method", method_name)
        summary_df.to_csv(metrics_dir / f"{run_name}_per_category.csv", index=False)

        breakdown_df = breakdown_by_defect_type(bundle["df"], thresholds)
        breakdown_df.insert(0, "run_name", run_name)
        breakdown_df.to_csv(metrics_dir / f"{run_name}_defect_breakdown.csv", index=False)

        errors_df = find_error_cases(bundle["df"], thresholds, top_k=5)
        errors_df.insert(0, "run_name", run_name)
        errors_df.to_csv(metrics_dir / f"{run_name}_error_cases.csv", index=False)

        mean_row = summary_df[summary_df["category"] == "mean"].iloc[0]
        main_rows.append(
            {
                "run_name": run_name,
                "method": method_name,
                "image_auroc": float(mean_row["image_auroc"]),
                "f1": float(mean_row["f1"]),
                "precision": float(mean_row["precision"]),
                "recall": float(mean_row["recall"]),
                "fpr": float(mean_row["fpr"]),
                "accuracy": float(mean_row["accuracy"]),
                "pixel_auroc": float(mean_row["pixel_auroc"]),
                "pixel_pro": float(mean_row["pixel_pro"]),
                "infer_ms_per_image": meta.get("mean_infer_ms_per_image", float("nan")),
                "artifact_size_mb": meta.get("mean_artifact_size_mb", float("nan")),
                "total_fit_seconds": meta.get("total_fit_seconds", float("nan")),
            }
        )

        if bundle["anomaly_maps"] is not None:
            loc_sub = summary_df[
                ["run_name", "method", "category", "pixel_auroc", "pixel_pro"]
            ].copy()
            loc_tables.append(loc_sub)

    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(metrics_dir / "main_comparison_table.csv", index=False)

    if loc_tables:
        loc_df = pd.concat(loc_tables, ignore_index=True)
        loc_df.to_csv(metrics_dir / "pixel_localization_table.csv", index=False)
    else:
        loc_df = pd.DataFrame()

    print(f"Saved main comparison table to {metrics_dir / 'main_comparison_table.csv'}")
    return {"main_comparison": main_df, "pixel_localization": loc_df}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved prediction bundles and generate summary tables."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts",
        help="Directory containing predictions/<run_name>/ bundles.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Output directory for CSV metric tables.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "configs" / "protocol.yaml",
        help="Path to shared protocol.yaml.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional list of run names to evaluate (defaults to all runs in artifacts/predictions).",
    )
    parser.add_argument(
        "--no-pro",
        action="store_true",
        help="Disable PRO calculation if fast pixel AUROC-only check is desired.",
    )
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    out = evaluate_runs(
        artifacts_dir=resolve_path(args.artifacts_dir),
        results_dir=resolve_path(args.results_dir),
        protocol=protocol,
        runs=args.runs,
        compute_pro=not args.no_pro,
    )
    print(out["main_comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
