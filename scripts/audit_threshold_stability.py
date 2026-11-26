"""Audit validation-threshold stability (bootstrap CI) and compare P90 vs P95 operating points."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import (
    bootstrap_threshold_stability,
    compare_operating_percentiles,
    load_prediction_bundle,
)
from src.utils import ensure_dir, resolve_path


def plot_threshold_stability(
    boot_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot P90 vs P95 Recall & FPR across categories (PatchCore) and across methods (mean)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # Panel A: PatchCore per-category P90 vs P95 Recall & FPR with 95% bootstrap CI
    pc_boot = boot_df[boot_df["run_name"] == "patchcore"]
    if not pc_boot.empty:
        cats = sorted(pc_boot["category"].unique())
        x = np.arange(len(cats))
        w = 0.2

        def _extract(pct: float, metric: str) -> tuple[np.ndarray, np.ndarray]:
            sub = pc_boot[pc_boot["percentile"] == pct].set_index("category").loc[cats]
            vals = sub[metric].to_numpy()
            low_err = np.maximum(0.0, vals - sub[f"{metric}_ci_low"].to_numpy())
            high_err = np.maximum(0.0, sub[f"{metric}_ci_high"].to_numpy() - vals)
            return vals, np.vstack([low_err, high_err])

        rec90, rec90_err = _extract(90.0, "recall")
        rec95, rec95_err = _extract(95.0, "recall")
        fpr90, fpr90_err = _extract(90.0, "fpr")
        fpr95, fpr95_err = _extract(95.0, "fpr")

        ax1.bar(x - 1.5 * w, rec90, width=w, yerr=rec90_err, capsize=2.5, label="Recall (P90)", color="#2563EB")
        ax1.bar(x - 0.5 * w, rec95, width=w, yerr=rec95_err, capsize=2.5, label="Recall (P95)", color="#93C5FD")
        ax1.bar(x + 0.5 * w, fpr90, width=w, yerr=fpr90_err, capsize=2.5, label="FPR (P90)", color="#DC2626")
        ax1.bar(x + 1.5 * w, fpr95, width=w, yerr=fpr95_err, capsize=2.5, label="FPR (P95)", color="#FCA5A5")
        ax1.set_xticks(x)
        ax1.set_xticklabels(cats)
        ax1.set_ylim(0.0, 1.12)
        ax1.set_ylabel("Rate (with 95% Val Bootstrap CI)")
        ax1.set_title("PatchCore: P90 vs P95 Recall & FPR by Category")
        ax1.grid(axis="y", alpha=0.3)
        ax1.legend(fontsize=7.5, ncol=2)

    # Panel B: Mean P90 vs P95 Recall & FPR across all evaluated runs
    mean_comp = comp_df[comp_df["category"] == "mean"].copy()
    if not mean_comp.empty:
        runs = mean_comp["run_name"].tolist()
        x2 = np.arange(len(runs))
        w2 = 0.2
        ax2.bar(x2 - 1.5 * w2, mean_comp["recall_p90"], width=w2, label="Recall (P90)", color="#2563EB")
        ax2.bar(x2 - 0.5 * w2, mean_comp["recall_p95"], width=w2, label="Recall (P95)", color="#93C5FD")
        ax2.bar(x2 + 0.5 * w2, mean_comp["fpr_p90"], width=w2, label="FPR (P90)", color="#DC2626")
        ax2.bar(x2 + 1.5 * w2, mean_comp["fpr_p95"], width=w2, label="FPR (P95)", color="#FCA5A5")
        ax2.set_xticks(x2)
        ax2.set_xticklabels(runs, rotation=12, ha="right")
        ax2.set_ylim(0.0, 1.08)
        ax2.set_ylabel("Mean Rate across 5 Categories")
        ax2.set_title("Cross-Method Mean Trade-off: P90 vs P95")
        ax2.grid(axis="y", alpha=0.3)
        ax2.legend(fontsize=7.5, ncol=2)

    fig.suptitle("Validation Threshold Stability & Operating-Point Comparison (P90 vs P95)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run_threshold_audit(
    artifacts_dir: Path,
    results_dir: Path,
    runs: list[str] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run bootstrap stability audit and P90-vs-P95 comparison across prediction bundles."""
    pred_root = artifacts_dir / "predictions"
    if not pred_root.exists():
        raise FileNotFoundError(f"Predictions directory not found: {pred_root}")

    run_names = runs or sorted(p.name for p in pred_root.iterdir() if (p / "predictions.csv").exists())
    if not run_names:
        raise ValueError(f"No prediction bundles found in {pred_root}")

    boot_rows: list[pd.DataFrame] = []
    comp_rows: list[pd.DataFrame] = []

    for r_name in run_names:
        bundle = load_prediction_bundle(pred_root / r_name)
        df = bundle["df"]
        method_name = bundle["metadata"].get("method", r_name)

        for cat, grp in df.groupby("category", sort=True):
            val_scores = grp[grp["split"] == "val"]["score"].to_numpy()
            test_grp = grp[grp["split"] == "test"]
            bdf = bootstrap_threshold_stability(
                val_scores=val_scores,
                test_labels=test_grp["label"].to_numpy(),
                test_scores=test_grp["score"].to_numpy(),
                percentiles=(90.0, 95.0),
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            bdf.insert(0, "run_name", r_name)
            bdf.insert(1, "method", method_name)
            bdf.insert(2, "category", str(cat))
            boot_rows.append(bdf)

        cdf = compare_operating_percentiles(
            df,
            low_p=90.0,
            high_p=95.0,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        cdf.insert(0, "run_name", r_name)
        cdf.insert(1, "method", method_name)
        comp_rows.append(cdf)

    boot_df = pd.concat(boot_rows, ignore_index=True)
    comp_df = pd.concat(comp_rows, ignore_index=True)

    metrics_dir = ensure_dir(results_dir / "metrics")
    figures_dir = ensure_dir(results_dir / "figures")
    boot_df.to_csv(metrics_dir / "threshold_stability_bootstrap.csv", index=False)
    comp_df.to_csv(metrics_dir / "p90_vs_p95_comparison.csv", index=False)
    plot_threshold_stability(boot_df, comp_df, figures_dir / "threshold_stability_p90_p95.png")
    return boot_df, comp_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    boot_df, comp_df = run_threshold_audit(
        artifacts_dir=resolve_path(args.artifacts_dir),
        results_dir=resolve_path(args.results_dir),
        runs=args.runs,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print("=== P90 vs P95 Operating Point Comparison (Mean across 5 Categories) ===")
    print(comp_df[comp_df["category"] == "mean"].to_string(index=False))
    print("\n=== PatchCore Per-Category P90 vs P95 & Bootstrap Stability ===")
    print(comp_df[comp_df["run_name"] == "patchcore"].to_string(index=False))
    print("\n=== PatchCore Bootstrap 95% CI Details ===")
    cols = ["category", "percentile", "n_val", "threshold", "threshold_std", "recall", "recall_ci_low", "recall_ci_high", "fpr", "fpr_ci_low", "fpr_ci_high"]
    print(boot_df[boot_df["run_name"] == "patchcore"][cols].to_string(index=False))


if __name__ == "__main__":
    main()
