"""Run normal-sample sensitivity study and false-alarm cost analysis for I-01."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_and_predict import DEFAULT_METHOD_CONFIGS, run_single_method
from src.data import build_manifest, load_protocol
from src.evaluation import (
    analyze_false_alarm_cost,
    evaluate_predictions,
    load_prediction_bundle,
)
from src.utils import ensure_dir, load_yaml, plot_cost_tradeoff_curve, resolve_path


def plot_sensitivity_curve(sens_df: pd.DataFrame, output_path: Path) -> Path:
    """Plot mean Image AUROC and Pixel PRO vs. number of normal training samples."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mean_df = sens_df[sens_df["category"] == "mean"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for method, grp in mean_df.groupby("method", sort=True):
        grp = grp.sort_values("n_normal_sort")
        ax1.plot(grp["sample_label"], grp["image_auroc"], marker="o", linewidth=2, label=method)
        if grp["pixel_pro"].notna().any():
            ax2.plot(grp["sample_label"], grp["pixel_pro"], marker="s", linewidth=2, label=method)

    ax1.set_title("Image-Level AUROC vs. #Normal Training Samples")
    ax1.set_xlabel("Normal Training Samples per Category")
    ax1.set_ylabel("Mean Image AUROC")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_title("Pixel-Level PRO vs. #Normal Training Samples")
    ax2.set_xlabel("Normal Training Samples per Category")
    ax2.set_ylabel("Mean Pixel PRO")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("Sensitivity to Normal Training Sample Count (I-01)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run_cost_analysis_on_saved_bundles(
    artifacts_dir: Path,
    results_dir: Path,
    exp_cfg: dict,
    runs: list[str] | None = None,
) -> pd.DataFrame:
    """Compute false-alarm cost trade-off across validation percentiles on saved prediction bundles."""
    pred_root = artifacts_dir / "predictions"
    cost_cfg = exp_cfg.get("false_alarm_cost", {})
    percentiles = cost_cfg.get("percentiles", [80.0, 85.0, 90.0, 92.5, 95.0, 97.5, 99.0, 99.5])
    cost_ratios = cost_cfg.get("cost_ratios", [1.0, 5.0, 10.0])

    target_runs = runs or [
        p.name for p in sorted(pred_root.iterdir()) if (p / "predictions.csv").exists()
    ]
    all_cost_rows: list[pd.DataFrame] = []

    for r_name in target_runs:
        bundle = load_prediction_bundle(pred_root / r_name)
        df = bundle["df"]
        method_name = bundle["metadata"].get("method", r_name)

        cat_dfs: list[pd.DataFrame] = []
        for cat, grp in df.groupby("category", sort=True):
            val_scores = grp[grp["split"] == "val"]["score"].to_numpy()
            test_grp = grp[grp["split"] == "test"]
            c_df = analyze_false_alarm_cost(
                test_grp["label"].to_numpy(),
                test_grp["score"].to_numpy(),
                val_scores=val_scores,
                cost_ratios=cost_ratios,
                percentiles=percentiles,
            )
            c_df.insert(0, "category", str(cat))
            cat_dfs.append(c_df)

        combined_cat = pd.concat(cat_dfs, ignore_index=True)
        mean_cost = combined_cat.groupby("percentile", as_index=False).mean(numeric_only=True)
        mean_cost.insert(0, "category", "mean")
        run_cost_df = pd.concat([combined_cat, mean_cost], ignore_index=True)
        run_cost_df.insert(0, "run_name", r_name)
        run_cost_df.insert(1, "method", method_name)
        all_cost_rows.append(run_cost_df)

        fig_path = results_dir / "figures" / f"false_alarm_cost_{r_name}.png"
        plot_cost_tradeoff_curve(
            mean_cost,
            fig_path,
            title=f"False-Alarm Rate & Expected Cost Trade-off ({r_name})",
        )

    out_df = pd.concat(all_cost_rows, ignore_index=True) if all_cost_rows else pd.DataFrame()
    if not out_df.empty:
        metrics_dir = ensure_dir(results_dir / "metrics")
        out_df.to_csv(metrics_dir / "false_alarm_cost.csv", index=False)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sample-count sensitivity and false-alarm cost analysis (I-01)."
    )
    parser.add_argument(
        "--exp-config",
        type=Path,
        default=REPO_ROOT / "configs" / "experiments" / "few_shot.yaml",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "configs" / "protocol.yaml",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional category subset.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional method subset (defaults to methods in few_shot.yaml).",
    )
    parser.add_argument(
        "--cost-only",
        action="store_true",
        help="Only run false-alarm cost analysis on already saved prediction bundles.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    exp_cfg = load_yaml(args.exp_config)
    artifacts_dir = resolve_path(args.artifacts_dir)
    results_dir = resolve_path(args.results_dir)
    metrics_dir = ensure_dir(results_dir / "metrics")
    figures_dir = ensure_dir(results_dir / "figures")

    if not args.cost_only:
        sens_cfg = exp_cfg.get("sensitivity", {})
        methods = args.methods or sens_cfg.get("methods", ["autoencoder", "patchcore", "one_class"])
        sample_counts = sens_cfg.get("sample_counts", [10, 25, 50, 100, None])
        manifest = build_manifest(protocol)

        sens_rows: list[pd.DataFrame] = []
        for m_name in methods:
            m_cfg = load_yaml(DEFAULT_METHOD_CONFIGS[m_name])
            for count in sample_counts:
                label = str(count) if count is not None else "full"
                sort_val = int(count) if count is not None else 999999
                r_name = f"{m_name}_n{label}"
                res = run_single_method(
                    method_name=m_name,
                    protocol=protocol,
                    method_cfg=m_cfg,
                    run_name=r_name,
                    artifacts_dir=artifacts_dir / "sensitivity",
                    categories=args.categories,
                    max_train_samples=count,
                    device=args.device,
                    manifest=manifest,
                )
                summary_df, _ = evaluate_predictions(
                    res["df"],
                    anomaly_maps=res["anomaly_maps"],
                    gt_masks=res["gt_masks"],
                    threshold_cfg=protocol.get("threshold", {}),
                )
                summary_df.insert(0, "method", m_name)
                summary_df.insert(1, "sample_label", label)
                summary_df.insert(2, "n_normal_sort", sort_val)
                sens_rows.append(summary_df)

        sens_df = pd.concat(sens_rows, ignore_index=True)
        sens_df.to_csv(metrics_dir / "sample_sensitivity.csv", index=False)
        plot_sensitivity_curve(sens_df, figures_dir / "sample_sensitivity_curve.png")
        print(f"Saved sensitivity study to {metrics_dir / 'sample_sensitivity.csv'}")

    run_cost_analysis_on_saved_bundles(
        artifacts_dir=artifacts_dir,
        results_dir=results_dir,
        exp_cfg=exp_cfg,
    )


if __name__ == "__main__":
    main()
