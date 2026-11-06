"""I-01: sensitivity to the number of normal training images, and false-alarm cost analysis.

    python scripts/run_sensitivity.py                  # train with fewer normal images, then cost analysis
    python scripts/run_sensitivity.py --cost-only      # only the false-alarm table, from saved predictions

Sensitivity reuses the shared runner and harness: val and test stay fixed, only train/good is capped.
The cost analysis tabulates operating points at several validation percentiles. It is descriptive:
the final operating point must not be chosen by minimising a cost on the official test set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import build_manifest, load_protocol  # noqa: E402
from src.evaluation import Predictions, analyze_false_alarm_cost, evaluate_method  # noqa: E402
from src.methods.runner import run_method  # noqa: E402
from src.utils import ensure_dir, load_yaml, plot_cost_tradeoff_curve  # noqa: E402


def plot_sensitivity_curve(df: pd.DataFrame, output_path) -> Path:
    """Mean image AUROC and PRO versus the number of normal training images (overall rows)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    overall = df[df["category"] == "overall"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for method, g in overall.groupby("method", sort=True):
        g = g.sort_values("n_normal_sort")
        ax1.plot(g["sample_label"], g["image_auroc"], marker="o", label=method)
        if g["pro"].notna().any():
            ax2.plot(g["sample_label"], g["pro"], marker="s", label=method)
    for ax, title, ylab in ((ax1, "Image AUROC", "mean image AUROC"), (ax2, "PRO", "mean PRO")):
        ax.set_title(f"{title} vs. number of normal training images")
        ax.set_xlabel("normal training images per category")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run_sensitivity(protocol: dict, exp_cfg: dict, sens_pred_dir, methods: list[str],
                    categories: list[str] | None, device: str,
                    method_cfgs: dict[str, dict] | None = None) -> pd.DataFrame:
    sens = exp_cfg.get("sensitivity", {})
    counts = sens.get("sample_counts", [10, 25, 50, 100, None])
    cats = categories or list(protocol["dataset"]["categories"])
    eval_cfg = {"threshold_percentile": float(protocol["threshold"]["percentile"]),
                "seed": int(protocol["split"]["seed"])}
    manifest = build_manifest(protocol)
    frames = []
    for method in methods:
        cfg_path = f"configs/{method}.yaml"
        cfg = (method_cfgs or {}).get(method) or load_yaml(cfg_path)
        for count in counts:
            label = "full" if count is None else str(count)
            pred_dir = Path(sens_pred_dir) / f"n{label}"
            run_method(method, cfg, protocol, pred_dir=pred_dir, ckpt_dir=None, categories=cats,
                       max_train=count, device=device, config_path=cfg_path, manifest=manifest)
            rows = pd.DataFrame(evaluate_method(pred_dir, method, cats, eval_cfg))
            rows.insert(1, "sample_label", label)
            rows.insert(2, "n_normal_sort", 10**9 if count is None else int(count))
            frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def run_cost_analysis(pred_dir, protocol: dict, exp_cfg: dict, methods: list[str],
                      figures_dir) -> pd.DataFrame:
    cost = exp_cfg.get("false_alarm_cost", {})
    kwargs = {"cost_ratios": cost.get("cost_ratios", (1.0, 5.0, 10.0))}
    if "percentiles" in cost:
        kwargs["percentiles"] = cost["percentiles"]
    out_frames = []
    for method in methods:
        per_cat = []
        for cat in protocol["dataset"]["categories"]:
            val = Predictions.load(pred_dir, method, cat, "val")
            test = Predictions.load(pred_dir, method, cat, "test")
            per_cat.append(analyze_false_alarm_cost(val, test, **kwargs))
        per_cat = pd.concat(per_cat, ignore_index=True)
        mean = (per_cat.drop(columns=["category", "method"]).groupby("percentile", as_index=False).mean())
        mean.insert(0, "category", "mean")
        mean.insert(0, "method", method)
        out_frames += [per_cat, mean]
        plot_cost_tradeoff_curve(mean, Path(figures_dir) / f"false_alarm_cost_{method}.png",
                                 title=f"Operating points on the test set ({method}, mean over categories)")
    return pd.concat(out_frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--exp-config", default="configs/experiments/few_shot.yaml")
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--pred-dir", default="results/predictions", help="full runs, for the cost analysis")
    ap.add_argument("--sens-pred-dir", default="results/predictions_sensitivity")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--categories", nargs="*")
    ap.add_argument("--cost-only", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    protocol = load_protocol(args.protocol)
    exp_cfg = load_yaml(args.exp_config)
    methods = args.methods or exp_cfg.get("sensitivity", {}).get("methods", ["autoencoder", "patchcore", "oneclass"])
    metrics_dir = ensure_dir(Path(args.results_dir) / "metrics")
    figures_dir = ensure_dir(Path(args.results_dir) / "figures")

    if not args.cost_only:
        df = run_sensitivity(protocol, exp_cfg, args.sens_pred_dir, methods, args.categories, args.device)
        df.to_csv(metrics_dir / "sample_sensitivity.csv", index=False)
        plot_sensitivity_curve(df, figures_dir / "sample_sensitivity.png")
        print(f"Saved {metrics_dir / 'sample_sensitivity.csv'}")

    cost_df = run_cost_analysis(args.pred_dir, protocol, exp_cfg, methods, figures_dir)
    cost_df.to_csv(metrics_dir / "false_alarm_cost.csv", index=False)
    print(f"Saved {metrics_dir / 'false_alarm_cost.csv'}")


if __name__ == "__main__":
    main()
