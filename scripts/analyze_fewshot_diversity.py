"""Analyze few-shot performance degradation alongside PatchCore memory-bank diversity & coverage."""
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

from src.utils import ensure_dir, resolve_path


def compute_memory_bank_diversity(memory_bank: np.ndarray, knn_k: int = 5, max_eval: int = 2500, seed: int = 42) -> dict[str, float | int]:
    """Compute geometric dispersion, effective rank, and local density radius of a coreset memory bank."""
    x = np.asarray(memory_bank, dtype=np.float32)
    n_patches, dim = x.shape
    centroid = x.mean(axis=0, keepdims=True)
    dists_to_centroid = np.linalg.norm(x - centroid, axis=1)
    mean_dispersion = float(dists_to_centroid.mean())

    # Effective rank from singular value spectrum of centered embeddings
    centered = (x - centroid).astype(np.float64)
    if n_patches > max_eval:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_patches, size=max_eval, replace=False)
        centered_sub = centered[idx]
        x_sub = x[idx]
    else:
        centered_sub = centered
        x_sub = x

    s = np.linalg.svd(centered_sub, compute_uv=False)
    eig = s**2
    eig_sum = float(eig.sum())
    if eig_sum > 0:
        p = eig / eig_sum
        p = p[p > 1e-12]
        entropy = float(-np.sum(p * np.log(p)))
        effective_rank = float(np.exp(entropy))
    else:
        effective_rank = 0.0

    # Mean k-NN radius within the memory bank (measures patch redundancy vs coverage sparsity)
    sq_norms = np.sum(x_sub**2, axis=1, keepdims=True)
    dist_sq = np.maximum(0.0, sq_norms + sq_norms.T - 2.0 * (x_sub @ x_sub.T))
    dists = np.sqrt(dist_sq)
    np.fill_diagonal(dists, np.inf)
    k_eff = min(knn_k, max(1, len(x_sub) - 1))
    knn_dists = np.partition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
    mean_knn_radius = float(knn_dists.mean())

    return {
        "n_coreset_patches": int(n_patches),
        "embed_dim": int(dim),
        "mean_patch_l2_dispersion": mean_dispersion,
        "effective_rank": effective_rank,
        "mean_knn_radius": mean_knn_radius,
    }


def plot_fewshot_category_degradation(
    sens_df: pd.DataFrame,
    div_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot per-category PatchCore few-shot curves and memory-bank diversity vs AUROC drop."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    pc_df = sens_df[(sens_df["method"] == "patchcore") & (sens_df["category"] != "mean")].copy()
    palette = {
        "bottle": "#2563EB",
        "carpet": "#059669",
        "leather": "#0284C7",
        "screw": "#DC2626",
        "transistor": "#D97706",
    }
    for cat, grp in pc_df.groupby("category", sort=True):
        grp = grp.sort_values("n_normal_sort")
        ax1.plot(
            grp["sample_label"],
            grp["image_auroc"],
            marker="o",
            linewidth=2.2 if cat == "screw" else 1.8,
            color=palette.get(str(cat), "#334155"),
            label=str(cat),
        )
    ax1.set_title("PatchCore Few-Shot Degradation by Category")
    ax1.set_xlabel("Normal Training Samples per Category (N)")
    ax1.set_ylabel("Test Image AUROC")
    ax1.set_ylim(0.50, 1.03)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8.5)

    # Panel B: Effective Rank / Mean k-NN Radius vs Few-Shot AUROC Drop (Full -> N=10)
    x_vals = div_df["mean_knn_radius"].to_numpy()
    y_vals = div_df["patchcore_auroc_drop_full_to_10"].to_numpy()
    for _, row in div_df.iterrows():
        c = palette.get(str(row["category"]), "#334155")
        ax2.scatter(row["mean_knn_radius"], row["patchcore_auroc_drop_full_to_10"], s=95, color=c, zorder=3)
        ax2.annotate(
            f"{row['category']} (eff_rank={row['effective_rank']:.0f})",
            (row["mean_knn_radius"], row["patchcore_auroc_drop_full_to_10"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8.2,
            color="#0F172A",
        )
    ax2.set_title("Memory-Bank Coverage Radius vs. Few-Shot AUROC Drop")
    ax2.set_xlabel("Mean Coreset 5-NN Radius (Patch Diversity)")
    ax2.set_ylabel("Image AUROC Drop (Full → N=10)")
    ax2.set_xlim(min(x_vals) * 0.90, max(x_vals) * 1.22)
    ax2.set_ylim(-0.02, max(y_vals) * 1.18)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Few-Shot Sensitivity Analysis: Reference Diversity vs. Patch Count",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run_fewshot_diversity_analysis(artifacts_dir: Path, results_dir: Path) -> pd.DataFrame:
    """Combine per-category few-shot degradation with PatchCore memory-bank diversity statistics."""
    sens_path = results_dir / "metrics" / "sample_sensitivity.csv"
    if not sens_path.exists():
        raise FileNotFoundError(f"Sample sensitivity CSV not found: {sens_path}")
    sens_df = pd.read_csv(sens_path)

    ckpt_root = artifacts_dir / "checkpoints" / "patchcore"
    categories = sorted(c for c in sens_df["category"].unique() if c != "mean")

    rows: list[dict[str, float | int | str]] = []
    for cat in categories:
        mb_path = ckpt_root / cat / "memory_bank.npz"
        if not mb_path.exists():
            raise FileNotFoundError(f"PatchCore memory bank not found: {mb_path}")
        npz = np.load(mb_path)
        key = "memory_bank" if "memory_bank" in npz.files else npz.files[0]
        div_stats = compute_memory_bank_diversity(npz[key])

        sub = sens_df[sens_df["category"] == cat]

        def _get(method: str, label: str, col: str) -> float:
            m = sub[(sub["method"] == method) & (sub["sample_label"] == str(label))]
            return float(m[col].iloc[0]) if not m.empty else float("nan")

        pc_n10 = _get("patchcore", "10", "image_auroc")
        pc_n25 = _get("patchcore", "25", "image_auroc")
        pc_n50 = _get("patchcore", "50", "image_auroc")
        pc_n100 = _get("patchcore", "100", "image_auroc")
        pc_full = _get("patchcore", "full", "image_auroc")

        pro_n10 = _get("patchcore", "10", "pixel_pro")
        pro_full = _get("patchcore", "full", "pixel_pro")

        oc_n10 = _get("one_class", "10", "image_auroc")
        oc_full = _get("one_class", "full", "image_auroc")

        ae_n10 = _get("autoencoder", "10", "image_auroc")
        ae_full = _get("autoencoder", "full", "image_auroc")

        rows.append(
            {
                "category": str(cat),
                "patchcore_auroc_n10": pc_n10,
                "patchcore_auroc_n25": pc_n25,
                "patchcore_auroc_n50": pc_n50,
                "patchcore_auroc_n100": pc_n100,
                "patchcore_auroc_full": pc_full,
                "patchcore_auroc_drop_full_to_10": float(pc_full - pc_n10),
                "patchcore_pro_n10": pro_n10,
                "patchcore_pro_full": pro_full,
                "patchcore_pro_drop_full_to_10": float(pro_full - pro_n10),
                "one_class_auroc_n10": oc_n10,
                "one_class_auroc_full": oc_full,
                "autoencoder_auroc_n10": ae_n10,
                "autoencoder_auroc_full": ae_full,
                **div_stats,
            }
        )

    out_df = pd.DataFrame(rows)
    metrics_dir = ensure_dir(results_dir / "metrics")
    figures_dir = ensure_dir(results_dir / "figures")
    out_df.to_csv(metrics_dir / "fewshot_diversity_analysis.csv", index=False)
    plot_fewshot_category_degradation(sens_df, out_df, figures_dir / "fewshot_category_degradation.png")
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()

    out_df = run_fewshot_diversity_analysis(
        artifacts_dir=resolve_path(args.artifacts_dir),
        results_dir=resolve_path(args.results_dir),
    )
    print("=== Few-Shot Degradation vs. Memory-Bank Diversity ===")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
