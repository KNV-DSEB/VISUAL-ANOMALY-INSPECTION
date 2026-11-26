"""Audit defect-mask truncation caused by Resize(256) + CenterCrop(224) preprocessing."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_protocol, resolve_root
from src.evaluation import audit_mask_crop_loss
from src.utils import ensure_dir, resolve_path


def download_ground_truth_masks(protocol: dict, root: Path, categories: list[str]) -> None:
    """Download only ground_truth mask folders from Hugging Face when missing locally."""
    missing = [c for c in categories if not (root / c / "ground_truth").is_dir()]
    if not missing:
        return
    from huggingface_hub import snapshot_download

    hf_repo = protocol["dataset"]["hf_repo"]
    patterns = [f"{c}/ground_truth/*" for c in missing]
    print(f"Downloading ground-truth masks for {missing} from {hf_repo} into {root}...")
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=hf_repo,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(root),
    )


def plot_crop_audit(cat_df: pd.DataFrame, output_path: Path) -> Path:
    """Plot per-category percentage of border-affected defect masks and mean area lost."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub = cat_df[cat_df["category"] != "overall"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    colors = ["#2563EB", "#1D4ED8", "#0284C7", "#D97706", "#059669"]

    bars1 = ax1.bar(sub["category"], sub["pct_affected_images"], color=colors[: len(sub)], alpha=0.9)
    ax1.set_title("Defect Masks Touching Cropped Border (%)")
    ax1.set_ylabel("% of Defect Images with Lost Pixels")
    ax1.set_ylim(0, max(15.0, float(sub["pct_affected_images"].max()) * 1.25))
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, sub["pct_affected_images"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.4, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5)

    bars2 = ax2.bar(sub["category"], sub["mean_area_loss_pct"], color=colors[: len(sub)], alpha=0.9)
    ax2.set_title("Mean Defect Mask Area Lost by CenterCrop (%)")
    ax2.set_ylabel("Mean % Mask Area Lost")
    ax2.set_ylim(0, max(5.0, float(sub["mean_area_loss_pct"].max()) * 1.25))
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, sub["mean_area_loss_pct"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.12, f"{val:.2f}%", ha="center", va="bottom", fontsize=8.5)

    fig.suptitle("Preprocessing Audit: Resize(256) + CenterCrop(224) on Ground-Truth Masks", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run_crop_audit(
    protocol: dict,
    results_dir: Path,
    categories: list[str] | None = None,
    download_masks_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit all ground-truth masks across selected categories."""
    root = resolve_root(protocol)
    cats = categories or list(protocol["dataset"]["categories"])
    if download_masks_only:
        download_ground_truth_masks(protocol, root, cats)

    pre = protocol["preprocessing"]
    resize_size = int(pre["resize"])
    crop_size = int(pre["crop"])

    records: list[dict] = []
    for cat in cats:
        gt_root = root / cat / "ground_truth"
        if not gt_root.is_dir():
            raise FileNotFoundError(
                f"Ground-truth directory not found: {gt_root}. "
                "Re-run with --download-masks-only or download MVTec AD first."
            )
        for defect_dir in sorted(p for p in gt_root.iterdir() if p.is_dir()):
            for mask_path in sorted(defect_dir.glob("*.png")):
                with Image.open(mask_path) as img:
                    stats = audit_mask_crop_loss(img, resize_size=resize_size, crop_size=crop_size)
                records.append(
                    {
                        "category": cat,
                        "defect_type": defect_dir.name,
                        "mask_path": mask_path.relative_to(root).as_posix(),
                        **stats,
                    }
                )

    raw_df = pd.DataFrame(records)
    if raw_df.empty:
        raise ValueError("No ground-truth masks found to audit.")

    def _summarize(grp: pd.DataFrame) -> dict[str, float | int]:
        affected = grp[grp["is_border_affected"]]
        return {
            "n_defect_images": int(len(grp)),
            "n_affected_images": int(grp["is_border_affected"].sum()),
            "pct_affected_images": float(100.0 * grp["is_border_affected"].mean()),
            "n_completely_lost": int(grp["is_completely_lost"].sum()),
            "mean_area_loss_pct": float(100.0 * grp["area_loss_fraction"].mean()),
            "max_area_loss_pct": float(100.0 * grp["area_loss_fraction"].max()),
            "mean_area_loss_among_affected_pct": float(100.0 * affected["area_loss_fraction"].mean())
            if not affected.empty
            else 0.0,
        }

    cat_rows = []
    for cat, grp in raw_df.groupby("category", sort=True):
        cat_rows.append({"category": str(cat), **_summarize(grp)})
    cat_rows.append({"category": "overall", **_summarize(raw_df)})
    cat_df = pd.DataFrame(cat_rows)

    defect_rows = []
    for (cat, defect), grp in raw_df.groupby(["category", "defect_type"], sort=True):
        defect_rows.append({"category": str(cat), "defect_type": str(defect), **_summarize(grp)})
    defect_df = pd.DataFrame(defect_rows)

    metrics_dir = ensure_dir(results_dir / "metrics")
    figures_dir = ensure_dir(results_dir / "figures")
    cat_df.to_csv(metrics_dir / "preprocessing_crop_audit.csv", index=False)
    defect_df.to_csv(metrics_dir / "preprocessing_crop_audit_by_defect.csv", index=False)
    plot_crop_audit(cat_df, figures_dir / "preprocessing_crop_audit.png")
    return cat_df, defect_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "configs" / "protocol.yaml")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument(
        "--download-masks-only",
        action="store_true",
        help="Download missing ground_truth mask folders from Hugging Face mirror.",
    )
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    cat_df, defect_df = run_crop_audit(
        protocol=protocol,
        results_dir=resolve_path(args.results_dir),
        categories=args.categories,
        download_masks_only=args.download_masks_only,
    )
    print("=== Preprocessing Crop Audit (Category Summary) ===")
    print(cat_df.to_string(index=False))
    affected_defects = defect_df[defect_df["n_affected_images"] > 0]
    if not affected_defects.empty:
        print("\n=== Defect Types Affected by CenterCrop ===")
        print(affected_defects.to_string(index=False))


if __name__ == "__main__":
    main()
