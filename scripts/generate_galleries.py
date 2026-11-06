"""Generate qualitative localization galleries comparing Input, GT Mask, AE, and PatchCore (L-02, E-03)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import build_transforms, load_protocol, resolve_root
from src.evaluation import compute_category_thresholds, load_prediction_bundle
from src.utils import denormalize_image, ensure_dir, resolve_path, save_localization_gallery


def build_galleries_from_bundles(
    artifacts_dir: Path,
    results_dir: Path,
    protocol: dict,
    ae_run: str = "autoencoder",
    pc_run: str = "patchcore",
    max_samples_per_category: int = 4,
) -> list[Path]:
    """Generate per-category qualitative localization galleries including both defects and normals."""
    pred_root = artifacts_dir / "predictions"
    ae_bundle = load_prediction_bundle(pred_root / ae_run) if (pred_root / ae_run).exists() else None
    pc_bundle = load_prediction_bundle(pred_root / pc_run) if (pred_root / pc_run).exists() else None

    primary = pc_bundle or ae_bundle
    if primary is None:
        raise FileNotFoundError(
            f"Neither {ae_run!r} nor {pc_run!r} prediction bundle found in {pred_root}"
        )

    data_root = resolve_root(protocol)
    image_tf, _ = build_transforms(protocol)
    norm_cfg = protocol["preprocessing"]["normalize"]
    fig_dir = ensure_dir(results_dir / "figures")

    ae_map_lookup: dict[str, np.ndarray] = {}
    if ae_bundle and ae_bundle["anomaly_maps"] is not None:
        for idx, p in enumerate(ae_bundle["df"]["path"]):
            ae_map_lookup[str(p)] = ae_bundle["anomaly_maps"][idx]

    pc_map_lookup: dict[str, np.ndarray] = {}
    if pc_bundle and pc_bundle["anomaly_maps"] is not None:
        for idx, p in enumerate(pc_bundle["df"]["path"]):
            pc_map_lookup[str(p)] = pc_bundle["anomaly_maps"][idx]

    df = primary["df"]
    gt_masks = primary["gt_masks"]
    thresholds = compute_category_thresholds(
        df,
        rule=protocol["threshold"]["rule"],
        percentile=float(protocol["threshold"]["percentile"]),
    )

    saved_paths: list[Path] = []
    test_mask = df["split"] == "test"
    test_indices = np.flatnonzero(test_mask.to_numpy())

    for cat in sorted(df.loc[test_mask, "category"].unique()):
        cat_indices = [i for i in test_indices if df.iloc[i]["category"] == cat]
        defect_indices = [i for i in cat_indices if df.iloc[i]["label"] == 1]
        normal_indices = [i for i in cat_indices if df.iloc[i]["label"] == 0]

        # Select top-scoring defect (clear hit), lowest-scoring defect (hard/missed case),
        # highest-scoring normal (false alarm candidate), and a clean normal.
        defect_sorted = sorted(defect_indices, key=lambda idx: df.iloc[idx]["score"], reverse=True)
        normal_sorted = sorted(normal_indices, key=lambda idx: df.iloc[idx]["score"], reverse=True)

        chosen: list[int] = []
        if defect_sorted:
            chosen.append(defect_sorted[0])
        if len(defect_sorted) > 1:
            chosen.append(defect_sorted[-1])
        if len(defect_sorted) > 2 and len(chosen) < max_samples_per_category - 1:
            chosen.append(defect_sorted[len(defect_sorted) // 2])
        if normal_sorted:
            chosen.append(normal_sorted[0])
        chosen = chosen[:max_samples_per_category]

        samples = []
        tau = thresholds[cat]
        for idx in chosen:
            row = df.iloc[idx]
            rel_path = str(row["path"])
            img_path = data_root / rel_path
            if img_path.exists():
                pil_img = Image.open(img_path).convert("RGB")
                img_tensor = image_tf(pil_img)
                rgb = denormalize_image(img_tensor, mean=norm_cfg["mean"], std=norm_cfg["std"])
            else:
                h, w = gt_masks[idx].shape[-2:] if gt_masks is not None else (224, 224)
                rgb = np.zeros((h, w, 3), dtype=np.uint8)

            gt = gt_masks[idx] if gt_masks is not None else np.zeros(rgb.shape[:2], dtype=np.uint8)
            verdict = "DEFECT" if row["score"] >= tau else "NORMAL"
            caption = f"{cat}/{row['defect_type']}\nGT={row['label']} | Pred={verdict}\n({Path(rel_path).name})"
            samples.append(
                {
                    "image_rgb": rgb,
                    "gt_mask": gt,
                    "ae_map": ae_map_lookup.get(rel_path),
                    "patchcore_map": pc_map_lookup.get(rel_path),
                    "caption": caption,
                }
            )

        out_path = fig_dir / f"gallery_{cat}.png"
        save_localization_gallery(
            samples,
            out_path,
            title=f"Localization Comparison — Category: {cat}",
        )
        saved_paths.append(out_path)
        print(f"Saved localization gallery to {out_path}")

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate localization heatmap galleries.")
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "configs" / "protocol.yaml")
    parser.add_argument("--ae-run", type=str, default="autoencoder")
    parser.add_argument("--pc-run", type=str, default="patchcore")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    build_galleries_from_bundles(
        artifacts_dir=resolve_path(args.artifacts_dir),
        results_dir=resolve_path(args.results_dir),
        protocol=protocol,
        ae_run=args.ae_run,
        pc_run=args.pc_run,
    )


if __name__ == "__main__":
    main()
