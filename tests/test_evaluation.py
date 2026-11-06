"""Unit tests for Chặng 1: Evaluation harness, PRO metric, threshold & utilities (E-01, L-01)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.mvtec import subsample_train_manifest
from src.evaluation import (
    analyze_false_alarm_cost,
    breakdown_by_defect_type,
    compute_image_metrics,
    compute_pixel_auroc,
    compute_pro_score,
    compute_validation_threshold,
    evaluate_predictions,
    find_error_cases,
    load_prediction_bundle,
    save_prediction_bundle,
    validate_predictions,
)
from src.utils import (
    FeatureExtractor,
    denormalize_image,
    overlay_heatmap,
    plot_cost_tradeoff_curve,
    save_localization_gallery,
    seed_everything,
)


def _make_toy_predictions() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = []
    for cat in ("bottle", "carpet"):
        for i in range(10):
            rows.append(
                {
                    "path": f"{cat}/train/good/{i:03d}.png",
                    "category": cat,
                    "split": "val",
                    "label": 0,
                    "defect_type": "good",
                    "score": 0.10 + 0.02 * i,
                }
            )
        for i in range(5):
            rows.append(
                {
                    "path": f"{cat}/test/good/{i:03d}.png",
                    "category": cat,
                    "split": "test",
                    "label": 0,
                    "defect_type": "good",
                    "score": 0.12 + 0.02 * i,
                }
            )
        for i in range(5):
            rows.append(
                {
                    "path": f"{cat}/test/broken/{i:03d}.png",
                    "category": cat,
                    "split": "test",
                    "label": 1,
                    "defect_type": "broken",
                    "score": 0.70 + 0.05 * i,
                }
            )
    df = pd.DataFrame(rows)
    n = len(df)
    gt_masks = np.zeros((n, 16, 16), dtype=np.uint8)
    anomaly_maps = np.full((n, 16, 16), 0.1, dtype=np.float32)

    for idx, row in df.iterrows():
        if row["label"] == 1:
            gt_masks[idx, 4:10, 4:10] = 1
            anomaly_maps[idx, 4:10, 4:10] = 0.9
    return df, anomaly_maps, gt_masks


def test_schema_validation_and_persistence(tmp_path: Path) -> None:
    df, amaps, masks = _make_toy_predictions()
    save_prediction_bundle(
        tmp_path / "bundle",
        df=df,
        anomaly_maps=amaps,
        gt_masks=masks,
        metadata={"method": "toy"},
    )
    loaded = load_prediction_bundle(tmp_path / "bundle")
    assert len(loaded["df"]) == len(df)
    assert loaded["anomaly_maps"].shape == amaps.shape
    assert loaded["gt_masks"].shape == masks.shape
    assert loaded["metadata"]["method"] == "toy"

    bad_val_df = df.copy()
    bad_val_df.loc[bad_val_df["split"] == "val", "label"] = 1
    with pytest.raises(ValueError, match="normal samples"):
        validate_predictions(bad_val_df)


def test_validation_threshold_and_image_metrics() -> None:
    val_scores = np.linspace(0.0, 1.0, 101)
    tau = compute_validation_threshold(val_scores, percentile=95.0)
    assert pytest.approx(tau, abs=1e-6) == 0.95

    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.99])
    metrics = compute_image_metrics(labels, scores, threshold=0.5)
    assert metrics["image_auroc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["fpr"] == pytest.approx(0.0)


def test_pixel_auroc_and_pro_region_weighting() -> None:
    gt = np.zeros((1, 20, 20), dtype=np.uint8)
    gt[0, 1:3, 1:3] = 1     # Small region (4 pixels)
    gt[0, 10:18, 10:18] = 1 # Large region (64 pixels)

    perfect_map = gt.astype(np.float32) * 0.9 + 0.05
    assert compute_pixel_auroc(gt, perfect_map) == pytest.approx(1.0)
    assert compute_pro_score(gt, perfect_map) > 0.99

    # Background has scores in [0.1, 0.3], large defect has 0.95, small defect is missed (0.0)
    rng = np.random.default_rng(42)
    missed_small_map = rng.uniform(0.1, 0.3, size=gt.shape).astype(np.float32)
    missed_small_map[0, 1:3, 1:3] = 0.0
    missed_small_map[0, 10:18, 10:18] = 0.95
    pix_auroc = compute_pixel_auroc(gt, missed_small_map)
    pro_score = compute_pro_score(gt, missed_small_map)
    assert pix_auroc > 0.93
    assert pro_score == pytest.approx(0.5, abs=0.02)


def test_evaluate_predictions_and_analysis() -> None:
    df, amaps, masks = _make_toy_predictions()
    summary_df, thresholds = evaluate_predictions(
        df, anomaly_maps=amaps, gt_masks=masks, threshold_cfg={"percentile": 95.0}
    )
    assert set(summary_df["category"]) == {"bottle", "carpet", "mean"}
    mean_row = summary_df[summary_df["category"] == "mean"].iloc[0]
    assert mean_row["image_auroc"] == pytest.approx(1.0)
    assert mean_row["pixel_auroc"] == pytest.approx(1.0)
    assert mean_row["pixel_pro"] > 0.98

    breakdown = breakdown_by_defect_type(df, thresholds)
    assert not breakdown.empty
    errors = find_error_cases(df, thresholds, top_k=2)
    assert not errors.empty


def test_false_alarm_cost_and_subsampling(tmp_path: Path) -> None:
    df, _, _ = _make_toy_predictions()
    val_scores = df[(df["category"] == "bottle") & (df["split"] == "val")]["score"].to_numpy()
    test_sub = df[(df["category"] == "bottle") & (df["split"] == "test")]
    cost_df = analyze_false_alarm_cost(
        test_sub["label"].to_numpy(),
        test_sub["score"].to_numpy(),
        val_scores=val_scores,
        cost_ratios=(1.0, 5.0, 10.0),
    )
    assert "cost_ratio_5" in cost_df.columns
    fig_path = plot_cost_tradeoff_curve(cost_df, tmp_path / "cost.png")
    assert fig_path.exists()

    manifest = pd.DataFrame(
        [
            {"path": f"bottle/train/good/{i}.png", "category": "bottle", "split": "train", "label": 0}
            for i in range(20)
        ]
        + [
            {"path": f"bottle/val/good/{i}.png", "category": "bottle", "split": "val", "label": 0}
            for i in range(5)
        ]
    )
    sub = subsample_train_manifest(manifest, max_train_samples=7, seed=42)
    assert (sub["split"] == "train").sum() == 7
    assert (sub["split"] == "val").sum() == 5


def test_tiny_backbone_and_visualization(tmp_path: Path) -> None:
    seed_everything(42)
    extractor = FeatureExtractor("tiny_test", layers=("layer2", "layer3"), pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    feats = extractor(x)
    assert feats["layer2"].shape == (2, 32, 8, 8)
    assert feats["layer3"].shape == (2, 64, 4, 4)

    rgb = denormalize_image(x[0])
    amap = np.random.rand(64, 64).astype(np.float32)
    blended = overlay_heatmap(rgb, amap)
    assert blended.shape == (64, 64, 3)

    out_gallery = save_localization_gallery(
        [
            {
                "image_rgb": rgb,
                "gt_mask": np.zeros((64, 64), dtype=np.uint8),
                "ae_map": amap,
                "patchcore_map": amap,
                "caption": "bottle / broken",
            }
        ],
        tmp_path / "gallery.png",
    )
    assert out_gallery.exists()
