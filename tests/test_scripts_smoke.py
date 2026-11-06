"""Smoke tests for Chặng 3 experiment & evaluation scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate import evaluate_runs
from scripts.generate_galleries import build_galleries_from_bundles
from scripts.run_sensitivity import plot_sensitivity_curve, run_cost_analysis_on_saved_bundles
from src.data import load_protocol
from src.evaluation import save_prediction_bundle
from src.utils import load_yaml


def _create_mock_bundles(artifacts_dir: Path) -> None:
    rows = []
    for cat in ("bottle", "carpet"):
        for i in range(8):
            rows.append(
                {
                    "path": f"{cat}/train/good/{i:03d}.png",
                    "category": cat,
                    "split": "val",
                    "label": 0,
                    "defect_type": "good",
                    "score": 0.1 + 0.02 * i,
                }
            )
        for i in range(4):
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
        for i in range(4):
            rows.append(
                {
                    "path": f"{cat}/test/crack/{i:03d}.png",
                    "category": cat,
                    "split": "test",
                    "label": 1,
                    "defect_type": "crack",
                    "score": 0.75 + 0.04 * i,
                }
            )
    df = pd.DataFrame(rows)
    n = len(df)
    masks = np.zeros((n, 16, 16), dtype=np.uint8)
    amaps = np.full((n, 16, 16), 0.1, dtype=np.float32)
    for idx, row in df.iterrows():
        if row["label"] == 1:
            masks[idx, 4:12, 4:12] = 1
            amaps[idx, 4:12, 4:12] = 0.9

    for method in ("autoencoder", "patchcore", "one_class"):
        has_map = method != "one_class"
        save_prediction_bundle(
            artifacts_dir / "predictions" / method,
            df=df,
            anomaly_maps=amaps if has_map else None,
            gt_masks=masks if has_map else None,
            metadata={
                "run_name": method,
                "method": method,
                "mean_infer_ms_per_image": 12.5,
                "mean_artifact_size_mb": 4.2,
                "total_fit_seconds": 15.0,
            },
        )


def test_evaluate_galleries_and_cost_scripts(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    results_dir = tmp_path / "results"
    _create_mock_bundles(artifacts_dir)

    protocol = load_protocol()
    eval_out = evaluate_runs(
        artifacts_dir=artifacts_dir,
        results_dir=results_dir,
        protocol=protocol,
    )
    main_df = eval_out["main_comparison"]
    loc_df = eval_out["pixel_localization"]
    assert set(main_df["method"]) == {"autoencoder", "patchcore", "one_class"}
    assert set(loc_df["method"]) == {"autoencoder", "patchcore"}
    assert (results_dir / "metrics" / "main_comparison_table.csv").exists()
    assert (results_dir / "metrics" / "pixel_localization_table.csv").exists()

    galleries = build_galleries_from_bundles(
        artifacts_dir=artifacts_dir,
        results_dir=results_dir,
        protocol=protocol,
        max_samples_per_category=2,
    )
    assert len(galleries) == 2
    assert all(p.exists() for p in galleries)

    exp_cfg = load_yaml("configs/experiments/few_shot.yaml")
    cost_df = run_cost_analysis_on_saved_bundles(
        artifacts_dir=artifacts_dir,
        results_dir=results_dir,
        exp_cfg=exp_cfg,
    )
    assert not cost_df.empty
    assert (results_dir / "metrics" / "false_alarm_cost.csv").exists()

    toy_sens = pd.DataFrame(
        [
            {"method": "patchcore", "sample_label": "10", "n_normal_sort": 10, "category": "mean", "image_auroc": 0.88, "pixel_pro": 0.82},
            {"method": "patchcore", "sample_label": "full", "n_normal_sort": 999, "category": "mean", "image_auroc": 0.98, "pixel_pro": 0.94},
            {"method": "autoencoder", "sample_label": "10", "n_normal_sort": 10, "category": "mean", "image_auroc": 0.65, "pixel_pro": 0.58},
            {"method": "autoencoder", "sample_label": "full", "n_normal_sort": 999, "category": "mean", "image_auroc": 0.84, "pixel_pro": 0.79},
        ]
    )
    sens_fig = plot_sensitivity_curve(toy_sens, results_dir / "figures" / "sens.png")
    assert sens_fig.exists()
