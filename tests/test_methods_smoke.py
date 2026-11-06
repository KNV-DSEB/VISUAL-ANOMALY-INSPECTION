"""End-to-end smoke tests for Autoencoder, PatchCore, and One-Class methods (Chặng 2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation import evaluate_predictions, save_prediction_bundle
from src.methods import build_method
from src.utils import seed_everything


class _SyntheticMVTecDataset(Dataset):
    """Lightweight synthetic dataset mimicking MVTecDataset dict items (64x64)."""

    def __init__(self, split: str, category: str = "bottle", img_size: int = 64) -> None:
        self.split = split
        self.category = category
        self.img_size = img_size
        gen = torch.Generator().manual_seed(42 if split != "test" else 99)

        if split == "train":
            n_normal, n_defect = 8, 0
        elif split == "val":
            n_normal, n_defect = 4, 0
        else:
            n_normal, n_defect = 3, 3

        self.items: list[dict] = []
        for i in range(n_normal):
            img = torch.randn(3, img_size, img_size, generator=gen) * 0.1
            mask = torch.zeros(1, img_size, img_size, dtype=torch.float32)
            self.items.append(
                {
                    "image": img,
                    "mask": mask,
                    "label": 0,
                    "category": category,
                    "defect_type": "good",
                    "path": f"{category}/{split}/good/{i:03d}.png",
                    "split": split,
                }
            )
        for i in range(n_defect):
            img = torch.randn(3, img_size, img_size, generator=gen) * 0.1
            img[:, 16:40, 16:40] += 3.5  # Strong localized anomaly patch
            mask = torch.zeros(1, img_size, img_size, dtype=torch.float32)
            mask[:, 16:40, 16:40] = 1.0
            self.items.append(
                {
                    "image": img,
                    "mask": mask,
                    "label": 1,
                    "category": category,
                    "defect_type": "broken_large",
                    "path": f"{category}/{split}/broken_large/{i:03d}.png",
                    "split": split,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def _method_test_configs() -> dict[str, dict]:
    return {
        "autoencoder": {
            "model": {"in_channels": 3, "base_channels": 8, "latent_channels": 16, "image_size": 64},
            "training": {"epochs": 2, "batch_size": 4, "lr": 1e-3, "l1_weight": 0.5, "seed": 42},
            "scoring": {"gaussian_sigma": 1.0, "image_score_reduction": "topk_mean", "topk_fraction": 0.05},
        },
        "patchcore": {
            "backbone": {"name": "tiny_test", "layers": ["layer2", "layer3"], "pretrained": False},
            "patch": {"patch_size": 3, "proj_dim": 32, "seed": 42},
            "coreset": {"ratio": 0.25, "max_patches": 64, "seed": 42},
            "scoring": {"n_neighbors": 3, "gaussian_sigma": 1.0},
        },
        "one_class": {
            "backbone": {"name": "tiny_test", "layers": ["layer2", "layer3", "layer4"], "pretrained": False},
            "features": {"pooling": "avg_plus_max", "use_scaler": True, "pca_components": 4, "seed": 42},
            "classifier": {"kernel": "rbf", "nu": 0.1, "gamma": "scale"},
        },
    }


@pytest.mark.parametrize("method_name", ["autoencoder", "patchcore", "one_class"])
def test_method_fit_predict_save_load_and_evaluate(method_name: str, tmp_path: Path) -> None:
    seed_everything(42)
    configs = _method_test_configs()
    cfg = configs[method_name]

    train_loader = DataLoader(_SyntheticMVTecDataset("train"), batch_size=4, shuffle=False)
    val_loader = DataLoader(_SyntheticMVTecDataset("val"), batch_size=4, shuffle=False)
    test_loader = DataLoader(_SyntheticMVTecDataset("test"), batch_size=4, shuffle=False)

    method = build_method(method_name, cfg, device="cpu")
    fit_info = method.fit(train_loader, val_loader)
    assert isinstance(fit_info, dict)

    val_pred = method.predict_dataloader(val_loader)
    test_pred = method.predict_dataloader(test_loader)

    combined_df = pd.concat([val_pred["df"], test_pred["df"]], ignore_index=True)
    if method.supports_localization:
        assert val_pred["anomaly_maps"] is not None
        assert test_pred["anomaly_maps"] is not None
        combined_maps = np.concatenate([val_pred["anomaly_maps"], test_pred["anomaly_maps"]], axis=0)
        combined_masks = np.concatenate([val_pred["gt_masks"], test_pred["gt_masks"]], axis=0)
    else:
        assert val_pred["anomaly_maps"] is None
        assert test_pred["anomaly_maps"] is None
        combined_maps = None
        combined_masks = None

    bundle_dir = save_prediction_bundle(
        tmp_path / f"preds_{method_name}",
        df=combined_df,
        anomaly_maps=combined_maps,
        gt_masks=combined_masks,
        metadata={"method": method_name},
    )
    assert (bundle_dir / "predictions.csv").exists()

    summary_df, thresholds = evaluate_predictions(
        combined_df,
        anomaly_maps=combined_maps,
        gt_masks=combined_masks,
        threshold_cfg={"percentile": 95.0},
    )
    assert "bottle" in thresholds
    mean_row = summary_df[summary_df["category"] == "mean"].iloc[0]
    assert np.isfinite(mean_row["image_auroc"])
    if method.supports_localization:
        assert np.isfinite(mean_row["pixel_auroc"])
        assert np.isfinite(mean_row["pixel_pro"])
    else:
        assert np.isnan(mean_row["pixel_auroc"])

    # Verify save() and load() round-trip consistency
    ckpt_dir = method.save(tmp_path / f"ckpt_{method_name}")
    reloaded = build_method(method_name, cfg, device="cpu")
    reloaded.load(ckpt_dir)

    sample_img = _SyntheticMVTecDataset("test")[0]["image"]
    score_1, map_1 = method.predict_single(sample_img)
    score_2, map_2 = reloaded.predict_single(sample_img)
    assert score_1 == pytest.approx(score_2, rel=1e-5, abs=1e-5)
    if method.supports_localization:
        assert map_1 is not None and map_2 is not None
        assert np.allclose(map_1, map_2, atol=1e-5)
    else:
        assert map_1 is None and map_2 is None
