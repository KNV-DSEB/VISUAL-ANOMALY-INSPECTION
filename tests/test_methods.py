"""Autoencoder / PatchCore / One-class behind the shared interface, plus the runner.

No MVTec download and no pretrained weights are needed (tiny backbone / random weights).
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.data import subsample_train_manifest
from src.evaluation import Predictions, evaluate_category, percentile_threshold
from src.methods import build_method, load_method, save_method
from src.methods.runner import run_method, to_predictions
from src.utils import set_seed

CFGS = {
    "autoencoder": {
        "model": {"in_channels": 3, "base_channels": 8, "latent_channels": 16, "image_size": 64},
        "training": {"epochs": 2, "batch_size": 4, "lr": 1e-3, "l1_weight": 0.5, "seed": 42},
        "scoring": {"gaussian_sigma": 1.0, "image_score_reduction": "topk_mean", "topk_fraction": 0.05},
    },
    "patchcore": {
        "backbone": {"name": "tiny_test", "layers": ["layer2", "layer3"], "pretrained": False},
        "patch": {"patch_size": 3, "proj_dim": 32, "seed": 42},
        "coreset": {"ratio": 0.25, "max_patches": 64, "seed": 42},
        "scoring": {"n_neighbors": 3, "gaussian_sigma": 1.0, "batch_size": 4},
    },
    "oneclass": {
        "seed": 42,
        "features": {"backbone": "resnet18", "weights": None, "layers": ["layer2", "layer3"],
                     "batch_size": 4},
        "model": {"l2_normalize": True, "standardize": True, "pca_components": None,
                  "ocsvm": {"kernel": "rbf", "nu": 0.1, "gamma": "scale"}},
    },
}


class _Synthetic(Dataset):
    """64x64 tensors with the same dict items as MVTecDataset."""

    def __init__(self, split: str, category: str = "bottle", size: int = 64):
        gen = torch.Generator().manual_seed(99 if split == "test" else 42)
        n_normal, n_defect = {"train": (8, 0), "val": (4, 0), "test": (3, 3)}[split]
        self.items = []
        for i in range(n_normal + n_defect):
            img = torch.randn(3, size, size, generator=gen) * 0.1
            mask = torch.zeros(1, size, size)
            defect = i >= n_normal
            if defect:
                img[:, 16:40, 16:40] += 3.5
                mask[:, 16:40, 16:40] = 1.0
            kind = "broken_large" if defect else "good"
            self.items.append({"image": img, "mask": mask, "label": int(defect), "category": category,
                               "defect_type": kind, "path": f"{category}/{split}/{kind}/{i:03d}.png",
                               "split": split})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


@pytest.mark.parametrize("name", list(CFGS))
def test_fit_predict_checkpoint_roundtrip(name, tmp_path):
    set_seed(42)
    cfg = CFGS[name]
    loaders = {s: DataLoader(_Synthetic(s), batch_size=4) for s in ("train", "val", "test")}
    method = build_method(name, cfg, device="cpu")
    assert isinstance(method.fit(loaders["train"], loaders["val"]), dict)

    val = to_predictions(name, "bottle", "val", method.predict_dataloader(loaders["val"]), keep_maps=False)
    test = to_predictions(name, "bottle", "test", method.predict_dataloader(loaders["test"]),
                          keep_maps=method.supports_localization)
    assert test.has_maps == method.supports_localization and not val.has_maps
    row = evaluate_category(val, test, {"threshold_percentile": 95.0})
    assert np.isfinite(row["image_auroc"])
    if method.supports_localization:
        assert np.isfinite(row["pixel_auroc"]) and np.isfinite(row["pro"])
    else:
        assert np.isnan(row["pixel_auroc"])

    method.threshold_ = percentile_threshold(val, 95.0)
    save_method(method, tmp_path / "ckpt")
    # backbones are not stored in checkpoints (pinned pretrained weights are reloaded by name);
    # here weights=None means random init, so reseed to rebuild the same backbone
    set_seed(42)
    reloaded = load_method(name, tmp_path / "ckpt", device="cpu")
    img = _Synthetic("test")[0]["image"]
    s1, m1 = method.predict_single(img)
    s2, m2 = reloaded.predict_single(img)
    assert s1 == pytest.approx(s2, rel=1e-5, abs=1e-5)
    assert reloaded.threshold_ == pytest.approx(method.threshold_)
    if method.supports_localization:
        assert np.allclose(m1, m2, atol=1e-5)
    else:
        assert m1 is None and m2 is None


def test_load_method_rejects_wrong_method(tmp_path):
    method = build_method("oneclass", CFGS["oneclass"], device="cpu")
    method.fit(DataLoader(_Synthetic("train"), batch_size=4))
    save_method(method, tmp_path / "ckpt")
    with pytest.raises(ValueError, match="belongs to"):
        load_method("patchcore", tmp_path / "ckpt")
    with pytest.raises(ValueError, match="Unknown method"):
        build_method("yolo", {})


# --------------------------------------------------------------- runner
@pytest.mark.parametrize("name", ["patchcore", "autoencoder"])
def test_run_method_writes_schema_predictions_and_checkpoint(toy, name):
    protocol, tmp = toy
    (summary,) = run_method(name, CFGS[name], protocol, pred_dir=tmp / "pred", ckpt_dir=tmp / "ckpt",
                            device="cpu", config_path=f"configs/{name}.yaml")
    assert summary["category"] == "toy" and summary["n_test"] == 6

    val = Predictions.load(tmp / "pred", name, "toy", "val")
    test = Predictions.load(tmp / "pred", name, "toy", "test")
    assert (val.labels == 0).all() and not val.has_maps  # val maps are not stored
    assert test.has_maps and test.anomaly_maps.shape[1:] == (32, 32)
    assert test.meta["val_threshold"] == pytest.approx(percentile_threshold(val, 95.0))
    assert test.meta["config_path"] == f"configs/{name}.yaml"
    assert test.meta["seed"] == 42 and test.meta["n_train"] == 8
    assert str(tmp) not in json.dumps(test.meta)  # no personal absolute paths
    assert (tmp / "ckpt" / name / "toy" / "method_config.json").exists()
    assert load_method(name, tmp / "ckpt" / name / "toy", device="cpu").threshold_ == pytest.approx(
        test.meta["val_threshold"])


def test_run_method_is_reproducible(toy):
    protocol, tmp = toy
    run_method("autoencoder", CFGS["autoencoder"], protocol, pred_dir=tmp / "a", device="cpu")
    run_method("autoencoder", CFGS["autoencoder"], protocol, pred_dir=tmp / "b", device="cpu")
    a = Predictions.load(tmp / "a", "autoencoder", "toy", "test").scores
    b = Predictions.load(tmp / "b", "autoencoder", "toy", "test").scores
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_run_method_max_train_only_caps_train(toy):
    protocol, tmp = toy
    (s,) = run_method("oneclass", CFGS["oneclass"], protocol, pred_dir=tmp / "p", device="cpu", max_train=3)
    assert (s["n_train"], s["n_val"], s["n_test"]) == (3, 2, 6)


# --------------------------------------------------------------- data helper
def test_subsample_train_manifest_is_per_category_and_seeded():
    rows = [{"path": f"{c}/{s}/{i}.png", "category": c, "split": s, "label": 0}
            for c in ("a", "b") for s, n in (("train", 20), ("val", 5)) for i in range(n)]
    manifest = pd.DataFrame(rows)
    sub = subsample_train_manifest(manifest, 7, seed=42)
    assert sub[sub.split == "train"].groupby("category").size().to_dict() == {"a": 7, "b": 7}
    assert (sub.split == "val").sum() == 10
    assert sub["path"].tolist() == subsample_train_manifest(manifest, 7, seed=42)["path"].tolist()
    only_a = subsample_train_manifest(manifest, 7, seed=42, categories=["a"])
    assert (only_a[only_a.split == "train"].groupby("category").size().to_dict()) == {"a": 7, "b": 20}
    assert subsample_train_manifest(manifest, None) is manifest
