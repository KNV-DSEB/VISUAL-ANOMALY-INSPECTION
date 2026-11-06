"""One-class method tests. No MVTec download and no pretrained weights needed."""
import copy

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.mvtec import build_manifest, load_protocol
from src.evaluation import Predictions, evaluate_category, image_auroc
from src.methods.one_class import FeatureExtractor, OneClassDetector, extract_features
from src.methods.one_class.pipeline import METHOD, run, subsample_train


# ---------------------------------------------------------------- detector
def _blobs(seed=0, n_train=200, dim=16):
    rng = np.random.default_rng(seed)
    train = rng.normal(0, 1, (n_train, dim))
    inliers = rng.normal(0, 1, (50, dim))
    outliers = rng.normal(6, 1, (50, dim))
    return train, inliers, outliers


@pytest.mark.parametrize("pca", [None, 8, 0.9])
def test_outliers_score_higher(pca):
    train, inliers, outliers = _blobs()
    det = OneClassDetector(l2_normalize=False, pca_components=pca).fit(train)
    scores = det.score(np.vstack([inliers, outliers]))
    labels = [0] * len(inliers) + [1] * len(outliers)
    assert image_auroc(labels, scores) == 1.0  # higher == more anomalous


def test_nu_bounds_training_outlier_fraction():
    train, _, _ = _blobs(n_train=500)
    det = OneClassDetector(nu=0.1, l2_normalize=False).fit(train)
    frac = (det.score(train) > 0).mean()  # score > 0 <=> outside the learned boundary
    assert frac <= 0.1 + 0.05


def test_detector_is_deterministic():
    train, inliers, _ = _blobs()
    a = OneClassDetector(pca_components=8).fit(train).score(inliers)
    b = OneClassDetector(pca_components=8).fit(train).score(inliers)
    np.testing.assert_array_equal(a, b)


def test_from_config_and_guards():
    det = OneClassDetector.from_config(
        {"l2_normalize": False, "standardize": True, "pca_components": None,
         "ocsvm": {"kernel": "rbf", "nu": 0.2, "gamma": "scale"}})
    assert det.pipeline.named_steps["ocsvm"].nu == 0.2
    assert "l2" not in det.pipeline.named_steps
    with pytest.raises(RuntimeError):
        det.score(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        det.fit(np.zeros((1, 3)))


# ---------------------------------------------------------------- features
@pytest.mark.parametrize("layers,dim", [(["layer2", "layer3"], 128 + 256), (["layer4"], 512)])
def test_feature_extractor_shapes(layers, dim):
    ex = FeatureExtractor("resnet18", layers, weights=None)
    out = ex(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, dim)
    assert torch.isfinite(out).all()


def test_feature_extractor_stays_frozen():
    ex = FeatureExtractor("resnet18", ["layer2"], weights=None)
    ex.train()
    assert not ex.training
    assert not any(p.requires_grad for p in ex.parameters())
    x = torch.randn(3, 3, 64, 64)
    torch.testing.assert_close(ex(x), ex(x))


def test_feature_extractor_rejects_bad_layers():
    with pytest.raises(ValueError):
        FeatureExtractor("resnet18", ["layer5"], weights=None)
    with pytest.raises(ValueError):
        FeatureExtractor("resnet18", [], weights=None)


def test_extract_features_keeps_dataset_order():
    data = [{"image": torch.full((3, 32, 32), float(i)), "label": i % 2,
             "path": f"img{i}.png", "defect_type": "good" if i % 2 == 0 else "crack"}
            for i in range(5)]
    out = extract_features(FeatureExtractor("resnet18", ["layer1"], weights=None),
                           data, torch.device("cpu"), batch_size=2)
    assert out["features"].shape == (5, 64)
    assert out["paths"].tolist() == [f"img{i}.png" for i in range(5)]
    assert out["labels"].tolist() == [0, 1, 0, 1, 0]


# ---------------------------------------------------------- end to end (toy)
def _toy_mvtec(root, n_train=10):
    rng = np.random.default_rng(0)

    def save(path, arr):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(path)

    def normal():
        return np.clip(rng.normal(128, 5, (40, 40, 3)), 0, 255).astype(np.uint8)

    for i in range(n_train):
        save(root / "toy" / "train" / "good" / f"{i:03d}.png", normal())
    for i in range(3):
        save(root / "toy" / "test" / "good" / f"{i:03d}.png", normal())
        img = normal()
        img[10:25, 10:25] = 255
        mask = np.zeros((40, 40), np.uint8)
        mask[10:25, 10:25] = 255
        save(root / "toy" / "test" / "crack" / f"{i:03d}.png", img)
        save(root / "toy" / "ground_truth" / "crack" / f"{i:03d}_mask.png", mask)


@pytest.fixture
def toy(tmp_path):
    _toy_mvtec(tmp_path / "mvtec")
    protocol = copy.deepcopy(load_protocol())
    protocol["dataset"]["root"] = str(tmp_path / "mvtec")
    protocol["dataset"]["categories"] = ["toy"]
    protocol["preprocessing"].update(resize=36, crop=32)
    cfg = {
        "pred_dir": str(tmp_path / "pred"), "checkpoint_dir": str(tmp_path / "ckpt"),
        "seed": 42, "device": "cpu", "config_path": "configs/oneclass.yaml",
        "features": {"backbone": "resnet18", "weights": None, "layers": ["layer2", "layer3"],
                     "batch_size": 4, "num_workers": 0},
        "model": {"l2_normalize": True, "standardize": True, "pca_components": None,
                  "ocsvm": {"kernel": "rbf", "nu": 0.1, "gamma": "scale"}},
        "train": {"max_train": None},
    }
    return protocol, cfg, tmp_path


def test_run_writes_schema_compliant_predictions(toy):
    protocol, cfg, tmp = toy
    (summary,) = run(cfg, protocol)
    assert summary == {"category": "toy", "n_train": 8, "feature_dim": 384, "n_val": 2, "n_test": 6}

    val = Predictions.load(cfg["pred_dir"], METHOD, "toy", "val")
    test = Predictions.load(cfg["pred_dir"], METHOD, "toy", "test")
    assert (val.labels == 0).all()
    assert sorted(test.labels.tolist()) == [0, 0, 0, 1, 1, 1]
    assert set(test.defect_types) == {"good", "crack"}
    assert not test.has_maps  # image-level method
    assert test.meta["backbone"] == "resnet18" and test.meta["n_train"] == 8
    assert not any(str(tmp) in str(v) for v in test.meta.values())  # no absolute paths
    assert (tmp / "ckpt" / "toy.joblib").exists()

    row = evaluate_category(val, test, {"threshold_percentile": 95})
    assert 0.0 <= row["image_auroc"] <= 1.0
    assert np.isnan(row["pixel_auroc"]) and np.isnan(row["pro"])


def test_run_is_reproducible(toy):
    protocol, cfg, _ = toy
    run(cfg, protocol)
    first = Predictions.load(cfg["pred_dir"], METHOD, "toy", "test").scores
    run(cfg, protocol)
    second = Predictions.load(cfg["pred_dir"], METHOD, "toy", "test").scores
    np.testing.assert_allclose(first, second, rtol=1e-6)


def test_max_train_subsamples_only_train_split(toy):
    protocol, cfg, _ = toy
    manifest = build_manifest(protocol)
    sub = subsample_train(manifest, "toy", 3, seed=42)
    counts = sub.groupby("split").size().to_dict()
    assert counts == {"train": 3, "val": 2, "test": 6}
    again = subsample_train(manifest, "toy", 3, seed=42)
    assert sub["path"].tolist() == again["path"].tolist()
    assert subsample_train(manifest, "toy", None, 42) is manifest
    assert len(subsample_train(manifest, "toy", 99, 42)) == len(manifest)
