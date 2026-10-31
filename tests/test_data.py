import pytest
import torch

from src.data.mvtec import MVTecDataset, build_manifest, load_protocol, resolve_root

protocol = load_protocol()
pytestmark = pytest.mark.skipif(
    not resolve_root(protocol).is_dir(), reason="MVTec AD data not downloaded"
)


@pytest.fixture(scope="module")
def manifest():
    return build_manifest(protocol)


def test_split_is_normal_only_and_disjoint(manifest):
    trainval = manifest[manifest["split"].isin(["train", "val"])]
    assert (trainval["label"] == 0).all()
    assert trainval["path"].is_unique
    assert not set(manifest[manifest.split == "train"].path) & set(manifest[manifest.split == "val"].path)


def test_val_fraction_and_determinism(manifest):
    again = build_manifest(protocol)
    assert manifest.equals(again)
    frac = protocol["split"]["val_fraction"]
    for cat, g in manifest[manifest.split != "test"].groupby("category"):
        assert abs((g.split == "val").sum() - round(len(g) * frac)) <= 1


def test_defect_images_have_masks(manifest):
    defects = manifest[(manifest.split == "test") & (manifest.label == 1)]
    assert len(defects) > 0
    assert (defects["mask_path"] != "").all()


def test_sample_shapes_and_values():
    cat = protocol["dataset"]["categories"][0]
    size = protocol["preprocessing"]["crop"]
    ds = MVTecDataset(protocol, "test", [cat])
    labels = {ds[i]["label"] for i in range(len(ds))}
    assert labels == {0, 1}
    item = next(ds[i] for i in range(len(ds)) if ds[i]["label"] == 1)
    assert item["image"].shape == (3, size, size)
    assert item["mask"].shape == (1, size, size)
    assert set(torch.unique(item["mask"]).tolist()) <= {0.0, 1.0}
    assert item["mask"].sum() > 0
    normal = next(ds[i] for i in range(len(ds)) if ds[i]["label"] == 0)
    assert normal["mask"].sum() == 0


def test_invalid_split_raises():
    with pytest.raises(ValueError):
        MVTecDataset(protocol, "validation")
