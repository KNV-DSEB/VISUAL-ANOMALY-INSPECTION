"""MVTec AD manifest + dataset shared by all three methods.

Splits:
  train : train/good minus the validation portion (normal only)
  val   : val_fraction of train/good (normal only), chosen with a fixed seed
  test  : official test set (normal + defect); use only for final reporting
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

REPO_ROOT = Path(__file__).resolve().parents[2]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_protocol(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "protocol.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_root(protocol: dict) -> Path:
    root = Path(protocol["dataset"]["root"])
    return root if root.is_absolute() else REPO_ROOT / root


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def _mask_for(root: Path, category: str, defect: str, image: Path) -> Path | None:
    if defect == "good":
        return None
    mask = root / category / "ground_truth" / defect / f"{image.stem}_mask.png"
    return mask if mask.exists() else None


def build_manifest(protocol: dict) -> pd.DataFrame:
    """One row per image: path, mask_path, category, split, label, defect_type."""
    root = resolve_root(protocol)
    categories = protocol["dataset"]["categories"]
    val_fraction = protocol["split"]["val_fraction"]
    seed = protocol["split"]["seed"]
    rows = []
    for cat in categories:
        cat_dir = root / cat
        if not cat_dir.is_dir():
            raise FileNotFoundError(
                f"Category folder not found: {cat_dir}. Run scripts/download_mvtec.py first."
            )
        train = _images(cat_dir / "train" / "good")
        order = list(range(len(train)))
        random.Random(f"{seed}-{cat}").shuffle(order)
        n_val = round(len(train) * val_fraction)
        val_idx = set(order[:n_val])
        for i, img in enumerate(train):
            rows.append(_row(root, img, cat, "val" if i in val_idx else "train", "good"))
        for defect_dir in sorted(p for p in (cat_dir / "test").iterdir() if p.is_dir()):
            for img in _images(defect_dir):
                rows.append(_row(root, img, cat, "test", defect_dir.name))
    return pd.DataFrame(rows)


def _row(root: Path, img: Path, cat: str, split: str, defect: str) -> dict:
    mask = _mask_for(root, cat, defect, img)
    return {
        "path": img.relative_to(root).as_posix(),
        "mask_path": mask.relative_to(root).as_posix() if mask else "",
        "category": cat,
        "split": split,
        "label": 0 if defect == "good" else 1,
        "defect_type": defect,
    }


def subsample_train_manifest(
    manifest: pd.DataFrame,
    max_train_samples: int | None,
    seed: int = 42,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """Keep at most `max_train_samples` normal train images per category (few-shot study).

    Only rows with split == "train" are reduced; val and test stay untouched so results remain
    comparable across sample counts. Seeded and deterministic. Returns `manifest` itself when
    there is nothing to cap.
    """
    if max_train_samples is None or max_train_samples <= 0:
        return manifest
    keep = pd.Series(True, index=manifest.index)
    for cat in categories or list(manifest["category"].unique()):
        idx = list(manifest.index[(manifest["split"] == "train") & (manifest["category"] == cat)])
        if len(idx) > max_train_samples:
            chosen = set(random.Random(f"{seed}-subsample-{cat}-{max_train_samples}")
                         .sample(idx, max_train_samples))
            keep.loc[[i for i in idx if i not in chosen]] = False
    return manifest[keep]


def build_transforms(protocol: dict):
    pre = protocol["preprocessing"]
    norm = pre["normalize"]
    image_tf = T.Compose([
        T.Resize(pre["resize"], interpolation=InterpolationMode.BILINEAR),
        T.CenterCrop(pre["crop"]),
        T.ToTensor(),
        T.Normalize(norm["mean"], norm["std"]),
    ])
    mask_tf = T.Compose([
        T.Resize(pre["resize"], interpolation=InterpolationMode.NEAREST),
        T.CenterCrop(pre["crop"]),
        T.PILToTensor(),
    ])
    return image_tf, mask_tf


class MVTecDataset(Dataset):
    """Returns dict(image [3,H,W], mask [1,H,W] in {0,1}, label, category, defect_type, path, split)."""

    def __init__(self, protocol: dict, split: str, categories: list[str] | None = None,
                 manifest: pd.DataFrame | None = None):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        manifest = build_manifest(protocol) if manifest is None else manifest
        df = manifest[manifest["split"] == split]
        if categories:
            df = df[df["category"].isin(categories)]
        if df.empty:
            raise ValueError(f"No samples for split={split!r}, categories={categories}")
        self.df = df.reset_index(drop=True)
        self.root = resolve_root(protocol)
        self.image_tf, self.mask_tf = build_transforms(protocol)
        self.size = protocol["preprocessing"]["crop"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = self.image_tf(Image.open(self.root / row["path"]).convert("RGB"))
        if row["mask_path"]:
            mask = self.mask_tf(Image.open(self.root / row["mask_path"]).convert("L"))
            mask = (mask > 0).float()
        else:
            mask = torch.zeros(1, self.size, self.size)
        return {
            "image": image,
            "mask": mask,
            "label": int(row["label"]),
            "category": row["category"],
            "defect_type": row["defect_type"],
            "path": row["path"],
            "split": row["split"],
        }
