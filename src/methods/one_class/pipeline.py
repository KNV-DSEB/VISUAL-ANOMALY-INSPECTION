"""Fit the one-class detector per category and write val/test predictions.

Output follows docs/evaluation_schema.md:
    <pred_dir>/oneclass/<category>_{val,test}.npz
Image-level method: no anomaly_maps, so pixel metrics are NaN in the evaluation table.
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd

from src.data.mvtec import REPO_ROOT, MVTecDataset, build_manifest
from src.evaluation import Predictions
from src.utils.repro import environment_info, resolve_device, set_seed

from .features import FeatureExtractor, extract_features
from .model import OneClassDetector

METHOD = "oneclass"


def _resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def subsample_train(manifest: pd.DataFrame, category: str, max_train: int | None,
                    seed: int) -> pd.DataFrame:
    """Keep at most max_train train/good images of one category (seeded); other rows untouched."""
    if max_train is None:
        return manifest
    is_train = (manifest["split"] == "train") & (manifest["category"] == category)
    train = manifest[is_train]
    if max_train >= len(train):
        return manifest
    keep = train.sample(n=max_train, random_state=seed).index
    return manifest[~is_train | manifest.index.isin(keep)]


def run_category(category: str, cfg: dict, protocol: dict, manifest: pd.DataFrame,
                 extractor: FeatureExtractor, device, env: dict | None = None) -> dict:
    seed = cfg.get("seed", 42)
    fcfg = cfg["features"]
    max_train = cfg.get("train", {}).get("max_train")
    manifest = subsample_train(manifest, category, max_train, seed)

    def feats(split):
        ds = MVTecDataset(protocol, split, [category], manifest=manifest)
        return extract_features(extractor, ds, device, fcfg.get("batch_size", 32),
                                fcfg.get("num_workers", 0))

    t0 = time.perf_counter()
    train = feats("train")
    detector = OneClassDetector.from_config(cfg["model"], seed=seed).fit(train["features"])
    fit_seconds = time.perf_counter() - t0

    meta = {
        "config": cfg.get("config_path"),
        "backbone": fcfg["backbone"],
        "weights": fcfg.get("weights"),
        "layers": list(fcfg["layers"]),
        "feature_dim": int(train["features"].shape[1]),
        "model": cfg["model"],
        "n_train": int(len(train["features"])),
        "max_train": max_train,
        "seed": seed,
        "fit_seconds": round(fit_seconds, 2),
        **(env or {}),
    }

    pred_dir = _resolve(cfg["pred_dir"])
    out = {"category": category, "n_train": meta["n_train"], "feature_dim": meta["feature_dim"]}
    for split in ("val", "test"):
        t0 = time.perf_counter()
        data = feats(split)
        scores = detector.score(data["features"])
        Predictions(
            method=METHOD, category=category, split=split,
            scores=scores, labels=data["labels"], paths=data["paths"],
            defect_types=data["defect_types"],
            meta={**meta, "infer_seconds": round(time.perf_counter() - t0, 2)},
        ).save(pred_dir)
        out[f"n_{split}"] = len(scores)

    ckpt_dir = cfg.get("checkpoint_dir")
    if ckpt_dir:
        ckpt = _resolve(ckpt_dir) / f"{category}.joblib"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"detector": detector, "meta": meta}, ckpt)
    return out


def run(cfg: dict, protocol: dict, categories: list[str] | None = None) -> list[dict]:
    """Fit + predict every category. Thresholds/metrics are left to src.evaluation."""
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = resolve_device(cfg.get("device", "auto"))
    fcfg = cfg["features"]
    extractor = FeatureExtractor(fcfg["backbone"], fcfg["layers"], fcfg.get("weights"))
    env = environment_info(device)
    manifest = build_manifest(protocol)
    categories = categories or protocol["dataset"]["categories"]
    return [run_category(c, cfg, protocol, manifest, extractor, device, env) for c in categories]
