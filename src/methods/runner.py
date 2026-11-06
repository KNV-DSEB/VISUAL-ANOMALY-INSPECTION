"""Train/predict runner shared by all methods.

For every category: fit on train/good, score val and test, then write
  <pred_dir>/<method>/<category>_{val,test}.npz   (schema: docs/evaluation_schema.md)
  <ckpt_dir>/<method>/<category>/                  (checkpoint + config + validation threshold)
Thresholds are only ever derived from val scores through `percentile_threshold`, the same
function the evaluation harness uses. The official test split is scored but never used to fit.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.data import MVTecDataset, build_manifest, subsample_train_manifest
from src.evaluation import Predictions
from src.evaluation.thresholds import percentile_threshold
from src.methods import build_method, save_method
from src.utils.io import resolve_path
from src.utils.repro import environment_info, resolve_device, set_seed

BATCH_SECTIONS = ("training", "scoring", "runtime", "features")


def batch_and_workers(cfg: dict[str, Any]) -> tuple[int, int]:
    """Batch size / num_workers from the first config section that defines them."""
    for section in BATCH_SECTIONS:
        sub = cfg.get(section)
        if isinstance(sub, dict) and "batch_size" in sub:
            return int(sub["batch_size"]), int(sub.get("num_workers", 0))
    return 16, 0


def folder_size_mb(folder: Path) -> float:
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / (1024.0 * 1024.0)


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def jsonable(obj: dict) -> dict:
    return json.loads(json.dumps(obj, default=_json_default))


def to_predictions(method: str, category: str, split: str, bundle: dict,
                   keep_maps: bool) -> Predictions:
    """Convert BaseAnomalyMethod.predict_dataloader output to the shared Predictions schema."""
    df: pd.DataFrame = bundle["df"]
    maps = bundle["anomaly_maps"] if keep_maps else None
    masks = bundle["gt_masks"] if maps is not None else None
    return Predictions(
        method=method, category=category, split=split,
        scores=df["score"].to_numpy(), labels=df["label"].to_numpy(), paths=df["path"].to_numpy(),
        defect_types=df["defect_type"].to_numpy(), anomaly_maps=maps, gt_masks=masks,
    )


def run_method(
    method_name: str,
    cfg: dict[str, Any],
    protocol: dict[str, Any],
    pred_dir: str | Path,
    ckpt_dir: str | Path | None = None,
    categories: list[str] | None = None,
    max_train: int | None = None,
    device: str = "auto",
    config_path: str | None = None,
    manifest: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Fit + predict every category for one method. Returns one summary dict per category."""
    seed = int(protocol["split"]["seed"])
    percentile = float(protocol["threshold"]["percentile"])
    dev = resolve_device(device)
    env = environment_info(dev)
    manifest = build_manifest(protocol) if manifest is None else manifest
    manifest = subsample_train_manifest(manifest, max_train, seed)
    categories = categories or list(protocol["dataset"]["categories"])
    batch_size, num_workers = batch_and_workers(cfg)
    pred_dir = resolve_path(pred_dir)
    summaries = []

    for cat in categories:
        set_seed(seed)  # every category starts from the same RNG state, whatever ran before
        print(f"[{method_name}] category={cat} device={dev}", flush=True)
        ds = {s: MVTecDataset(protocol, s, [cat], manifest=manifest) for s in ("train", "val", "test")}
        train_loader = DataLoader(ds["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers)
        loaders = {s: DataLoader(ds[s], batch_size=batch_size, shuffle=False, num_workers=num_workers)
                   for s in ("val", "test")}

        method = build_method(method_name, cfg, device=dev)
        t0 = time.perf_counter()
        fit_info = method.fit(train_loader, loaders["val"])
        fit_seconds = time.perf_counter() - t0

        preds: dict[str, Predictions] = {}
        infer_seconds = 0.0
        for split, loader in loaders.items():
            t1 = time.perf_counter()
            bundle = method.predict_dataloader(loader, collect_masks=True)
            infer_seconds += time.perf_counter() - t1
            # val maps are not needed (val is only used for the threshold), so they are not stored
            preds[split] = to_predictions(method_name, cat, split, bundle, keep_maps=(split == "test"))

        method.threshold_ = percentile_threshold(preds["val"], percentile)
        artifact_mb = 0.0
        if ckpt_dir is not None:
            out = save_method(method, resolve_path(ckpt_dir) / method_name / cat)
            artifact_mb = folder_size_mb(out)

        n_infer = len(ds["val"]) + len(ds["test"])
        meta = jsonable({
            "method": method_name, "category": cat, "config": cfg, "config_path": config_path,
            "seed": seed, "threshold_percentile": percentile,
            "val_threshold": method.threshold_, "max_train": max_train,
            "n_train": len(ds["train"]), "n_val": len(ds["val"]), "n_test": len(ds["test"]),
            "fit_seconds": round(fit_seconds, 2),
            "infer_ms_per_image": round(infer_seconds * 1000.0 / max(n_infer, 1), 2),
            "artifact_size_mb": round(artifact_mb, 3),
            "fit_info": fit_info, **env,
        })
        for p in preds.values():
            p.meta = meta
            p.save(pred_dir)
        summaries.append({"category": cat, "n_train": len(ds["train"]), "n_val": len(ds["val"]),
                          "n_test": len(ds["test"]), "val_threshold": method.threshold_,
                          "fit_seconds": meta["fit_seconds"]})
        del method
    return summaries
