"""Train and generate validation/test predictions for Autoencoder, PatchCore, or One-Class."""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import MVTecDataset, build_manifest, load_protocol, subsample_train_manifest
from src.evaluation import compute_validation_threshold, save_prediction_bundle
from src.methods import METHOD_REGISTRY, build_method
from src.utils import ensure_dir, get_device, load_yaml, resolve_path, seed_everything

DEFAULT_METHOD_CONFIGS = {
    "autoencoder": "configs/autoencoder.yaml",
    "patchcore": "configs/patchcore.yaml",
    "one_class": "configs/one_class.yaml",
}


def _folder_size_mb(folder: Path) -> float:
    if not folder.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    return float(total_bytes / (1024.0 * 1024.0))


def _batch_size_and_workers(cfg: dict[str, Any]) -> tuple[int, int]:
    for section in ("training", "scoring", "runtime"):
        if section in cfg:
            sub = cfg[section]
            bs = int(sub.get("batch_size", 16))
            nw = int(sub.get("num_workers", 0))
            return bs, nw
    return 16, 0


def run_single_method(
    method_name: str,
    protocol: dict[str, Any],
    method_cfg: dict[str, Any],
    run_name: str,
    artifacts_dir: Path,
    categories: list[str] | None = None,
    max_train_samples: int | None = None,
    device: torch.device | str | None = None,
    manifest: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Train and predict across all selected categories for one method."""
    seed = int(protocol.get("split", {}).get("seed", 42))
    seed_everything(seed)
    dev = get_device(str(device) if device else None)

    full_manifest = build_manifest(protocol) if manifest is None else manifest.copy()
    active_manifest = subsample_train_manifest(full_manifest, max_train_samples, seed=seed)
    cats = categories or list(protocol["dataset"]["categories"])
    batch_size, num_workers = _batch_size_and_workers(method_cfg)

    t_cfg = protocol.get("threshold", {})
    t_rule = t_cfg.get("rule", "percentile_of_val_normal_scores")
    t_percentile = float(t_cfg.get("percentile", 95.0))

    all_dfs: list[pd.DataFrame] = []
    all_maps: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    per_category_meta: dict[str, Any] = {}

    total_fit_sec = 0.0
    total_infer_sec = 0.0
    total_infer_imgs = 0
    total_artifact_mb = 0.0

    for cat in cats:
        print(f"[{run_name}] Category={cat} | Device={dev}")
        train_ds = MVTecDataset(protocol, "train", [cat], manifest=active_manifest)
        val_ds = MVTecDataset(protocol, "val", [cat], manifest=active_manifest)
        test_ds = MVTecDataset(protocol, "test", [cat], manifest=active_manifest)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        method = build_method(method_name, method_cfg, device=dev)

        t0 = time.perf_counter()
        fit_info = method.fit(train_loader, val_loader)
        fit_sec = time.perf_counter() - t0
        total_fit_sec += fit_sec

        t1 = time.perf_counter()
        val_pred = method.predict_dataloader(val_loader, collect_masks=True)
        test_pred = method.predict_dataloader(test_loader, collect_masks=True)
        infer_sec = time.perf_counter() - t1

        n_infer = len(val_ds) + len(test_ds)
        total_infer_sec += infer_sec
        total_infer_imgs += n_infer

        tau = compute_validation_threshold(
            val_pred["df"]["score"].to_numpy(),
            rule=t_rule,
            percentile=t_percentile,
        )
        method.threshold_ = tau

        ckpt_dir = ensure_dir(artifacts_dir / "checkpoints" / run_name / cat)
        method.save(ckpt_dir)
        cat_mb = _folder_size_mb(ckpt_dir)
        total_artifact_mb += cat_mb

        cat_df = pd.concat([val_pred["df"], test_pred["df"]], ignore_index=True)
        all_dfs.append(cat_df)
        if method.supports_localization and val_pred["anomaly_maps"] is not None:
            all_maps.append(
                np.concatenate([val_pred["anomaly_maps"], test_pred["anomaly_maps"]], axis=0)
            )
            all_masks.append(
                np.concatenate([val_pred["gt_masks"], test_pred["gt_masks"]], axis=0)
            )

        per_category_meta[cat] = {
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "threshold": tau,
            "fit_seconds": round(fit_sec, 3),
            "infer_ms_per_image": round((infer_sec * 1000.0) / max(n_infer, 1), 3),
            "artifact_size_mb": round(cat_mb, 3),
            "fit_info": fit_info,
        }

        del method
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_maps = np.concatenate(all_maps, axis=0) if all_maps else None
    combined_masks = np.concatenate(all_masks, axis=0) if all_masks else None

    metadata = {
        "run_name": run_name,
        "method": method_name,
        "categories": cats,
        "max_train_samples": max_train_samples,
        "total_fit_seconds": round(total_fit_sec, 3),
        "mean_infer_ms_per_image": round(
            (total_infer_sec * 1000.0) / max(total_infer_imgs, 1), 3
        ),
        "total_artifact_size_mb": round(total_artifact_mb, 3),
        "mean_artifact_size_mb": round(total_artifact_mb / max(len(cats), 1), 3),
        "method_config": method_cfg,
        "per_category": per_category_meta,
    }

    pred_dir = artifacts_dir / "predictions" / run_name
    save_prediction_bundle(
        pred_dir,
        df=combined_df,
        anomaly_maps=combined_maps,
        gt_masks=combined_masks,
        metadata=metadata,
    )
    print(f"[{run_name}] Saved prediction bundle to {pred_dir}")
    return {
        "df": combined_df,
        "anomaly_maps": combined_maps,
        "gt_masks": combined_masks,
        "metadata": metadata,
        "pred_dir": pred_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and generate predictions for MVTec AD anomaly detection methods."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", *METHOD_REGISTRY.keys()],
        help="Method to run or 'all' for all three baseline methods.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional custom YAML config path (overrides default config for single method).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional identifier for checkpoints/predictions folder (defaults to method name).",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "configs" / "protocol.yaml",
        help="Path to shared protocol.yaml.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional subset of categories (defaults to all categories in protocol.yaml).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts",
        help="Directory to store checkpoints and prediction bundles.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on normal training images per category (for I-01).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto, cuda, or cpu.",
    )
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    artifacts_dir = resolve_path(args.artifacts_dir)

    methods = list(METHOD_REGISTRY.keys()) if args.method == "all" else [args.method]
    for m_name in methods:
        cfg_path = args.config if (args.config and len(methods) == 1) else DEFAULT_METHOD_CONFIGS[m_name]
        method_cfg = load_yaml(cfg_path)
        r_name = args.run_name if (args.run_name and len(methods) == 1) else m_name
        run_single_method(
            method_name=m_name,
            protocol=protocol,
            method_cfg=method_cfg,
            run_name=r_name,
            artifacts_dir=artifacts_dir,
            categories=args.categories,
            max_train_samples=args.max_train_samples,
            device=args.device,
        )


if __name__ == "__main__":
    main()
