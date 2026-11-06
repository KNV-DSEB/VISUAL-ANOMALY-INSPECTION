"""Prediction output schema validation and persistence for E-01."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, load_json, save_json

REQUIRED_COLUMNS: tuple[str, ...] = (
    "path",
    "category",
    "split",
    "label",
    "defect_type",
    "score",
)
VALID_SPLITS = {"train", "val", "test"}


def validate_predictions(
    df: pd.DataFrame,
    anomaly_maps: np.ndarray | None = None,
    gt_masks: np.ndarray | None = None,
) -> None:
    """Validate that a prediction table and optional pixel maps obey the shared schema."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Prediction DataFrame missing required columns: {missing}")
    if df.empty:
        raise ValueError("Prediction DataFrame cannot be empty.")

    invalid_splits = set(df["split"].unique()) - VALID_SPLITS
    if invalid_splits:
        raise ValueError(f"Invalid split values {invalid_splits}; must be in {VALID_SPLITS}.")

    invalid_labels = set(df["label"].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"Invalid label values {invalid_labels}; must be in {{0, 1}}.")

    val_rows = df[df["split"] == "val"]
    if not val_rows.empty and (val_rows["label"] != 0).any():
        raise ValueError("Validation split must contain normal samples (label == 0) only.")

    if df["score"].isna().any() or not np.isfinite(df["score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Prediction scores contain NaN or infinite values.")

    n_rows = len(df)
    if anomaly_maps is not None:
        if anomaly_maps.ndim != 3 or anomaly_maps.shape[0] != n_rows:
            raise ValueError(
                f"anomaly_maps must have shape [N, H, W] with N={n_rows}, got {anomaly_maps.shape}."
            )
        if not np.isfinite(anomaly_maps).all():
            raise ValueError("anomaly_maps contain NaN or infinite values.")

    if gt_masks is not None:
        if gt_masks.ndim != 3 or gt_masks.shape[0] != n_rows:
            raise ValueError(
                f"gt_masks must have shape [N, H, W] with N={n_rows}, got {gt_masks.shape}."
            )
        if anomaly_maps is not None and gt_masks.shape != anomaly_maps.shape:
            raise ValueError(
                f"gt_masks shape {gt_masks.shape} does not match anomaly_maps shape {anomaly_maps.shape}."
            )


def save_prediction_bundle(
    bundle_dir: str | Path,
    df: pd.DataFrame,
    anomaly_maps: np.ndarray | None = None,
    gt_masks: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Validate and save predictions.csv, optional maps.npz, and metadata.json."""
    validate_predictions(df, anomaly_maps=anomaly_maps, gt_masks=gt_masks)
    out_dir = ensure_dir(bundle_dir)
    df.to_csv(out_dir / "predictions.csv", index=False)

    npz_payload: dict[str, np.ndarray] = {}
    if anomaly_maps is not None:
        npz_payload["anomaly_maps"] = np.asarray(anomaly_maps, dtype=np.float32)
    if gt_masks is not None:
        npz_payload["gt_masks"] = np.asarray(gt_masks, dtype=np.uint8)
    if npz_payload:
        np.savez_compressed(out_dir / "pixel_maps.npz", **npz_payload)

    if metadata is not None:
        save_json(metadata, out_dir / "metadata.json")
    return out_dir


def load_prediction_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Load and validate a saved prediction bundle from disk."""
    in_dir = Path(bundle_dir)
    csv_path = in_dir / "predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)

    anomaly_maps = None
    gt_masks = None
    npz_path = in_dir / "pixel_maps.npz"
    if npz_path.exists():
        with np.load(npz_path) as data:
            if "anomaly_maps" in data:
                anomaly_maps = data["anomaly_maps"]
            if "gt_masks" in data:
                gt_masks = data["gt_masks"]

    meta_path = in_dir / "metadata.json"
    metadata = load_json(meta_path) if meta_path.exists() else {}
    validate_predictions(df, anomaly_maps=anomaly_maps, gt_masks=gt_masks)
    return {
        "df": df,
        "anomaly_maps": anomaly_maps,
        "gt_masks": gt_masks,
        "metadata": metadata,
    }
