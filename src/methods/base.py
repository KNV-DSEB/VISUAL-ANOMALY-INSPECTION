"""Base interface shared by Autoencoder, PatchCore, and One-Class methods."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.repro import resolve_device


class BaseAnomalyMethod(ABC):
    """Abstract base class ensuring all 3 methods share a uniform fit/predict/save contract."""

    name: str = "base"
    supports_localization: bool = False

    def __init__(self, config: dict[str, Any], device: str | torch.device | None = None) -> None:
        self.config = config
        self.device = torch.device(device) if device is not None else resolve_device("auto")
        self.threshold_: float | None = None

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        """Fit the model using normal training images only."""

    @abstractmethod
    def predict_batch(
        self, images: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute image-level anomaly scores [B] and optional pixel maps [B, H, W].

        Higher scores must always indicate higher anomaly likelihood.
        """

    @torch.inference_mode()
    def predict_single(
        self, image: torch.Tensor
    ) -> tuple[float, np.ndarray | None]:
        """Score a single preprocessed image tensor [3, H, W] or [1, 3, H, W]."""
        if image.ndim == 3:
            image = image.unsqueeze(0)
        scores, amaps = self.predict_batch(image.to(self.device))
        score = float(scores[0])
        amap = amaps[0] if amaps is not None else None
        return score, amap

    @torch.inference_mode()
    def predict_dataloader(
        self,
        dataloader: DataLoader,
        collect_masks: bool = True,
    ) -> dict[str, Any]:
        """Run inference over a DataLoader and return a standardized prediction bundle.

        Returns a dict with:
          - df: pd.DataFrame(columns=['path', 'category', 'split', 'label', 'defect_type', 'score'])
          - anomaly_maps: np.ndarray [N, H, W] float32 or None
          - gt_masks: np.ndarray [N, H, W] uint8 or None
        """
        rows: list[dict[str, Any]] = []
        maps_list: list[np.ndarray] = []
        masks_list: list[np.ndarray] = []

        for batch in dataloader:
            images = batch["image"].to(self.device)
            scores, batch_maps = self.predict_batch(images)
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)

            paths = batch["path"]
            categories = batch["category"]
            splits = batch.get("split", ["test"] * len(paths))
            labels = batch["label"]
            defect_types = batch["defect_type"]

            for i in range(len(paths)):
                rows.append(
                    {
                        "path": str(paths[i]),
                        "category": str(categories[i]),
                        "split": str(splits[i]),
                        "label": int(labels[i]),
                        "defect_type": str(defect_types[i]),
                        "score": float(scores[i]),
                    }
                )

            if self.supports_localization and batch_maps is not None:
                maps_arr = np.asarray(batch_maps, dtype=np.float32)
                if maps_arr.ndim == 4 and maps_arr.shape[1] == 1:
                    maps_arr = maps_arr[:, 0]
                maps_list.append(maps_arr)

            if collect_masks and "mask" in batch:
                mask_arr = batch["mask"].detach().cpu().numpy()
                if mask_arr.ndim == 4 and mask_arr.shape[1] == 1:
                    mask_arr = mask_arr[:, 0]
                masks_list.append((mask_arr > 0.5).astype(np.uint8))

        df = pd.DataFrame(rows)
        anomaly_maps = np.concatenate(maps_list, axis=0) if maps_list else None
        gt_masks = np.concatenate(masks_list, axis=0) if masks_list else None
        return {
            "df": df,
            "anomaly_maps": anomaly_maps,
            "gt_masks": gt_masks,
        }

    @abstractmethod
    def save(self, artifact_dir: str | Path) -> Path:
        """Save trained weights / memory bank / classifier to `artifact_dir`."""

    @abstractmethod
    def load(self, artifact_dir: str | Path) -> None:
        """Load trained weights / memory bank / classifier from `artifact_dir`."""
