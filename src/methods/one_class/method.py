"""One-class method behind the shared BaseAnomalyMethod interface.

Thin adapter over `features.FeatureExtractor` and `model.OneClassDetector` so that the unified
training script and the demo can treat all three methods the same way. No extra logic here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.methods.base import BaseAnomalyMethod
from src.utils.io import ensure_dir

from .features import FeatureExtractor, extract_features
from .model import OneClassDetector

ARTIFACT = "oneclass.joblib"


class OneClassMethod(BaseAnomalyMethod):
    """Frozen ImageNet ResNet features + One-Class SVM. Image-level only (no anomaly map)."""

    name = "oneclass"
    supports_localization = False

    def __init__(self, config: dict[str, Any], device: str | torch.device | None = None) -> None:
        super().__init__(config, device=device)
        f = config["features"]
        self.batch_size = int(f.get("batch_size", 32))
        self.extractor = FeatureExtractor(f["backbone"], f["layers"], f.get("weights")).to(self.device)
        self.detector = OneClassDetector.from_config(config["model"], seed=int(config.get("seed", 42)))

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> dict[str, Any]:
        data = extract_features(self.extractor, train_loader.dataset, self.device, self.batch_size)
        self.detector.fit(data["features"])
        return {"n_train": int(len(data["features"])), "feature_dim": int(data["features"].shape[1])}

    @torch.inference_mode()
    def predict_batch(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray | None]:
        feats = self.extractor(images.to(self.device)).float().cpu().numpy()
        return self.detector.score(feats).astype(np.float32), None

    def save(self, artifact_dir: str | Path) -> Path:
        out = ensure_dir(artifact_dir)
        joblib.dump({"detector": self.detector}, out / ARTIFACT)
        return out

    def load(self, artifact_dir: str | Path) -> None:
        self.detector = joblib.load(Path(artifact_dir) / ARTIFACT)["detector"]
