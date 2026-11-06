"""One-Class SVM pipeline on pretrained CNN representations (OC-01, OC-02)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader

from src.methods.base import BaseAnomalyMethod
from src.utils.io import ensure_dir, load_json, save_json

from .extractor import GlobalFeatureExtractor


class OneClassMethod(BaseAnomalyMethod):
    """Image-level anomaly detection using pretrained multi-scale features + One-Class SVM."""

    name = "one_class"
    supports_localization = False

    def __init__(self, config: dict[str, Any], device: str | torch.device | None = None) -> None:
        super().__init__(config, device=device)
        b_cfg = config.get("backbone", {})
        f_cfg = config.get("features", {})
        c_cfg = config.get("classifier", {})

        self.backbone_name = str(b_cfg.get("name", "wide_resnet50_2"))
        self.layers = list(b_cfg.get("layers", ["layer2", "layer3", "layer4"]))
        self.pretrained = bool(b_cfg.get("pretrained", True))

        self.pooling = str(f_cfg.get("pooling", "avg_plus_max"))
        self.use_scaler = bool(f_cfg.get("use_scaler", True))
        self.pca_components = f_cfg.get("pca_components", 0.95)
        self.seed = int(f_cfg.get("seed", 42))

        self.kernel = str(c_cfg.get("kernel", "rbf"))
        self.nu = float(c_cfg.get("nu", 0.05))
        self.gamma = c_cfg.get("gamma", "scale")

        self.feature_extractor = GlobalFeatureExtractor(
            backbone_name=self.backbone_name,
            layers=self.layers,
            pretrained=self.pretrained,
            pooling=self.pooling,
        ).to(self.device)

        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.ocsvm_: OneClassSVM | None = None
        self.fit_info_: dict[str, Any] = {}

    @torch.inference_mode()
    def _extract_loader_features(self, loader: DataLoader) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for batch in loader:
            images = batch["image"].to(self.device)
            feats = self.feature_extractor(images)
            chunks.append(feats.detach().cpu().numpy().astype(np.float64))
        return np.concatenate(chunks, axis=0)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        """Extract features from normal training images and fit Scaler + PCA + One-Class SVM."""
        x_train = self._extract_loader_features(train_loader)
        n_samples, raw_dim = x_train.shape

        if self.use_scaler:
            self.scaler_ = StandardScaler()
            x_train = self.scaler_.fit_transform(x_train)
        else:
            self.scaler_ = None

        if self.pca_components is not None and n_samples > 2:
            if isinstance(self.pca_components, float) and 0.0 < self.pca_components < 1.0:
                self.pca_ = PCA(n_components=self.pca_components, random_state=self.seed)
            else:
                n_comp = min(int(self.pca_components), n_samples - 1, raw_dim)
                self.pca_ = PCA(n_components=max(n_comp, 1), random_state=self.seed)
            x_train = self.pca_.fit_transform(x_train)
        else:
            self.pca_ = None

        self.ocsvm_ = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)
        self.ocsvm_.fit(x_train)

        self.fit_info_ = {
            "backbone": self.backbone_name,
            "layers": self.layers,
            "n_train_samples": int(n_samples),
            "raw_feature_dim": int(raw_dim),
            "reduced_feature_dim": int(x_train.shape[1]),
            "nu": self.nu,
            "gamma": str(self.gamma),
            "n_support_vectors": int(len(self.ocsvm_.support_)),
        }
        return self.fit_info_

    @torch.inference_mode()
    def predict_batch(
        self, images: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute image-level anomaly score = -decision_function(x). No pixel map."""
        if self.ocsvm_ is None:
            raise RuntimeError("OneClassSVM is not fitted; call fit() or load() first.")

        images = images.to(self.device)
        feats = self.feature_extractor(images).detach().cpu().numpy().astype(np.float64)
        if self.scaler_ is not None:
            feats = self.scaler_.transform(feats)
        if self.pca_ is not None:
            feats = self.pca_.transform(feats)

        # OneClassSVM decision_function is positive for inliers and negative for outliers.
        # Negate so higher score strictly means higher anomaly probability (OC-02).
        anomaly_scores = -self.ocsvm_.decision_function(feats)
        return anomaly_scores.astype(np.float32), None

    def save(self, artifact_dir: str | Path) -> Path:
        if self.ocsvm_ is None:
            raise RuntimeError("Cannot save OneClassMethod before fit().")
        out_dir = ensure_dir(artifact_dir)
        joblib.dump(
            {
                "scaler": self.scaler_,
                "pca": self.pca_,
                "ocsvm": self.ocsvm_,
                "threshold": self.threshold_,
            },
            out_dir / "one_class_model.joblib",
        )
        save_json(
            {
                "fit_info": self.fit_info_,
                "threshold": self.threshold_,
                "config": self.config,
            },
            out_dir / "one_class_meta.json",
        )
        return out_dir

    def load(self, artifact_dir: str | Path) -> None:
        in_dir = Path(artifact_dir)
        payload = joblib.load(in_dir / "one_class_model.joblib")
        self.scaler_ = payload["scaler"]
        self.pca_ = payload["pca"]
        self.ocsvm_ = payload["ocsvm"]
        self.threshold_ = payload.get("threshold")
        meta_path = in_dir / "one_class_meta.json"
        if meta_path.exists():
            self.fit_info_ = load_json(meta_path).get("fit_info", {})
