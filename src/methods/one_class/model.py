"""One-class classifier on image features (fit on normal images only).

Pipeline: [L2 normalise] -> [standardise] -> [PCA] -> One-Class SVM.
Anomaly score = -decision_function, so higher == more anomalous (evaluation convention).
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import OneClassSVM


class OneClassDetector:
    def __init__(self, nu: float = 0.1, kernel: str = "rbf", gamma: str | float = "scale",
                 l2_normalize: bool = True, standardize: bool = True,
                 pca_components: int | float | None = None, seed: int = 42):
        steps = []
        if l2_normalize:
            steps.append(("l2", Normalizer(norm="l2")))
        if standardize:
            steps.append(("scale", StandardScaler()))
        if pca_components is not None:
            steps.append(("pca", PCA(n_components=pca_components, svd_solver="full",
                                     random_state=seed)))
        steps.append(("ocsvm", OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)))
        self.pipeline = Pipeline(steps)
        self.fitted = False

    @classmethod
    def from_config(cls, model_cfg: dict, seed: int = 42) -> "OneClassDetector":
        return cls(**model_cfg.get("ocsvm", {}),
                   l2_normalize=model_cfg.get("l2_normalize", True),
                   standardize=model_cfg.get("standardize", True),
                   pca_components=model_cfg.get("pca_components"),
                   seed=seed)

    def fit(self, features: np.ndarray) -> "OneClassDetector":
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or len(features) < 2:
            raise ValueError(f"need a 2-D array with >= 2 rows, got shape {features.shape}")
        self.pipeline.fit(features)
        self.fitted = True
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("call fit() before score()")
        return -self.pipeline.decision_function(np.asarray(features, dtype=np.float64))
