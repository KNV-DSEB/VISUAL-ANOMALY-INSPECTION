"""Prediction schema shared by all methods (AE, PatchCore, One-class).

One .npz file per (method, category, split). Layout on disk:
    <pred_dir>/<method>/<category>_<split>.npz
Required arrays : scores (N,), labels (N,), paths (N,)
Optional arrays : defect_types (N,), anomaly_maps (N,H,W), gt_masks (N,H,W)
Convention      : higher score / higher map value == more anomalous.
labels: 0 = good, 1 = defective. gt_masks: 0/1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SPLITS = ("val", "test")


@dataclass
class Predictions:
    method: str
    category: str
    split: str
    scores: np.ndarray
    labels: np.ndarray
    paths: np.ndarray
    defect_types: np.ndarray | None = None
    anomaly_maps: np.ndarray | None = None
    gt_masks: np.ndarray | None = None
    meta: dict = field(default_factory=dict)  # config path, checkpoint id, ...

    def __post_init__(self):
        self.scores = np.asarray(self.scores, dtype=np.float64)
        self.labels = np.asarray(self.labels).astype(int)
        self.paths = np.asarray(self.paths).astype(str)
        self.validate()

    @property
    def has_maps(self) -> bool:
        return self.anomaly_maps is not None and self.gt_masks is not None

    def validate(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")
        n = len(self.scores)
        if self.scores.ndim != 1:
            raise ValueError("scores must be 1-D")
        if not np.isfinite(self.scores).all():
            raise ValueError("scores contain NaN/inf")
        for name in ("labels", "paths", "defect_types", "anomaly_maps", "gt_masks"):
            arr = getattr(self, name)
            if arr is not None and len(arr) != n:
                raise ValueError(f"{name} has length {len(arr)}, expected {n}")
        if not set(np.unique(self.labels)) <= {0, 1}:
            raise ValueError("labels must be 0/1")
        if (self.anomaly_maps is None) != (self.gt_masks is None):
            raise ValueError("anomaly_maps and gt_masks must be given together")
        if self.has_maps and self.anomaly_maps.shape != self.gt_masks.shape:
            raise ValueError("anomaly_maps and gt_masks must have the same shape")

    def save(self, pred_dir):
        out = Path(pred_dir) / self.method / f"{self.category}_{self.split}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        data = dict(
            scores=self.scores,
            labels=self.labels,
            paths=self.paths,
            meta=np.array(json.dumps(self.meta)),
        )
        if self.defect_types is not None:
            data["defect_types"] = np.asarray(self.defect_types).astype(str)
        if self.has_maps:
            data["anomaly_maps"] = self.anomaly_maps.astype(np.float32)
            data["gt_masks"] = (self.gt_masks > 0).astype(np.uint8)
        np.savez_compressed(out, **data)
        return out

    @classmethod
    def load(cls, pred_dir, method, category, split):
        path = Path(pred_dir) / method / f"{category}_{split}.npz"
        with np.load(path, allow_pickle=False) as z:
            return cls(
                method=method,
                category=category,
                split=split,
                scores=z["scores"],
                labels=z["labels"],
                paths=z["paths"],
                defect_types=z["defect_types"] if "defect_types" in z else None,
                anomaly_maps=z["anomaly_maps"] if "anomaly_maps" in z else None,
                gt_masks=z["gt_masks"] if "gt_masks" in z else None,
                meta=json.loads(str(z["meta"])) if "meta" in z else {},
            )
