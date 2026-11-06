"""PatchCore memory bank construction and nearest-neighbor anomaly scoring (PC-01, PC-02)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from src.methods.base import BaseAnomalyMethod
from src.utils.io import ensure_dir, load_json, save_json

from .coreset import greedy_coreset_subsampling
from .extractor import PatchFeatureExtractor


class PatchCoreMethod(BaseAnomalyMethod):
    """PatchCore-style feature memory bank with k-Center Greedy coreset subsampling."""

    name = "patchcore"
    supports_localization = True

    def __init__(self, config: dict[str, Any], device: str | torch.device | None = None) -> None:
        super().__init__(config, device=device)
        b_cfg = config.get("backbone", {})
        p_cfg = config.get("patch", {})
        c_cfg = config.get("coreset", {})
        s_cfg = config.get("scoring", {})

        self.backbone_name = str(b_cfg.get("name", "wide_resnet50_2"))
        self.layers = list(b_cfg.get("layers", ["layer2", "layer3"]))
        self.pretrained = bool(b_cfg.get("pretrained", True))
        self.weights = str(b_cfg.get("weights", "IMAGENET1K_V1"))

        self.patch_size = int(p_cfg.get("patch_size", 3))
        self.proj_dim = p_cfg.get("proj_dim", 384)
        self.patch_seed = int(p_cfg.get("seed", 42))

        self.coreset_ratio = float(c_cfg.get("ratio", 0.10))
        self.max_patches = c_cfg.get("max_patches", 10000)
        self.coreset_seed = int(c_cfg.get("seed", 42))

        self.n_neighbors = int(s_cfg.get("n_neighbors", 9))
        self.gaussian_sigma = float(s_cfg.get("gaussian_sigma", 4.0))

        self.feature_extractor = PatchFeatureExtractor(
            backbone_name=self.backbone_name,
            layers=self.layers,
            pretrained=self.pretrained,
            weights=self.weights,
            patch_size=self.patch_size,
            proj_dim=self.proj_dim,
            seed=self.patch_seed,
        ).to(self.device)

        self.memory_bank_: torch.Tensor | None = None
        self.bank_stats_: dict[str, Any] = {}

    @torch.inference_mode()
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        """Extract normal patch embeddings and build a coreset-subsampled memory bank."""
        all_patches: list[np.ndarray] = []
        grid_size: tuple[int, int] = (28, 28)
        n_images = 0

        for batch in train_loader:
            images = batch["image"].to(self.device)
            patches, grid_size = self.feature_extractor(images)
            b, n_p, d = patches.shape
            all_patches.append(patches.reshape(b * n_p, d).detach().cpu().numpy())
            n_images += b

        raw_bank = np.concatenate(all_patches, axis=0)
        coreset_bank, _ = greedy_coreset_subsampling(
            raw_bank,
            ratio=self.coreset_ratio,
            max_patches=self.max_patches,
            seed=self.coreset_seed,
        )
        self.memory_bank_ = torch.from_numpy(coreset_bank).to(self.device)
        self.bank_stats_ = {
            "backbone": self.backbone_name,
            "weights": self.weights if self.pretrained else None,
            "layers": self.layers,
            "raw_embed_dim": self.feature_extractor.raw_embed_dim_,
            "embed_dim": int(coreset_bank.shape[1]),
            "n_train_images": n_images,
            "patch_grid": list(grid_size),
            "raw_num_patches": int(raw_bank.shape[0]),
            "coreset_num_patches": int(coreset_bank.shape[0]),
            "coreset_ratio": float(coreset_bank.shape[0] / max(raw_bank.shape[0], 1)),
        }
        return self.bank_stats_

    @torch.inference_mode()
    def predict_batch(
        self, images: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute nearest-neighbor patch distances, density-reweighted image score, and 2D map."""
        if self.memory_bank_ is None:
            raise RuntimeError("PatchCore memory_bank_ is empty; call fit() or load() first.")

        images = images.to(self.device)
        b, _, img_h, img_w = images.shape
        patches, (h_p, w_p) = self.feature_extractor(images)  # [B, P, D]
        _, n_p, d = patches.shape

        flat_queries = patches.reshape(b * n_p, d)
        dists = torch.cdist(flat_queries, self.memory_bank_, p=2.0)  # [B*P, M]

        k = min(max(self.n_neighbors, 1), self.memory_bank_.shape[0])
        knn_dists, _ = torch.topk(dists, k=k, dim=1, largest=False, sorted=True)  # [B*P, k]
        min_patch_dists = knn_dists[:, 0].reshape(b, n_p)  # [B, P]
        knn_reshaped = knn_dists.reshape(b, n_p, k)

        # PatchCore Eq. 6: density-aware reweighting on the most anomalous patch per image
        max_patch_idx = torch.argmax(min_patch_dists, dim=1)  # [B]
        batch_indices = torch.arange(b, device=images.device)
        top_patch_knn = knn_reshaped[batch_indices, max_patch_idx]  # [B, k]
        s_star = top_patch_knn[:, 0]
        if k > 1:
            exp_diff = torch.exp(top_patch_knn - s_star.unsqueeze(1))
            weights = 1.0 - (1.0 / torch.clamp(exp_diff.sum(dim=1), min=1e-8))
            image_scores = (weights * s_star).detach().cpu().numpy().astype(np.float32)
        else:
            image_scores = s_star.detach().cpu().numpy().astype(np.float32)

        patch_maps = min_patch_dists.reshape(b, h_p, w_p).detach().cpu().numpy().astype(np.float32)
        anomaly_maps = np.empty((b, img_h, img_w), dtype=np.float32)
        for i in range(b):
            upsampled = cv2.resize(
                patch_maps[i], (img_w, img_h), interpolation=cv2.INTER_LINEAR
            )
            if self.gaussian_sigma > 0:
                upsampled = gaussian_filter(upsampled, sigma=self.gaussian_sigma)
            anomaly_maps[i] = upsampled

        return image_scores, anomaly_maps

    def save(self, artifact_dir: str | Path) -> Path:
        if self.memory_bank_ is None:
            raise RuntimeError("Cannot save PatchCore before fit().")
        out_dir = ensure_dir(artifact_dir)
        np.savez_compressed(
            out_dir / "memory_bank.npz",
            memory_bank=self.memory_bank_.detach().cpu().numpy().astype(np.float32),
            projection=(
                self.feature_extractor.projection.detach().cpu().numpy().astype(np.float32)
                if self.feature_extractor.projection is not None
                else np.array([], dtype=np.float32)
            ),
        )
        save_json(
            {
                "bank_stats": self.bank_stats_,
                "threshold": self.threshold_,
                "config": self.config,
            },
            out_dir / "patchcore_meta.json",
        )
        return out_dir

    def load(self, artifact_dir: str | Path) -> None:
        in_dir = Path(artifact_dir)
        with np.load(in_dir / "memory_bank.npz") as data:
            bank_np = data["memory_bank"].astype(np.float32)
            self.memory_bank_ = torch.from_numpy(bank_np).to(self.device)
            proj_np = data["projection"]
            if proj_np.size > 0:
                self.feature_extractor.projection = torch.from_numpy(proj_np).to(self.device)
                self.feature_extractor.raw_embed_dim_ = int(proj_np.shape[0])
                self.feature_extractor.embed_dim_ = int(proj_np.shape[1])
        meta_path = in_dir / "patchcore_meta.json"
        if meta_path.exists():
            meta = load_json(meta_path)
            self.bank_stats_ = meta.get("bank_stats", {})
            self.threshold_ = meta.get("threshold")
