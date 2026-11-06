"""Autoencoder training, checkpoint selection, and reconstruction scoring pipeline (AE-01, AE-02)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from src.methods.base import BaseAnomalyMethod
from src.utils.io import ensure_dir, load_json, save_json
from src.utils.seed import seed_everything

from .model import ConvAutoencoder


class AutoencoderMethod(BaseAnomalyMethod):
    """Reconstruction-based anomaly detection and pixel localization."""

    name = "autoencoder"
    supports_localization = True

    def __init__(self, config: dict[str, Any], device: str | torch.device | None = None) -> None:
        super().__init__(config, device=device)
        m_cfg = config.get("model", {})
        self.model = ConvAutoencoder(
            in_channels=int(m_cfg.get("in_channels", 3)),
            base_channels=int(m_cfg.get("base_channels", 32)),
            latent_channels=int(m_cfg.get("latent_channels", 128)),
        ).to(self.device)

        t_cfg = config.get("training", {})
        self.epochs = int(t_cfg.get("epochs", 40))
        self.lr = float(t_cfg.get("lr", 1e-3))
        self.weight_decay = float(t_cfg.get("weight_decay", 1e-5))
        self.l1_weight = float(t_cfg.get("l1_weight", 0.5))
        self.seed = int(t_cfg.get("seed", 42))

        s_cfg = config.get("scoring", {})
        self.gaussian_sigma = float(s_cfg.get("gaussian_sigma", 2.0))
        self.reduction = str(s_cfg.get("image_score_reduction", "topk_mean")).lower()
        self.topk_fraction = float(s_cfg.get("topk_fraction", 0.02))
        self.history_: list[dict[str, float]] = []

    def _reconstruction_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(recon, target, reduction="mean")
        if self.l1_weight > 0:
            l1 = F.l1_loss(recon, target, reduction="mean")
            return mse + self.l1_weight * l1
        return mse

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, Any]:
        """Train on normal images and select best checkpoint strictly by validation loss."""
        seed_everything(self.seed)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.epochs, 1)
        )

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(self.model.state_dict())
        self.history_.clear()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss_sum = 0.0
            n_train = 0
            for batch in train_loader:
                images = batch["image"].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                recon = self.model(images)
                loss = self._reconstruction_loss(recon, images)
                loss.backward()
                optimizer.step()
                bs = images.size(0)
                train_loss_sum += float(loss.item()) * bs
                n_train += bs
            scheduler.step()
            train_loss = train_loss_sum / max(n_train, 1)

            if val_loader is not None:
                val_loss = self._evaluate_recon_loss(val_loader)
            else:
                val_loss = train_loss

            self.history_.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_state)
        self.model.eval()
        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "epochs": self.epochs,
            "history": self.history_,
        }

    @torch.inference_mode()
    def _evaluate_recon_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        loss_sum = 0.0
        n_total = 0
        for batch in loader:
            images = batch["image"].to(self.device)
            recon = self.model(images)
            loss = self._reconstruction_loss(recon, images)
            bs = images.size(0)
            loss_sum += float(loss.item()) * bs
            n_total += bs
        return loss_sum / max(n_total, 1)

    @torch.inference_mode()
    def predict_batch(
        self, images: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute pixel reconstruction error map [B, H, W] and image-level score [B]."""
        self.model.eval()
        images = images.to(self.device)
        recon = self.model(images)
        diff = recon - images
        pixel_err = (diff.pow(2) + self.l1_weight * diff.abs()).mean(dim=1)
        maps_np = pixel_err.detach().cpu().numpy().astype(np.float32)

        if self.gaussian_sigma > 0:
            for i in range(maps_np.shape[0]):
                maps_np[i] = gaussian_filter(maps_np[i], sigma=self.gaussian_sigma)

        flat = maps_np.reshape(maps_np.shape[0], -1)
        if self.reduction == "mean":
            scores = flat.mean(axis=1)
        elif self.reduction == "max":
            scores = flat.max(axis=1)
        else:
            k = max(int(round(flat.shape[1] * self.topk_fraction)), 1)
            part = np.partition(flat, -k, axis=1)[:, -k:]
            scores = part.mean(axis=1)

        return scores.astype(np.float32), maps_np

    def save(self, artifact_dir: str | Path) -> Path:
        out_dir = ensure_dir(artifact_dir)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "threshold": self.threshold_,
            },
            out_dir / "autoencoder.pt",
        )
        save_json({"history": self.history_, "threshold": self.threshold_}, out_dir / "training_log.json")
        return out_dir

    def load(self, artifact_dir: str | Path) -> None:
        in_dir = Path(artifact_dir)
        ckpt = torch.load(in_dir / "autoencoder.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.threshold_ = ckpt.get("threshold")
        self.model.eval()
        log_path = in_dir / "training_log.json"
        if log_path.exists():
            self.history_ = load_json(log_path).get("history", [])
