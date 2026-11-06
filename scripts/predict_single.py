"""Single-image anomaly inspection CLI and inference engine for Demo UI (DM-01)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_and_predict import DEFAULT_METHOD_CONFIGS
from src.data import build_transforms, load_protocol
from src.evaluation import compute_validation_threshold
from src.methods import METHOD_REGISTRY, BaseAnomalyMethod, build_method
from src.utils import (
    denormalize_image,
    ensure_dir,
    get_device,
    load_yaml,
    normalize_anomaly_map,
    overlay_heatmap,
    resolve_path,
    save_json,
    seed_everything,
)

_MODEL_CACHE: dict[tuple[str, str, str], tuple[BaseAnomalyMethod, str]] = {}


def load_or_validate_image(image_input: str | Path | Image.Image | np.ndarray) -> Image.Image:
    """Validate and convert an input image (path, PIL Image, or numpy array) to RGB PIL Image."""
    if image_input is None:
        raise ValueError("No image provided. Please upload or select a valid image.")
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Image file not found: {p}")
        try:
            img = Image.open(p).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Corrupted or unsupported image file: {p}") from exc
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim not in (2, 3) or image_input.size == 0:
            raise ValueError(f"Invalid numpy image shape: {image_input.shape}")
        arr = image_input
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input).__name__}")

    w, h = img.size
    if w < 16 or h < 16:
        raise ValueError(f"Image dimensions too small ({w}x{h}); minimum required is 16x16.")
    return img


def _build_demo_fallback_config(method_name: str) -> dict[str, Any]:
    """Return an ultra-fast fallback config when no trained checkpoint is found on disk."""
    if method_name == "autoencoder":
        return {
            "model": {"in_channels": 3, "base_channels": 16, "latent_channels": 32, "image_size": 224},
            "training": {"epochs": 2, "batch_size": 4, "lr": 2e-3, "l1_weight": 0.5, "seed": 42},
            "scoring": {"gaussian_sigma": 2.0, "image_score_reduction": "topk_mean", "topk_fraction": 0.02},
        }
    if method_name == "patchcore":
        return {
            "backbone": {"name": "tiny_test", "layers": ["layer2", "layer3"], "pretrained": False},
            "patch": {"patch_size": 3, "proj_dim": 64, "seed": 42},
            "coreset": {"ratio": 0.5, "max_patches": 256, "seed": 42},
            "scoring": {"n_neighbors": 3, "gaussian_sigma": 3.0},
        }
    return {
        "backbone": {"name": "tiny_test", "layers": ["layer2", "layer3", "layer4"], "pretrained": False},
        "features": {"pooling": "avg_plus_max", "use_scaler": True, "pca_components": 4, "seed": 42},
        "classifier": {"kernel": "rbf", "nu": 0.1, "gamma": "scale"},
    }


def _fit_fallback_model(
    method_name: str, img_size: int = 224, device: str | torch.device = "cpu"
) -> BaseAnomalyMethod:
    """Fit a fast deterministic baseline on synthetic normal tensors so demo never crashes."""
    seed_everything(42)
    cfg = _build_demo_fallback_config(method_name)
    method = build_method(method_name, cfg, device=device)

    gen = torch.Generator().manual_seed(42)
    normal_imgs = torch.randn(8, 3, img_size, img_size, generator=gen) * 0.15

    class _DictWrapper(torch.utils.data.Dataset):
        def __init__(self, tensors: torch.Tensor) -> None:
            self.tensors = tensors

        def __len__(self) -> int:
            return len(self.tensors)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return {"image": self.tensors[idx]}

    loader = DataLoader(_DictWrapper(normal_imgs), batch_size=4, shuffle=False)
    method.fit(loader, loader)
    val_scores = []
    for i in range(len(normal_imgs)):
        s, _ = method.predict_single(normal_imgs[i])
        val_scores.append(s)
    method.threshold_ = compute_validation_threshold(val_scores, percentile=95.0)
    return method


def get_loaded_method(
    method_name: str,
    category: str,
    artifacts_dir: Path,
    run_name: str | None = None,
    device: str | torch.device = "cpu",
    allow_demo_fallback: bool = False,
) -> tuple[BaseAnomalyMethod, str]:
    """Load trained checkpoint for `(run_name, category)` or build fallback if enabled."""
    r_name = run_name or method_name
    cache_key = (str(artifacts_dir.resolve()), r_name, category)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    ckpt_dir = artifacts_dir / "checkpoints" / r_name / category
    if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
        cfg_path = DEFAULT_METHOD_CONFIGS[method_name]
        cfg = load_yaml(cfg_path)
        for meta_name in ("patchcore_meta.json", "one_class_meta.json"):
            meta_file = ckpt_dir / meta_name
            if meta_file.exists():
                import json

                saved_cfg = json.loads(meta_file.read_text(encoding="utf-8")).get("config")
                if saved_cfg:
                    cfg = saved_cfg
        ae_ckpt = ckpt_dir / "autoencoder.pt"
        if ae_ckpt.exists():
            ckpt_data = torch.load(ae_ckpt, map_location=device, weights_only=False)
            if ckpt_data.get("config"):
                cfg = ckpt_data["config"]

        method = build_method(method_name, cfg, device=device)
        method.load(ckpt_dir)
        source = f"checkpoint ({ckpt_dir.relative_to(REPO_ROOT).as_posix() if ckpt_dir.is_relative_to(REPO_ROOT) else ckpt_dir.name})"
        _MODEL_CACHE[cache_key] = (method, source)
        return method, source

    if allow_demo_fallback:
        method = _fit_fallback_model(method_name, device=device)
        source = "demo_fallback (no saved checkpoint found)"
        _MODEL_CACHE[cache_key] = (method, source)
        return method, source

    raise FileNotFoundError(
        f"Checkpoint not found at {ckpt_dir}. Run `python scripts/train_and_predict.py --method {method_name}` "
        "or pass `--allow-fallback` for a quick demo test."
    )


def draw_defect_contours(
    rgb_image: np.ndarray,
    anomaly_map: np.ndarray,
    is_defect: bool,
    hotspot_quantile: float = 0.92,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw red defect contours + bounding boxes over `rgb_image` and return `(annotated_rgb, binary_mask_rgb)`."""
    norm_map = normalize_anomaly_map(anomaly_map)
    q = float(np.clip(hotspot_quantile, 0.50, 0.995))
    cutoff = max(float(np.quantile(norm_map, q)), 0.55 if is_defect else 0.75)
    mask_u8 = ((norm_map >= cutoff).astype(np.uint8)) * 255

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    annotated = rgb_image.copy()
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = (235, 45, 45) if is_defect else (255, 165, 0)
    for cnt in contours:
        if cv2.contourArea(cnt) < 12:
            continue
        cv2.drawContours(annotated, [cnt], -1, color, 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 1)

    mask_rgb = np.zeros_like(rgb_image)
    mask_rgb[:, :, 0] = mask_u8
    mask_rgb[:, :, 1] = (mask_u8 * 0.25).astype(np.uint8)
    return annotated, mask_rgb


def inspect_single_image(
    image_input: str | Path | Image.Image | np.ndarray,
    category: str,
    method_name: str = "patchcore",
    protocol: dict[str, Any] | None = None,
    artifacts_dir: str | Path = REPO_ROOT / "artifacts",
    run_name: str | None = None,
    custom_threshold: float | None = None,
    alpha: float = 0.45,
    hotspot_quantile: float = 0.92,
    device: str = "auto",
    allow_demo_fallback: bool = False,
) -> dict[str, Any]:
    """Run end-to-end single-image inspection and return scores, verdict, and visualizations."""
    protocol = protocol or load_protocol()
    valid_categories = protocol["dataset"]["categories"]
    cat = category.lower().strip()
    if cat not in valid_categories:
        raise ValueError(
            f"Unsupported category {category!r}. Must be one of {valid_categories}."
        )

    m_key = method_name.lower().strip()
    if m_key == "patchcore_refined":
        base_method = "patchcore"
        effective_run = run_name or "patchcore_refined"
    else:
        base_method = m_key
        effective_run = run_name or m_key

    if base_method not in METHOD_REGISTRY:
        raise ValueError(
            f"Unsupported method {method_name!r}. Choose from {list(METHOD_REGISTRY.keys())}."
        )

    pil_img = load_or_validate_image(image_input)
    dev = get_device(device)
    method, source = get_loaded_method(
        method_name=base_method,
        category=cat,
        artifacts_dir=resolve_path(artifacts_dir),
        run_name=effective_run,
        device=dev,
        allow_demo_fallback=allow_demo_fallback,
    )

    image_tf, _ = build_transforms(protocol)
    norm_cfg = protocol["preprocessing"]["normalize"]
    img_tensor = image_tf(pil_img)
    preprocessed_rgb = denormalize_image(img_tensor, mean=norm_cfg["mean"], std=norm_cfg["std"])

    t0 = time.perf_counter()
    score, anomaly_map = method.predict_single(img_tensor)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    val_threshold = float(method.threshold_) if method.threshold_ is not None else 0.5
    active_threshold = float(custom_threshold) if custom_threshold is not None else val_threshold
    is_defect = bool(score >= active_threshold)
    verdict = "DEFECT" if is_defect else "NORMAL"
    margin = float(score - active_threshold)
    ratio = float(score / max(abs(active_threshold), 1e-6))

    if anomaly_map is not None:
        heatmap_rgb = overlay_heatmap(preprocessed_rgb, anomaly_map, alpha=alpha)
        contour_rgb, binary_mask_rgb = draw_defect_contours(
            preprocessed_rgb,
            anomaly_map,
            is_defect=is_defect,
            hotspot_quantile=hotspot_quantile,
        )
    else:
        heatmap_rgb = None
        contour_rgb = preprocessed_rgb.copy()
        binary_mask_rgb = None

    return {
        "category": cat,
        "method": method_name,
        "score": float(score),
        "val_threshold": val_threshold,
        "threshold": active_threshold,
        "margin": margin,
        "anomaly_ratio": ratio,
        "is_defect": is_defect,
        "verdict": verdict,
        "supports_localization": bool(method.supports_localization),
        "latency_ms": round(latency_ms, 2),
        "checkpoint_source": source,
        "preprocessed_rgb": preprocessed_rgb,
        "contour_rgb": contour_rgb,
        "heatmap_rgb": heatmap_rgb,
        "binary_mask_rgb": binary_mask_rgb,
        "anomaly_map": anomaly_map,
    }


def compare_all_methods_on_image(
    image_input: str | Path | Image.Image | np.ndarray,
    category: str,
    protocol: dict[str, Any] | None = None,
    artifacts_dir: str | Path = REPO_ROOT / "artifacts",
    alpha: float = 0.45,
    allow_demo_fallback: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Run Autoencoder, PatchCore, and One-Class SVM on the same image for side-by-side comparison."""
    protocol = protocol or load_protocol()
    rows = []
    outputs: dict[str, dict[str, Any]] = {}

    for m_name in ("patchcore", "autoencoder", "one_class"):
        res = inspect_single_image(
            image_input=image_input,
            category=category,
            method_name=m_name,
            protocol=protocol,
            artifacts_dir=artifacts_dir,
            alpha=alpha,
            allow_demo_fallback=allow_demo_fallback,
        )
        outputs[m_name] = res
        rows.append(
            {
                "Method": m_name,
                "Anomaly Score": round(res["score"], 4),
                "Val Threshold (P95)": round(res["val_threshold"], 4),
                "Margin (Score - Thr)": round(res["margin"], 4),
                "Verdict": "🚨 DEFECT" if res["is_defect"] else "✅ NORMAL",
                "Pixel Localization": "Yes" if res["supports_localization"] else "No (Image-only)",
                "Latency (ms)": res["latency_ms"],
                "Model Source": res["checkpoint_source"],
            }
        )

    return pd.DataFrame(rows), outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a single image for visual anomalies (DM-01 CLI)."
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to input image.")
    parser.add_argument("--category", type=str, required=True, help="MVTec category name.")
    parser.add_argument(
        "--method",
        type=str,
        default="patchcore",
        choices=["autoencoder", "patchcore", "one_class", "patchcore_refined"],
    )
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional folder to save visual outputs.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional custom threshold override.")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow running with a lightweight fallback model if checkpoint is not yet trained.",
    )
    args = parser.parse_args()

    res = inspect_single_image(
        image_input=args.image,
        category=args.category,
        method_name=args.method,
        artifacts_dir=args.artifacts_dir,
        custom_threshold=args.threshold,
        allow_demo_fallback=args.allow_fallback,
    )
    summary = {
        k: v
        for k, v in res.items()
        if k
        not in {
            "preprocessed_rgb",
            "contour_rgb",
            "heatmap_rgb",
            "binary_mask_rgb",
            "anomaly_map",
        }
    }
    print(pd.Series(summary).to_string())

    if args.output_dir:
        out_dir = ensure_dir(args.output_dir)
        save_json(summary, out_dir / "prediction.json")
        Image.fromarray(res["contour_rgb"]).save(out_dir / "contour_overlay.png")
        if res["heatmap_rgb"] is not None:
            Image.fromarray(res["heatmap_rgb"]).save(out_dir / "heatmap_overlay.png")
        print(f"Saved inspection visualizations to {out_dir}")


if __name__ == "__main__":
    main()
