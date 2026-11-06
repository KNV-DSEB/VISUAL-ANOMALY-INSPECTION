"""Inspect one image with a trained checkpoint (DM-01). Used by the CLI and by demo/app.py.

    python scripts/predict_single.py --image path/to/img.png --category bottle --method patchcore

Checkpoints come from scripts/train_and_predict.py (checkpoints/<method>/<category>/). There is no
fallback model: without a trained checkpoint the call fails with instructions instead of guessing.
The decision threshold is the validation threshold stored with the checkpoint.
"""
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import build_transforms, load_protocol  # noqa: E402
from src.methods import METHOD_REGISTRY, BaseAnomalyMethod, load_method  # noqa: E402
from src.utils import (  # noqa: E402
    denormalize_image,
    ensure_dir,
    normalize_anomaly_map,
    overlay_heatmap,
    resolve_device,
    resolve_path,
    save_json,
)

METHODS = tuple(METHOD_REGISTRY)
_MODEL_CACHE: dict[tuple[str, str, str], BaseAnomalyMethod] = {}


def load_or_validate_image(image_input: str | Path | Image.Image | np.ndarray) -> Image.Image:
    """Validate and convert a path, PIL image or numpy array to an RGB PIL image."""
    if image_input is None:
        raise ValueError("No image provided. Please upload or select a valid image.")
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
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
        arr = image_input if image_input.dtype == np.uint8 else np.clip(image_input, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input).__name__}")
    if min(img.size) < 16:
        raise ValueError(f"Image too small ({img.size[0]}x{img.size[1]}); minimum is 16x16.")
    return img


def get_loaded_method(method_name: str, category: str, checkpoint_dir: Path,
                      device: torch.device) -> BaseAnomalyMethod:
    """Load checkpoints/<method>/<category>; raise with instructions when it does not exist."""
    key = (str(checkpoint_dir.resolve()), method_name, category)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    ckpt = checkpoint_dir / method_name / category
    if not ckpt.is_dir() or not any(ckpt.iterdir()):
        raise FileNotFoundError(
            f"No trained checkpoint for method={method_name!r}, category={category!r} in {ckpt}. "
            f"Train one with: python scripts/train_and_predict.py --method {method_name} "
            f"--categories {category}"
        )
    method = load_method(method_name, ckpt, device=device)
    if method.threshold_ is None:
        raise RuntimeError(f"Checkpoint {ckpt} has no validation threshold; retrain it.")
    _MODEL_CACHE[key] = method
    return method


def draw_defect_contours(rgb_image: np.ndarray, anomaly_map: np.ndarray, is_defect: bool,
                         hotspot_quantile: float = 0.92) -> tuple[np.ndarray, np.ndarray]:
    """Outline the hottest region of the map. Only drawn when the image is judged defective."""
    mask_rgb = np.zeros_like(rgb_image)
    if not is_defect:
        return rgb_image.copy(), mask_rgb
    norm_map = normalize_anomaly_map(anomaly_map)
    cutoff = max(float(np.quantile(norm_map, float(np.clip(hotspot_quantile, 0.5, 0.995)))), 0.55)
    mask_u8 = ((norm_map >= cutoff).astype(np.uint8)) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    annotated = rgb_image.copy()
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 12:
            continue
        cv2.drawContours(annotated, [cnt], -1, (235, 45, 45), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (235, 45, 45), 1)
    mask_rgb[:, :, 0] = mask_u8
    return annotated, mask_rgb


def inspect_single_image(
    image_input: str | Path | Image.Image | np.ndarray,
    category: str,
    method_name: str = "patchcore",
    protocol: dict[str, Any] | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    custom_threshold: float | None = None,
    alpha: float = 0.45,
    hotspot_quantile: float = 0.92,
    device: str = "auto",
) -> dict[str, Any]:
    """Score one image with a trained checkpoint and return score, verdict and visualizations."""
    protocol = protocol or load_protocol()
    cat = category.lower().strip()
    if cat not in protocol["dataset"]["categories"]:
        raise ValueError(f"Unsupported category {category!r}. Must be one of "
                         f"{protocol['dataset']['categories']}.")
    name = method_name.lower().strip()
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unsupported method {method_name!r}. Choose from {list(METHOD_REGISTRY)}.")

    pil_img = load_or_validate_image(image_input)
    method = get_loaded_method(name, cat, resolve_path(checkpoint_dir), resolve_device(device))

    image_tf, _ = build_transforms(protocol)
    norm = protocol["preprocessing"]["normalize"]
    tensor = image_tf(pil_img)
    rgb = denormalize_image(tensor, mean=norm["mean"], std=norm["std"])

    t0 = time.perf_counter()
    score, amap = method.predict_single(tensor)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    val_threshold = float(method.threshold_)
    threshold = float(custom_threshold) if custom_threshold is not None else val_threshold
    is_defect = bool(score >= threshold)
    if amap is not None:
        heatmap = overlay_heatmap(rgb, amap, alpha=alpha)
        contour, mask = draw_defect_contours(rgb, amap, is_defect, hotspot_quantile)
    else:
        heatmap, contour, mask = None, rgb.copy(), None
    return {
        "category": cat, "method": name, "score": float(score), "val_threshold": val_threshold,
        "threshold": threshold, "threshold_is_override": custom_threshold is not None,
        "margin": float(score - threshold), "is_defect": is_defect,
        "verdict": "DEFECT" if is_defect else "NORMAL",
        "supports_localization": bool(method.supports_localization),
        "latency_ms": round(latency_ms, 2),
        "preprocessed_rgb": rgb, "contour_rgb": contour, "heatmap_rgb": heatmap,
        "binary_mask_rgb": mask, "anomaly_map": amap,
    }


def compare_all_methods_on_image(
    image_input: str | Path | Image.Image | np.ndarray,
    category: str,
    protocol: dict[str, Any] | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    alpha: float = 0.45,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Run every method on one image. A method without a checkpoint is reported, not faked."""
    protocol = protocol or load_protocol()
    rows, outputs = [], {}
    for name in METHODS:
        try:
            res = inspect_single_image(image_input, category, name, protocol, checkpoint_dir, alpha=alpha)
        except FileNotFoundError as exc:
            if "No trained checkpoint" not in str(exc):
                raise
            rows.append({"Method": name, "Anomaly Score": None, "Threshold (val)": None,
                         "Verdict": "no checkpoint", "Pixel Localization": "-", "Latency (ms)": None})
            continue
        outputs[name] = res
        rows.append({"Method": name, "Anomaly Score": round(res["score"], 4),
                     "Threshold (val)": round(res["val_threshold"], 4),
                     "Verdict": res["verdict"],
                     "Pixel Localization": "yes" if res["supports_localization"] else "no (image-level)",
                     "Latency (ms)": res["latency_ms"]})
    return pd.DataFrame(rows), outputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--method", default="patchcore", choices=METHODS)
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--output-dir", type=Path, help="save prediction.json and overlays here")
    ap.add_argument("--threshold", type=float, help="manual threshold override (default: validation threshold)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    res = inspect_single_image(args.image, args.category, args.method, checkpoint_dir=args.checkpoint_dir,
                               custom_threshold=args.threshold, device=args.device)
    images = {"preprocessed_rgb", "contour_rgb", "heatmap_rgb", "binary_mask_rgb", "anomaly_map"}
    summary = {k: v for k, v in res.items() if k not in images}
    print(pd.Series(summary).to_string())
    if args.output_dir:
        out = ensure_dir(args.output_dir)
        save_json(summary, out / "prediction.json")
        Image.fromarray(res["contour_rgb"]).save(out / "contour_overlay.png")
        if res["heatmap_rgb"] is not None:
            Image.fromarray(res["heatmap_rgb"]).save(out / "heatmap_overlay.png")
        print(f"Saved visualizations to {out}")


if __name__ == "__main__":
    main()
