"""Qualitative localization galleries: input | GT mask | Autoencoder map | PatchCore map (L-02).

Per category the examples follow a fixed rule (not hand-picked), taken from PatchCore's test scores:
the highest-scoring defect, the lowest-scoring defect (hardest/missed), the median defect and the
highest-scoring normal image (worst false-alarm candidate). Failures are therefore always shown.

    python scripts/generate_galleries.py --pred-dir results/predictions --out-dir results/figures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import build_transforms, load_protocol, resolve_root  # noqa: E402
from src.evaluation import Predictions, percentile_threshold  # noqa: E402
from src.utils import denormalize_image, ensure_dir, save_localization_gallery  # noqa: E402


def choose_examples(test: Predictions, max_samples: int = 4) -> list[int]:
    """Indices of the examples to show (fixed rule, see module docstring)."""
    order = np.argsort(-test.scores)
    defects = [int(i) for i in order if test.labels[i] == 1]
    normals = [int(i) for i in order if test.labels[i] == 0]
    chosen = []
    if defects:
        chosen.append(defects[0])
    if len(defects) > 1:
        chosen.append(defects[-1])
    if len(defects) > 2:
        chosen.append(defects[len(defects) // 2])
    if normals:
        chosen.append(normals[0])
    return chosen[:max_samples]


def build_galleries(pred_dir, out_dir, protocol: dict, ae_method: str = "autoencoder",
                    pc_method: str = "patchcore", categories: list[str] | None = None,
                    max_samples: int = 4) -> list[Path]:
    root = resolve_root(protocol)
    image_tf, _ = build_transforms(protocol)
    norm = protocol["preprocessing"]["normalize"]
    percentile = float(protocol["threshold"]["percentile"])
    out_dir = ensure_dir(out_dir)
    saved = []

    for cat in categories or protocol["dataset"]["categories"]:
        runs = {}
        for key, method in (("ae", ae_method), ("pc", pc_method)):
            if (Path(pred_dir) / method / f"{cat}_test.npz").exists():
                runs[key] = (Predictions.load(pred_dir, method, cat, "val"),
                             Predictions.load(pred_dir, method, cat, "test"))
        if not runs:
            raise FileNotFoundError(f"No {ae_method!r}/{pc_method!r} predictions for {cat} in {pred_dir}")
        primary_val, primary = runs.get("pc") or runs["ae"]
        if not primary.has_maps:
            raise ValueError(f"{primary.method} predictions for {cat} have no anomaly maps")

        index = {key: {p: i for i, p in enumerate(test.paths)} for key, (_, test) in runs.items()}
        thr = {key: percentile_threshold(val, percentile) for key, (val, _) in runs.items()}
        # one color scale per method and category, so rows are comparable
        scale = {key: (float(test.anomaly_maps.min()), float(test.anomaly_maps.max()))
                 for key, (_, test) in runs.items() if test.has_maps}
        samples = []
        for i in choose_examples(primary, max_samples):
            rel = str(primary.paths[i])
            rgb = denormalize_image(image_tf(Image.open(root / rel).convert("RGB")),
                                    mean=norm["mean"], std=norm["std"])
            verdicts = []
            maps = {}
            for key, (_, test) in runs.items():
                j = index[key].get(rel)
                if j is not None and test.has_maps:
                    maps[key] = test.anomaly_maps[j]
                    verdicts.append(f"{key.upper()}={'DEFECT' if test.scores[j] >= thr[key] else 'NORMAL'}")
            defect = primary.defect_types[i] if primary.defect_types is not None else ""
            samples.append({
                "image_rgb": rgb, "gt_mask": primary.gt_masks[i], "ae_map": maps.get("ae"),
                "patchcore_map": maps.get("pc"),
                "ae_range": scale.get("ae"), "patchcore_range": scale.get("pc"),
                "caption": f"{cat}/{defect}\nGT={'DEFECT' if primary.labels[i] else 'NORMAL'} | "
                           f"{' '.join(verdicts)}\n({Path(rel).name})",
            })
        path = out_dir / f"gallery_{cat}.png"
        save_localization_gallery(samples, path, title=f"Localization comparison: {cat}")
        saved.append(path)
        print(f"Saved {path}")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pred-dir", default="results/predictions")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--categories", nargs="*")
    args = ap.parse_args()
    build_galleries(args.pred_dir, args.out_dir, load_protocol(args.protocol), categories=args.categories)


if __name__ == "__main__":
    main()
