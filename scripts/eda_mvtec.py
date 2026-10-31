"""EDA for the selected MVTec AD categories.

Writes:
  results/eda_counts.csv      images per category / split / defect type
  results/eda_samples.png     one normal + one defect (with mask overlay) per category
Also validates that every image opens and every defect image has a mask.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.mvtec import REPO_ROOT, build_manifest, load_protocol, resolve_root  # noqa: E402


def main() -> None:
    protocol = load_protocol()
    root = resolve_root(protocol)
    df = build_manifest(protocol)

    problems = []
    sizes = {}
    for _, r in df.iterrows():
        try:
            with Image.open(root / r["path"]) as im:
                im.verify()
            with Image.open(root / r["path"]) as im:
                sizes.setdefault(r["category"], set()).add((im.size, im.mode))
        except Exception as e:  # noqa: BLE001
            problems.append(f"{r['path']}: {e}")
        if r["label"] == 1 and not r["mask_path"]:
            problems.append(f"{r['path']}: defect image without mask")

    counts = (df.groupby(["category", "split", "defect_type"]).size()
              .rename("n_images").reset_index())
    out = REPO_ROOT / "results"
    out.mkdir(exist_ok=True)
    counts.to_csv(out / "eda_counts.csv", index=False)

    print(df.groupby(["category", "split"]).size().unstack(fill_value=0))
    print("\nImage size/mode per category:")
    for cat, s in sizes.items():
        print(f"  {cat}: {sorted(s)}")
    print(f"\nProblems: {len(problems)}")
    for p in problems[:20]:
        print("  ", p)

    cats = protocol["dataset"]["categories"]
    fig, axes = plt.subplots(2, len(cats), figsize=(3 * len(cats), 6))
    for j, cat in enumerate(cats):
        good = df[(df.category == cat) & (df.split == "train")].iloc[0]
        bad = df[(df.category == cat) & (df.label == 1)].iloc[0]
        axes[0, j].imshow(Image.open(root / good["path"]).convert("RGB"))
        axes[0, j].set_title(f"{cat} / normal", fontsize=9)
        img = np.array(Image.open(root / bad["path"]).convert("RGB"))
        mask = np.array(Image.open(root / bad["mask_path"]).convert("L")) > 0
        img[mask] = (0.5 * img[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        axes[1, j].imshow(img)
        axes[1, j].set_title(f"{bad['defect_type']} (mask in red)", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "eda_samples.png", dpi=110)
    print(f"\nSaved {out / 'eda_counts.csv'} and {out / 'eda_samples.png'}")


if __name__ == "__main__":
    main()
