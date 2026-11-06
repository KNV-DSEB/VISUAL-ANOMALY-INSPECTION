"""Evaluate saved predictions for one or more methods.

    python scripts/evaluate.py --config configs/evaluation.yaml
    python scripts/evaluate.py --methods patchcore
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.harness import evaluate_method, write_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/evaluation.yaml")
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--methods", nargs="*")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    categories = cfg["categories"]
    proto = Path(args.protocol)
    if proto.exists():
        categories = yaml.safe_load(proto.read_text(encoding="utf-8")).get("categories", categories)
    methods = args.methods or cfg["methods"]

    all_rows = []
    for m in methods:
        rows = evaluate_method(cfg["pred_dir"], m, categories, cfg)
        write_table(rows, Path(cfg["out_dir"]) / f"{m}.csv", Path(cfg["out_dir"]) / f"{m}.json")
        all_rows += rows
        o = rows[-1]
        print(f"{m:12s} img-AUROC={o['image_auroc']:.4f} F1={o['f1']:.4f} "
              f"pix-AUROC={o['pixel_auroc']:.4f} PRO={o['pro']:.4f}")
    write_table(all_rows, Path(cfg["out_dir"]) / "comparison.csv")


if __name__ == "__main__":
    main()
