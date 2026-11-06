"""Fit the one-class detector (CNN features + OC-SVM) and write val/test predictions.

    python scripts/train_oneclass.py --config configs/oneclass.yaml
    python scripts/train_oneclass.py --categories bottle --device cpu
    python scripts/train_oneclass.py --max-train 50 --pred-dir results/predictions_n50

Then: python scripts/evaluate.py --methods oneclass
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.mvtec import load_protocol  # noqa: E402
from src.methods.one_class.pipeline import _resolve, run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/oneclass.yaml")
    ap.add_argument("--categories", nargs="*", help="default: categories in the protocol")
    ap.add_argument("--max-train", type=int, help="override train.max_train")
    ap.add_argument("--device", help="override device (auto/cuda/mps/cpu)")
    ap.add_argument("--pred-dir", help="override pred_dir")
    args = ap.parse_args()

    cfg_path = _resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["config_path"] = args.config  # relative, as typed: no personal absolute paths in meta
    if args.max_train is not None:
        cfg.setdefault("train", {})["max_train"] = args.max_train
    if args.device:
        cfg["device"] = args.device
    if args.pred_dir:
        cfg["pred_dir"] = args.pred_dir

    protocol = load_protocol(_resolve(cfg["protocol"]))
    print(f"oneclass: backbone={cfg['features']['backbone']} layers={cfg['features']['layers']} "
          f"nu={cfg['model']['ocsvm']['nu']} max_train={cfg.get('train', {}).get('max_train')}")
    for r in run(cfg, protocol, args.categories):
        print(f"  {r['category']:12s} train={r['n_train']:4d} val={r['n_val']:3d} "
              f"test={r['n_test']:4d} dim={r['feature_dim']}")
    print(f"Predictions written to {cfg['pred_dir']}/oneclass/. "
          f"Evaluate with: python scripts/evaluate.py --methods oneclass")


if __name__ == "__main__":
    main()
