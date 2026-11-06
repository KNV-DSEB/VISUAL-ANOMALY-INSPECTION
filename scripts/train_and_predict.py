"""Train a method on normal images and write val/test predictions in the shared schema.

    python scripts/train_and_predict.py --method patchcore --categories bottle
    python scripts/train_and_predict.py --method all
    python scripts/train_and_predict.py --method patchcore --config configs/experiments/patchcore_refined.yaml \
        --pred-dir results/predictions_refined

Then evaluate with: python scripts/evaluate.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import load_protocol  # noqa: E402
from src.methods import METHOD_REGISTRY  # noqa: E402
from src.methods.runner import run_method  # noqa: E402
from src.utils.io import load_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--method", default="all", choices=["all", *METHOD_REGISTRY])
    ap.add_argument("--config", help="method config (single method only); default configs/<method>.yaml")
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--categories", nargs="*", help="default: categories in the protocol")
    ap.add_argument("--pred-dir", default="results/predictions", help="must match configs/evaluation.yaml")
    ap.add_argument("--checkpoint-dir", default="checkpoints", help="never committed")
    ap.add_argument("--max-train", type=int, help="cap on train images per category (few-shot study)")
    ap.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    args = ap.parse_args()

    methods = list(METHOD_REGISTRY) if args.method == "all" else [args.method]
    if args.config and len(methods) > 1:
        ap.error("--config can only be used with a single --method")
    protocol = load_protocol(args.protocol)

    for name in methods:
        config_path = args.config or f"configs/{name}.yaml"
        cfg = load_yaml(config_path)
        print(f"== {name} (config: {config_path})")
        for r in run_method(name, cfg, protocol, pred_dir=args.pred_dir, ckpt_dir=args.checkpoint_dir,
                            categories=args.categories, max_train=args.max_train, device=args.device,
                            config_path=config_path):
            print(f"   {r['category']:12s} train={r['n_train']:4d} val={r['n_val']:3d} test={r['n_test']:4d} "
                  f"val_threshold={r['val_threshold']:.4f} fit={r['fit_seconds']}s")
    print(f"Predictions written to {args.pred_dir}. Evaluate with: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
