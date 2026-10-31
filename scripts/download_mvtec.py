"""Download the selected MVTec AD categories from the Hugging Face mirror.

Usage:
    python scripts/download_mvtec.py                  # categories from configs/protocol.yaml
    python scripts/download_mvtec.py --categories bottle carpet
"""
import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "protocol.yaml"))
    parser.add_argument("--categories", nargs="*", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["dataset"]
    categories = args.categories or cfg["categories"]
    root = REPO_ROOT / cfg["root"]
    root.mkdir(parents=True, exist_ok=True)

    patterns = [f"{c}/*" for c in categories]
    print(f"Downloading {categories} from {cfg['hf_repo']} into {root}")
    snapshot_download(
        repo_id=cfg["hf_repo"],
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(root),
    )
    print("Done.")


if __name__ == "__main__":
    main()
