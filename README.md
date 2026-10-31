# Visual Anomaly Inspection

Course project (Computer Vision): **Automated visual quality inspection with few defect samples** on MVTec AD.

Compared methods (trained on normal images only):
1. Autoencoder reconstruction
2. PatchCore-style feature memory
3. One-class classification

Status: bootstrap. See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for current phase and decisions.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Data
MVTec AD (non-commercial, CC BY-NC-SA 4.0) is not committed. Download the selected categories
from the Hugging Face mirror `foersben/mvtec-ad` (about 1 GB for 5 categories):
```bash
python scripts/download_mvtec.py          # categories from configs/protocol.yaml
python scripts/eda_mvtec.py               # counts, sanity checks, sample grid -> results/
python -m pytest                          # data tests
```
If Hugging Face answers HTTP 429, set `HF_HUB_DISABLE_XET=1` and rerun (the download resumes).

Shared protocol (categories, preprocessing, validation split, seed): `configs/protocol.yaml`.
Use `src.data.MVTecDataset(protocol, split)` with `split` in `train`, `val`, `test`.

## Layout
- `configs/` experiment configs (YAML)
- `src/` data, methods, evaluation, utils
- `scripts/` train / evaluate entry points
- `notebooks/` exploration only
- `demo/` inspection demo
- `docs/` proposal & reports
- `results/` small metric summaries
- `tests/` unit tests
