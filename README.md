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
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Google Colab (used for full runs): no venv needed.
```bash
!git clone https://github.com/KNV-DSEB/VISUAL-ANOMALY-INSPECTION.git
%cd VISUAL-ANOMALY-INSPECTION
!pip install -q -r requirements.txt
!python scripts/download_mvtec.py
```
Keep checkpoints and full predictions on Drive; only small CSV/MD summaries go into `results/`.

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

## Methods

All methods write predictions in the shared schema (`docs/evaluation_schema.md`) to
`results/predictions/<method>/<category>_{val,test}.npz`, then one harness computes the metrics:
```bash
python scripts/evaluate.py --methods oneclass      # -> results/metrics/oneclass.csv
```

### One-class (`oneclass`)
Frozen ImageNet ResNet features (global-average-pooled `layer2` + `layer3`, concatenated),
L2-normalised and standardised, then a One-Class SVM (RBF) fit on `train/good` of each category.
Anomaly score = negative SVM decision value (higher = more anomalous). Image-level only, so
pixel AUROC / PRO are NaN for this method. Config: `configs/oneclass.yaml`.
```bash
python scripts/train_oneclass.py                                  # all categories
python scripts/train_oneclass.py --categories bottle --device cpu
python scripts/train_oneclass.py --max-train 50 --pred-dir results/predictions_n50   # sensitivity to #normal images
```
Backbone fallback for small GPUs / CPU: set `features.backbone: resnet18`.
Hyperparameters (`nu`, `gamma`, layers) are fixed in the config; validation contains normal
images only and the test set is never used for tuning.

## Reproducibility
- One protocol for every method: `configs/protocol.yaml` (categories, preprocessing, 20% val split, seed 42).
- Every run is config-driven; CLI flags only override config keys.
- `src/utils/repro.py`: `set_seed` (python / numpy / torch, deterministic cuDNN), `resolve_device`
  (`auto` = cuda > mps > cpu) and `environment_info` (git commit, library versions, device).
- Each prediction file stores its run metadata in `meta` (config path, backbone, pinned weights,
  hyperparameters, #train images, git commit, versions, timings). No absolute paths.
- Pretrained weights are pinned by name (`IMAGENET1K_V1`).
- Data, checkpoints (`checkpoints/`) and `.npz` predictions are never committed.
- Tests: `python -m pytest` (data tests are skipped when MVTec AD is not downloaded).

## Layout
- `configs/` experiment configs (YAML)
- `src/` data, methods, evaluation, utils
- `scripts/` train / evaluate entry points
- `notebooks/` exploration only
- `demo/` inspection demo
- `docs/` proposal & reports
- `results/` small metric summaries
- `tests/` unit tests
