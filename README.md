# Visual Anomaly Inspection

Course project (Computer Vision), Topic 2: **automated visual quality inspection with few defect samples** on MVTec AD.

Three methods, all trained on normal images only and compared under one protocol:
1. Autoencoder reconstruction
2. PatchCore-style feature memory
3. One-class classification (pretrained features + One-Class SVM)

Current phase and decisions: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```
Full runs use Google Colab: open `notebooks/colab_runner.ipynb` (GPU runtime) and run the cells in order. Predictions, checkpoints
and result tables are kept on Drive; only light CSV/PNG tables go into `results/metrics` and `results/figures`.

## Data
MVTec AD (non-commercial, CC BY-NC-SA 4.0) is not committed. The five selected categories come from the Hugging Face mirror
`foersben/mvtec-ad` (about 1 GB):
```bash
python scripts/download_mvtec.py          # categories from configs/protocol.yaml
python scripts/eda_mvtec.py               # counts, sanity checks, sample grid -> results/
```
If Hugging Face answers HTTP 429, set `HF_HUB_DISABLE_XET=1` and rerun (the download resumes).

Protocol (categories, preprocessing, 20% validation split, seed, threshold percentile): `configs/protocol.yaml`, explained in
[docs/protocol_note.md](docs/protocol_note.md). `src.data.MVTecDataset(protocol, split)` with `split` in `train`, `val`, `test`.

## Run the benchmark
```bash
python scripts/train_and_predict.py --method all            # or autoencoder | patchcore | oneclass; --categories bottle
python scripts/evaluate.py                                   # results/metrics/<method>.csv and comparison.csv
python scripts/analyze_errors.py                             # defect_breakdown.csv, error_cases.csv
python scripts/generate_galleries.py                         # results/figures/gallery_<category>.png
python scripts/run_sensitivity.py                            # sample_sensitivity.csv, false_alarm_cost.csv
```
Each method writes `results/predictions/<method>/<category>_{val,test}.npz` (schema in
[docs/evaluation_schema.md](docs/evaluation_schema.md)) and a checkpoint in `checkpoints/<method>/<category>/`. One harness
(`src/evaluation`) computes every metric. Thresholds come from validation scores only.

## Methods
| Method | Config | What it does | Output |
|---|---|---|---|
| `autoencoder` | `configs/autoencoder.yaml` | small conv autoencoder, MSE + L1 loss, best checkpoint by validation loss; score = mean of the top 2% smoothed error pixels | image score + pixel map |
| `patchcore` | `configs/patchcore.yaml` | WideResNet50-2 `layer2`+`layer3` patch features, random projection, k-center greedy coreset, k-NN score with density re-weighting | image score + pixel map |
| `oneclass` | `configs/oneclass.yaml` | frozen ResNet features (global average pooled `layer2`+`layer3`), L2 normalise, standardise, One-Class SVM (RBF) | image score only |

Backbone fallback for small GPUs or CPU: `resnet18` in the config. Pretrained weights are pinned by name (`IMAGENET1K_V1`).
`configs/experiments/` holds the study configs (few-shot counts, a refined PatchCore declared in advance).
`scripts/train_oneclass.py` is the original standalone entry point for the one-class method; prefer `train_and_predict.py`.

## Demo
```bash
python demo/app.py                                           # Gradio, needs trained checkpoints
python scripts/predict_single.py --image path/to/img.png --category bottle --method patchcore
```
The demo only uses trained checkpoints and saved evaluation tables. A missing checkpoint is reported with the command to train it;
nothing is simulated.

## Reproducibility
- One protocol for every method; every run is config-driven (CLI flags only override config keys).
- `src/utils/repro.py`: `set_seed` (python, numpy, torch; the runner reseeds before every category), `resolve_device`, `environment_info`.
- Each prediction file stores run metadata in `meta`: method, config path (relative), seed, backbone and weights, sample counts,
  fit and inference time, checkpoint size, git commit and library versions. No absolute paths.
- Data, checkpoints and `.npz` predictions are never committed.
- Tests: `python -m pytest` (data tests are skipped when MVTec AD is not downloaded; the other tests use tiny synthetic data).

## Layout
- `configs/` protocol, method and experiment configs
- `src/` data, methods (autoencoder, patchcore, one_class, runner), evaluation, utils
- `scripts/` train, evaluate, analysis, galleries, sensitivity, single-image inference
- `demo/` Gradio app
- `notebooks/` Colab runner
- `docs/` protocol note, evaluation schema, proposal, milestone and report outlines
- `results/` light metric tables and figures
- `tests/` unit and end-to-end tests
