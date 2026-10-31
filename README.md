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
Dataset (MVTec AD) is not committed; download instructions will be added in the Data phase.

## Layout
- `configs/` experiment configs (YAML)
- `src/` data, methods, evaluation, utils
- `scripts/` train / evaluate entry points
- `notebooks/` exploration only
- `demo/` inspection demo
- `docs/` proposal & reports
- `results/` small metric summaries
- `tests/` unit tests
