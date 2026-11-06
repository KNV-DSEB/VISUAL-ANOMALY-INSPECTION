# Visual Anomaly Inspection (MVTec AD — Topic 2)

Course project for **Computer Vision**: **Automated visual quality inspection with few defect samples** on the [MVTec Anomaly Detection (MVTec AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad) benchmark.

We implement, validate, and compare three anomaly detection approaches trained **exclusively on normal (`good`) images** under a strictly shared evaluation protocol:
1. **Convolutional Autoencoder Reconstruction (`autoencoder`)** — image-level scoring + pixel-level reconstruction error maps.
2. **PatchCore-style Feature Memory (`patchcore`)** — multi-scale local patch embeddings (`layer2 + layer3`), k-Center Greedy coreset subsampling, density-reweighted k-NN scoring, and pixel-level localization maps.
3. **One-Class Classification (`one_class`)** — multi-scale global CNN representations (`StandardScaler + PCA + One-Class SVM`).

**Extensions included:**
- Pixel-level defect localization evaluated with both **Pixel AUROC** and **Per-Region Overlap (PRO)**.
- Systematic sensitivity study as normal training sample counts drop ($N \in \{10, 25, 50, 100, \text{full}\}$).
- Asymmetric **false-alarm vs. missed-defect cost analysis** across validation percentile operating points.
- Interactive **3-Tab Gradio Industrial QA Dashboard** (`demo/app.py`) ready for local execution or **Hugging Face Spaces** deployment.

---

## 1. Repository Architecture

```text
VISUAL-ANOMALY-INSPECTION/
├── configs/
│   ├── protocol.yaml                  # Shared dataset categories, preprocessing, split & P95 threshold rule
│   ├── autoencoder.yaml               # Autoencoder architecture, loss & top-k scoring config
│   ├── patchcore.yaml                 # PatchCore backbone, patch pooling & coreset config
│   ├── one_class.yaml                 # One-Class SVM feature extraction, PCA & RBF config
│   └── experiments/
│       ├── few_shot.yaml              # Normal-sample sensitivity & false-alarm cost config (I-01)
│       └── patchcore_refined.yaml     # Refined PatchCore configuration (I-02)
├── src/
│   ├── data/mvtec.py                  # Shared manifest builder, MVTecDataset & train subsampling
│   ├── methods/                       # BaseAnomalyMethod + autoencoder, patchcore, one_class modules
│   ├── evaluation/                    # Schema validation, val-only thresholds, AUROC/F1/PRO & error analysis
│   └── utils/                         # Seed, YAML/JSON I/O, backbones & heatmap/gallery visualization
├── scripts/
│   ├── download_mvtec.py              # Download selected MVTec AD categories from Hugging Face
│   ├── eda_mvtec.py                   # Category/defect count summary & sample grid
│   ├── train_and_predict.py           # Train & export standardized prediction bundles
│   ├── evaluate.py                    # Compute comparison tables, localization tables & error cases
│   ├── generate_galleries.py          # Generate 4-column localization galleries (Input|GT|AE|PatchCore)
│   ├── run_sensitivity.py             # Run sample-count sensitivity & false-alarm cost study
│   └── predict_single.py              # Single-image inspection CLI entry point
├── demo/
│   └── app.py                         # 3-Tab Gradio Industrial QA Dashboard
├── notebooks/
│   └── colab_runner.ipynb             # 1-click Google Colab GPU execution notebook
├── docs/                              # Protocol specification, Proposal, Milestone & Final Report outlines
├── results/                           # Lightweight CSV metric tables & figures
└── tests/                             # Fast, low-RAM unit & smoke tests (pytest)
```

---

## 2. Local Setup & Verification

```powershell
python -m venv .venv
.\.venv\Scripts\activate               # Windows PowerShell
pip install -r requirements.txt

# Run the full unit & smoke test suite (runs in ~35s on CPU, <200 MB RAM, no dataset required)
python -m pytest -v
```

---

## 3. Running Full Experiments (Local GPU or Google Colab)

### Option A: Running on Google Colab (Recommended for Low-RAM / No-GPU Machines)
Open [notebooks/colab_runner.ipynb](file:///d:/code/Python/DSEB_ex/ComputerVision/VISUAL-ANOMALY-INSPECTION/notebooks/colab_runner.ipynb) in Google Colab (Runtime $\rightarrow$ Change runtime type $\rightarrow$ **T4 GPU**) and click **Run all**. It automatically mounts Google Drive, downloads the 5 MVTec AD categories, trains all methods, runs evaluation/galleries/sensitivity, and packages `results_package.zip`.

### Option B: Running via CLI Commands
```bash
# 1. Download MVTec AD (bottle, carpet, leather, screw, transistor) & run EDA
python scripts/download_mvtec.py
python scripts/eda_mvtec.py

# 2. Train and generate predictions for all 3 baseline methods
python scripts/train_and_predict.py --method all --artifacts-dir artifacts

# 3. Run refined PatchCore configuration (I-02)
python scripts/train_and_predict.py \
    --method patchcore \
    --config configs/experiments/patchcore_refined.yaml \
    --run-name patchcore_refined \
    --artifacts-dir artifacts

# 4. Compute unified comparison tables, Pixel AUROC / PRO, and error analysis (E-02, E-03, L-01)
python scripts/evaluate.py --artifacts-dir artifacts --results-dir results

# 5. Generate qualitative localization galleries (L-02)
python scripts/generate_galleries.py --artifacts-dir artifacts --results-dir results

# 6. Run normal-sample sensitivity & false-alarm cost trade-off analysis (I-01)
python scripts/run_sensitivity.py --artifacts-dir artifacts --results-dir results
```

---

## 4. Single-Image Inference CLI & Interactive Gradio Demo (DM-01)

### Single-Image CLI
```bash
python scripts/predict_single.py \
    --image path/to/image.png \
    --category bottle \
    --method patchcore \
    --output-dir results/single_inspection \
    --allow-fallback
```

### Launching the 3-Tab Gradio Web UI Locally
```bash
python demo/app.py
```
Then open `http://127.0.0.1:7860` in your browser.

### Deploying to Hugging Face Spaces (So the Instructor Can Open Directly via Link)
1. Create a new **Space** on [Hugging Face Spaces](https://huggingface.co/spaces) and select **Gradio** as the Space SDK.
2. In your Space settings or `README.md` YAML header, set `app_file: demo/app.py`.
3. Push this repository (along with your lightweight trained checkpoints in `artifacts/checkpoints/` if desired, or let the built-in sample gallery + fallback mode run automatically) to the Space repository.
