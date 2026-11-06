# PROJECT_CONTEXT

## Topic
Topic 2: Automated visual quality inspection with few defect samples (MVTec AD).
Core task: implement and compare (1) Autoencoder reconstruction, (2) PatchCore-style feature memory, (3) One-class classification.
Extensions (secondary): pixel-level localization, sensitivity to #normal training samples, false-alarm cost.

## Team
| Member | Role |
|---|---|
| Khánh | Lead / integration, final comparison, repo quality |
| Vũ | Data/EDA + Autoencoder |
| Hoàng | Data pipeline + PatchCore |
| Phong | One-class + reproducibility (configs, README) |
| Kiên | Evaluation lead (metrics, harness, thresholds) |
| Huy | Localization + demo |

Ownership changes require the user's approval.

## Phase
All 5 implementation stages completed on branch `hoang_ver` and verified with `pytest`:
1. Data pipeline & deterministic train subsampling (`D-01`, `D-02`).
2. Evaluation harness with image metrics, `Pixel AUROC`, `PRO`, validation thresholds & false-alarm cost analysis (`E-01`, `L-01`).
3. Three anomaly detection methods: Autoencoder (`AE-01..02`), PatchCore (`PC-01..02`), One-Class SVM (`OC-01..02`).
4. Experiment scripts, gallery generation, sensitivity study, refined PatchCore config & Colab runner (`E-02..03`, `L-02`, `I-01..02`).
5. Single-image CLI, 3-tab Gradio QA Dashboard (`DM-01`) & documentation/report templates (`docs/`, `README.md`).

## Decisions made
- Remote is `KNV-DSEB/VISUAL-ANOMALY-INSPECTION`.
- Categories: `bottle`, `carpet`, `leather`, `screw`, `transistor`.
- Preprocessing: resize 256, center-crop 224, ImageNet normalization; masks resized with nearest.
- Validation: 20% of `train/good` (seed 42). Test set only for final reporting.
- Dataset source: Hugging Face mirror `foersben/mvtec-ad`.
- Threshold rule: P95 percentile of validation normal scores; asymmetric cost trade-off explored across validation percentiles in `I-01`.
- Localization metrics: both `Pixel AUROC` and `PRO` (integrated up to FPR = 0.3).
- Backbone: `wide_resnet50_2` default on GPU, configurable to `resnet18` or `tiny_test` for lightweight local testing.
- Demo UI: 3-tab Gradio dashboard (`demo/app.py`) deployable to Hugging Face Spaces.

## Next tasks
1. Run `notebooks/colab_runner.ipynb` on Google Colab GPU to generate full benchmark artifacts and CSV/figure outputs.
2. Populate final tables in `docs/milestone/` and `docs/final_report/` from `results/metrics/*.csv`.
