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

Ownership changes require the user's approval. The timeline sheet (25 tasks) has no owners assigned yet.

## Phase
Data phase merged on `main`. Teammate work (`Kien`, `Phong`, `hoang_ver`) has been combined on branch `integration`
(not merged into `main`, not pushed until the user approves): Kien's harness and schema, Phong's one-class and reproducibility
helpers, and Hoang's autoencoder, PatchCore, runner, scripts and demo ported onto the shared schema.
Everything runs and is tested on small synthetic data and on one MVTec category with light backbones.
No full benchmark has been run yet: there are no results to report.

Next: review and merge `integration`, run the full benchmark on Colab (E-02), then analysis, milestone and report.

## Decisions made
- Legacy name is obsolete; remote is `KNV-DSEB/VISUAL-ANOMALY-INSPECTION`.
- Categories: bottle, carpet, leather, screw, transistor.
- Preprocessing: resize 256, center-crop 224, ImageNet normalization; masks resized with nearest.
- Validation: 20% of train/good (seed 42). Test set only for final reporting; never used for thresholds, checkpoints or hyperparameters.
- Dataset source: Hugging Face mirror `foersben/mvtec-ad`.
- Threshold rule: per category, P95 of validation normal scores (`percentile_threshold`, shared by runner and harness).
- Evaluation schema: Kien's per-(method, category, split) `.npz` (`docs/evaluation_schema.md`), one harness for all methods.
- One-class: Phong's implementation (pinned weights, hyperparameters fixed in config). Hoang's multi-scale variant is not a baseline.
- Pretrained weights pinned by name (IMAGENET1K_V1).
- Demo: Gradio, only real checkpoints and saved tables (no fallback model, no simulated data).
- Commits carry no Claude attribution (see .claude/settings.json).
- Full runs use Google Colab; the local machine (GTX 1650 4 GB, CPU-only torch) is for development and tests.

## Open items
- Ownership of the autoencoder and the demo (currently implemented by Hoang; Vu and Huy have no branch).
- `scripts/train_oneclass.py` and `src/methods/one_class/pipeline.py` overlap with the shared runner; retire after Phong agrees.
- PRO convention (needs a decision before the full run): the harness measures the false-positive rate on pixels of anomaly-free
  images only. Counting all non-defect pixels of all test images (the dataset-level reading of the metric) gives a different value:
  on `bottle` with a light PatchCore run, 0.989 versus 0.909 (pixel AUROC identical); for the autoencoder 0.133 versus 0.129.
  Both implementations match their own exact reference, so the gap is the convention. The first one flatters methods whose false
  positives sit next to defects.
- Deadlines for proposal, milestone and final (the timeline assumes everything fits in 31/10 to 30/11).
- Backbone default WideResNet50-2 may be heavy for a 4 GB GPU; ResNet18 is the fallback.

## Next tasks
1. Review and merge `integration` (needs the user's decision on the open items).
2. E-02: full benchmark on Colab with `notebooks/colab_runner.ipynb`.
3. Assign owners in the timeline sheet.
