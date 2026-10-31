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

Ownership changes require the user's approval. The timeline sheet (25 tasks) currently has no owners assigned.

## Phase
Data phase (D-01, D-02) implemented, pending commit. Next: evaluation harness (E-01), then the three methods.

## Decisions made
- Legacy name is obsolete; remote is `KNV-DSEB/VISUAL-ANOMALY-INSPECTION`.
- Categories: bottle, carpet, leather, screw, transistor.
- Preprocessing: resize 256, center-crop 224, ImageNet normalization; masks resized with nearest.
- Validation: 20% of train/good (seed 42). Test set only for final reporting.
- Dataset source: Hugging Face mirror `foersben/mvtec-ad` (counts match official MVTec AD: train+val 209/280/245/320/213, test 83/117/124/160/100).
- Threshold rule: percentile of validation normal scores (P95 initially, revisit on validation only).
- Commits carry no Claude attribution (see .claude/settings.json).

## Environment notes
- Training and full runs will use Google Colab (decided by the user). Local machine (GTX 1650 4 GB, Python 3.14, CPU-only torch) is for development and tests only.
- On Colab: clone the repo, `pip install -r requirements.txt`, run `scripts/download_mvtec.py`, keep checkpoints/results on Drive (never commit them).
- 4 GB VRAM may limit PatchCore with WideResNet50; ResNet18 is the fallback.

## Undecided
- One-class method (default plan: pretrained features + OC-SVM)
- Backbone (WideResNet50 vs ResNet18)
- Localization metric beyond pixel AUROC (PRO)
- Demo framework
- Final deadlines for proposal/milestone/final (timeline assumes everything fits in 31/10 to 30/11)

## Next tasks
1. Commit and push the Data phase.
2. E-01: evaluation harness (metrics + tests).
3. Assign owners in the timeline sheet.
