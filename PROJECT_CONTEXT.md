# PROJECT_CONTEXT

## Topic
Topic 2 — Automated visual quality inspection with few defect samples (MVTec AD).
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
Bootstrap done. Waiting for approval of technical plan, then Data phase.

## Decisions made
- Legacy name is obsolete; remote is already `KNV-DSEB/VISUAL-ANOMALY-INSPECTION`.
- No test-set tuning; validation split from normal train images.

## Undecided (need user approval)
- MVTec category scope
- Exact one-class method
- Image size / backbone details
- Threshold policy and localization metric
- Demo framework

## Next tasks
1. Approve plan.
2. Data phase: MVTec download, loader, train/val split, EDA.
