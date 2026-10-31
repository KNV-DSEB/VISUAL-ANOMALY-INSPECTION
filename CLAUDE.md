# CLAUDE.md

Project: MVTec AD visual anomaly inspection (Topic 2). Read PROJECT_CONTEXT.md first.
Legacy name "DETECTING-AI-GENERATED-IMAGES" is obsolete; never use it.

Rules:
- Reply to the user in Vietnamese.
- Course project: build → validate → compare → demo. No literature review / novelty work.
- Same categories, preprocessing, metric code for all 3 methods.
- Never tune hyperparameters/thresholds/checkpoints on the MVTec test set; use a validation split from normal train data.
- Config-driven (configs/*.yaml); reusable logic in src/, notebooks only for exploration.
- Don't commit data, checkpoints, caches, absolute personal paths.
- Do not change team ownership without asking. Ask before locking major decisions.
- Small commits with conventional prefixes (feat(data): ..., docs: ...).
