# Model Search Experiment

## Title

Bootstrap demo: trial conversion (synthetic)

## Goal

Smoke-test the agentic-ml-loop engine (runner, sealed holdout, validation, evidence chain) on a
small bundled dataset with a trivial baseline and a minimal linear model.

## Baseline

**Majority-class** predictor: every row gets the training-set positive rate.

## Dataset

Synthetic rows from `demos/demo_bootstrap/generate_data.py`.

## Data Profile

- **Row count**: 800
- **Feature count**: 4 (3 numeric + 1 categorical)
- **Feature types**: numeric engagement counts, categorical `segment`
- **Missing values**: none in generated data
- **Class balance**: roughly 35-45% positive (synthetic logit)

## Keys

key_columns: user_id

## Target

`converted`

## Problem Type

binary classification

## Split Strategy

Time-ordered 60% train / 20% validation / sealed 20% holdout on `event_date` (no shuffling).

## Objective Metric

Validation ROC-AUC (`val_auc`).

## Candidate Families

- constant / majority baseline
- logistic regression (numeric + one-hot)

## Constraints

- deterministic bundled data
- no external services
- no leakage: `user_id` excluded from features

## Success Definition

- `majority-baseline` and `logreg-tiny` both run and appear in `results.json`
- validation AUC for logistic regression above majority baseline
