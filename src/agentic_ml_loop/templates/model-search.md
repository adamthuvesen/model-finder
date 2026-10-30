# Model Search Experiment

## Title

Short, specific experiment title.

## Goal

What model behavior or business outcome are we trying to improve?

## Baseline

Describe the current baseline, for example:

- rule-based heuristic
- incumbent deployed model
- constant baseline

## Dataset

Where does the dataset come from? Include any local paths or table names.

## Data Source

If this experiment is backed by a warehouse snapshot, record its provenance here.
Research reads only the frozen snapshot, never the live warehouse.

- **Source**: _local_ (or: snowflake | bigquery | redshift | databricks | postgres | duckdb)
- **Query / table**: _n/a_
- **As-of**: _n/a_ (time-travel point, or the modeled time predicate)
- **Materialized snapshot**: _n/a_ (e.g. `data/snapshot.parquet`, frozen via `experiment pull`)

Provenance is captured in `data/dataset_manifest.json`; `experiment validate`
checks the snapshot against it.

## Data Profile

Fill in after the data audit pass:

- **Row count**: _unknown_
- **Feature count**: _unknown_
- **Feature types**: _unknown_ (e.g., 12 numeric, 5 categorical, 2 datetime)
- **Missing values**: _unknown_ (e.g., 3 columns with >10% missing)
- **Class balance**: _unknown_ (e.g., 70/30 positive/negative)

## Keys

Name entity columns for the overlap report: `key_columns: account_id`. Delete this line if none.

## Target

What is the prediction target?

## Problem Type

Examples:

- binary classification
- regression
- multiclass classification
- ranking

## Split Strategy

How should train, validation, and the sealed holdout be defined?

## Objective Metric

What metric determines "better"? Name it with a parenthesized, validation-scoped token:
`val_<name>` or `validation_<name>` (e.g. `(val_auc)`, `(val_r2)`). The engine extracts
this token and checks every candidate's `objective_metric` against it, so it
must be the validation metric each candidate reports. Keep the bare name (`auc`) as the key
inside each candidate's `metrics["validation"]`.

## Candidate Families

Examples:

- logistic regression
- random forest
- xgboost
- lightgbm
- simple neural baseline

## Constraints

Examples:

- interpretability
- max training time
- latency
- monotonicity
- limited feature availability
- no leakage from future data

## Known Risks

Potential issues to watch for:

- **Leakage sources**: _unknown_ (e.g., features derived from future data, target-correlated IDs)
- **Distribution shift**: _unknown_ (e.g., train from 2023, test from 2024)
- **Label noise**: _unknown_ (e.g., labels from heuristic, not ground truth)

## Evaluation Strategy

- **CV scheme**: _unknown_ (e.g., 5-fold stratified, time-based split)
- **Secondary metrics**: _unknown_ (e.g., Brier score, F1, precision@k)
- **Significance requirements**: _unknown_ (e.g., bootstrap p < 0.05 vs baseline)

## Research Pointers

Similar problems, relevant papers, or Kaggle competitions that may inform the search:

- _none yet_

Keep detailed, reusable outside research in `research_sources.md`. Use this
spec for the concise problem framing and research direction.
When new evidence changes the story, rewrite the top synthesis in
`research_sources.md` instead of leaving contradictory takeaways unresolved.
Source cards are supporting provenance; `Reusable Takeaways` is the current synthesis.

## Research Hypotheses

Ranked directions to explore, informed by data profiling and prior knowledge.
Treat these as starting priors: verify, refine, or discard based on evidence.

- _none yet_

## Success Definition

What would make this experiment useful?

- a trustworthy baseline
- a leaderboard with comparable candidates
- clear next moves

## Deliverables

Stakeholder-facing artefacts this experiment should produce. Every path lives
under `outputs/<filename>`, never flat at the experiment root.
`agentic_ml_loop.paths.outputs_dir(exp_dir)` locates this folder.

- `outputs/<final_report>.md`: write-up of findings
- `outputs/<final_results>.csv`: final ranked / scored output

(Replace these placeholders with the concrete deliverables for this experiment.)
