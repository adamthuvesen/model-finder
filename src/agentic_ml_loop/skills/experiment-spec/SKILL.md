---
name: experiment-spec
description: Design and scaffold an agentic-ml-loop experiment from raw data and context. Profiles the data, proposes a falsifiable spec, generates candidate code, seals the holdout, runs baselines, and validates. Use when a user wants to turn a dataset and problem description into a runnable experiment. Use research to run one.
---

# Experiment Spec

Produce a validated experiment directory with a sealed holdout, a trivial baseline on the
leaderboard, and a spec the user confirmed.

Done means `uv run experiment validate experiments/<slug>` passes and the checks at the end of
[references/generate.md](references/generate.md) hold.

## NEVER

- Generate complex models as baselines. The trivial anchor is the job here.
- Invent column names, row counts, or metrics. Every number comes from profiling you ran.
- Write a hypothesis you cannot ground and falsify. Write at most 5.
- Hand-roll locking, CLI parsing, or a result model. The engine provides them.
- Skip the confirmation before generating files.
- Read holdout rows while profiling. Seal first when the data can support the split, and profile `data/`.

## Workflow

1. **Locate data.** Use a local file plus `context.md`, or pull a warehouse snapshot:
   `uv run experiment pull --experiment experiments/<slug> --source <type> --query @q.sql`.
   Dry-run first with `--dry-run`. Write explicit columns and an `ORDER BY` on a stable key.
   `uv run experiment sources enable <type>` prints the path of that source's setup steps.
2. **Seal.** This is the recommended path. `--no-holdout` keeps all rows for search, and results stay labelled exploratory. A sealed experiment refuses later pulls. For a snapshot, run `uv run experiment seal --experiment experiments/<slug> --time-column <column>`.
3. **Profile** search rows: dtypes, cardinality, missingness (flag over 10%), target distribution,
   positives and the feature budget (N/10 to N/20), time range, and suspiciously high correlations.
4. **Discover.** Read `learnings.md`. Find 2-4 references for this problem shape.
5. **Propose** with the template in [references/proposal.md](references/proposal.md). Stop and get confirmation.
6. **Generate** by following [references/generate.md](references/generate.md).

## Machine-parsed spec fields

- `## Objective Metric` holds a parenthesized token with a `val_` or `validation_` prefix, such as `(val_auc)`.
- `objective_direction: minimize` on its own line overrides the default of maximize.
- `key_columns: a, b` on its own line names entity columns for the overlap report.
- A bootstrap interval is stored as `hyperparameters["{objective_metric}_ci_95"] = [lo, hi]` and requires `bootstrap_count`.
