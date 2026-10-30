# Retire a read holdout

A user replaces a read holdout with unseen rows. Search reopens and the history records the old estimate as exploratory.

Source: src/model_finder/final_evaluation.py, src/model_finder/cli/main.py
Inventory-id: experiment:reseal, experiment:validate

## Sub-features

- `reseal` seals 80 new rows, writes `work/holdout_history/0001/consumed.json`, and makes `validate` warn.

## How to get to it (user POV)

- Run `experiment reseal --experiment <dir> --rows <parquet> --note <text>`.

## Driving it with verify

Preconditions: `final-after-freeze` passed in this session.

- **Reseal.** Run `verify drive reseal`. Exit 0, the history file exists, and `validate` prints `holdouts retired`.
- **Proof.** Artifacts in the run directory: `reseal.out`, `reseal-warning.out`, and the manifest blocks.

## Gotchas

- New rows must not repeat search rows or old holdout rows. The drive shifts dates to avoid that.
