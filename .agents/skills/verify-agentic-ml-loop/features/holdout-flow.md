# Seal, search, select, evaluate

A demo seals its holdout outside the workspace, records search metrics only, refuses holdout evaluation before a freeze, and scores the sealed rows once after it.

Source: demos/demo_bootstrap/__main__.py, src/agentic_ml_loop/cli/runner.py, src/agentic_ml_loop/final_evaluation.py
Inventory-id: runner:init-demo, runner:run-candidate, runner:freeze-candidate, runner:evaluate-final

## Sub-features

- `seal-workspace` seals 160 rows outside the workspace and leaves 640 search rows in `data/`.
- `search-metrics` records a candidate with only `train` and `validation` metrics.
- `final-before-freeze` exits 1 with `No frozen selection`.
- `final-after-freeze` freezes, evaluates, and then refuses further search.

## How to get to it (user POV)

- Run `python -m demos.demo_bootstrap <subcommand> --experiment <dir>`.

## Driving it with verify

Preconditions: `verify launch`; `verify doctor` exits 0.

- **Seal.** Run `verify drive seal-workspace`. Exit 0, and the row count is 640.
- **Search.** Run `verify drive search-metrics`. Exit 0, and `results.json` has only the two split keys.
- **Refuse.** Run `verify drive final-before-freeze`. Exit 1, and stderr names the missing selection.
- **Evaluate.** Run `verify drive final-after-freeze`. Evaluation exits 0, and a later search exits 1.
- **Proof.** Artifacts in the run directory: each `.out` and `.err`, `seal-workspace.rows`, and the manifest blocks.

## Gotchas

- `uv run` with a custom `XDG_DATA_HOME` rebuilds `.venv`. Use the helper.
