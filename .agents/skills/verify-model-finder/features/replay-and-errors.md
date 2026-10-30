# Replay and errors

The replay renders results and the journal to one HTML file. A missing experiment produces one stderr line and exit 1, with no traceback.

Source: src/model_finder/viz/bundle.py, src/model_finder/cli/errors.py
Inventory-id: experiment:replay

## Sub-features

- `replay` writes `outputs/replay.html` naming the leaderboard's candidates.
- `missing-experiment` exits 1 with `does not exist`.

## How to get to it (user POV)

- Run `experiment replay <dir>`.
- Run any runner subcommand against a missing directory.

## Driving it with verify

Preconditions: `search-metrics` passed in this session.

- **Replay.** Run `verify drive replay`. Exit 0, and the copied HTML contains `logreg-tiny`.
- **Error.** Run `verify drive missing-experiment`. Exit 1, and stderr has no traceback.
- **Proof.** Artifacts in the run directory: `replay.html`, `missing-experiment.err`, and the manifest blocks.

## Gotchas

- The replay's scenes come from `## Cycle NNNN:` journal headings. A journal without them still shows the final leaderboard.
