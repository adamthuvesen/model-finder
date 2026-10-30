---
name: verify-model-finder
description: Drives the real model-finder CLI and demo runner offline, and keeps evidence. Use to run, launch, drive, or verify model-finder, or to prove a change to it works.
---

# Verify model-finder

The surface is a short-lived CLI: `experiment` and `python -m demos.<id>`. Each drive runs it in its
own temp HOME and data home, offline, on CPU. Run `make setup` first.

## Run

- `make verify` after a change. It runs the drives that cover files changed against `main`, plus
  uncommitted and untracked files, or the smoke drive when nothing matches.
- `make verify ARGS=all` before handing over larger work. Drives run in parallel, about 25s.
- `make verify ARGS=list` shows the drives. `make verify ARGS="<drive> ..."` runs named drives.
- `make verify ARGS=control` re-drives one check against a wrong exit code. It exits 0 only when it
  observes FAIL.

The helper is `.agents/skills/verify-model-finder/verify`. The drives live in `drives.py` beside it.

## Reading results

Each drive prints `PASS|FAIL <drive> <seconds>s <detail>`. A FAIL names the step, the expected exit
and text, and the last line the command printed. The run ends with `evidence: <dir>`. That is
`.verify/runs/<utc>-<sha7>[-dirty]/`, with one folder per drive holding each step's `.cmd`, `.out`,
`.err`, and `.exit`, plus `summary.json`. The helper does not retry, so there is no FLAKY result.

A `tripwire` FAIL means the run called `claude`, `codex`, `modal`, or another stub on `PATH`, or it
changed the real holdout store or `~/.config/model-finder/sources.json`.

## Limits

- No drive pulls from a warehouse or seals a snapshot (`experiment pull`, `experiment seal`).
  `tests/test_sources.py` covers those with DuckDB.
- The torch and GBDT demos are not driven here. `make slow` runs them in `tests/verify`.
- The guard is driven only through the Codex hook. The Cursor and Claude Code clients are not run.

## Gotchas

- `uv build` and `uv pip install` run with the real HOME, because uv keeps its managed Python and
  cache there. They run with `--offline`, so a cold uv cache fails the `wheel` drive. Run `make setup`.
- The first import in a new venv can take 10s on macOS. The `wheel` drive pays this on every run.
- `complete`, `reseal`, and `replay` copy one demo experiment that the run builds once. The copy
  works because the experiment and the holdout store hold no absolute paths.
