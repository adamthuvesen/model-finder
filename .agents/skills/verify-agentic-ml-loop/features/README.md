# Features

## Baseline preconditions

Run `verify launch` and `verify doctor`. All data lives under `.verify/state/`.

## Driving conventions

Drive in the table's order. The drives share one experiment, `run_a`. Start again from `launch`.

## Inventory

`.venv/bin/experiment --help` and `.venv/bin/python -m demos.demo_bootstrap --help`. Each subcommand
name is an `Inventory-id`, prefixed `experiment:` or `runner:`.

Unmapped: `experiment:complete-check` (`tests/test_end_to_end.py` covers it, because it needs a written
review), `experiment:seal` and `experiment:pull` (need a warehouse snapshot, tested with DuckDB in
`tests/test_sources.py`), `experiment:sources`,
`runner:list-candidates` (read-only listings).

## Proof, skip, and flake reporting

Record the feature ID and entry point with every artifact. Report an unreachable path with the command
and the unmet prerequisite. A pass on retry is `flaky`.

## Control

`verify control` uses `missing-experiment` and mutates the expected exit code to 0.

## Features

| Feature | File | Inventory ids | Last verified |
| --- | --- | --- | --- |
| Seal, search, select, evaluate | [holdout-flow.md](holdout-flow.md) | runner:init-demo, runner:run-candidate, runner:freeze-candidate, runner:evaluate-final | 49cf9e6-dirty 2026-11-04 created |
| Retire a read holdout | [reseal.md](reseal.md) | experiment:reseal, experiment:validate | 49cf9e6-dirty 2026-11-04 created |
| Replay and errors | [replay-and-errors.md](replay-and-errors.md) | experiment:replay | 49cf9e6-dirty 2026-11-04 created |
