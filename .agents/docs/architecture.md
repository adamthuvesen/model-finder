# Architecture

A native client `/goal` runs the agent. The agent reads the experiment directory and calls the
engine's commands. The engine holds no long-lived state of its own.

```text
/goal in Claude Code or Codex
  -> research skill
  -> experiments/<slug>/model/run.py (or your own runner) -> model_finder.api
  -> experiment next | new | candidate new | seal | reseal | complete-check | pull
```

## Layers

`make boundaries` enforces this order, from `engine-layers` in `pyproject.toml`. A module imports only from layers below it.

## A candidate

A candidate is one function: `@candidates.add def name(splits) -> Predict`. `Candidates` in `candidates.py` scores it on
train and validation, builds the result, and hands the engine `score_rows` for later. `run-candidate` validates it, writes
one run record, and updates the leaderboard. Everything else reads those records.
