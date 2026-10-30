# Architecture

A native client `/goal` runs the agent. The agent reads the experiment directory and calls the
engine's commands. The engine holds no long-lived state of its own.

```text
/goal in Claude Code or Codex
  -> research skill
  -> python -m demos.<id> (or your own runner) -> agentic_ml_loop.runner
  -> experiment validate | seal | reseal | complete-check | replay | pull | sources
```

## Layers

`make boundaries` enforces this order. A module imports only from layers below it.

| Layer | Modules |
| --- | --- |
| CLI | `cli/` |
| Consumers | `completion`, `viz` |
| Workflows | `final_evaluation`, `runner` |
| Checks | `validation`, `data_audit` |
| Warehouse | `sources/` |
| Evidence | `run_evidence`, `holdout`, `spec` |
| Records | `results`, `ledger` |
| Models | `candidate_result`, then `splits` |
| I/O | `io`, `paths` |

## A candidate

```python
def run_baseline(splits: SearchSplits) -> FittedCandidate: ...
```

A candidate receives `SearchSplits` and returns a `CandidateResult` plus a `score_rows` function.
It never sees holdout rows. After a freeze, the engine refits the candidate and passes the sealed
rows to `score_rows`. See `demos/common.py` for a worked helper.

## Experiment directory

```text
experiment.md         Spec. `(val_auc)` in `## Objective Metric` is machine-parsed.
research_journal.md   Head each entry `## Cycle NNNN: <title>`. Replay reads these.
research_sources.md   Reusable research notes.
results.json          The leaderboard. Only the engine writes it.
holdout_seal.json     Experiment uid, holdout hash, and row count.
data/                 Search rows and warehouse snapshots.
work/                 Scratch, run records, selection, holdout history.
outputs/              Deliverables, final evaluation, completion record.
scripts/              One-shot scripts.
```

Sealed rows and the ledger live under `$XDG_DATA_HOME/agentic-ml-loop/holdouts/`.
Read [evidence.md](../../docs/evidence.md) for the chain and its limits.
