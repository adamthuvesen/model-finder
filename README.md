# Model Finder

![License](https://img.shields.io/github/license/adamthuvesen/model-finder) ![Python](https://img.shields.io/badge/python-3.13%2B-blue)

A model-search engine for a coding agent running under a native client `/goal`.
The client keeps the agent going. This repo makes the result trustworthy:

- Candidates receive search rows only. The holdout is sealed outside the workspace.
- Selection is frozen before the holdout is read, and a read holdout leaves a permanent record.
- Every result is bound to its specification, data, and code, and a completion check verifies the chain.

This repo was named `agentic-ml-loop` while it was a supervisor loop.
[Read why that loop was replaced](.agents/docs/decisions/native-goals.md).

## Migrating from `python -m loop`

The loop is gone. To keep using it, check out the git tag `loop-final`. Old `results.json` files fail validation, so
rerun candidates in a new experiment.

## Try it offline

Requires Python 3.13+ and `uv`. This costs nothing and calls no LLM.

```bash
uv sync --extra models
demo() { uv run --extra models python -m demos.demo_bootstrap "$@"; }

# First leaderboard: three commands, with no holdout. Results stay labelled exploratory.
demo init-demo --experiment experiments/quick --no-holdout
demo run-candidate --experiment experiments/quick --candidate logreg-tiny
uv run experiment leaderboard experiments/quick

# The recommended path seals a holdout:
demo init-demo --experiment experiments/my_run          # seeds files, seals the holdout
demo run-candidate --experiment experiments/my_run --candidate majority-baseline
demo run-candidate --experiment experiments/my_run --candidate logreg-tiny
demo freeze-candidate --experiment experiments/my_run --candidate logreg-tiny
demo evaluate-final --experiment experiments/my_run     # reads the holdout once
uv run experiment validate experiments/my_run
uv run experiment replay experiments/my_run           # writes outputs/replay.html
```

`evaluate-final` needs a frozen selection. A second read needs `--allow-reread --reason <text>`, and it labels the
estimate exploratory. Sealed rows are written under
`$XDG_DATA_HOME/model-finder/holdouts/`, defaulting to `~/.local/share`.
Open [`examples/demo_bootstrap_replay.html`](examples/demo_bootstrap_replay.html) to see a replay
without running anything.

## Use your own CSV

```bash
uv run experiment new churn --csv rows.csv --target label   # prints a first leaderboard
uv run experiment candidate new gbdt                        # adds one small function to experiments/churn/model/candidates.py
uv run python experiments/churn/model/run.py run-candidate --candidate gbdt
uv run experiment next                                      # says what could come next
```

A candidate is one function. It gets train and validation rows and returns a function that predicts. The engine scores it
and records the result. Lines marked `CHANGE` are the choices to review. A date-like column sets the split order. Pass
`--no-holdout` to keep every row for search. The result is then labelled exploratory.

## Use it from another repository

```bash
uv add model-finder                       # or, to develop against a local engine checkout:
uv add --editable ../model-finder
uv run experiment setup                      # links skills, adds the holdout deny rule and the agent hooks
```

Your candidates, loaders, and metrics live in your own modules: `RunnerConfig`, `Candidates`, and `Scoring(..., evaluate=...)`
take them with no engine change.

## Run research

Inside Claude Code or Codex, start one goal that names the skill and the experiment:

```text
/goal Use the research skill to answer the question in experiments/my_run/experiment.md.
```

To create a new experiment from a dataset, use the `experiment-spec` skill.
Finish with `uv run experiment complete-check experiments/my_run`. It prints `complete`, `complete, exploratory`,
or `not complete`.
`uv run experiment contract evidence` explains the labels and what they do not prove.

## Demos

| Demo | Problem | Install |
| --- | --- | --- |
| `demo_bootstrap` | Small classification smoke test | `--extra models` |
| `demo_classification` | Binary classification | `--extra models` |
| `demo_regression` | Zero-inflated revenue regression | `--extra models` |
| `demo_deep` | Nonlinear classification with PyTorch MLPs | `--extra deep` |

All data is synthetic and generated in memory. Run `python -m demos.<demo> list-candidates`.
Each `init-demo` generates new rows, so you can run a demo as many times as you like.

## Warehouse snapshots (optional)

Pull once from Snowflake, BigQuery, Redshift, Databricks, Postgres, or DuckDB, then run offline.

```bash
uv sync --extra postgres              # or: snowflake | bigquery | databricks | redshift
uv run experiment sources list
uv run experiment pull --experiment experiments/<id> --source duckdb \
  --database path/to.duckdb --query "SELECT * FROM events ORDER BY id"
uv run experiment seal --experiment experiments/<id> --time-column event_date
```

Pulls refuse any write, and no flag overrides that. Each source has a `SETUP.md` under
[`src/model_finder/sources/bundles/`](src/model_finder/sources/bundles/). DuckDB and local Postgres are tested end
to end. The other connectors are tested against mocked clients.

## Develop

```bash
uv sync --extra models --extra deep
make quick    # only changed Python files, no tests
make check    # the gate: lock, ruff, strict types, import contracts, ast-grep, deptry, vulture, tests
```
