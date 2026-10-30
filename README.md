# Agentic ML Loop

![License](https://img.shields.io/github/license/adamthuvesen/agentic-ml-loop) ![Python](https://img.shields.io/badge/python-3.13%2B-blue)

A model-search engine for a coding agent running under a native client `/goal`.
The client keeps the agent going. This repo makes the result trustworthy:

- Candidates receive search rows only. The holdout is sealed outside the workspace.
- Selection is frozen before the holdout is read, and a read holdout leaves a permanent record.
- Every result is bound to its specification, data, and code, and a completion check verifies the chain.

The repo keeps its name from the supervisor loop it used to be.
[Read why that loop was replaced](.agents/docs/decisions/native-goals.md).

## Migrating from `python -m loop`

The loop, its runner presets, the referee, and the benchmark are gone. To keep using them, check
out the git tag `loop-final`. To move forward, start a `/goal` as shown below. Old `results.json`
files fail validation, because many hold test metrics. Preserve them and rerun candidates in a new
experiment.

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
`$XDG_DATA_HOME/agentic-ml-loop/holdouts/`, defaulting to `~/.local/share`.
Open [`examples/demo_bootstrap_replay.html`](examples/demo_bootstrap_replay.html) to see a replay
without running anything.

## Use your own CSV

Put this file at `my_model/run.py`. The runner fingerprints its own folder, so keep data and experiments outside it.

```python
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from agentic_ml_loop.candidate_result import CandidateResult, FittedCandidate, SearchMetrics
from agentic_ml_loop.cli.runner import RunnerConfig, run_runner_main
from agentic_ml_loop.runner import ExperimentSeed

CSV, TARGET, FEATURES = Path("rows.csv"), "label", ["x"]


def logreg(splits):
    model = LogisticRegression().fit(splits.train[FEATURES], splits.train[TARGET])

    def score(rows):
        return {"auc": float(roc_auc_score(rows[TARGET], model.predict_proba(rows[FEATURES])[:, 1]))}

    metrics = score(splits.validation)
    result = CandidateResult(
        candidate_id="logreg",
        objective_metric="val_auc",
        objective_score=metrics["auc"],
        metrics=SearchMetrics(validation=metrics),
    )
    return FittedCandidate(result=result, score_rows=score)


raise SystemExit(
    run_runner_main(
        RunnerConfig(
            seed=ExperimentSeed("my_run"),
            candidate_runners={"logreg": logreg},
            rows_loader=lambda seed: pd.read_csv(CSV),
            rows_source=CSV,
            time_column="when",
            target_column=TARGET,
            code_root=Path(__file__).parent,
        )
    )
)
```

```bash
uv run --extra models python my_model/run.py init-demo --experiment experiments/my_run --no-holdout
uv run --extra models python my_model/run.py run-candidate --experiment experiments/my_run --candidate logreg
uv run experiment leaderboard experiments/my_run
```

Drop `--no-holdout` to seal a holdout. The sealed rows stay readable in `rows.csv`, and the seal warns about that.

## Run research

Inside Claude Code or Codex, start one goal that names the skill and the experiment:

```text
/goal Use the research skill to answer the question in experiments/my_run/experiment.md.
```

To create a new experiment from a dataset, use the `experiment-spec` skill.
Finish with `uv run experiment complete-check experiments/my_run`. It prints `complete`, `complete, exploratory`,
or `not complete`.
[docs/evidence.md](docs/evidence.md) explains the chain and what it does not prove.

## Replacing a read holdout

If a holdout informed a decision, seal new rows:

```bash
uv run experiment reseal --experiment experiments/my_run --rows new_rows.parquet --note "Why, and where the rows came from"
```

Search reopens, and the old estimate stays on record as exploratory. A holdout's identity is its sealed file's
hash. Reading a file a second time, in any experiment, needs `--allow-reread --reason <text>`. The engine does not recognize similar data: [docs/evidence.md](docs/evidence.md) lists the limits.

## Demos

| Demo | Problem | Install |
| --- | --- | --- |
| `demo_bootstrap` | Small classification smoke test | `--extra models` |
| `demo_classification` | Binary classification | `--extra models` |
| `demo_regression` | Zero-inflated revenue regression | `--extra models` |
| `demo_deep` | Nonlinear classification with PyTorch MLPs | `--extra deep` |

All data is synthetic and generated in memory. Run `python -m demos.<demo> list-candidates`.
Each `init-demo` generates new rows, so you can run a demo as many times as you like.
Evaluating a repeated `--data-seed` whose holdout was read needs `--allow-reread --reason <text>`.

## Warehouse snapshots (optional)

Pull once from Snowflake, BigQuery, Redshift, Databricks, Postgres, or DuckDB, then run offline.

```bash
uv sync --extra postgres              # or: snowflake | bigquery | databricks | redshift
uv run experiment sources list
uv run experiment pull --experiment experiments/<id> --source duckdb \
  --database path/to.duckdb --query "SELECT * FROM events ORDER BY id"
uv run experiment seal --experiment experiments/<id> --time-column event_date
```

A pull creates the directory and writes `data/snapshot.parquet` and `dataset_manifest.json`. The
manifest holds the query, as-of point, schema hash, and content hash. Pulls refuse any write, and no
flag overrides that. They also refuse queries that are not reproducible, unless you pass `--allow-nondeterministic --reason <text>`, which labels the result exploratory, and queries that sqlglot cannot parse.
A pull into a sealed experiment is refused, because it would put holdout rows back. Each source has a `SETUP.md` under
[`src/agentic_ml_loop/sources/bundles/`](src/agentic_ml_loop/sources/bundles/). DuckDB and local
Postgres are tested end to end. The other connectors are tested against mocked clients.

## Develop

```bash
uv sync --extra models --extra deep
make check    # lock, ruff, strict types, import contracts, ast-grep, deptry, vulture, tests
```
