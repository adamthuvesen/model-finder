# Generate an experiment

Candidate code lives in your own package, outside the engine. A checkout of the engine has a worked example in `demos/demo_bootstrap/`. The wheel does not ship it.

## Candidates

```python
from agentic_ml_loop.candidate_result import (
    CandidateResult, CandidateRunner, FittedCandidate, SearchMetrics, train_validation_gap,
)
from agentic_ml_loop.splits import SearchSplits

def run_majority_baseline(splits: SearchSplits) -> FittedCandidate:
    rate = float(splits.train[TARGET].mean())
    def score_rows(rows):
        return evaluate(rows[TARGET], np.full(len(rows), rate))
    metrics = SearchMetrics(train=score_rows(splits.train), validation=score_rows(splits.validation))
    return FittedCandidate(
        result=CandidateResult(
            candidate_id="majority-baseline", model_family="constant", feature_set="none",
            objective_metric="val_auc", objective_score=metrics.validation["auc"],
            split_strategy="time_split_60_20_20", notes="Train prevalence.", metrics=metrics,
            hyperparameters={"train_positive_rate": rate}, selected_features=[],
            random_seed=None, bootstrap_count=None,
            train_validation_gap=train_validation_gap(metrics, "val_auc"),
        ),
        score_rows=score_rows,
    )

CANDIDATE_RUNNERS: dict[str, CandidateRunner] = {"majority-baseline": run_majority_baseline}
```

Only `candidate_id`, `objective_metric`, `objective_score`, and `metrics.validation` are required. Record a
seed, a bootstrap count, and train metrics when you have them. Metric names may not contain a `test` or
`holdout` token.

## Runner

```python
from agentic_ml_loop.cli.runner import RunnerConfig, run_runner_main
from agentic_ml_loop.runner import ExperimentSeed

CONFIG = RunnerConfig(
    seed=ExperimentSeed(experiment_id="my_slug", template_path=HERE / "experiment_template.md"),
    candidate_runners=CANDIDATE_RUNNERS,
    rows_loader=load_rows,        # takes a seed, returns the full dataset; used only to seal
    rows_source=HERE / "rows.csv",  # the local file it reads, or None. Seal warns that it stays readable
    time_column="event_date",
    target_column="converted",    # read by the split audit at freeze
    code_root=HERE,               # this whole tree is fingerprinted for run evidence
    config_files=(HERE / "params.json",),  # any config outside that tree
)
raise SystemExit(run_runner_main(CONFIG))
```

For a sealed warehouse snapshot, `rows_loader` is not called, because `init-demo` skips sealing when
`holdout_seal.json` exists.

## Spec

Start from the output of `uv run experiment contract spec-template`. Fill it from the confirmed proposal.
Seed `research_sources.md` with the discovery references.

## Run and check

```bash
uv run --extra models python -m <package> init-demo --experiment experiments/<slug>
uv run --extra models python -m <package> run-candidate --experiment experiments/<slug> --candidate majority-baseline
uv run experiment validate experiments/<slug>
```

Confirm each item:

- `validate` reports no errors and no stray root files.
- The leaderboard has a recognizable baseline: `model_family` is `constant` or `rule_based`, or the ID contains `baseline`.
- Every number in the spec traces to profiling.
- At least one hypothesis has a real falsifier.
- Known Risks rules out each leakage type with a reason, and the split matches the data structure.
- Deliverables are under `outputs/`.

Report the files created, problem, objective, baseline score, and hypothesis count. Then give the
start command: a native `/goal` that names the `research` skill and the experiment path.
