---
name: experiment-spec
description: Design and scaffold an model-finder experiment from raw data and context. Profiles the data, proposes a falsifiable spec, seals the holdout, runs baselines, and validates. Use when a user wants to turn a dataset and problem description into a runnable experiment. Use research to run one.
---

# Experiment Spec

1. **Locate data.** Use a local file, or pull a snapshot with `uv run experiment pull`. `uv run experiment sources enable <type>`
   prints that source's setup steps.
2. **Start.** Run `uv run experiment new <slug> --csv <file> --target <column>`. It writes the runner, candidates, and spec,
   seals a holdout, and prints a first leaderboard. For a snapshot, run `uv run experiment seal`, then your runner's `init-demo`.
3. **Profile** search rows in `data/`: dtypes, cardinality, missingness, target distribution, time range, and suspiciously
   high correlations. Do not read holdout rows while profiling.
4. **Propose.** Stop and get confirmation for: problem type, target, split rationale, objective metric and the decision it
   reflects, leakage controls, risks, and 3-5 hypotheses. Each hypothesis has a claim, a reason grounded in profiling or a
   cited source, an expected effect against the validation noise floor, and a falsifier.
5. **Review** every line marked `CHANGE` in `experiments/<slug>/model/` and in `experiment.md`.

Do not invent column names, row counts, or metrics. Every number comes from profiling you ran. Then use `uv run experiment next`.
`uv run experiment contract candidate-schema` lists the candidate fields.
