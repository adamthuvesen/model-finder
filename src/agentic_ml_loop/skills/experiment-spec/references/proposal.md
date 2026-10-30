# Experiment proposal template

Every populated field traces to profiling, `learnings.md`, or a cited source.

```text
## Experiment Proposal: {slug}

Problem type:
Target column:
Split strategy: temporal cutoff or grouped by key, with the rationale. Never random k-fold on
  temporal data. Keep each repeated entity in one split.
Objective metric: {metric} with the (val_{metric}) token, and why it fits.
Operating metric: precision@k, F-beta, Brier, or MAE, and the decision it reflects.
Baselines: the trivial anchor, plus any human baseline.
Leakage controls: clean split, train-only preprocessing, no duplicates across splits, each feature
  available at prediction time, split respects time and entity grouping.
Constraints and non-goals:
Data profile: rows, features, missingness, class balance or target distribution, positives and the
  feature budget, key correlations.
Known risks: imbalance, scale confounders, drift, label noise.
Research pointers: 2-4 references with what transfers and what does not.
Evaluation strategy: split sizes, paired bootstrap on the metric and on the delta, cohorts.
Success definition: specific and checkable.
Research hypotheses, most promising first:
```

## Hypothesis format

Write 3-5, ordered by prior probability. Drop any you cannot ground.

- **Claim.** One bounded change.
- **Why it might work.** A profiling result, a `learnings.md` pattern, or a cited source.
- **Expected effect and noise.** Direction and magnitude against the validation noise floor.
  If the gain is under one bootstrap CI width, say so.
- **Falsifier.** The observation that disconfirms it, such as "val R2 gain under 0.02".
- **Confidence.** High, medium, or speculative.
