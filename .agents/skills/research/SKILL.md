---
name: research
description: Run or resume an agentic-ml-loop experiment under a native goal. Use for bounded hypothesis tests, research recovery, evidence synthesis, and method review of an existing experiment. Use experiment-spec to create a new experiment.
---

# Research

Answer the experiment's question with reproducible evidence under the authorized budget.
Done means the answer, deliverables, limitations, and an independent method review support the
completion claim. A budget stop leaves a checkpoint and names unresolved work.

## Orient or recover

Native `/goal` owns continuation. Read `experiment.md`, `research_journal.md`, `research_sources.md`,
`results.json`, `status.md` when present, [program.md](../../../program.md),
[guidelines.md](../../../guidelines.md), and relevant `learnings.md` sections.
Check the question, decision, objective direction, prediction time, label horizon, split policy,
practical improvement threshold, and budget. Resolve missing constraints before expensive work.
Check inherited feature selection for label overlap with validation or holdout data.

On recovery, reconcile unfinished run IDs with processes and artifacts.
Check status before you repeat possibly billable work. Update `status.md`; create no other handoff file.

## Test a hypothesis

1. Choose one bounded question from the journal, error analysis, sources, or data audit.
2. Write the hypothesis, expected observation, comparison, and budget in the journal, under `## Cycle NNNN: <title>`.
3. Audit grain, missingness, label maturity, and feature availability. Check duplicates and split boundaries too. `freeze-candidate` reports drift, and the final output reports overlap. Read both.
4. Read primary sources for uncertain technical claims.
5. Run the smallest informative test.
6. Record the result, failures, interpretation, evidence paths, and next decision.

If a source is unavailable, state what stays unverified and continue offline work.
Use `learnings.md` as a prior and test whether its conditions apply. A score alone is not a rule.
Record parent candidate, rationale, and changed configuration. Keep failed runs visible.
Repeated failures require diagnosis, not another run of the same assumption.

## Compare and review

Compare candidates on paired predictions from the same rows. Use entity or time-block resampling
when observations depend on each other. Record the estimand, resampling unit, seed, method, and
invalid resamples. Report the difference against the practical threshold.
An interval for the selected candidate alone does not correct adaptive validation reuse.

Select on validation. Then run `freeze-candidate --experiment <path> --candidate <id>` and
`evaluate-final --experiment <path>`. A frozen comparison rejects further candidate writes.
Do not tune after reading the holdout. If a decision must change, use `experiment reseal` with
unseen rows, as [docs/evidence.md](../../../docs/evidence.md) describes. Old estimates stay exploratory.

Ask a reviewer with fresh context to trace the question, implementation, split provenance,
comparison, and conclusion. Review is internal work, not a user approval gate.
If review is unavailable, disclose that and do not claim reviewed completion.

## Checkpoint or finish

Write `outputs/<report>.md` with the decision, baseline comparison, uncertainty, scope, limitations,
and reproduction commands. Create the records in [docs/evidence.md](../../../docs/evidence.md).
Run `uv run experiment complete-check experiments/<id>` before you claim reviewed completion.
It checks references and freshness, not scientific validity.

Update `status.md` at each checkpoint: current question, last verified result and paths, next action,
unfinished run IDs, remaining budget. No demonstrated improvement and a supported inconclusive
result are acceptable answers.
