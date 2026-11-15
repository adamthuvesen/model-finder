---
name: research
description: Run or resume an model-finder experiment under a native goal. Use for bounded hypothesis tests, research recovery, evidence synthesis, and method review of an existing experiment. Use experiment-spec to create a new experiment.
---

# Research

Answer the experiment's question with reproducible evidence, under the authorized budget. Follow any hypothesis in any
order. The engine provides tools, not a workflow.

Run `uv run experiment next` at any time. It reads the state and prints useful steps with exact commands.
Read `uv run experiment contract method` for craft, and `uv run experiment contract evidence` for what each label means.

## Orient or recover

Read `experiment.md`, `research_journal.md`, `research_sources.md`, `status.md` when present, and `learnings.md` when
present. Reconcile unfinished work before you repeat possibly billable work. Update `status.md`.

## Test a hypothesis

Write the hypothesis, expected observation, comparison, and budget in the journal, under `## Cycle NNNN: <title>`. Run the
smallest informative test. Record the result, failures, interpretation, evidence paths, and next decision. Add a
candidate with `uv run experiment candidate new <name>`. Keep failed runs visible. A repeated failure needs diagnosis, not
another run of the same assumption. If a source is unavailable, state what stays unverified and continue offline.

## Compare and review

Compare candidates on paired predictions from the same rows. Report the difference against the practical threshold.
Prefer not to tune after reading the holdout. If a decision must change, seal new rows, or pass
`--allow-reread --reason <text>` and accept the exploratory label. Ask a reviewer with fresh context to trace the question,
implementation, split provenance, comparison, and conclusion. If review is unavailable, disclose that.

## Finish

Write `outputs/<report>.md` with the decision, baseline comparison, uncertainty, and limitations. Report the label from
`uv run experiment complete-check <dir>` with your answer. A budget stop leaves a checkpoint and names unresolved work.
No demonstrated improvement and a supported inconclusive result are acceptable answers.
