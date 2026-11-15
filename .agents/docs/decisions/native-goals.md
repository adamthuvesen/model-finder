# Use Native Goals For Experiment Execution

Date: 2026-11-04

## What the loop was

`python -m loop` was a supervisor. It started a fresh `claude`, `codex`, or `cursor-agent`
subprocess per cycle. It rebuilt the prompt each time from `program.md`, `guidelines.md`, and
experiment state. The agent ended a cycle by writing `cycle_status.json`. The supervisor validated
the attempt, rolled back on failure, and retried up to three times.

## What it measured

- **Progress.** A hash of `research_journal.md` plus leaderboard changes. Three cycles without
  progress stopped the run.
- **Contract.** Exit code, status file, a changed journal, and an unchanged spec.
- **Quality.** An advisory referee scorecard over six text heuristics, and a runner benchmark
  ranked by that scorecard.

These measured writing, not research. A scorecard over journal words cannot detect leakage, and
the committed demo results carried test metrics that passed every check.

## Why it was replaced

The clients now provide persistence and continuation, so the loop was a second, weaker execution
system. Each cycle discarded context. Unattended runs required shell access granted up front. The
supervisor, invoker, prompts, referee, and benchmark were about 4,000 lines that did not protect the
result. The boundary that matters is around the holdout, and the loop's freeze was a flag in a state
file.

The sibling repo reached the same decision first. Read its record:
[agentic-research-loop native goals](https://github.com/adamthuvesen/agentic-research-loop/blob/main/.agents/docs/decisions/native-goals.md).
It warns against treating one failed custom run as proof, and this record repeats that warning.
The argument is the failure surface of a second execution system. It is not a benchmark result.

## Consequences

- Start inside the client with one `/goal` that names the `research` skill.
- The experiment directory is the only durable record.
- **Unattended execution is gone.** `/goal` needs a live client session. If a workflow needs
  unattended runs, add a thin entry point outside this engine. Do not revive the cycle loop.
- **Per-cycle transcripts are gone.** Debugging a run means reading the client's transcript.
- **Advisory modules are gone.** Signals, referee, and learnings extraction only fed the loop's
  prompts. `learnings.md` stays as a file the agent reads. The valuable part of the evaluation review,
  split overlap and drift, runs at freeze as `data_audit.py`.
- **Client goal behavior is an unpinned dependency.** The engine's commands stay usable without it.
- The last loop version stays at the git tag `loop-final`.
