# Research evidence

This tool helps people get good models. It is not a publication gate.

Safety is hard: no agent runner, no writes to external systems, and read-only pulls. Rigour is a default that
a person can override with one flag and a short reason. The override is recorded, and the result is labelled.
A check blocks only when it would otherwise produce a silently misleading number. Everything else warns.

Commands: `experiment contract commands`. Candidate fields: `experiment contract candidate-schema`.
Next step, at any time: `experiment next`.

## What blocks and what labels

`experiment complete-check` gives one result: `complete`, `complete, exploratory`, or `not complete`.
It checks traceability, not scientific validity.

| Situation | Result |
| --- | --- |
| No holdout, or no final evaluation | `complete, exploratory` |
| A second read of a holdout, with `--allow-reread --reason` | `complete, exploratory`, reason in the ledger |
| Search or selection after a read, with `--allow-reread --reason` | `complete, exploratory`, reason in the ledger |
| A second read or later search without the flag | Refused, because the number would look clean |
| Earlier holdouts retired | `complete, exploratory` |
| A data pull with `--allow-nondeterministic --reason` | `complete, exploratory`, reason in the manifest |
| A rebuild with `--allow-changes --reason` | `complete, exploratory`, reason in `work/results_rebuilds.json` |
| A rebuild that changes results, without the flag | Refused |
| At least 10% of holdout rows in search rows, with a reason | `complete, exploratory` |
| The same overlap with no reason | `not complete` until `--acknowledged-overlap` gives one line |
| Smaller overlap, drift, or a comparison that could not run | Warning |
| A hyperparameter that names a holdout score | Warning |
| Edited code, config, or specification since a run | Warning. `--strict` fails |
| No review, report, or reproduction notes | Warning |
| A ledger that never saw this read, such as on another machine | `complete, exploratory` |
| An evidence copy in another directory | Warning |
| An edited final, an edited leaderboard, or changed search rows | `not complete` |
| A candidate named in the claim but not frozen or not run | `not complete` |
| A review that requests changes | `not complete` |

`--strict` fails on every label and warning. Use it when a result must stand on its own. The 10% is a judgement
default with no source.

## The holdout

A holdout is optional and recommended. It costs nothing: `experiment new` seals one unless you pass `--no-holdout`.
To add one later, run `experiment reseal --experiment <dir> --rows <parquet> --note <text>`.

Sealing writes holdout rows to one file outside the workspace, under
`$XDG_DATA_HOME/model-finder/holdouts/<uid>/`, defaulting to `~/.local/share`. `holdout_seal.json` records only the
uid, the content hash, and the row count. `data/` holds search rows. A candidate receives `SearchSplits`, so its type
has no holdout. A holdout's identity is the sha256 of its file. `holdouts/ledger.jsonl` records every read and
retirement, across all experiments.

Order of writes, so that a crash cannot hide a read:

- `evaluate-final`: `evaluated` entry, attempt file, scores, `finalized` entry with the final's hash, final file.
- `experiment reseal`: `retired` entry, history, new sealed file, seal record. A retry resumes. A retry after the
  commit finds its rows live and does nothing.

A store from the engine's old name, `agentic-ml-loop`, stays usable. Its ledger still counts, its path stays denied, and a
holdout left there fails with the exact `mv` to run. The engine does not move it on its own.

A corrupt ledger fails closed, because it could hide an earlier read. The error names the file and the line.

## Limits

- A new-seed resplit or a schema change makes a different file, so the ledger sees a different holdout.
- The ledger and run records are plain files. A deliberate agent can edit them. An edited run record goes unseen if the
  leaderboard is also damaged or edited to match.
- A read the ledger never saw gets a label, not a refusal. A hand-written final with a new seal gets that label too.
- A metric name such as `q4_auc` passes. A name cannot prove which rows a metric used.
- The engine cannot delete your source file, so sealed rows stay readable there. Name it in `RunnerConfig.rows_source`
  to get a warning at seal time, and move it out of the agent's reach.
- Claude Code has no sandbox. `experiment setup` adds path-independent deny rules to `.claude/settings.json`, so they
  cover any data home and a committed file works for every user. Other clients get no rule. A real Claude Code client
  refused a read under a custom directory. The `Edit` and Bash rules were not run in a client. These rules stop honest
  and shortest-path agents, not a deliberate one.
- Run records fingerprint the engine tree, the code root, and declared config files. An edit to an installed
  dependency's source goes unseen, because versions are what identify dependencies.
- Overlap uses plain value comparison, so a zero does not prove the rows are unrelated.
- Text dates parse with pandas inference. An ambiguous `01/02/2025` reads month first. Convert it in `rows_loader` if not.

## Holdout-based selection

To select on the holdout, say so: pass `--allow-reread --reason <text>` to `run-candidate` or `freeze-candidate`, and
to `evaluate-final`. The result is labelled exploratory. It is never silent. Otherwise, seal new rows with
`experiment reseal`, which keeps the old estimate on record as exploratory.

## Repairs

`experiment rebuild-results <dir>` repairs a damaged leaderboard from run records in `work/runs/`. It lists each added,
removed, and changed value. Do not edit the leaderboard by hand.

For a supervised feature ranking, write `feature_selection_provenance.json` beside the artifact. It maps the file name
to its `sha256` and to the `label_dates` the ranking used. `experiment check-features` fails on a label date outside the
train rows. It checks what the file declares, not what the ranking read.

## Review

Have a fresh reviewer inspect the specification, code, split provenance, evaluation, and report, and then run
`experiment record-review`. The checker cannot authenticate the reviewer or detect every leakage path.
