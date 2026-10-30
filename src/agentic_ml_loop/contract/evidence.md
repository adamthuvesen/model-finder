# Research evidence

This tool helps people get good models. It is not a publication gate.

Safety stays hard: no agent runner, no writes to external systems, and read-only pulls. Rigour is a
default that a person can override with one flag and a short reason. The override is recorded, and the
result is labelled. A rigour check blocks only when it would otherwise produce a silently misleading
number. Everything else warns.

`experiment validate` checks structure. `experiment complete-check` gives one result: `complete`,
`complete, exploratory`, or `not complete`. Neither proves that a method is scientifically valid.

## The shortest path

```bash
uv run --extra models python -m demos.demo_bootstrap init-demo --experiment experiments/my_run --no-holdout
uv run --extra models python -m demos.demo_bootstrap run-candidate --experiment experiments/my_run --candidate logreg-tiny
uv run experiment leaderboard experiments/my_run
```

Scores from this path come from validation rows and stay labelled exploratory. A candidate must report
only `candidate_id`, `objective_metric`, `objective_score`, and `metrics.validation`. Record a seed, a bootstrap
count, and train metrics when you have them.

## What blocks and what labels

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
| A hyperparameter that names a holdout score | Warning |
| Large overlap between holdout and search rows, with a reason | `complete, exploratory` |
| Large overlap with no reason | `not complete` until `--acknowledged-overlap` gives one line |
| Small overlap, drift, or a comparison that could not run | Warning |
| Edited code, config, or specification since a run | Warning. `--strict` fails |
| No review, report, or reproduction notes | Warning |
| A ledger that never saw this read, such as on another machine | `complete, exploratory` |
| An evidence copy in another directory | Warning |
| An edited final, an edited leaderboard, or changed search rows | `not complete` |
| A candidate named in the claim but not frozen or not run | `not complete` |
| A review that requests changes | `not complete` |

`complete-check --strict` fails on every label and warning. Use it when a result must stand on its own.

## Candidate runs

The engine writes one immutable record per run under `work/runs/<run_id>.json`.
`results.json` is the current leaderboard. Repeating a candidate replaces its entry and keeps
earlier run records.

Each result records its objective direction, specification hash, and train and validation data
identities. It records a random seed, a bootstrap count, and a train-validation gap when a candidate
reports them. Each run record adds runtime,
the installed-package hash, and a fingerprint of the engine and candidate code.
Data identities hold hashes and dimensions, not rows.

Each run fingerprints the engine package tree, the experiment package tree, and any declared config
files. That covers lazily imported helpers and JSON config. An edit in those trees gives a warning. It
does not force a rerun. `complete-check --strict` fails on it. Symlinked helper directories are followed.
A changed set of installed names and versions also warns. An edit to an installed dependency's source does
not: versions are what identify dependencies.
Run records are not committed. Each run writes its own.

## The sealed holdout

A holdout is optional, and it is the recommended path. It costs one command. `init-demo --no-holdout` and
`experiment seal --no-holdout` keep every row for search. To add a holdout later, run `experiment reseal --rows
<parquet> --note <text>`.

Sealing writes holdout rows to one file outside the workspace:
`$XDG_DATA_HOME/agentic-ml-loop/holdouts/<experiment uid>/`, defaulting to `~/.local/share`. The uid is
a random identity, so experiments with the same name never share a file. `holdout_seal.json` records
only the uid, the content hash, and the row count. `data/` then holds search rows only.

Demos seal in `init-demo`, with new synthetic rows per experiment unless `--data-seed` is set. For a
warehouse pull, run `experiment seal --experiment <dir> --time-column <column>`. It removes the full
snapshot. A pull into a sealed experiment is refused. If a crash leaves the snapshot behind,
`validate` and `run-candidate` fail, and running `seal` again finishes the job.

Candidates receive `SearchSplits`, so the type checker rejects holdout access in candidate code.
Metrics allow only `train` and `validation` split keys. Metric and objective names may not contain a `test`
or `holdout` token, because a holdout score on the leaderboard would look like a validation score. A name
such as `q4_auc` passes. A name cannot prove which rows a metric used.

## The ledger

A holdout's identity is the sha256 of its sealed file. `holdouts/ledger.jsonl` records every file that
was read or retired, across all experiments. The engine enforces the workflow around that file. It does
not try to recognize similar data.

- `evaluate-final` refuses a file with any ledger entry, from any experiment, unless you pass
  `--allow-reread --reason <text>`. The reason goes in the ledger, and the estimate is labelled exploratory.
- `evaluate-final` writes in this order: `evaluated` entry with the attempt's hash, attempt file,
  scores, `finalized` entry with the final file's hash, then the final file. After a crash, retry with
  `--allow-reread --reason <text>`.
- `run-candidate` and `freeze-candidate` after a read also need the flag. They record a `reopened` entry and
  copy the old evidence to `work/rereads/<n>/`. Before any read, they withdraw or replace the selection freely.
- `experiment reseal` validates first. It then writes in this order: `retired` entry, history, new sealed
  file, seal record. A retry resumes. A retry after the commit finds its rows live and does nothing.
- Seal and reseal warn about an exact file that the ledger knows. `evaluate-final` then needs the flag.
- `complete-check` compares the ledger with the final file's hash, the attempt's hash, the holdout hash, and
  the experiment uid. It also rehashes the search rows. A mismatch blocks.
- If the ledger has no entry for the file, as on another machine, the estimate is labelled unverified. So
  a hand-written final with a new seal gets that label too, and not a refusal.
- The prior-holdout count comes from the ledger, so deleting `work/holdout_history` does not change it.

A corrupt ledger fails closed, because it could hide an earlier read. The error names the file and the
line. Every evaluation waits until that line is repaired.

### Limits of file identity

- **A new-seed resplit of the same source is a new holdout.** It makes a different file, so the ledger
  sees a different holdout, even when the rows overlap.
- **A column schema change is a new holdout.** So is a cast, an added column, a new ID column, or a
  new library version that writes different bytes.
- **The ledger is a plain file, and an agent has a shell.** A deliberate agent can edit or delete it.

### What the seal does not stop

Claude Code has no sandbox. A determined agent with a shell could find the file. `experiment setup` adds deny rules to the repository's
`.claude/settings.json`. They use the `~/` form when the store is under the home directory, so a committed file works for
every user. A store outside home gets an absolute `//` rule, which names that one path. Run setup again after you
change the data home. It replaces its earlier rules and leaves your rules alone. Other agent clients get no rule. Synthetic demo generators can
also recreate rows. The seal and the ledger stop honest and shortest-path agents. They do not stop a
deliberate one.

### Rows that stay readable in a source file

The engine seals its own copy of the rows. It cannot delete your source file. If a `rows_loader` reads a
local file, or `experiment reseal --rows` reads one, the sealed rows stay readable in that file.
Name that file in `RunnerConfig.rows_source`. Seal and reseal then print a warning that names the path. Move the file out of the agent's reach, or deny it in the agent's
settings. No code check can prove that an agent did not read it.

## Selection and final evaluation

```bash
uv run --extra models python -m demos.<id> freeze-candidate --experiment <dir> --candidate <id>
uv run --extra models python -m demos.<id> evaluate-final --experiment <dir>
```

Freezing binds the selected run and the leaderboard. A new run before any read withdraws the selection, and
a new freeze replaces it. It also reports target and feature drift between train and validation in
`work/selection.json`. It does not read the holdout, and it does not block. `evaluate-final` requires a frozen selection. It verifies the seal hash before it parses
a row, writes an attempt marker, refits the candidate, rejects a changed result, and then scores the
sealed rows once. The result goes to `outputs/final_evaluation.json`, not the leaderboard.

An interrupted attempt fails loudly on retry, and the message names the flag. A completed evaluation
returns its checked evidence without refitting.

To select on the holdout, say so. Run `freeze-candidate` or `run-candidate`, and then `evaluate-final`, each with
`--allow-reread --reason <text>`. The result is labelled `complete, exploratory`. It is never silent.

The final output reports overlap as a warning. Counts use plain value comparison of exact rows, and of
any columns the spec declares with a `key_columns: a, b` line. Numeric columns compare by value across
int and float. Any other type difference lands in `not_comparable`. A missing key never counts as shared,
and a misspelled key fails the freeze. A zero does not prove the rows are unrelated. Shared rows and failed
comparisons warn. If at least 10% of holdout rows also appear in search rows, `complete-check` asks for one line
in `--acknowledged-overlap`, and the result is labelled exploratory. The 10% is a judgement default with no source.

## Repairing the leaderboard

This command repairs a damaged leaderboard from run records in `work/runs/`. Do not edit the leaderboard by hand.

```bash
uv run experiment rebuild-results <dir>
```

It takes each candidate's latest run record. If the old leaderboard is invalid, it rebuilds and says so. If the old
leaderboard is readable and disagrees with the records, it lists each added, removed, and changed value and writes
nothing. To accept the records, pass `--allow-changes --reason <text>`, which labels the result exploratory. It also
writes nothing if a run record is invalid, if the runs compare different specifications or data, or if a frozen
selection covers a different leaderboard. `init-demo --force` removes run records, so do not use it for this repair.

A run record is a plain file with no signature. If the leaderboard is also damaged or edited to match, an edited record
goes unseen.

## Feature artifacts

A supervised feature ranking that saw validation or holdout labels leaks them into search. Next to the
artifact, write `feature_selection_provenance.json`. It maps the artifact's file name to its `sha256` and to the
`label_dates` that the ranking used, each in `YYYY-MM-DD` form. Then run:

```bash
uv run experiment check-features --experiment <dir> --artifact <path> --time-column <column>
```

The command fails on missing provenance, a changed artifact, or a label date outside the train rows. A holdout
date is outside the train rows, so the command needs no holdout rows. It checks what the file declares. It
cannot see which labels the ranking really read. Packages that know their split dates can call
`agentic_ml_loop.feature_provenance.validate_feature_provenance` directly.

## Replacing a read holdout

A holdout that informed a decision no longer gives an unbiased estimate. New rows can give a new one:

```bash
uv run experiment reseal --experiment <dir> --rows new_rows.parquet --note "Why, and where the rows came from"
```

A short note is enough. The rows must have the search columns. The
command copies the selection and the final evidence to `work/holdout_history/<n>/`, and writes
`consumed.json` with the old seal, the note, and the run IDs that are now exploratory for that holdout.
Search reopens. `init-demo --force` never removes the seal, the history, or `data/`.

Later evaluations record `prior_holdouts`. `experiment validate` warns while any holdout is retired, and
`complete-check` labels the result exploratory.

## Completion record

Create `outputs/completion.json` with this command. It hashes each file, and it writes nothing if something blocks:

```bash
uv run experiment record-completion --experiment <dir> --outcome supported --conclusion "What the evidence supports"
```

An outcome and a conclusion are enough. The outcome is `supported`, `no_demonstrated_improvement`, or
`inconclusive`. Add `--selected`, `--baseline`, `--comparison`, `--limitation`, `--report`, and `--reproduction` when
you have them. The command picks up `outputs/final_evaluation.json` and `outputs/review.json` when they exist.

## Independent review

Have a fresh reviewer inspect the specification, code, split provenance, evaluation, and report.
The reviewer then writes `outputs/review.json`. It records the reviewer, the verdict, hashes of what was
reviewed, findings, and limitations:

```bash
uv run experiment record-review --experiment <dir> --reviewer <name> --verdict accepted \
  --report outputs/report.md --limitation "One limitation" --finding "One finding"
```

A review is recommended. Without one, `complete-check` warns. A `changes_requested` verdict blocks, and a stale
review warns. The checker cannot authenticate the reviewer, establish independence, infer statistical validity,
or detect every leakage path.

## Recovery

Inspect its artifacts before you repeat
an expensive command. The journal records the hypothesis, the result, and its interpretation.
Native goals own continuation. Evidence files are not another scheduler.
