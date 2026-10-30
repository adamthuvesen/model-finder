# Research evidence

`experiment validate` checks structure. `experiment complete-check` checks whether reviewed evidence
still matches the conclusion. Neither proves that a method is scientifically valid.

## Candidate runs

The engine writes one immutable record per run under `work/runs/<run_id>.json`.
`results.json` is the current leaderboard. Repeating a candidate replaces its entry and keeps
earlier run records.

Each result records its objective direction, specification hash, train and validation data
identities, random seed, bootstrap count, and train-validation gap. Each run record adds runtime,
the installed-package hash, and a fingerprint of the engine and candidate code.
Data identities hold hashes and dimensions, not rows.

Each run fingerprints the engine package tree, the experiment package tree, and any declared config
files. That covers lazily imported helpers and JSON config. Any edit in those trees makes evidence
stale, even an unrelated one. That is the accepted cost. Symlinked helper directories are followed.
A changed set of installed names and versions also makes it stale. An edit to an installed
dependency's source does not: versions are what identify dependencies.
Run records are not committed. Each run writes its own.

## The sealed holdout

Sealing writes holdout rows to one file outside the workspace:
`$XDG_DATA_HOME/agentic-ml-loop/holdouts/<experiment uid>/`, defaulting to `~/.local/share`. The uid is
a random identity, so experiments with the same name never share a file. `holdout_seal.json` records
only the uid, the content hash, and the row count. `data/` then holds search rows only.

Demos seal in `init-demo`, with new synthetic rows per experiment unless `--data-seed` is set. For a
warehouse pull, run `experiment seal --experiment <dir> --time-column <column>`. It removes the full
snapshot. A pull into a sealed experiment is refused. If a crash leaves the snapshot behind,
`validate` and `run-candidate` fail, and running `seal` again finishes the job.

Candidates receive `SearchSplits`, so the type checker rejects holdout access in candidate code.
Metrics allow only `train` and `validation` split keys. Metric, objective, and hyperparameter names
may not contain a `test` or `holdout` token, apart from `test_size`.

## The ledger

A holdout's identity is the sha256 of its sealed file. `holdouts/ledger.jsonl` records every file that
was read or retired, across all experiments. The engine enforces the workflow around that file. It does
not try to recognize similar data.

- `evaluate-final` refuses a file with any ledger entry, from any experiment.
- `evaluate-final` writes in this order: `evaluated` entry with the attempt's hash, attempt file,
  scores, `finalized` entry with the final file's hash, then the final file. A crash after the first
  entry costs a reseal with a note. It never allows a second read, and it leaves nothing to complete.
- `experiment reseal` validates everything first, so a refused reseal changes nothing. It then writes in
  this order: `retired` entry, history, new sealed file, seal record. A crash before the seal commits
  leaves the old seal in place, and evaluation refuses it. A retry resumes. A retry after the commit
  finds its rows live and does nothing.
- Seal and reseal refuse an exact file that the ledger knows.
- `complete-check` requires entries that match the final file's hash, the attempt's hash, the holdout
  hash, the experiment uid, and the experiment path relative to the repository root. It also rehashes the
  search rows. A moved repository passes, and a copy into another directory fails. Another machine fails
  until its data home holds this ledger, because the ledger is the record of the read.
- The prior-holdout count comes from the ledger, so deleting `work/holdout_history` does not change it.

An unavailable store, a corrupt ledger, or a ledger that lacks this experiment's own history fails closed.
The corrupt-ledger error names the file and the line. Every experiment waits until that line is repaired.

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
`RunnerConfig.rows_source` is required, so that an author names the file or passes `None`. Seal and reseal
print a warning that names the path. Move the file out of the agent's reach, or deny it in the agent's
settings. No code check can prove that an agent did not read it.

## Selection and final evaluation

```bash
uv run --extra models python -m demos.<id> freeze-candidate --experiment <dir> --candidate <id>
uv run --extra models python -m demos.<id> evaluate-final --experiment <dir>
```

Freezing binds the selected run, the leaderboard, the specification, and the code. It blocks further
search writes. It also reports target and feature drift between train and validation in
`work/selection.json`. It does not read the holdout, and it does not block. `evaluate-final` requires a frozen selection. It verifies the seal hash before it parses
a row, writes an attempt marker, refits the candidate, rejects a changed result, and then scores the
sealed rows once. The result goes to `outputs/final_evaluation.json`, not the leaderboard.

An interrupted attempt fails loudly on retry. Inspect its artifacts before you decide on recovery.
A completed evaluation returns its checked evidence without refitting.

The final output reports overlap as a warning. Counts use plain value comparison of exact rows, and of
any columns the spec declares with a `key_columns: a, b` line. Numeric columns compare by value across
int and float. Any other type difference lands in `not_comparable`. A missing key never counts as shared,
and a misspelled key fails the freeze. A zero does not prove the rows are unrelated. If rows are shared
or a comparison could not run, `complete-check` fails until `acknowledged_overlap` explains it.

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

A holdout that informed a decision no longer gives an unbiased estimate. Nothing restores that, but
new rows can give a new one:

```bash
uv run experiment reseal --experiment <dir> --rows new_rows.parquet --note "Why it is retired, and where the rows came from"
```

The note is required. The rows must come from outside the experiment and have the search columns. The
command copies the selection and the final evidence to `work/holdout_history/<n>/`, and writes
`consumed.json` with the old seal, the note, and the run IDs that are now exploratory for that holdout.
Search reopens. `init-demo --force` never removes the seal, the history, or `data/`.

Later evaluations record `prior_holdouts`. `experiment validate` warns while any holdout is retired.
`complete-check` fails until the completion record sets `acknowledged_prior_holdouts` to the count.
Say in the limitations what the old holdout influenced. The check cannot verify that sentence.

## Completion record

Create `outputs/completion.json` with this command. It hashes each file, and it writes nothing unless the
completion check passes:

```bash
uv run experiment record-completion --experiment <dir> --outcome supported \
  --conclusion "What the evidence supports" --limitation "One limitation" \
  --selected <id> --baseline <id> --report outputs/report.md --reproduction outputs/reproduce.md
```

The record requires:

- An outcome: `supported`, `no_demonstrated_improvement`, or `inconclusive`.
- A concrete conclusion and limitations.
- References to `experiment.md`, `results.json`, the report, evaluation evidence, review, and reproduction instructions.
- Selected and baseline candidate IDs for a supported comparison.
- For `supported` and `no_demonstrated_improvement`: `evaluation` must be `outputs/final_evaluation.json`,
  the evidence must match its freeze, its seal, and the ledger, and the selected candidate must be the
  frozen one.

When the conclusion retains the baseline, selected and baseline IDs may match. Set
`comparison_candidate_id` to the distinct tested challenger. All referenced files must stay inside
the experiment. The checker rejects missing, empty, stale, or escaping references.
An inconclusive report may name no candidates when the evidence explains why. A budget expiration
alone is not such an explanation.

## Independent review

Have a fresh reviewer inspect the specification, code, split provenance, evaluation, and report.
The reviewer then writes `outputs/review.json`. It records the reviewer, the verdict, hashes of what was
reviewed, findings, and limitations:

```bash
uv run experiment record-review --experiment <dir> --reviewer <name> --verdict accepted \
  --report outputs/report.md --limitation "One limitation" --finding "One finding"
```

The checker rejects a stale review or a `changes_requested` verdict. It cannot authenticate the
reviewer, establish independence, infer statistical validity, or detect every leakage path.
The actual review remains necessary.

## Recovery

Treat a run without a valid completion record as unfinished. Inspect its artifacts before you repeat
an expensive command. The journal records the hypothesis, the result, and its interpretation.
Native goals own continuation. Evidence files are not another scheduler.
