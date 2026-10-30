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
stale, even an unrelated one. That is the accepted cost. A changed package set also makes it stale.
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
- `evaluate-final` writes in this order: ledger entry, attempt file, then scores. The entry holds the
  attempt's hash. A crash after the entry costs a reseal with a note. It never allows a second read.
- `experiment reseal` writes in this order: `retired` entry, history, new sealed file, seal record.
  A crash before the seal commits leaves the old seal in place, and evaluation refuses it.
- Seal and reseal refuse an exact file that the ledger knows.
- `complete-check` requires an `evaluated` entry that matches the final evidence's holdout hash,
  experiment uid, and attempt hash.

An unavailable store, a corrupt ledger, or a ledger that lacks this experiment's own history fails closed.

### Limits of file identity

- **A new-seed resplit of the same source is a new holdout.** It makes a different file, so the ledger
  sees a different holdout, even when the rows overlap.
- **A column schema change is a new holdout.** So is a cast, an added column, a new ID column, or a
  new library version that writes different bytes.
- **The ledger is a plain file, and an agent has a shell.** A deliberate agent can edit or delete it.

### What the seal does not stop

Claude Code has no sandbox. A determined agent with a shell could find the file. The deny rule in
`.claude/settings.json` covers only the default path and the commands that match it. Synthetic demo generators can
also recreate rows. The seal and the ledger stop honest and shortest-path agents. They do not stop a
deliberate one.

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
any columns the spec declares with a `key_columns: a, b` line. A column whose type differs between
frames gives `null`, not a guess. A zero does not prove the rows are unrelated. If rows are shared,
`complete-check` fails until the completion record explains them in `acknowledged_overlap`.

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

Create `outputs/completion.json` with `agentic_ml_loop.completion.CompletionRecord` and
`save_completion`. Use `evidence_file(experiment_dir, relative_path)` to bind each reference to its hash.
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
Save `agentic_ml_loop.completion.ResearchReview` as JSON, commonly `outputs/review.json`. It records
the reviewer, the verdict, hashes of what was reviewed, findings, and limitations.

The checker rejects a stale review or a `changes_requested` verdict. It cannot authenticate the
reviewer, establish independence, infer statistical validity, or detect every leakage path.
The actual review remains necessary.

## Recovery

Treat a run without a valid completion record as unfinished. Inspect its artifacts before you repeat
an expensive command. The journal records the hypothesis, the result, and its interpretation.
Native goals own continuation. Evidence files are not another scheduler.
