# model-finder

A model-search engine. An agent searches candidates on search rows, and the engine keeps the evidence: a sealed holdout, run records, and a completion check.

## Core

**Experiment**:
One directory under `experiments/<slug>/`, with its `experiment.md` spec, search rows in `data/`, and candidates in `model/`. A change to the spec starts a new comparison.

**Candidate**:
One fit function registered with `@candidates.add`. Its function name is its id, and its docstring is its notes.
_Avoid_: model, run

**Sealed holdout**:
Rows the engine sealed into one file outside the workspace. Its identity is the sha256 of that file, and a candidate never receives it.
_Avoid_: test set, holdout split

**Ledger**:
The append-only `holdouts/ledger.jsonl` record, across all experiments, of each read and retirement of a sealed holdout. Its events are `evaluated`, `finalized`, `reopened`, and `retired`.

## Terms that collide with fact-finder

**Source**:
A warehouse that a pull reads, named by a `source_type` that is a sqlglot dialect.
Not the same as in fact-finder, where a source is one recorded piece of evidence.

**Bundle**:
One warehouse connector directory, `sources/bundles/<name>/`, with `source.json` and `SETUP.md`.
Not the same as in fact-finder, where a bundle is one `<name>.toml` server definition.

**Finding**:
One free-text line a reviewer records in a review.
Not the same as in fact-finder, where a finding is a typed validation result with a kind.

**Exploratory**:
A label the engine applies to a result after the fact, from its evidence, such as search that continued after a holdout read, or a nondeterministic pull.
Not the same as in fact-finder, where exploratory is a rigour level chosen before the work.
