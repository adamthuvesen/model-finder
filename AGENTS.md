# AGENTS.md

`model-finder` is a model-search engine for an agent running under a native client `/goal`.
The client owns continuation. This repo owns the evidence: search rows, a sealed holdout,
run records, and a completion check. Docs: [architecture](.agents/docs/architecture.md),
[labels and limits](docs/evidence.md), [why native goals](.agents/docs/decisions/native-goals.md).

## Where things go

| Task | File |
| --- | --- |
| New experiment on a CSV | `experiment new <slug> --csv <file>` |
| New candidate | One function in `experiments/<slug>/model/candidates.py`: `experiment candidate new <name>` |
| New result field | Optional: one line in `CandidateResult`, plus `ignore_names` for vulture if nothing reads it. Required: also `scored_candidate` and `result_fields()` in `tests/conftest.py`. That is 4 edits when the value is at hand. With no value at hand, it also needs a source, such as a field on `Scoring` |
| New source | `sources/bundles/<name>/source.json` and `SETUP.md`, an extra in `pyproject.toml`, and a `source_type` that is a sqlglot dialect. A non-ADBC driver also needs an adapter class. `experiment sources check` reports what is missing |
| A label or an exploratory reason | `labels.py` |
| Freeze and evaluate | `final_evaluation.py` |
| Reseal, or rebuild the leaderboard | `reseal.py`, `rebuild.py` |
| Skills and documents shipped in the wheel | `src/model_finder/skills/`, `contract/` |

## Commands

```bash
make quick            # only changed Python files, no tests: 0.7s for one file, 2.7s for 49
make check            # the gate; CI runs the same targets
uv run experiment next
uv run experiment contract commands
```

Prove user-facing changes with `.agents/skills/verify-model-finder/`. Build test results with `candidate_result()` or `result_fields()` in
`tests/conftest.py`. Every error message follows
`<file>:<line>: <what is wrong>. Fix: <exact change or command>`.

## Conventions

- `experiment.md` is a spec. Change it only when the human asks. A change starts a new comparison.
- Write one-shot scripts under `experiments/<id>/scripts/`. Keep candidates in `experiments/<id>/model/`.
- Code for one experiment stays out of the engine. Nothing specific to one company enters this repo.
- Never commit secrets, `.env`, generated notebooks, or local data.
- If a doc disagrees with code, fix the doc in the same change.
- Keep torch and the GBDT libraries in separate processes. Together they can deadlock on OpenMP on macOS.

## Stop and ask

These are safety rules. A task's wording cannot authorize any of them. Stop, and say which rule the
request reverses.

- An agent runner, an agent SDK, or a spawned coding agent. The native goal runs the agent.
- A write to a warehouse or other external system. Pulls stay read-only.
- An edit to a check, rule, or contract that would allow one of these.

Rigour is different. It is a default with an override. Selecting on the holdout, reading it twice, or skipping it
are labelled choices, so do them when a person asks: `--allow-reread --reason <text>` or `--no-holdout`. Say the
result is exploratory. Never produce a holdout number with no label.

## Enforced rules

Labels and warnings are in `experiment contract evidence`. This table lists only what code refuses.

| Rule | Enforced by | Strength |
| --- | --- | --- |
| No agent SDK or spawned process | Ruff `TID251`, `no-agent-runtime` rule and contract | Safety. Limit: `getattr(os, ...)` passes |
| Pulls are read-only, with no override | `require_read_only_query`, `verify_snapshot` | Safety. Defence in depth: the warehouse role is the real guard |
| The agent does not read the holdout store | Deny rule that `experiment setup` adds | Safety for Claude Code. Limit: a rule does not match every command form |
| Candidates receive no holdout rows, and predict never receives the target | `SearchSplits` type, `scored_candidate`, `demos-cannot-reach-holdout-store` | Design. Limit: a script can still read the store, and a hand-built `FittedCandidate` scores its own rows |
| A holdout read needs a reason after the first | Ledger, `claim_for_evaluation`, `withdraw_selection` | Hard unless the ledger is edited. Limit: a new resplit is a new file |
| Edited evidence cannot complete | `load_verified_final`, `_require_ledger_chain`, search-row rehash | Hard unless the ledger is edited |
| A holdout score cannot enter search metrics by name | `SearchMetrics`, `names_holdout` | Hard for split keys. Limit: a neutral name such as `q4_auc` passes |
| A reported gap and interval match the metrics | `CandidateResult` validators | Hard when reported |
| Models reject unknown fields | `pydantic-extra-forbid` | Lint |
| CLI above domain above I/O, and the engine never imports demos | `engine-layers`, `engine-independent-of-demos` | Hard for imports |
| Engine code passes strict types, and has no dead code | `make types`, `make dead` | Hard. Vulture has 4 known false positives in `ignore_names`, for names only generated code uses |
