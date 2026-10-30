# AGENTS.md

`agentic-ml-loop` is a model-search engine for an agent running under a native client `/goal`.
The client owns continuation. This repo owns the evidence: search rows, a sealed holdout,
run records, and a completion check. Docs: [architecture](.agents/docs/architecture.md),
[evidence](docs/evidence.md), [why native goals](.agents/docs/decisions/native-goals.md).

## Layout

```text
src/agentic_ml_loop/  The engine. The `experiment` console script lives in cli/.
demos/                Free, offline demos. They import the engine. It never imports them.
experiments/<id>/     One experiment: spec, journal, sources, leaderboard, data, outputs.
rules/, rule-tests/   ast-grep rules and their tests.
src/agentic_ml_loop/skills/, contract/  Packaged skills and documents. `experiment setup` links the skills.
.agents/skills/       Links to the packaged skills, plus verify-agentic-ml-loop.
```

## Commands

```bash
make check                                       # the gate; CI runs the same targets
uv run --extra models python -m demos.demo_bootstrap init-demo --experiment experiments/my_run
uv run --extra models python -m demos.demo_bootstrap run-candidate --experiment experiments/my_run --candidate logreg-tiny
uv run experiment validate experiments/my_run
```

Prove user-facing changes with the `verify-agentic-ml-loop` skill at
`.agents/skills/verify-agentic-ml-loop/`.

## Conventions

- `experiment.md` is a spec. Change it only when the human asks. A change starts a new comparison.
- Write one-shot scripts under `experiments/<id>/scripts/`.
- Code for one experiment stays out of the engine. Nothing specific to one company enters this repo.
- Never commit secrets, `.env`, generated notebooks, or local data.
- If a doc disagrees with code, fix the doc in the same change.
- Keep torch and the GBDT libraries in separate processes. Together they can deadlock on OpenMP on macOS.
- With a custom `XDG_DATA_HOME`, call `.venv/bin` directly. `uv run` stores its Python there and rebuilds `.venv`.

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

"Honest code" means the check stops a normal path. Deliberate code can still get past it. "Label" means the
check does not refuse: the result says `complete, exploratory` or prints a warning.

| Rule | Enforced by | Strength | Added |
| --- | --- | --- | --- |
| Search results hold no holdout metric | `SearchMetrics` and name checks in `candidate_result.py` | Hard for split keys; honest code for names. Limit: a neutral or period-labelled name such as `q4_auc` passes, and it can stand in for a holdout. A name cannot prove which rows a metric used | 2026-11-04 |
| Candidate code receives no holdout rows | `SearchSplits` type, `no-holdout-access-in-demos`, `demos-cannot-reach-holdout-store` | Honest code: a script can still read the file | 2026-11-04 |
| Holdout rows stay outside the workspace | `holdout.py`, pull refusal, deny rule that `experiment setup` adds to `.claude/settings.json` | Honest code: Claude Code only, and a rule does not match every command form | 2026-11-04 |
| A local rows file is named at seal time | `RunnerConfig.rows_source`, warning at seal and reseal | Report only: the engine cannot delete a source file, so its rows stay readable | 2026-11-04 |
| Supervised feature artifacts used only training labels | `experiment check-features` | Honest code: it checks declared dates, when an agent runs it | 2026-11-04 |
| Selection comes before holdout evaluation | `evaluate_final` needs a frozen selection | Hard in the engine; choosing again after a read is a labelled override | 2026-11-04 |
| A second read of a sealed file needs a reason | Ledger keyed by file hash, `claim_for_evaluation`, `--allow-reread --reason` | Label. Limit: a new-seed resplit or schema change makes a different file, so a different holdout. Limit: the ledger is a plain file, and a deliberate agent can edit it | 2026-11-04 |
| Search after a holdout read needs a reason | `withdraw_selection`, `--allow-reread --reason` | Label | 2026-11-04 |
| A holdout is optional | `--no-holdout`, `assess_completion` | Label: `complete, exploratory` | 2026-11-04 |
| Retiring a holdout leaves a record | `reseal_holdout`, ledger count | Label, and a short note | 2026-11-04 |
| Edited evidence cannot complete | `load_verified_final`, `_require_ledger_chain`, search-row rehash | Hard, short of editing the ledger. Limit: a read the ledger never saw is labelled unverified, not refused, so another machine works | 2026-11-04 |
| Overlap and drift are reported | `data_audit.py`, `complete-check` | Warning. At least 10% of holdout rows shared needs a one-line reason, and it labels the result | 2026-11-04 |
| Edits since a run are reported | `run_evidence_staleness`, `complete-check --strict` | Warning; `--strict` fails | 2026-11-04 |
| Only the engine writes the leaderboard | `results-json-single-writer`, `results-json-engine-only` | Honest code | 2026-11-04 |
| A damaged leaderboard is rebuilt from run records, not edited | `experiment rebuild-results`, `--allow-changes --reason` | Label: a change to a readable leaderboard needs a reason. Hard for invalid records and a frozen selection. Limit: run records are plain files, so an edit to a record and the leaderboard together goes unseen | 2026-11-04 |
| A reported gap and interval match the metrics | `CandidateResult` validators | Hard when reported. All are optional | 2026-11-04 |
| Models reject unknown fields | `pydantic-extra-forbid` | Lint | 2026-11-04 |
| The engine never imports demos | `engine-independent-of-demos`, `no-dynamic-experiment-import` | Hard for imports; honest code for runtime paths | 2026-11-04 |
| CLI above domain above I/O | `engine-layers` | Hard | 2026-11-04 |
| No agent SDK or spawned process | Ruff `TID251`, `no-agent-runtime` rule and contract | Honest code: `getattr(os, ...)` passes. Covers `experiments/*/scripts`, which get no other lint | 2026-11-04 |
| Pulls are read-only | `require_read_only_query`, `verify_snapshot` | Hard in the engine, with no override. Defence in depth: the warehouse role is the real guard | 2026-11-04 |
| Pulls are reproducible | `require_deterministic_query`, `--allow-nondeterministic --reason` | Label: the reason goes in the manifest | 2026-11-04 |
| Engine code passes strict types | `make types` | Hard | 2026-11-04 |
