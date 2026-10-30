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
| New result field | One optional field in `CandidateResult`. A required field also needs a value in `scored_candidate` and in `result_fields()` in `tests/conftest.py` |
| New source | `sources/bundles/<name>/source.json` and `SETUP.md`, an extra in `pyproject.toml`, and a `source_type` that is a sqlglot dialect. A non-ADBC driver also needs an adapter class. `experiment sources check` reports what is missing |
| A label or an exploratory reason | `labels.py` |
| Freeze and evaluate | `final_evaluation.py` |
| Reseal, or rebuild the leaderboard | `reseal.py`, `rebuild.py` |
| Skills and documents shipped in the wheel | `src/model_finder/skills/`, `contract/` |

## Commands

```bash
make quick            # only changed Python files, no tests: under 1s for one file
make fast             # the fast lane, about 13s; the first CI job
make verify           # slow tests in tests/verify, and coverage over the whole suite
make check            # fast, then verify: the full offline gate
uv run experiment next
uv run experiment contract commands
```

Put a test that starts a process or runs a demo in `tests/verify/`. Prove user-facing changes with
`.agents/skills/verify-model-finder/`. Build test results with `candidate_result()` or `result_fields()` in
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
`tests/test_agents_table.py` fails when a name in backticks under "Enforced by" does not exist.

| Rule | Enforced by | Added | Limit |
| --- | --- | --- | --- |
| No agent SDK or spawned process | Ruff `TID251` in `pyproject.toml`, `rules/no-agent-runtime.yml` | 2026-11-04 | `getattr(os, ...)` passes |
| Pulls are read-only, with no override | `require_read_only_query`, `verify_snapshot` | 2026-06-07 | Defence in depth: the warehouse role is the real guard |
| The agent does not read the holdout store | `HOLDOUT_DENY_RULES`, which `experiment setup` writes | 2026-11-04 | Claude Code only, and a rule does not match every command form |
| A consumer's declared claude.ai connector writes are denied | `_connector_rules`, `CONNECTOR_RULE_PREFIX` | 2026-11-06 | Claude Code only. A tool the declaration omits stays allowed |
| Candidates receive no holdout rows, and predict never receives the target | `SearchSplits`, `scored_candidate`, `demos-cannot-reach-holdout-store` | 2026-11-04 | A script can still read the store, and a hand-built `FittedCandidate` scores its own rows |
| A holdout read needs a reason after the first | `claim_for_evaluation`, `withdraw_selection` | 2026-11-04 | An edited ledger passes, and a new resplit is a new file |
| Edited evidence cannot complete | `load_verified_final`, `_require_ledger_chain` | 2026-11-04 | An edited ledger passes |
| A holdout score cannot enter search metrics by name | `SearchMetrics`, `names_holdout` | 2026-11-04 | A neutral name such as `q4_auc` passes |
| A reported gap and interval match the metrics | `_gap_matches_metrics`, `_interval_declares_bootstrap_count` | 2026-11-04 | Only when reported |
| Models reject unknown fields | `rules/pydantic-extra-forbid.yml` | 2026-11-04 | Lint |
| CLI above domain above I/O, and the engine never imports demos | `engine-layers`, `engine-independent-of-demos` | 2026-11-04 | Imports only |
| Engine code passes strict types, and has no dead code | `[tool.basedpyright]`, `scripts/dead_code.py` | 2026-11-04 | Vulture counts every Pydantic field as used |
