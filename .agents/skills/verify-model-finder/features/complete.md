# Record and check completion

A reviewer records a verdict, the agent records the conclusion, and `complete-check` passes. An edit to the final file then fails the check.

Source: src/model_finder/completion.py, src/model_finder/cli/main.py
Inventory-id: experiment:record-review, experiment:record-completion, experiment:complete-check

## Sub-features

- `complete` writes both records, passes `complete-check`, and shows `Stale evaluation` after an edit.

## How to get to it (user POV)

- Run `experiment record-review`, `experiment record-completion`, then `experiment complete-check <dir>`.

## Driving it with verify

Preconditions: `final-after-freeze` passed in this session.

- **Complete.** Run `verify drive complete`. The check exits 0, and it exits 1 after the edit. The drive restores the file.
- **Proof.** Artifacts in the run directory: `complete-check.out`, `complete-check-edited.err`, and the manifest blocks.

## Gotchas

- The drive needs both candidates, so `search-metrics` also runs `majority-baseline`.
