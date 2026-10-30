# Examples

## `demo_bootstrap_replay.html`

A self-contained replay of a `demo_bootstrap` run. Open it in any browser. It walks through the
example research journal and ends with the leaderboard.

Regenerate it by running every candidate in a new experiment, then:

```bash
uv run experiment replay experiments/my_run --output examples/demo_bootstrap_replay.html
```

The replay reads only `experiment.md`, `results.json`, and `research_journal.md`.
