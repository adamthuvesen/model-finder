---
name: verify-agentic-ml-loop
description: Drives the agentic-ml-loop CLI offline through a real demo run and writes evidence. Use to run, launch, drive, or verify agentic-ml-loop, or to prove a change to it works.
---

# Verify agentic-ml-loop

The surface is a short-lived CLI: `experiment` and `python -m demos.<id>`. Every run is free and
offline, and it uses a temp HOME and data home. The helper calls `.venv/bin` directly, because
`uv run` keeps its managed Python under `XDG_DATA_HOME` and would rebuild `.venv` from the temp
directory. Install first with `uv sync --extra models --extra deep`.

## Launch / Reset

`verify launch` creates the temp home and the run directory, and prints `ready: <run dir>`. There is
no server. `verify reset` is the same as a fresh launch.

## Doctor

`verify doctor` exits non-zero when the data home is outside this run's temp directory, when the
`experiment` script does not run, or when the engine resolves outside this checkout.

## Drive

`verify drive <feature-id>` runs one sub-feature. The session is `launch`, `doctor`, the drives in the
order that [features/README.md](features/README.md) lists, `control`, `clean`. Drives build on one
experiment, so the order matters. Assertions use exit codes and exact strings.

## Evidence

The run directory is `.verify/runs/<utc>-<sha7>[-dirty]/`. Each drive writes `<feature>.out`,
`<feature>.err`, and a `manifest.yml` block, and some add a second read such as a row count or
`replay.html`. No drive handles a secret.

- **Control.** `verify control` re-drives `evaluate-final` on a missing experiment and expects exit 0.
  The real exit is 1, so the manifest must show `result: fail-observed`.

## Cleanup

`verify clean` writes `tripwire` to the run directory, removes `.verify/state/`, and leaves the
evidence. The tripwire must read `empty`. Otherwise the run changed the real holdout store.

## Helper

- `verify launch`, `verify reset`: create the temp environment.
- `verify doctor`: check the environment and build identity.
- `verify drive <feature-id>`: run one sub-feature.
- `verify control`: run the mutated assertion.
- `verify clean`: stop and record the tripwire.
- `verify exec -- <command>`: run a command in the isolated environment.
- `VERIFY_TARGET`: overrides the data home, only to prove that doctor rejects it.
