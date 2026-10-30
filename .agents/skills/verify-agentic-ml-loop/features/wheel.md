# Install from a built wheel

A consumer installs the wheel, links the packaged skills, and prints the contract without a checkout.

Source: src/agentic_ml_loop/cli/setup.py, src/agentic_ml_loop/cli/main.py, pyproject.toml
Inventory-id: experiment:setup, experiment:contract

## Sub-features

- `wheel` builds the wheel, installs it in a temp consumer, links skills, finds the deny rule, prints `contract evidence`, and finds `py.typed`.

## How to get to it (user POV)

- Run `experiment setup` in the consumer repository, then `experiment contract evidence`.

## Driving it with verify

Preconditions: `launch` and `doctor` passed. The drive needs the `uv` cache or the network to install dependencies.

- **Wheel.** Run `verify drive wheel`. The link resolves, the contract prints, and the engine resolves inside the consumer.
- **Proof.** Artifacts in the run directory: `wheel-setup.out`, `wheel-contract.out`, `wheel.origin`, and the manifest blocks.

## Gotchas

- The build runs outside the temp home, because `uv` keeps its managed Python under the real data home.
