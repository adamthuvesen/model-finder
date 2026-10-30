"""The ``experiment`` console script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from agentic_ml_loop.cli.errors import run_reporting_errors
from agentic_ml_loop.completion import check_completion
from agentic_ml_loop.final_evaluation import reseal_holdout
from agentic_ml_loop.paths import DATA_DIRNAME, HOLDOUT_SEAL_FILE_NAME, OUTPUTS_DIRNAME
from agentic_ml_loop.sources.adapters import SUPPORTED_SOURCE_TYPES
from agentic_ml_loop.sources.query import (
    apply_as_of,
    require_deterministic_query,
    require_read_only_query,
)
from agentic_ml_loop.sources.registry import enable_source, get_bundle, is_enabled, list_bundles
from agentic_ml_loop.sources.snapshot import (
    MANIFEST_FILENAME,
    SNAPSHOT_FILENAME,
    SnapshotFreezeRequest,
    freeze_snapshot,
    seal_snapshot,
)
from agentic_ml_loop.validation import validate_experiment
from agentic_ml_loop.viz.bundle import write_replay


def _validate(args: argparse.Namespace) -> int:
    messages = validate_experiment(Path(args.experiment), strict_completion=args.strict_completion)
    warnings = [message for message in messages if message.startswith("warning: ")]
    errors = [message for message in messages if not message.startswith("warning: ")]
    if errors:
        print(f"Validation failed: {args.experiment}", file=sys.stderr)
        for message in [*errors, *warnings]:
            print(f"  - {message}", file=sys.stderr)
        return 1
    print(f"Validation passed{' with warnings' if warnings else ''}: {args.experiment}")
    for warning in warnings:
        print(f"  - {warning}")
    return 0


def _complete_check(args: argparse.Namespace) -> int:
    errors = check_completion(Path(args.experiment))
    if errors:
        print("Research evidence check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Research evidence check passed: {args.experiment}")
    print("Traceability verified. Scientific validity still requires independent review.")
    return 0


def _seal(args: argparse.Namespace) -> int:
    seal = seal_snapshot(
        Path(args.experiment),
        args.time_column,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    print(f"Sealed {seal.holdout_rows} holdout rows outside the workspace: {seal.holdout_sha256}")
    return 0


def _reseal(args: argparse.Namespace) -> int:
    experiment_dir = Path(args.experiment).resolve()
    rows_path = Path(args.rows).resolve()
    if rows_path.is_relative_to(experiment_dir):
        raise ValueError(
            "New holdout rows must come from outside the experiment. "
            "Rows inside it are visible to search."
        )
    seal = reseal_holdout(experiment_dir, pd.read_parquet(rows_path), args.note)
    print(f"Retired the old holdout. Sealed {seal.holdout_rows} new rows: {seal.holdout_sha256}")
    return 0


def _replay(args: argparse.Namespace) -> int:
    experiment_dir = Path(args.experiment)
    output = Path(args.output) if args.output else experiment_dir / OUTPUTS_DIRNAME / "replay.html"
    print(write_replay(experiment_dir, output))
    return 0


def _read_query(value: str) -> str:
    return Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value


def _pull_config(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {}
    if args.uri:
        config["uri"] = args.uri
    if args.database:
        config["database"] = args.database
    if args.max_bytes is not None:
        config["max_bytes_billed"] = args.max_bytes
    conn_pairs: list[str] = args.conn or []
    for pair in conn_pairs:
        key, separator, value = pair.partition("=")
        if not separator:
            raise ValueError(f"--conn expects key=value, got {pair!r}")
        config[key] = value
    return config


def _pull(args: argparse.Namespace) -> int:
    experiment_dir = Path(args.experiment).resolve()
    if (experiment_dir / HOLDOUT_SEAL_FILE_NAME).exists():
        raise ValueError(
            "This experiment is sealed. A pull here would put holdout rows back in the workspace. "
            "Pull new rows into another directory, then run `experiment reseal --rows <parquet>`."
        )
    query = _read_query(args.query)
    rows_dir = experiment_dir / DATA_DIRNAME
    if args.dry_run:
        require_read_only_query(query, args.source)
        require_deterministic_query(
            query, dialect=args.source, allow_nondeterministic=args.allow_nondeterministic
        )
        rewritten, recorded = apply_as_of(query, args.source, args.as_of)
        print(f"[dry-run] source={args.source} experiment={experiment_dir.name} as_of={recorded}")
        print(f"[dry-run] would write {rows_dir / SNAPSHOT_FILENAME}")
        print(f"[dry-run] query:\n{rewritten}")
        return 0
    manifest = freeze_snapshot(
        SnapshotFreezeRequest(
            source_type=args.source,
            query=query,
            out_dir=rows_dir,
            config=_pull_config(args),
            as_of=args.as_of,
            sample_seed=args.sample_seed,
            allow_nondeterministic=args.allow_nondeterministic,
        )
    )
    print(
        f"Wrote {manifest.row_count} rows x {manifest.column_count} cols "
        f"-> {rows_dir / manifest.snapshot_filename}"
    )
    print(f"Manifest: {rows_dir / MANIFEST_FILENAME}")
    return 0


def _sources_list(_: argparse.Namespace) -> int:
    for bundle in list_bundles():
        mark = "*" if is_enabled(bundle.source_type) else " "
        print(f"[{mark}] {bundle.source_type:<10} {bundle.display_name}: {bundle.description}")
    print("\n* = enabled. Enable with: experiment sources enable <name>")
    return 0


def _sources_enable(args: argparse.Namespace) -> int:
    bundle = get_bundle(args.name)
    enable_source(args.name)
    print(f"Enabled '{args.name}'. Follow setup steps in: {bundle.setup_path}")
    return 0


def _add_pull_arguments(pull: argparse.ArgumentParser) -> None:
    pull.add_argument("--experiment", required=True)
    pull.add_argument("--source", required=True, choices=SUPPORTED_SOURCE_TYPES)
    pull.add_argument("--query", required=True, help="SQL string, or @path to a .sql file")
    pull.add_argument("--as-of", default=None, help="As-of timestamp/version (see source SETUP.md)")
    pull.add_argument("--sample-seed", type=int, default=None, help="Seed used for sampling")
    pull.add_argument("--uri", default=None, help="Connection URI (e.g. postgres)")
    pull.add_argument("--database", default=None, help="Database path (e.g. duckdb file)")
    pull.add_argument("--conn", action="append", metavar="KEY=VALUE", help="Repeatable")
    pull.add_argument(
        "--max-bytes", type=int, default=None, help="BigQuery scan ceiling in bytes (best-effort)"
    )
    pull.add_argument("--dry-run", action="store_true", help="Show the plan without extracting")
    pull.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help="Override the reproducibility guard (unseeded LIMIT/sampling)",
    )
    pull.set_defaults(func=_pull)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="experiment", description="ML experiment evidence tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Check an experiment's structure")
    validate.add_argument("experiment")
    validate.add_argument(
        "--strict-completion", action="store_true", help="Require at least one result entry"
    )
    validate.set_defaults(func=_validate)

    complete = subparsers.add_parser("complete-check", help="Check research completion evidence")
    complete.add_argument("experiment")
    complete.set_defaults(func=_complete_check)

    seal = subparsers.add_parser("seal", help="Seal the latest snapshot rows as the holdout")
    seal.add_argument("--experiment", required=True)
    seal.add_argument("--time-column", required=True)
    seal.add_argument("--train-fraction", type=float, default=0.6)
    seal.add_argument("--validation-fraction", type=float, default=0.2)
    seal.set_defaults(func=_seal)

    reseal = subparsers.add_parser("reseal", help="Retire a read holdout and seal new rows")
    reseal.add_argument("--experiment", required=True)
    reseal.add_argument("--rows", required=True, help="Parquet file of new, unseen rows")
    reseal.add_argument(
        "--note", required=True, help="Why the holdout is retired and where the rows came from"
    )
    reseal.set_defaults(func=_reseal)

    replay = subparsers.add_parser("replay", help="Render results and the journal to HTML")
    replay.add_argument("experiment")
    replay.add_argument("--output", default=None, help="Default: <experiment>/outputs/replay.html")
    replay.set_defaults(func=_replay)

    _add_pull_arguments(
        subparsers.add_parser("pull", help="Materialize a warehouse snapshot into an experiment")
    )

    sources = subparsers.add_parser("sources", help="List or enable source bundles")
    sources_commands = sources.add_subparsers(dest="sources_command", required=True)
    sources_commands.add_parser("list", help="List source bundles").set_defaults(func=_sources_list)
    enable = sources_commands.add_parser("enable", help="Enable a source bundle")
    enable.add_argument("name", choices=SUPPORTED_SOURCE_TYPES)
    enable.set_defaults(func=_sources_enable)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run_reporting_errors(lambda: args.func(args))
