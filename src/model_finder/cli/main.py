from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pandas as pd

from model_finder.candidate_result import CandidateResult
from model_finder.cli.errors import (
    override_reason,
    run_reporting_errors,
    warn_if_already_read,
    warn_rows_stay_readable,
)
from model_finder.cli.next_step import choose_experiment, command_prefix, next_steps
from model_finder.cli.scaffold import new_candidate, new_experiment
from model_finder.cli.setup import setup_repo
from model_finder.completion import (
    assess_completion,
    record_from_evidence,
    save_completion,
    write_review,
)
from model_finder.feature_provenance import validate_feature_provenance
from model_finder.holdout import load_search_splits
from model_finder.labels import leaderboard_label
from model_finder.paths import (
    DATA_DIRNAME,
    HOLDOUT_SEAL_FILE_NAME,
    OUTPUTS_DIRNAME,
    require_experiment_dir,
)
from model_finder.rebuild import rebuild_results
from model_finder.reseal import reseal_holdout
from model_finder.results import load_results, ranked_results
from model_finder.sources.adapters import SUPPORTED_SOURCE_TYPES
from model_finder.sources.query import (
    apply_as_of,
    require_deterministic_query,
    require_read_only_query,
)
from model_finder.sources.registry import (
    bundle_problems,
    enable_source,
    get_bundle,
    is_enabled,
    list_bundles,
)
from model_finder.sources.snapshot import (
    MANIFEST_FILENAME,
    SNAPSHOT_FILENAME,
    SnapshotFreezeRequest,
    freeze_snapshot,
    seal_snapshot,
    split_snapshot_without_holdout,
)
from model_finder.validation import validate_experiment
from model_finder.viz.bundle import write_replay

_PACKAGE_ROOT = Path(__file__).parents[1]
CONTRACT_DOCUMENTS = {
    "evidence": _PACKAGE_ROOT / "contract" / "evidence.md",
    "method": _PACKAGE_ROOT / "contract" / "method.md",
    "spec-template": _PACKAGE_ROOT / "templates" / "model-search.md",
}
GENERATED_DOCUMENTS = ("candidate-schema", "commands")


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
    status = assess_completion(Path(args.experiment), strict=args.strict)
    print(f"Result: {status.label}")
    for heading, lines in (
        ("Not complete because:", status.errors),
        ("Exploratory because:", status.exploratory),
        ("Warnings:", status.warnings),
    ):
        if lines:
            print(heading)
            print("\n".join(f"  - {line}" for line in lines))
    return 1 if status.errors else 0


def _leaderboard(args: argparse.Namespace) -> int:
    _print_leaderboard(Path(args.experiment))
    return 0


def _print_leaderboard(experiment_dir: Path) -> None:
    require_experiment_dir(experiment_dir)
    results = ranked_results(load_results(experiment_dir))
    if not results:
        print("No results yet. Run `experiment next` for the command.")
        return
    for rank, result in enumerate(results, start=1):
        print(f"{rank:>3}  {result.objective_score:>10.4f}  {result.candidate_id}")
    print(f"Metric: {results[0].objective_metric}, on validation rows.")
    label = leaderboard_label(experiment_dir)
    if label:
        print(label)


def _record_review(args: argparse.Namespace) -> int:
    require_experiment_dir(Path(args.experiment))
    path = write_review(
        Path(args.experiment),
        args.reviewer,
        args.verdict,
        args.report,
        args.finding,
        args.limitation,
    )
    print(f"Review recorded: {path}")
    return 0


def _record_completion(args: argparse.Namespace) -> int:
    experiment_dir = Path(args.experiment)
    require_experiment_dir(experiment_dir)
    record = record_from_evidence(
        experiment_dir,
        args.report,
        args.reproduction,
        outcome=args.outcome,
        conclusion=args.conclusion,
        limitations=args.limitation,
        selected_candidate_id=args.selected,
        baseline_candidate_id=args.baseline,
        comparison_candidate_id=args.comparison,
        acknowledged_overlap=args.acknowledged_overlap,
    )
    print(f"Completion recorded: {save_completion(experiment_dir, record)}")
    print(f"Result: {assess_completion(experiment_dir).label}")
    return 0


def _rebuild_results(args: argparse.Namespace) -> int:
    reason = override_reason("--allow-changes", allowed=args.allow_changes, reason=args.reason)
    print(rebuild_results(Path(args.experiment), change_reason=reason))
    return 0


def _new(args: argparse.Namespace) -> int:
    root = Path.cwd()
    experiment_dir = new_experiment(
        root,
        args.slug,
        Path(args.csv),
        target=args.target,
        time_column=args.time_column,
        no_holdout=args.no_holdout,
    )
    relative = experiment_dir.relative_to(root)
    print()
    _print_leaderboard(relative)
    print()
    _print_steps(relative)
    return 0


def _candidate_new(args: argparse.Namespace) -> int:
    root = Path.cwd()
    path, candidate_id = new_candidate(root, args.name, args.experiment)
    shown = path.relative_to(root) if path.is_relative_to(root) else path
    print(f"Added {candidate_id} to {shown}. Change the model, and its CHANGE docstring.")
    tool = command_prefix()
    print(
        f"Run it: {tool}python {shown.parent / 'run.py'} run-candidate --candidate {candidate_id}"
    )
    return 0


def _print_steps(experiment_dir: Path) -> None:
    for number, step in enumerate(next_steps(experiment_dir)):
        print(f"{'Next' if number == 0 else 'Or'}: {step.text}\n  {step.command}")


def _next(args: argparse.Namespace) -> int:
    root = Path.cwd()
    if args.experiment:
        require_experiment_dir(Path(args.experiment))
    experiment_dir = Path(args.experiment) if args.experiment else choose_experiment(root)
    if experiment_dir is None:
        found = sorted(str(path.parent) for path in root.glob("experiments/*/experiment.md"))
        tool = command_prefix()
        print("Next: Start an experiment, or pick one.")
        print(f"  {tool}experiment new my_run --csv rows.csv")
        for path in found:
            print(f"  {tool}experiment next {path}")
        return 0
    _print_steps(experiment_dir)
    return 0


def _setup(args: argparse.Namespace) -> int:
    print("\n".join(setup_repo(Path(args.repo).resolve()) or ["nothing to do"]))
    return 0


def _command_reference() -> str:
    parser = _parser()
    commands = cast("dict[str, argparse.ArgumentParser]", parser.get_default("command_parsers"))
    return "\n".join([parser.format_help(), *(sub.format_help() for sub in commands.values())])


def _contract(args: argparse.Namespace) -> int:
    if args.document == "candidate-schema":
        print(json.dumps(CandidateResult.model_json_schema(), indent=2))
    elif args.document == "commands":
        print(_command_reference())
    else:
        print(CONTRACT_DOCUMENTS[args.document].read_text(encoding="utf-8"), end="")
    return 0


def _row_dates(rows: pd.DataFrame, time_column: str) -> set[str]:
    return {str(date) for date in pd.DatetimeIndex(rows[time_column]).strftime("%Y-%m-%d")}


def _check_features(args: argparse.Namespace) -> int:
    search = load_search_splits(Path(args.experiment))
    if args.time_column not in search.train.columns:
        raise ValueError(
            f"Time column is missing from search rows: {args.time_column}. "
            f"Fix: use one of: {', '.join(str(column) for column in search.train.columns)}"
        )
    validate_feature_provenance(
        Path(args.artifact),
        selection_train_dates=_row_dates(search.train, args.time_column),
        validation_dates=_row_dates(search.validation, args.time_column),
        test_dates=(),
    )
    print(f"Feature provenance holds: {args.artifact}")
    return 0


def _seal(args: argparse.Namespace) -> int:
    if args.no_holdout:
        split_snapshot_without_holdout(Path(args.experiment), args.time_column)
        print("Wrote search rows with no holdout. Results stay labelled exploratory.")
        return 0
    seal = seal_snapshot(
        Path(args.experiment),
        args.time_column,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    print(f"Sealed {seal.holdout_rows} holdout rows outside the workspace: {seal.holdout_sha256}")
    return 0


def _reseal(args: argparse.Namespace) -> int:
    rows_path = Path(args.rows)
    seal = reseal_holdout(Path(args.experiment).resolve(), pd.read_parquet(rows_path), args.note)
    print(f"Sealed {seal.holdout_rows} new rows: {seal.holdout_sha256}")
    warn_rows_stay_readable(rows_path)
    warn_if_already_read(seal.holdout_sha256)
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
            raise ValueError(f"--conn expects key=value, got {pair!r}. Fix: write --conn key=value")
        config[key] = value
    return config


def _pull(args: argparse.Namespace) -> int:
    experiment_dir = Path(args.experiment).resolve()
    if (experiment_dir / HOLDOUT_SEAL_FILE_NAME).exists():
        raise ValueError(
            f"{experiment_dir}: This experiment is sealed, and a pull here would put holdout rows "
            "back in the workspace. Fix: pull new rows into another directory, then run "
            "`experiment reseal --rows <parquet>`."
        )
    query = _read_query(args.query)
    rows_dir = experiment_dir / DATA_DIRNAME
    nondeterministic_reason = override_reason(
        "--allow-nondeterministic", allowed=args.allow_nondeterministic, reason=args.reason
    )
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
            nondeterministic_reason=nondeterministic_reason,
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


def _sources_check(_: argparse.Namespace) -> int:
    problems = bundle_problems()
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print(f"All {len(list_bundles())} source bundles pass.")
    return 1 if problems else 0


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
    pull.add_argument("--reason", default=None, help="Why. Required with --allow-nondeterministic")
    pull.set_defaults(func=_pull)


def _add_review_arguments(review: argparse.ArgumentParser) -> None:
    review.add_argument("--experiment", required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--verdict", required=True, choices=["accepted", "changes_requested"])
    review.add_argument("--report", default=None, help="Path inside the experiment")
    review.add_argument("--finding", action="append", default=[], help="Repeat for each finding")
    review.add_argument("--limitation", action="append", default=[], help="Repeat for each")
    review.set_defaults(func=_record_review)


def _add_completion_arguments(completion: argparse.ArgumentParser) -> None:
    completion.add_argument("--experiment", required=True)
    completion.add_argument(
        "--outcome",
        required=True,
        choices=["supported", "no_demonstrated_improvement", "inconclusive"],
    )
    completion.add_argument("--conclusion", required=True)
    completion.add_argument("--limitation", action="append", default=[], help="Repeat for each")
    completion.add_argument("--report", default=None, help="Path inside the experiment")
    completion.add_argument("--reproduction", default=None, help="Path inside the experiment")
    completion.add_argument("--selected", default=None, help="Candidate ID")
    completion.add_argument("--baseline", default=None, help="Candidate ID")
    completion.add_argument("--comparison", default=None, help="Candidate ID when baseline stays")
    completion.add_argument("--acknowledged-overlap", default=None)
    completion.set_defaults(func=_record_completion)


def _add_seal_arguments(seal: argparse.ArgumentParser) -> None:
    seal.add_argument("--experiment", required=True)
    seal.add_argument("--time-column", required=True)
    seal.add_argument("--no-holdout", action="store_true", help="Keep all rows for search")
    seal.add_argument("--train-fraction", type=float, default=0.6)
    seal.add_argument("--validation-fraction", type=float, default=0.2)
    seal.set_defaults(func=_seal)


def _add_reseal_arguments(reseal: argparse.ArgumentParser) -> None:
    reseal.add_argument("--experiment", required=True)
    reseal.add_argument("--rows", required=True, help="Parquet file of new rows")
    reseal.add_argument(
        "--note", required=True, help="A short note: why, and where the rows came from"
    )
    reseal.set_defaults(func=_reseal)


def _add_start_commands(add_parser: Callable[..., argparse.ArgumentParser]) -> None:
    new = add_parser("new", help="Create an experiment from a CSV, and run its first candidates")
    new.add_argument("slug")
    new.add_argument("--csv", required=True)
    new.add_argument("--target", help="Default: a column named label or target, else the last")
    new.add_argument("--time-column", help="Default: a date-like column name, else file row order")
    new.add_argument("--no-holdout", action="store_true", help="Keep all rows for search")
    new.set_defaults(func=_new)

    candidate = add_parser("candidate", help="Add a candidate")
    candidate_commands = candidate.add_subparsers(dest="candidate_command", required=True)
    candidate_new = candidate_commands.add_parser("new", help="Add a working candidate stub")
    candidate_new.add_argument("name")
    candidate_new.add_argument("--experiment", help="Default: the only experiment here")
    candidate_new.set_defaults(func=_candidate_new)

    next_command = add_parser("next", help="Suggest useful next steps. Advice only")
    next_command.add_argument("experiment", nargs="?")
    next_command.set_defaults(func=_next)


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
    complete.add_argument(
        "--strict", action="store_true", help="Fail on every exploratory label and warning"
    )
    complete.set_defaults(func=_complete_check)

    _add_start_commands(subparsers.add_parser)

    leaderboard = subparsers.add_parser("leaderboard", help="Print candidates, best first")
    leaderboard.add_argument("experiment")
    leaderboard.set_defaults(func=_leaderboard)

    rebuild = subparsers.add_parser(
        "rebuild-results", help="Rebuild the leaderboard from run records and say what changed"
    )
    rebuild.add_argument("experiment")
    rebuild.add_argument(
        "--allow-changes",
        action="store_true",
        help="Accept results that differ from the leaderboard. Recorded, and labelled exploratory",
    )
    rebuild.add_argument("--reason", default=None, help="Why. Required with --allow-changes")
    rebuild.set_defaults(func=_rebuild_results)

    _add_review_arguments(
        subparsers.add_parser(
            "record-review", help="Write outputs/review.json with current evidence hashes"
        )
    )
    _add_completion_arguments(
        subparsers.add_parser(
            "record-completion", help="Write outputs/completion.json if the completion check passes"
        )
    )

    setup = subparsers.add_parser(
        "setup", help="Link the packaged skills and deny the holdout store to Claude Code"
    )
    setup.add_argument("--repo", default=".", help="Repository root. Default: this directory")
    setup.set_defaults(func=_setup)

    contract = subparsers.add_parser("contract", help="Print a packaged document")
    contract.add_argument("document", choices=sorted([*CONTRACT_DOCUMENTS, *GENERATED_DOCUMENTS]))
    contract.set_defaults(func=_contract)

    features = subparsers.add_parser(
        "check-features", help="Check that a feature artifact used only training labels"
    )
    features.add_argument("--experiment", required=True)
    features.add_argument("--artifact", required=True)
    features.add_argument("--time-column", required=True)
    features.set_defaults(func=_check_features)

    _add_seal_arguments(
        subparsers.add_parser("seal", help="Seal the latest snapshot rows as the holdout")
    )
    _add_reseal_arguments(
        subparsers.add_parser("reseal", help="Seal new rows, retiring any current holdout")
    )

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
    sources_commands.add_parser(
        "check", help="Report what each bundle still needs: JSON, dialect, extra, adapter"
    ).set_defaults(func=_sources_check)
    enable = sources_commands.add_parser("enable", help="Enable a source bundle")
    enable.add_argument("name", choices=SUPPORTED_SOURCE_TYPES)
    enable.set_defaults(func=_sources_enable)
    parser.set_defaults(command_parsers=subparsers.choices)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run_reporting_errors(lambda: args.func(args))
