"""The command line an experiment's runner module exposes."""

from __future__ import annotations

import argparse
import json
import secrets
import shlex
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from model_finder.candidate_result import CandidateRunner
from model_finder.cli.errors import (
    override_reason,
    run_reporting_errors,
    warn_if_already_read,
    warn_rows_stay_readable,
)
from model_finder.final_evaluation import (
    evaluate_final,
    freeze_candidate,
    load_verified_final,
    selection_staleness,
)
from model_finder.holdout import TRAIN_FILE_NAME, seal_holdout, write_search_rows
from model_finder.io import write_text
from model_finder.labels import assess_final
from model_finder.paths import DATA_DIRNAME, HOLDOUT_SEAL_FILE_NAME, RUNNER_COMMAND_PATH
from model_finder.runner import ExperimentSeed, init_experiment_dir, save_candidate_result
from model_finder.splits import temporal_search_split, temporal_split
from model_finder.validation import hyperparameter_warnings


@dataclass(frozen=True)
class RunnerConfig:
    seed: ExperimentSeed
    candidate_runners: Mapping[str, CandidateRunner]
    rows_loader: Callable[[int], pd.DataFrame]
    time_column: str | None
    target_column: str
    code_root: Path
    config_files: tuple[Path, ...] = ()
    # The local file that rows_loader reads, if any. Sealing warns that its rows stay readable.
    rows_source: Path | None = None
    # The default for --experiment, so a generated runner needs no flag.
    experiment_dir: Path | None = None


def _add_reread_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-reread",
        action="store_true",
        help="Continue after a holdout read. Recorded, and labelled exploratory",
    )
    parser.add_argument("--reason", default=None, help="Why. Required with --allow-reread")


def _reread_reason(args: argparse.Namespace) -> str | None:
    return override_reason("--allow-reread", allowed=args.allow_reread, reason=args.reason)


def _warn_stale(messages: list[str]) -> None:
    for message in messages:
        print(f"warning: {message}", file=sys.stderr)


def _parser(config: RunnerConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{config.seed.experiment_id} runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-candidates", help="List bundled candidates")

    init = subparsers.add_parser("init-demo", help="Create an experiment and seal its holdout")
    init.add_argument("--experiment", help="New experiment directory")
    init.add_argument("--force", action="store_true", help="Reseed files; keeps holdout evidence")
    init.add_argument(
        "--no-holdout", action="store_true", help="Seal nothing. Results stay labelled exploratory"
    )
    init.add_argument(
        "--data-seed", type=int, default=None, help="Rows to generate. Default: a new random seed"
    )

    candidates = sorted(config.candidate_runners)
    run = subparsers.add_parser("run-candidate", help="Run one candidate on search rows")
    run.add_argument("--experiment")
    run.add_argument("--candidate", required=True, choices=candidates)
    _add_reread_arguments(run)

    freeze = subparsers.add_parser("freeze-candidate", help="Freeze a verified candidate")
    freeze.add_argument("--experiment")
    freeze.add_argument("--candidate", required=True, choices=candidates)
    _add_reread_arguments(freeze)

    final = subparsers.add_parser("evaluate-final", help="Score the frozen candidate once")
    final.add_argument("--experiment")
    _add_reread_arguments(final)
    return parser


def initialize_experiment(
    config: RunnerConfig,
    experiment_dir: Path,
    *,
    force: bool,
    data_seed: int | None,
    no_holdout: bool,
) -> str:
    init_experiment_dir(experiment_dir, config.seed, force=force)
    if (experiment_dir / HOLDOUT_SEAL_FILE_NAME).exists():
        return f"Initialized {experiment_dir}. Its holdout was already sealed."
    if (experiment_dir / DATA_DIRNAME / TRAIN_FILE_NAME).exists():
        return f"Initialized {experiment_dir}. It already has search rows and no holdout."
    seed = secrets.randbits(32) if data_seed is None else data_seed
    rows = config.rows_loader(seed)
    if no_holdout:
        write_search_rows(experiment_dir, temporal_search_split(rows, config.time_column))
        return (
            f"Initialized {experiment_dir} with no holdout. Results stay labelled exploratory. "
            "Add a holdout later with `experiment reseal`."
        )
    search, holdout = temporal_split(rows, config.time_column)
    seal = seal_holdout(experiment_dir, search, holdout)
    warn_if_already_read(seal.holdout_sha256)
    if config.rows_source is not None:
        warn_rows_stay_readable(config.rows_source)
    return f"Initialized {experiment_dir}. Sealed {seal.holdout_rows} holdout rows."


def _experiment_dir(config: RunnerConfig, given: str | None) -> Path:
    if given is not None:
        return Path(given).resolve()
    if config.experiment_dir is None:
        raise ValueError(
            "this runner has no default experiment. Fix: pass --experiment experiments/<slug>"
        )
    return config.experiment_dir.resolve()


def _record_runner_command(experiment_dir: Path) -> None:
    """Remember how this runner was started, so that `experiment next` can print exact commands."""
    if not experiment_dir.is_dir() or not sys.argv[1:]:
        return
    invocation = sys.orig_argv[1 : -len(sys.argv[1:])]
    write_text(experiment_dir / RUNNER_COMMAND_PATH, shlex.join(["python", *invocation]) + "\n")


def run_runner_main(
    config_or_loader: RunnerConfig | Callable[[], RunnerConfig], argv: list[str] | None = None
) -> int:
    """Run one command. A loader lets a mistake in the candidates report its own line."""

    def dispatch() -> int:
        config = config_or_loader() if callable(config_or_loader) else config_or_loader
        args = _parser(config).parse_args(argv)
        if args.command == "list-candidates":
            print("\n".join(config.candidate_runners))
            return 0
        experiment_dir = _experiment_dir(config, args.experiment)
        if args.command == "init-demo":
            print(
                initialize_experiment(
                    config,
                    experiment_dir,
                    force=args.force,
                    data_seed=args.data_seed,
                    no_holdout=args.no_holdout,
                )
            )
            _record_runner_command(experiment_dir)
        elif args.command == "run-candidate":
            _record_runner_command(experiment_dir)
            result = save_candidate_result(
                experiment_dir,
                args.candidate,
                config.candidate_runners,
                config.code_root,
                config.config_files,
                reread_reason=_reread_reason(args),
            )
            _warn_stale(hyperparameter_warnings([result]))
            print(
                json.dumps(
                    {
                        "candidate_id": result.candidate_id,
                        "objective_metric": result.objective_metric,
                        "objective_score": result.objective_score,
                    },
                    indent=2,
                )
            )
        elif args.command == "freeze-candidate":
            selection = freeze_candidate(
                experiment_dir,
                args.candidate,
                config.target_column,
                reread_reason=_reread_reason(args),
            )
            print(selection.model_dump_json(indent=2))
            _warn_stale(selection_staleness(experiment_dir, selection))
        else:
            final = evaluate_final(
                experiment_dir, config.candidate_runners, reread_reason=_reread_reason(args)
            )
            print(final.model_dump_json(indent=2))
            selection, _ = load_verified_final(experiment_dir)
            _warn_stale(
                [
                    *selection_staleness(experiment_dir, selection),
                    *assess_final(experiment_dir, final).warnings,
                ]
            )
        return 0

    return run_reporting_errors(dispatch)
