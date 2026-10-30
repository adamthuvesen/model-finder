"""The command line an experiment's runner module exposes."""

from __future__ import annotations

import argparse
import json
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from agentic_ml_loop.candidate_result import CandidateRunner
from agentic_ml_loop.cli.errors import run_reporting_errors, warn_rows_stay_readable
from agentic_ml_loop.final_evaluation import evaluate_final, freeze_candidate
from agentic_ml_loop.holdout import seal_holdout
from agentic_ml_loop.paths import HOLDOUT_SEAL_FILE_NAME
from agentic_ml_loop.runner import ExperimentSeed, init_experiment_dir, save_candidate_result
from agentic_ml_loop.splits import temporal_split


@dataclass(frozen=True)
class RunnerConfig:
    seed: ExperimentSeed
    candidate_runners: Mapping[str, CandidateRunner]
    rows_loader: Callable[[int], pd.DataFrame]
    # Required, so that an author decides. None means no local file holds the rows.
    rows_source: Path | None
    time_column: str
    target_column: str
    code_root: Path
    config_files: tuple[Path, ...] = ()


def _parser(config: RunnerConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{config.seed.experiment_id} runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-candidates", help="List bundled candidates")

    init = subparsers.add_parser("init-demo", help="Create an experiment and seal its holdout")
    init.add_argument("--experiment", required=True, help="New experiment directory")
    init.add_argument("--force", action="store_true", help="Reseed files; keeps holdout evidence")
    init.add_argument(
        "--data-seed", type=int, default=None, help="Rows to generate. Default: a new random seed"
    )

    candidates = sorted(config.candidate_runners)
    run = subparsers.add_parser("run-candidate", help="Run one candidate on search rows")
    run.add_argument("--experiment", required=True)
    run.add_argument("--candidate", required=True, choices=candidates)

    freeze = subparsers.add_parser("freeze-candidate", help="Freeze a verified candidate")
    freeze.add_argument("--experiment", required=True)
    freeze.add_argument("--candidate", required=True, choices=candidates)

    final = subparsers.add_parser("evaluate-final", help="Score the frozen candidate once")
    final.add_argument("--experiment", required=True)
    return parser


def _init(config: RunnerConfig, experiment_dir: Path, *, force: bool, data_seed: int | None) -> str:
    init_experiment_dir(experiment_dir, config.seed, force=force)
    if (experiment_dir / HOLDOUT_SEAL_FILE_NAME).exists():
        return f"Initialized {experiment_dir}. Its holdout was already sealed."
    seed = secrets.randbits(32) if data_seed is None else data_seed
    search, holdout = temporal_split(config.rows_loader(seed), config.time_column)
    seal = seal_holdout(experiment_dir, search, holdout)
    if config.rows_source is not None:
        warn_rows_stay_readable(config.rows_source)
    return f"Initialized {experiment_dir}. Sealed {seal.holdout_rows} holdout rows."


def run_runner_main(config: RunnerConfig, argv: list[str] | None = None) -> int:
    args = _parser(config).parse_args(argv)

    def dispatch() -> int:
        if args.command == "list-candidates":
            print("\n".join(config.candidate_runners))
            return 0
        experiment_dir = Path(args.experiment).resolve()
        if args.command == "init-demo":
            print(_init(config, experiment_dir, force=args.force, data_seed=args.data_seed))
        elif args.command == "run-candidate":
            result = save_candidate_result(
                experiment_dir,
                args.candidate,
                config.candidate_runners,
                config.code_root,
                config.config_files,
            )
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
            selection = freeze_candidate(experiment_dir, args.candidate, config.target_column)
            print(selection.model_dump_json(indent=2))
        else:
            final = evaluate_final(experiment_dir, config.candidate_runners)
            print(final.model_dump_json(indent=2))
        return 0

    return run_reporting_errors(dispatch)
