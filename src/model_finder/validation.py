"""Structural checks on an experiment directory."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from model_finder.candidate_result import (
    CandidateResult,
    canonical_objective_metric,
    holdout_scored_names,
    metric_key,
    train_validation_gap,
)
from model_finder.holdout import TRAIN_FILE_NAME, consumed_holdout_count
from model_finder.paths import (
    DATA_DIRNAME,
    HOLDOUT_SEAL_FILE_NAME,
    JOURNAL_FILE_NAME,
    MODEL_DIRNAME,
    OUTPUTS_DIRNAME,
    RESEARCH_SOURCES_FILE_NAME,
    RESULTS_LOCK_FILE_NAME,
    SCRIPTS_DIRNAME,
    SPECIFICATION_FILE_NAME,
    WORK_DIRNAME,
)
from model_finder.results import (
    RESULTS_FILE_NAME,
    RecordedResult,
    comparison_errors,
    load_results,
)
from model_finder.sources.errors import SnapshotIntegrityError
from model_finder.sources.manifest import (
    MANIFEST_FILENAME,
    SNAPSHOT_FILENAME,
    DatasetManifest,
    verify_snapshot,
)
from model_finder.spec import get_objective_direction, get_objective_metric

REQUIRED_FILES = (SPECIFICATION_FILE_NAME, JOURNAL_FILE_NAME, RESULTS_FILE_NAME)

# Sealing replaces data/snapshot.parquet, so seeded files and directories exclude ``data/``.
SEEDED_FILES: frozenset[str] = frozenset(
    {
        SPECIFICATION_FILE_NAME,
        JOURNAL_FILE_NAME,
        RESEARCH_SOURCES_FILE_NAME,
        RESULTS_FILE_NAME,
        HOLDOUT_SEAL_FILE_NAME,
        RESULTS_LOCK_FILE_NAME,
    }
)
SEEDED_DIRECTORIES: frozenset[str] = frozenset({OUTPUTS_DIRNAME, WORK_DIRNAME, SCRIPTS_DIRNAME})
_ROOT_ALLOWLIST = (
    SEEDED_FILES
    | SEEDED_DIRECTORIES
    | {
        DATA_DIRNAME,
        MODEL_DIRNAME,
        "notebook.yaml",
        "status.md",
    }
)

# Source for both thresholds: the train-validation gap rule that method.md held until 2026-11.
_AUC_GAP_LIMIT = 0.05
_RELATIVE_GAP_LIMIT = 0.10


NO_HOLDOUT_WARNING = (
    "warning: No holdout is sealed. Scores come from validation rows and stay exploratory."
)


def stray_root_entries(experiment_dir: Path) -> list[str]:
    """Return non-hidden root entries outside the allowlist."""
    return sorted(
        entry.name
        for entry in experiment_dir.iterdir()
        if entry.name not in _ROOT_ALLOWLIST and not entry.name.startswith(".")
    )


def _snapshot_errors(experiment_dir: Path) -> list[str]:
    snapshot_path = experiment_dir / DATA_DIRNAME / SNAPSHOT_FILENAME
    manifest_path = experiment_dir / DATA_DIRNAME / MANIFEST_FILENAME
    if snapshot_path.exists() and (experiment_dir / HOLDOUT_SEAL_FILE_NAME).exists():
        return [
            "data/snapshot.parquet sits beside a seal and may hold sealed rows. Run `experiment seal`."
        ]
    if not manifest_path.exists() or not snapshot_path.exists():
        return []
    try:
        verify_snapshot(snapshot_path, DatasetManifest.load(manifest_path))
    except (SnapshotIntegrityError, ValueError, OSError) as exc:
        return [f"Snapshot verification failed: {exc}"]
    return []


def _gap_warnings(results: list[RecordedResult]) -> list[str]:
    warnings: list[str] = []
    for result in results:
        key = metric_key(result.objective_metric)
        gap = train_validation_gap(
            result.metrics, result.objective_metric, result.objective_direction
        )
        if gap is None:
            continue
        limit = (
            _AUC_GAP_LIMIT if "auc" in key else _RELATIVE_GAP_LIMIT * abs(result.metrics.train[key])
        )
        if gap > limit:
            warnings.append(
                f"warning: {result.candidate_id} train-validation gap {gap:.4f} exceeds {limit:.4f}. "
                "Investigate before accepting it."
            )
    return warnings


def hyperparameter_warnings(results: Sequence[CandidateResult]) -> list[str]:
    return [
        f"warning: {result.candidate_id} has a hyperparameter named {name}, which looks like a "
        "holdout score. Fix: remove it, and use evaluate-final for holdout scores"
        for result in results
        for name in holdout_scored_names(dict(result.hyperparameters))
    ]


def _objective_errors(experiment_dir: Path, results: list[RecordedResult]) -> list[str]:
    declared = get_objective_metric(experiment_dir)
    direction = get_objective_direction(experiment_dir)
    errors: list[str] = []
    for result in results:
        if result.objective_direction != direction:
            errors.append(
                f"{result.candidate_id} has objective_direction={result.objective_direction} "
                f"but the experiment declares {direction}"
            )
        if declared and canonical_objective_metric(
            result.objective_metric
        ) != canonical_objective_metric(declared):
            errors.append(
                f"{result.candidate_id} has objective_metric={result.objective_metric} "
                f"but the experiment declares {declared}"
            )
    return errors


def _next_step(experiment_dir: Path) -> str:
    seed = "run your runner's `init-demo --experiment <dir>` to seed these files"
    if (experiment_dir / DATA_DIRNAME / SNAPSHOT_FILENAME).exists():
        return (
            f"Next: run `experiment seal --experiment {experiment_dir} --time-column <column>`, "
            f"then {seed}."
        )
    return f"Next: {seed}."


def validate_experiment(experiment_dir: Path, *, strict_completion: bool = False) -> list[str]:
    """Return errors and ``warning:`` messages. An empty list means the directory is valid."""
    if not experiment_dir.is_dir():
        return [
            f"{experiment_dir}: experiment directory does not exist. "
            "Fix: run `experiment new <slug> --csv <file>`, or correct the path."
        ]
    missing = [
        f"{experiment_dir / name}: required file is missing. Fix: see the next step below."
        for name in REQUIRED_FILES
        if not (experiment_dir / name).exists()
    ]
    if missing:
        return [*missing, _next_step(experiment_dir)]
    messages: list[str] = []
    stray = stray_root_entries(experiment_dir)
    if stray:
        messages.append(
            f"warning: stray files in experiment root ({', '.join(stray)}); "
            "move deliverables to outputs/, scratch to work/, scripts to scripts/"
        )
    try:
        results = load_results(experiment_dir)
    except ValidationError as exc:
        return [
            *messages,
            *(
                f"{RESULTS_FILE_NAME}{json.dumps(list(error['loc']))}: {error['msg']}"
                for error in exc.errors()
            ),
            f"{RESULTS_FILE_NAME}: invalid. Fix: do not edit the leaderboard. "
            f"Run `experiment rebuild-results {experiment_dir}`.",
        ]
    retired = consumed_holdout_count(experiment_dir)
    if retired and not (experiment_dir / HOLDOUT_SEAL_FILE_NAME).exists():
        messages.append(
            "Holdout history exists but the seal is missing. Restore holdout_seal.json."
        )
    if retired:
        messages.append(
            f"warning: {retired} holdouts retired. Their final metrics are exploratory, "
            "and selection after reading them is holdout-informed."
        )
    has_search_rows = (experiment_dir / DATA_DIRNAME / TRAIN_FILE_NAME).is_file()
    if has_search_rows and not (experiment_dir / HOLDOUT_SEAL_FILE_NAME).exists():
        messages.append(NO_HOLDOUT_WARNING)
    if strict_completion and not results:
        messages.append("strict completion requires at least one result entry")
    messages.extend(comparison_errors(results))
    messages.extend(_gap_warnings(results))
    messages.extend(hyperparameter_warnings(results))
    try:
        messages.extend(_objective_errors(experiment_dir, results))
    except ValueError as exc:
        messages.append(str(exc))
    messages.extend(_snapshot_errors(experiment_dir))
    return messages
