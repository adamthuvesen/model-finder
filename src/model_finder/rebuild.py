"""Repair a damaged leaderboard from run records."""

from __future__ import annotations

import json
from pathlib import Path

from model_finder.final_evaluation import FrozenSelection
from model_finder.io import read_text, write_json
from model_finder.paths import REBUILDS_PATH, SELECTION_PATH, require_experiment_dir
from model_finder.results import (
    RecordedResult,
    load_results,
    results_lock,
    results_path,
    results_sha256,
    results_text_sha256,
    write_results,
)
from model_finder.run_evidence import latest_recorded_results


def _run_label(result: RecordedResult) -> str:
    return f"{result.candidate_id} ({result.run_id})"


def _describe_changes(previous: list[RecordedResult], rebuilt: list[RecordedResult]) -> list[str]:
    before = {result.run_id: result for result in previous}
    after = {result.run_id: result for result in rebuilt}
    changes = [f"Added {_run_label(after[run])}." for run in sorted(after.keys() - before.keys())]
    changes += [
        f"Removed {_run_label(before[run])}." for run in sorted(before.keys() - after.keys())
    ]
    for run in sorted(before.keys() & after.keys()):
        old, new = before[run].model_dump(mode="json"), after[run].model_dump(mode="json")
        changes += [
            f"Changed {_run_label(after[run])} {name}: {old[name]} -> {new[name]}."
            for name in old
            if old[name] != new[name]
        ]
    return changes


def rebuild_results(experiment_dir: Path, *, change_reason: str | None = None) -> str:
    """Repair a damaged leaderboard from run records, and say what changed.

    A readable leaderboard that disagrees with the records needs a recorded reason. A run record
    is a plain file, so an edit to it and to the leaderboard together goes unseen.
    """
    require_experiment_dir(experiment_dir)
    with results_lock(experiment_dir):
        rebuilt = latest_recorded_results(experiment_dir)
        rebuilt_sha256 = results_text_sha256(rebuilt)
        if (
            results_path(experiment_dir).exists()
            and results_sha256(experiment_dir) == rebuilt_sha256
        ):
            return f"The leaderboard already matches {len(rebuilt)} recorded candidates."
        selection_path = experiment_dir / SELECTION_PATH
        if selection_path.is_file():
            selection = FrozenSelection.model_validate_json(read_text(selection_path))
            if selection.results_sha256 != rebuilt_sha256:
                raise ValueError(
                    f"{results_path(experiment_dir)}: Selection is frozen, and the run records "
                    "give a different leaderboard. Nothing was rebuilt. "
                    "Fix: run a candidate to reopen search"
                )
        if not results_path(experiment_dir).exists():
            write_results(experiment_dir, rebuilt)
            return f"Rebuilt the missing leaderboard from {len(rebuilt)} recorded candidates."
        try:
            changes = _describe_changes(load_results(experiment_dir), rebuilt)
        except ValueError:
            write_results(experiment_dir, rebuilt)
            return (
                f"Rebuilt the leaderboard from {len(rebuilt)} recorded candidates. "
                "The old file was invalid, so what it held is unknown."
            )
        summary = " ".join(changes) or "Only the order changed."
        if changes and change_reason is None:
            raise ValueError(
                f"{results_path(experiment_dir)}: The run records disagree with the leaderboard. "
                f"{summary} Nothing was rebuilt. Fix: if these are right, pass "
                "`--allow-changes --reason <text>`, which labels the result exploratory"
            )
        if changes:
            path = experiment_dir / REBUILDS_PATH
            earlier: list[str] = json.loads(read_text(path)) if path.is_file() else []
            write_json(path, [*earlier, change_reason])
        write_results(experiment_dir, rebuilt)
        return f"Rebuilt the leaderboard from {len(rebuilt)} recorded candidates. {summary}"
