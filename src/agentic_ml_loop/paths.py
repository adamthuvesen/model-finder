"""Canonical experiment layout.

Each experiment directory holds:

- ``outputs/``: deliverables a stakeholder reads.
- ``work/``: intermediate artifacts and the engine's evidence records.
- ``scripts/``: one-shot Python scripts written during research.
- ``data/``: search rows and warehouse snapshots.
"""

from __future__ import annotations

from pathlib import Path

OUTPUTS_DIRNAME = "outputs"
WORK_DIRNAME = "work"
SCRIPTS_DIRNAME = "scripts"
DATA_DIRNAME = "data"

SPECIFICATION_FILE_NAME = "experiment.md"
JOURNAL_FILE_NAME = "research_journal.md"
RESEARCH_SOURCES_FILE_NAME = "research_sources.md"
HOLDOUT_SEAL_FILE_NAME = "holdout_seal.json"
RESULTS_LOCK_FILE_NAME = "results.lock"

SELECTION_PATH = "work/selection.json"
ATTEMPT_PATH = "work/final_evaluation_started.json"
FINAL_PATH = "outputs/final_evaluation.json"
COMPLETION_PATH = "outputs/completion.json"
HOLDOUT_HISTORY_DIR = "work/holdout_history"
SNAPSHOT_FILE_NAME = "snapshot.parquet"


def _mkdir_p(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def outputs_dir(experiment_dir: Path) -> Path:
    """Return ``experiment_dir/outputs``, creating it if needed."""
    return _mkdir_p(experiment_dir / OUTPUTS_DIRNAME)


def work_dir(experiment_dir: Path) -> Path:
    """Return ``experiment_dir/work``, creating it if needed."""
    return _mkdir_p(experiment_dir / WORK_DIRNAME)


def scripts_dir(experiment_dir: Path) -> Path:
    """Return ``experiment_dir/scripts``, creating it if needed."""
    return _mkdir_p(experiment_dir / SCRIPTS_DIRNAME)


def data_dir(experiment_dir: Path) -> Path:
    """Return ``experiment_dir/data``, creating it if needed."""
    return _mkdir_p(experiment_dir / DATA_DIRNAME)
