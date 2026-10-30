"""Why a result is exploratory. Every surface that shows a result reads its label here."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from model_finder.final_evaluation import FinalEvaluation
from model_finder.holdout import holdout_read_labels, read_holdout_seal
from model_finder.io import read_text
from model_finder.ledger import entries_for
from model_finder.paths import (
    DATA_DIRNAME,
    HOLDOUT_SEAL_FILE_NAME,
    REBUILDS_PATH,
    repository_relative_path,
)
from model_finder.sources.manifest import nondeterministic_pull_label


@dataclass(frozen=True)
class FinalAssessment:
    """Why an estimate is exploratory, and what a reader should know."""

    exploratory: list[str]
    warnings: list[str]


def rebuild_labels(experiment_dir: Path) -> list[str]:
    path = experiment_dir / REBUILDS_PATH
    if not path.is_file():
        return []
    return [
        f"A rebuild changed recorded results: {reason}" for reason in json.loads(read_text(path))
    ]


def evidence_labels(experiment_dir: Path) -> list[str]:
    """Label overrides that touched the rows or the leaderboard, whatever the claim."""
    pull = nondeterministic_pull_label(experiment_dir / DATA_DIRNAME)
    return [*([pull] if pull else []), *rebuild_labels(experiment_dir)]


def leaderboard_label(experiment_dir: Path) -> str | None:
    """Return one short line when the leaderboard's scores are exploratory."""
    if not (experiment_dir / HOLDOUT_SEAL_FILE_NAME).is_file():
        return "Exploratory: no holdout is sealed."
    if evidence_labels(experiment_dir):
        return "Exploratory: an override changed this work's data or results."
    seal = read_holdout_seal(experiment_dir)
    read_again = any(
        entry.event == "evaluated"
        and entry.provenance_note is not None
        and entry.experiment_uid == seal.experiment_uid
        for entry in entries_for(seal.holdout_sha256)
    )
    if read_again or holdout_read_labels(experiment_dir, seal):
        return "Exploratory: a holdout read informed or repeated this work."
    return None


def assess_final(experiment_dir: Path, final: FinalEvaluation) -> FinalAssessment:
    """Label a verified final. Nothing here refuses it."""
    exploratory = list(
        dict.fromkeys(
            [*final.exploratory_reasons, *holdout_read_labels(experiment_dir, final.holdout)]
        )
    )
    warnings: list[str] = []
    entries = [
        entry
        for entry in entries_for(final.holdout.holdout_sha256)
        if entry.experiment_uid == final.holdout.experiment_uid
    ]
    if not entries:
        exploratory.append(
            "This machine's ledger has no record of this read, so the estimate is unverified."
        )
    elif not any(
        entry.experiment_path == repository_relative_path(experiment_dir) for entry in entries
    ):
        warnings.append(f"This evidence was produced in {entries[0].experiment_path}, not here.")
    if final.overlap.is_large:
        exploratory.append(
            f"A large share of holdout rows also appears in search rows: {final.overlap.model_dump()}"
        )
    elif final.overlap.needs_warning:
        warnings.append(
            f"Splits share rows, or a comparison could not run: {final.overlap.model_dump()}"
        )
    return FinalAssessment(exploratory=exploratory, warnings=warnings)
