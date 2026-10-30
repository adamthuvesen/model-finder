"""Run one candidate on search rows and record the result. Seed new experiments."""

from __future__ import annotations

import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from agentic_ml_loop.candidate_result import CandidateRunner, canonical_objective_metric
from agentic_ml_loop.holdout import load_search_splits, withdraw_selection
from agentic_ml_loop.io import file_sha256, utc_now, write_text
from agentic_ml_loop.paths import (
    ATTEMPT_PATH,
    HOLDOUT_HISTORY_DIR,
    HOLDOUT_SEAL_FILE_NAME,
    JOURNAL_FILE_NAME,
    RESEARCH_SOURCES_FILE_NAME,
    SPECIFICATION_FILE_NAME,
    WORK_DIRNAME,
)
from agentic_ml_loop.results import (
    RecordedResult,
    comparison_errors,
    load_results,
    results_lock,
    results_path,
    write_results,
)
from agentic_ml_loop.run_evidence import (
    RunEvidence,
    code_fingerprint,
    comparison_context,
    write_run_evidence,
)
from agentic_ml_loop.spec import (
    DEFAULT_TEMPLATE,
    get_objective_direction,
    get_objective_metric,
    research_journal_template,
    research_sources_template,
)
from agentic_ml_loop.splits import search_data_identity
from agentic_ml_loop.validation import SEEDED_DIRECTORIES, SEEDED_FILES


def save_candidate_result(
    experiment_dir: Path,
    candidate_id: str,
    candidate_runners: Mapping[str, CandidateRunner],
    code_root: Path,
    config_files: Sequence[Path] = (),
    *,
    reread_reason: str | None = None,
) -> RecordedResult:
    """Run one candidate and merge its result. A frozen selection is withdrawn, since it no longer covers the leaderboard."""
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
    if candidate_id not in candidate_runners:
        known = ", ".join(sorted(candidate_runners)) or "none"
        raise ValueError(f"Unknown candidate '{candidate_id}'. Available: {known}")
    withdraw_selection(experiment_dir, reread_reason)
    comparison_problems = comparison_errors(load_results(experiment_dir))
    if comparison_problems:
        raise ValueError("; ".join(comparison_problems))
    declared_metric = get_objective_metric(experiment_dir)
    direction = get_objective_direction(experiment_dir)

    splits = load_search_splits(experiment_dir)
    data_identity = search_data_identity(splits)
    spec_path = experiment_dir / SPECIFICATION_FILE_NAME
    spec_sha256 = file_sha256(spec_path)
    fingerprint = code_fingerprint(experiment_dir, code_root, config_files)
    started_at = utc_now()
    started = time.perf_counter()
    candidate = candidate_runners[candidate_id](splits).result
    runtime_seconds = time.perf_counter() - started

    if candidate.candidate_id != candidate_id:
        raise ValueError("Candidate result candidate_id does not match requested candidate")
    if declared_metric and canonical_objective_metric(
        candidate.objective_metric
    ) != canonical_objective_metric(declared_metric):
        raise ValueError("Candidate result objective_metric does not match experiment")
    if candidate.objective_direction != direction:
        raise ValueError("Candidate result objective_direction does not match experiment")
    if search_data_identity(splits) != data_identity:
        raise ValueError("Candidate modified the search data during evaluation")
    if file_sha256(spec_path) != spec_sha256:
        raise ValueError("Experiment specification changed during evaluation")

    run_id = str(uuid4())
    recorded = RecordedResult(
        **candidate.model_dump(),
        comparison_context=comparison_context(
            experiment_dir, candidate.objective_metric, direction, data_identity
        ),
        run_id=run_id,
        run_evidence=f"work/runs/{run_id}.json",
    )
    evidence = RunEvidence(
        result=recorded,
        code_fingerprint=fingerprint,
        started_at=started_at,
        finished_at=utc_now(),
        runtime_seconds=runtime_seconds,
    )
    with results_lock(experiment_dir):
        withdraw_selection(experiment_dir, reread_reason)
        others = [r for r in load_results(experiment_dir) if r.candidate_id != candidate_id]
        write_run_evidence(experiment_dir, evidence)
        write_results(experiment_dir, [*others, recorded])
    return recorded


@dataclass(frozen=True)
class ExperimentSeed:
    experiment_id: str
    template_path: Path | None = None
    research_sources_template_path: Path | None = None
    research_journal_template_path: Path | None = None


def _remove_seeded_artifacts(experiment_dir: Path) -> None:
    """Remove files a fresh start replaces. The seal, data, and holdout history always stay."""
    if (experiment_dir / ATTEMPT_PATH).exists():
        raise ValueError(
            "Final evaluation consumed this holdout. Preserve the evidence: "
            "run `experiment reseal` with unseen rows."
        )
    for name in SEEDED_FILES - {HOLDOUT_SEAL_FILE_NAME}:
        (experiment_dir / name).unlink(missing_ok=True)
    for name in SEEDED_DIRECTORIES - {WORK_DIRNAME}:
        shutil.rmtree(experiment_dir / name, ignore_errors=True)
    work = experiment_dir / WORK_DIRNAME
    if work.is_dir():
        for entry in work.iterdir():
            if entry.resolve() == (experiment_dir / HOLDOUT_HISTORY_DIR).resolve():
                continue
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()


def _seed_text(path: Path, content: str, *, force: bool) -> None:
    if force or not path.exists():
        write_text(path, content)


def init_experiment_dir(experiment_dir: Path, seed: ExperimentSeed, *, force: bool = False) -> Path:
    """Create an experiment. ``force`` reseeds files and never touches holdout evidence."""
    if force and experiment_dir.exists():
        _remove_seeded_artifacts(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    template = seed.template_path or DEFAULT_TEMPLATE
    _seed_text(
        experiment_dir / SPECIFICATION_FILE_NAME, template.read_text(encoding="utf-8"), force=force
    )
    journal = (
        seed.research_journal_template_path.read_text(encoding="utf-8")
        if seed.research_journal_template_path
        else research_journal_template(seed.experiment_id)
    )
    _seed_text(experiment_dir / JOURNAL_FILE_NAME, journal, force=force)
    sources = (
        seed.research_sources_template_path.read_text(encoding="utf-8")
        if seed.research_sources_template_path
        else research_sources_template(seed.experiment_id)
    )
    _seed_text(experiment_dir / RESEARCH_SOURCES_FILE_NAME, sources, force=force)
    if force or not results_path(experiment_dir).exists():
        write_results(experiment_dir, [])
    for name in SEEDED_DIRECTORIES:
        target = experiment_dir / name
        target.mkdir(parents=True, exist_ok=True)
        (target / ".gitkeep").touch()
    return experiment_dir
