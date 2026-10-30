"""The leaderboard. This module owns the file name and is the only writer of the file."""

from __future__ import annotations

import fcntl
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from agentic_ml_loop.candidate_result import (
    CandidateResult,
    Direction,
    canonical_objective_metric,
)
from agentic_ml_loop.io import file_sha256, write_json
from agentic_ml_loop.paths import RESULTS_LOCK_FILE_NAME
from agentic_ml_loop.splits import SearchDataIdentity, Sha256

RESULTS_FILE_NAME = "results.json"

CanonicalUuid = Annotated[
    str, Field(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
]


class ComparisonContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    objective_metric: str
    objective_direction: Direction
    spec_sha256: Sha256
    data: SearchDataIdentity


class RecordedResult(CandidateResult):
    """A candidate result after the engine bound it to its run evidence."""

    comparison_context: ComparisonContext
    run_id: CanonicalUuid
    run_evidence: str

    @model_validator(mode="after")
    def _identity_is_consistent(self) -> Self:
        if self.run_evidence != f"work/runs/{self.run_id}.json":
            raise ValueError("run_evidence must be work/runs/<run_id>.json")
        if self.comparison_context.objective_metric != self.objective_metric:
            raise ValueError("comparison_context objective_metric does not match result")
        if self.comparison_context.objective_direction != self.objective_direction:
            raise ValueError("comparison_context objective_direction does not match result")
        return self


_RESULTS = TypeAdapter(list[RecordedResult])


def results_path(experiment_dir: Path) -> Path:
    return experiment_dir / RESULTS_FILE_NAME


def results_sha256(experiment_dir: Path) -> str:
    return file_sha256(results_path(experiment_dir))


def load_results(experiment_dir: Path) -> list[RecordedResult]:
    """Return validated results. An absent leaderboard has no results."""
    path = results_path(experiment_dir)
    if not path.exists():
        return []
    return _RESULTS.validate_json(path.read_bytes())


def comparison_errors(results: list[RecordedResult]) -> list[str]:
    errors: list[str] = []
    candidate_ids = [result.candidate_id for result in results]
    errors.extend(
        f"duplicate candidate_id: {candidate_id}"
        for candidate_id in sorted({c for c in candidate_ids if candidate_ids.count(c) > 1})
    )
    if len({canonical_objective_metric(result.objective_metric) for result in results}) > 1:
        errors.append("results contain incompatible objective_metric values")
    if len({result.objective_direction for result in results}) > 1:
        errors.append("results contain incompatible objective_direction values")
    contexts = {
        result.comparison_context.model_copy(update={"objective_metric": ""}).model_dump_json()
        for result in results
    }
    if len(contexts) > 1:
        errors.append(
            "results compare different specifications or data. "
            "Preserve them and start a new experiment."
        )
    return errors


def ranked_results(results: list[RecordedResult]) -> list[RecordedResult]:
    """Return results, best objective score first."""
    return sorted(
        results,
        key=lambda result: result.objective_score,
        reverse=not results or results[0].objective_direction == "maximize",
    )


def write_results(experiment_dir: Path, results: list[RecordedResult]) -> None:
    errors = comparison_errors(results)
    if errors:
        raise ValueError("; ".join(errors))
    write_json(
        results_path(experiment_dir),
        [result.model_dump(mode="json") for result in ranked_results(results)],
    )


@contextmanager
def results_lock(experiment_dir: Path) -> Generator[None]:
    with (experiment_dir / RESULTS_LOCK_FILE_NAME).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield
