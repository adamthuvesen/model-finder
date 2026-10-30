"""Bind each result to immutable evidence: specification, data, and code identity."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import site
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from agentic_ml_loop.candidate_result import Direction
from agentic_ml_loop.io import file_sha256, read_text
from agentic_ml_loop.paths import SPECIFICATION_FILE_NAME, WORK_DIRNAME
from agentic_ml_loop.results import ComparisonContext, RecordedResult
from agentic_ml_loop.splits import SearchDataIdentity, Sha256

ENGINE_ROOT = Path(__file__).resolve().parent


class CodeFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: Sha256
    code_root: str
    config_files: list[str]
    dependencies_sha256: Sha256
    python: str
    pandas: str


class RunEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    result: RecordedResult
    code_fingerprint: CodeFingerprint
    started_at: str
    finished_at: str
    runtime_seconds: float


def _tree_files(root: Path) -> list[tuple[str, Path]]:
    """Walk a tree, following symlinked directories once, and name files relative to it."""
    files: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for directory, subdirectories, names in os.walk(root, followlinks=True):
        real = os.path.realpath(directory)
        if real in seen:
            subdirectories.clear()
            continue
        seen.add(real)
        subdirectories[:] = sorted(name for name in subdirectories if name != "__pycache__")
        for name in sorted(names):
            path = Path(directory) / name
            if path.is_file() and path.suffix != ".pyc":
                files.append((path.relative_to(root).as_posix(), path))
    return files


def _sources_sha256(roots: list[Path], config_files: list[Path]) -> str | None:
    """Hash whole trees and declared files. A missing declared file returns None."""
    digest = hashlib.sha256()
    named = [named_file for root in roots for named_file in _tree_files(root)]
    for path in config_files:
        if not path.is_file():
            return None
        named.append((path.name, path))
    for name, path in named:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _dependencies_sha256() -> str:
    """Hash installed names and versions in site-packages only.

    ``python -m`` puts the working directory on ``sys.path``, so a stray egg-info there must not
    count. This identifies versions. An edit to an installed package's source goes unseen.
    """
    installed = sorted(
        {
            f"{distribution.name}=={distribution.version}"
            for distribution in importlib.metadata.distributions(path=site.getsitepackages())
        }
    )
    return hashlib.sha256("\n".join(installed).encode()).hexdigest()


def code_fingerprint(
    experiment_dir: Path, code_root: Path, config_files: Sequence[Path] = ()
) -> CodeFingerprint:
    """Fingerprint the engine tree, the experiment package tree, and declared config files.

    Whole trees catch lazily imported helpers and JSON config. An unrelated edit in those trees
    makes evidence stale, which is the accepted cost. Paths stay relative to the experiment, so
    evidence names no home directory.
    """
    root = code_root.resolve()
    configs = [path.resolve() for path in config_files]
    digest = _sources_sha256([ENGINE_ROOT, root], configs)
    if digest is None:
        raise ValueError("A declared config file is missing")
    base = experiment_dir.resolve()
    return CodeFingerprint(
        sha256=digest,
        code_root=os.path.relpath(root, base),
        config_files=[os.path.relpath(path, base) for path in configs],
        dependencies_sha256=_dependencies_sha256(),
        python=sys.version,
        pandas=pd.__version__,
    )


def comparison_context(
    experiment_dir: Path,
    objective_metric: str,
    objective_direction: Direction,
    data: SearchDataIdentity,
) -> ComparisonContext:
    return ComparisonContext(
        objective_metric=objective_metric,
        objective_direction=objective_direction,
        spec_sha256=file_sha256(experiment_dir / SPECIFICATION_FILE_NAME),
        data=data,
    )


def write_run_evidence(experiment_dir: Path, evidence: RunEvidence) -> None:
    """Create one run file exclusively. Existing run evidence is never overwritten."""
    path = experiment_dir / evidence.result.run_evidence
    content = json.dumps(evidence.model_dump(mode="json"), indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def latest_recorded_results(experiment_dir: Path) -> list[RecordedResult]:
    """Return each candidate's latest run, oldest first. Any invalid record refuses the whole read."""
    latest: dict[str, RunEvidence] = {}
    for path in sorted((experiment_dir / WORK_DIRNAME / "runs").glob("*.json")):
        try:
            evidence = RunEvidence.model_validate_json(read_text(path))
        except ValidationError as exc:
            raise ValueError(f"Run record {path} is invalid. Nothing was rebuilt.") from exc
        result = evidence.result
        if experiment_dir / result.run_evidence != path:
            raise ValueError(f"Run record {path} names another run. Nothing was rebuilt.")
        current = latest.get(result.candidate_id)
        if current is None or evidence.finished_at > current.finished_at:
            latest[result.candidate_id] = evidence
    return [run.result for run in sorted(latest.values(), key=lambda run: run.finished_at)]


def recorded_fingerprint_is_current(experiment_dir: Path, recorded: CodeFingerprint) -> bool:
    code_root = (experiment_dir / recorded.code_root).resolve()
    configs = [(experiment_dir / name).resolve() for name in recorded.config_files]
    return (
        code_root.is_dir()
        and _sources_sha256([ENGINE_ROOT, code_root], configs) == recorded.sha256
        and recorded.dependencies_sha256 == _dependencies_sha256()
        and recorded.python == sys.version
        and recorded.pandas == pd.__version__
    )


def validate_run_evidence(experiment_dir: Path, entry: RecordedResult) -> list[str]:
    """Verify the archived result. A mismatch here means the record is not what ran."""
    path = experiment_dir / entry.run_evidence
    try:
        evidence = RunEvidence.model_validate_json(read_text(path))
    except (OSError, ValidationError) as exc:
        return [f"cannot read run evidence for {entry.candidate_id}: {exc}"]
    if evidence.result != entry:
        return [f"run evidence does not match result: {entry.candidate_id}"]
    return []


def run_evidence_staleness(experiment_dir: Path, entry: RecordedResult) -> list[str]:
    """Report what changed since a run. These warn, unless a person asks for a strict check."""
    stale: list[str] = []
    spec = experiment_dir / SPECIFICATION_FILE_NAME
    if not spec.is_file() or entry.comparison_context.spec_sha256 != file_sha256(spec):
        stale.append(f"The specification changed since {entry.candidate_id} ran.")
    evidence = RunEvidence.model_validate_json(read_text(experiment_dir / entry.run_evidence))
    if not recorded_fingerprint_is_current(experiment_dir, evidence.code_fingerprint):
        stale.append(f"Code or dependencies changed since {entry.candidate_id} ran.")
    return stale
