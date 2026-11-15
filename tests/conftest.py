from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from model_finder.candidate_result import (
    CandidateResult,
    FittedCandidate,
    SearchMetrics,
    metric_key,
)
from model_finder.holdout import seal_holdout
from model_finder.results import ComparisonContext, RecordedResult
from model_finder.runner import ExperimentSeed, init_experiment_dir, save_candidate_result
from model_finder.splits import (
    FrameIdentity,
    SearchDataIdentity,
    SearchSplits,
    temporal_split,
)

# A skipped or expected-to-fail test proves nothing, and pytest exits 0 on one.
UNPROVEN: list[str] = []

_IDENTITY = FrameIdentity(sha256="a" * 64, rows=1, columns=1)
SPEC = "# Experiment\n\n## Objective Metric\n\nValidation AUC (`val_auc`).\n"


@pytest.fixture(autouse=True)
def isolated_holdout_store(tmp_path_factory, monkeypatch) -> Path:
    """No test may touch the real holdout store."""
    root = tmp_path_factory.mktemp("xdg")
    monkeypatch.setenv("XDG_DATA_HOME", str(root))
    return root


def result_fields(score: float = 0.7) -> dict[str, object]:
    """The fields of a valid result. A new required field needs one line here."""
    return {
        "candidate_id": "candidate-a",
        "objective_metric": "val_auc",
        "objective_score": score,
        "metrics": {"validation": {"auc": score}},
    }


def candidate_result(
    candidate_id: str = "candidate-a",
    objective_metric: str = "val_auc",
    score: float = 0.7,
    validation: dict[str, float] | None = None,
    **fields: object,
) -> CandidateResult:
    """The one place tests build a result."""
    validation = validation or {metric_key(objective_metric): score}
    return CandidateResult.model_validate(
        {
            **result_fields(score),
            "candidate_id": candidate_id,
            "objective_metric": objective_metric,
            "metrics": SearchMetrics(train=dict(validation), validation=validation),
            **fields,
        }
    )


def recorded_result(
    candidate_id: str = "candidate-a",
    objective_metric: str = "val_auc",
    score: float = 0.7,
    validation: dict[str, float] | None = None,
) -> RecordedResult:
    result = candidate_result(candidate_id, objective_metric, score, validation)
    run_id = str(uuid4())
    return RecordedResult(
        **result.model_dump(),
        comparison_context=ComparisonContext(
            objective_metric=objective_metric,
            objective_direction="maximize",
            spec_sha256="b" * 64,
            data=SearchDataIdentity(train=_IDENTITY, validation=_IDENTITY),
        ),
        run_id=run_id,
        run_evidence=f"work/runs/{run_id}.json",
    )


def rows(count: int = 60, seed: int = 8, start: str = "2025-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "when": pd.date_range(start, periods=count),
            "signal": rng.normal(size=count),
            "label": np.resize([0, 1], count),
        }
    )


def fitted(splits: SearchSplits, candidate_id: str = "candidate-a") -> FittedCandidate:
    """A deterministic candidate whose score depends on its rows."""
    score = float(splits.validation["signal"].abs().mean())
    train_score = float(splits.train["signal"].abs().mean())
    return FittedCandidate(
        result=candidate_result(
            candidate_id,
            score=score,
            metrics=SearchMetrics(train={"auc": train_score}, validation={"auc": score}),
        ),
        score_rows=lambda frame: {"auc": float(frame["signal"].abs().mean())},
    )


@pytest.fixture
def code_root(tmp_path: Path) -> Path:
    root = tmp_path / "candidate_code"
    root.mkdir()
    (root / "candidate_module.py").write_text("VERSION = 1\n")
    return root


@pytest.fixture
def sealed(tmp_path: Path) -> Path:
    """An initialized experiment with a sealed holdout and no results."""
    experiment = tmp_path / "experiment"
    init_experiment_dir(experiment, ExperimentSeed("experiment"))
    (experiment / "experiment.md").write_text(SPEC)
    search, holdout = temporal_split(rows(), "when")
    seal_holdout(experiment, search, holdout)
    return experiment


@pytest.fixture
def scored(sealed: Path, code_root: Path) -> Path:
    """A sealed experiment with one recorded candidate."""
    save_candidate_result(sealed, "candidate-a", {"candidate-a": fitted}, code_root)
    return sealed


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.skipped:
        UNPROVEN.append(f"{report.nodeid} (skipped)")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if hasattr(report, "wasxfail"):
        UNPROVEN.append(f"{report.nodeid} ({'xpassed' if report.passed else 'xfailed'})")
    elif report.skipped:
        UNPROVEN.append(f"{report.nodeid} (skipped)")


def pytest_sessionfinish(session: pytest.Session) -> None:
    if UNPROVEN and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    for test in UNPROVEN:
        terminalreporter.write_line(
            f"{test}: a test that does not run proves nothing. Make it pass or remove it.",
            red=True,
        )
