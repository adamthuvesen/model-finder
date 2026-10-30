"""Evaluate-final refits a candidate and requires the same result, so every demo must be deterministic."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys

import pytest

from demos.demo_bootstrap.__main__ import CONFIG as BOOTSTRAP
from demos.demo_classification.__main__ import CONFIG as CLASSIFICATION
from demos.demo_regression.__main__ import CONFIG as REGRESSION
from model_finder.splits import temporal_split

CONFIGS = [BOOTSTRAP, CLASSIFICATION, REGRESSION]

CASES = [(config, name) for config in CONFIGS for name in config.candidate_runners]


@pytest.mark.parametrize(
    ("config", "candidate_id"),
    CASES,
    ids=[f"{config.seed.experiment_id}-{name}" for config, name in CASES],
)
def test_candidate_is_deterministic_and_scores_unseen_rows(config, candidate_id: str) -> None:
    search, holdout = temporal_split(config.rows_loader(7), config.time_column)
    runner = config.candidate_runners[candidate_id]
    first = runner(search)
    second = runner(search)
    assert first.result == second.result
    assert first.result.candidate_id == candidate_id
    assert all(math.isfinite(value) for value in first.score_rows(holdout.holdout).values())


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs the deep extra")
def test_deep_demo_is_deterministic_in_its_own_process() -> None:
    """Torch and the GBDT libraries can deadlock on OpenMP in one process."""
    script = """
from model_finder.splits import temporal_split
from demos.demo_deep.__main__ import CONFIG
search, holdout = temporal_split(CONFIG.rows_loader(7), CONFIG.time_column)
run = CONFIG.candidate_runners["mlp-shallow"]
first = run(search)
assert first.result == run(search).result
print(first.score_rows(holdout.holdout)["auc"])
"""
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=240, check=False
    )
    assert done.returncode == 0, done.stderr
    assert 0.0 <= float(done.stdout.strip()) <= 1.0
