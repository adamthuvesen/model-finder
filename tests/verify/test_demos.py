"""Evaluate-final refits a candidate and requires the same result, so every demo must be deterministic."""

from __future__ import annotations

import math

import pytest

from demos.demo_bootstrap.__main__ import CONFIG as BOOTSTRAP
from model_finder.splits import temporal_split

CASES = [(BOOTSTRAP, name) for name in BOOTSTRAP.candidate_runners]


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
