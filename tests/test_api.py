"""Consumer repos import only from `model_finder.api`, so a name leaves it only on purpose."""

from __future__ import annotations

from model_finder import api

PUBLIC_NAMES = {
    "HOOK_CLIENTS",
    "CandidateRunner",
    "Candidates",
    "ExperimentSeed",
    "FittedCandidate",
    "Predict",
    "RunnerConfig",
    "Scoring",
    "SearchSplits",
    "assess_completion",
    "check_completion",
    "record_from_evidence",
    "run_runner_main",
    "save_completion",
    "scored_candidate",
    "setup_repo",
    "write_review",
}


def test_api_exports_exactly_the_public_names() -> None:
    assert set(api.__all__) == PUBLIC_NAMES, (
        "src/model_finder/api.py: __all__ changed. Fix: update PUBLIC_NAMES in tests/test_api.py, "
        "and repin every consumer that imports a removed name"
    )
    missing = [name for name in api.__all__ if not hasattr(api, name)]
    assert not missing, f"src/model_finder/api.py: {missing} are not defined. Fix: import them"
