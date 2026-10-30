"""One offline run of the real commands, with no LLM."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agentic_ml_loop.cli.main import main as experiment_main
from agentic_ml_loop.cli.runner import run_runner_main
from agentic_ml_loop.completion import (
    CompletionRecord,
    ResearchReview,
    evidence_file,
    save_completion,
)
from agentic_ml_loop.io import write_json
from demos.demo_bootstrap.__main__ import CONFIG


def _demo(*arguments: str) -> int:
    return run_runner_main(CONFIG, list(arguments))


def _record_reviewed_conclusion(experiment: Path) -> None:
    for name, text in (
        ("report.md", "logreg-tiny beats the majority baseline on validation AUC."),
        ("reproduce.md", "Run the commands in README.md."),
    ):
        (experiment / "outputs" / name).write_text(text)
    refs = {
        "specification": evidence_file(experiment, "experiment.md"),
        "results": evidence_file(experiment, "results.json"),
        "report": evidence_file(experiment, "outputs/report.md"),
        "evaluation": evidence_file(experiment, "outputs/final_evaluation.json"),
        "reproduction": evidence_file(experiment, "outputs/reproduce.md"),
    }
    review = ResearchReview(
        reviewer="test reviewer",
        verdict="accepted",
        specification_sha256=refs["specification"].sha256,
        results_sha256=refs["results"].sha256,
        report_sha256=refs["report"].sha256,
        evaluation_sha256=refs["evaluation"].sha256,
        findings=[],
        limitations=["Synthetic data."],
    )
    write_json(experiment / "outputs/review.json", review.model_dump())
    save_completion(
        experiment,
        CompletionRecord(
            outcome="supported",
            conclusion="logreg-tiny beats the baseline on search rows.",
            limitations=["Synthetic data."],
            selected_candidate_id="logreg-tiny",
            baseline_candidate_id="majority-baseline",
            review=evidence_file(experiment, "outputs/review.json"),
            specification=refs["specification"],
            results=refs["results"],
            report=refs["report"],
            evaluation=refs["evaluation"],
            reproduction=refs["reproduction"],
        ),
    )


def test_demo_runs_from_search_to_reviewed_completion(tmp_path: Path, capsys) -> None:
    experiment = tmp_path / "bootstrap_run"
    path = str(experiment)

    assert _demo("init-demo", "--experiment", path) == 0
    for candidate in ("majority-baseline", "logreg-tiny"):
        assert _demo("run-candidate", "--experiment", path, "--candidate", candidate) == 0

    results = json.loads((experiment / "results.json").read_text())
    assert all(set(entry["metrics"]) == {"train", "validation"} for entry in results)
    assert results[0]["candidate_id"] == "logreg-tiny"

    assert _demo("evaluate-final", "--experiment", path) == 1
    assert "No frozen selection" in capsys.readouterr().err
    assert not (experiment / "outputs/final_evaluation.json").exists()

    assert _demo("freeze-candidate", "--experiment", path, "--candidate", "logreg-tiny") == 0
    assert _demo("run-candidate", "--experiment", path, "--candidate", "logreg-tiny") == 1
    assert "Selection is frozen" in capsys.readouterr().err

    assert _demo("evaluate-final", "--experiment", path) == 0
    final = json.loads((experiment / "outputs/final_evaluation.json").read_text())
    assert 0.0 <= final["metrics"]["auc"] <= 1.0
    assert final["holdout"]["holdout_rows"] == 160
    assert json.loads((experiment / "results.json").read_text()) == results

    workspace_rows = sum(
        len(pd.read_parquet(experiment / "data" / name))
        for name in ("train.parquet", "validation.parquet")
    )
    assert workspace_rows == 640

    assert experiment_main(["complete-check", path]) == 1
    _record_reviewed_conclusion(experiment)
    assert experiment_main(["validate", path]) == 0
    assert experiment_main(["complete-check", path]) == 0

    replay = tmp_path / "replay.html"
    assert experiment_main(["replay", path, "--output", str(replay)]) == 0
    assert "logreg-tiny" in replay.read_text()
