"""One offline run of the real commands, with no LLM."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from demos.demo_bootstrap.__main__ import CONFIG
from model_finder.cli.main import main as experiment_main
from model_finder.cli.runner_main import run_runner_main


def _demo(*arguments: str) -> int:
    return run_runner_main(CONFIG, list(arguments))


def _record_reviewed_conclusion(path: str) -> int:
    """Follow the research skill: write the report, then run its two record commands."""
    report_args = ["--report", "outputs/report.md", "--limitation", "Synthetic data."]
    outputs = Path(path) / "outputs"
    (outputs / "report.md").write_text("logreg-tiny beats the majority baseline.")
    (outputs / "reproduce.md").write_text("Run the commands in README.md.")
    reviewed = experiment_main(
        [
            "record-review",
            "--experiment",
            path,
            "--reviewer",
            "test reviewer",
            "--verdict",
            "accepted",
            *report_args,
        ]
    )
    return reviewed or experiment_main(
        [
            "record-completion",
            "--experiment",
            path,
            "--outcome",
            "supported",
            "--conclusion",
            "logreg-tiny beats the baseline on search rows.",
            "--selected",
            "logreg-tiny",
            "--baseline",
            "majority-baseline",
            "--reproduction",
            "outputs/reproduce.md",
            *report_args,
        ]
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
    assert _demo("evaluate-final", "--experiment", path) == 0
    final = json.loads((experiment / "outputs/final_evaluation.json").read_text())
    assert 0.0 <= final["metrics"]["auc"] <= 1.0
    assert final["holdout"]["holdout_rows"] == 160
    assert json.loads((experiment / "results.json").read_text()) == results

    assert _demo("run-candidate", "--experiment", path, "--candidate", "logreg-tiny") == 1
    assert "holdout-informed" in capsys.readouterr().err

    workspace_rows = sum(
        len(pd.read_parquet(experiment / "data" / name))
        for name in ("train.parquet", "validation.parquet")
    )
    assert workspace_rows == 640

    assert experiment_main(["complete-check", path]) == 1
    assert _record_reviewed_conclusion(path) == 0
    assert experiment_main(["validate", path]) == 0
    capsys.readouterr()
    assert experiment_main(["complete-check", path]) == 0
    assert "Result: complete\n" in capsys.readouterr().out

    capsys.readouterr()
    final_path = experiment / "outputs/final_evaluation.json"
    edited = json.loads(final_path.read_text())
    edited["metrics"]["auc"] = 0.999
    final_path.write_text(json.dumps(edited))
    assert experiment_main(["complete-check", path]) == 1
    assert "Stale evaluation" in capsys.readouterr().out
