from __future__ import annotations

import json
from pathlib import Path

from model_finder.viz.bundle import bundle
from model_finder.viz.generate import (
    JsonObject,
    candidate_summaries,
    parse_experiment_md,
    replay_scenes,
)
from tests.conftest import recorded_result


class TestParseExperimentMd:
    def test_extracts_display_title_goal_and_metric(self, tmp_path: Path) -> None:
        path = tmp_path / "experiment.md"
        path.write_text(
            "# Spec\n\n"
            "## Title\n\n"
            "demo: fraud auc model\n\n"
            "## Goal\n\n"
            "Build a reliable fraud model, compare candidates carefully, and keep "
            "the replay text readable even when the spec goal is wordy.\n\n"
            "## Objective Metric\n\n"
            "`val_auc` — validation AUC.\n"
        )

        parsed = parse_experiment_md(path)

        assert parsed["name"] == "Autonomous Research Agent: Fraud AUC Model"
        assert parsed["goal"] == "Build a reliable fraud model, compare candidates carefully."
        assert parsed["metric"] == "val_auc"


def _summary(
    candidate_id: str, objective_metric: str, score: float, **validation: float
) -> JsonObject:
    return candidate_summaries(
        [recorded_result(candidate_id, objective_metric, score, validation)]
    )[0]


class TestCandidateSummaries:
    def test_captured_at_k_metrics(self) -> None:
        candidate = _summary(
            "value-model", "val_captured_at_20pct", 0.6789, captured_at_20pct=0.6789, auc=0.71
        )
        assert candidate["primary_metric"]["family"] == "captured_at_k"
        assert candidate["at_20"] == 0.679
        assert candidate["auc"] == 0.71

    def test_classification_metrics_have_no_capture_fields(self) -> None:
        candidate = _summary("logreg", "val_auc", 0.8123, auc=0.8123, precision=0.5)
        assert candidate["primary_metric"]["family"] == "classification"
        assert candidate["metrics"]["auc"] == 0.812
        assert "at_20" not in candidate

    def test_regression_metrics_have_no_capture_fields(self) -> None:
        candidate = _summary("ridge", "val_r2", 0.4369, r2=0.4369, rmse=12.345)
        assert candidate["primary_metric"]["family"] == "regression"
        assert candidate["metrics"]["rmse"] == 12.345
        assert "at_20" not in candidate

    def test_unknown_metrics_stay_generic(self) -> None:
        candidate = _summary("custom", "val_business_score", 2.5, business_score=2.5)
        assert candidate["primary_metric"]["family"] == "generic"

    def test_zero_scores_survive(self) -> None:
        candidate = _summary("zero", "val_auc", 0.0, auc=0.0)
        assert candidate["score"] == 0.0
        assert candidate["primary_metric"]["value"] == 0.0


class TestReplayScenes:
    def test_builds_basic_cycle_sequence_and_leaderboard(self, tmp_path: Path) -> None:
        experiment = {"name": "Replay", "goal": "Find the best model.", "metric": "val_auc"}
        cycles = [
            {
                "number": 1,
                "title": "baseline",
                "objective": "test learned models",
                "has_research": True,
                "findings": ["Baseline works."],
                "candidates": [{"name": "baseline", "at_20": 0.6}],
            }
        ]
        candidates = [
            {
                "id": "baseline",
                "score": 0.6,
                "primary_metric": {"name": "val_auc", "value": 0.6},
            }
        ]

        scenes = replay_scenes(experiment, cycles, candidates)

        assert [scene["type"] for scene in scenes] == [
            "intro",
            "hypothesis",
            "research",
            "training",
            "evaluation",
            "journal",
            "finale",
        ]
        assert scenes[1]["text"] == "test trained models"
        assert scenes[4]["leaderboard"][0]["name"] == "baseline"
        assert scenes[4]["leaderboard"][0]["score"] == 0.6


def test_bundle_embeds_escaped_json_payload(tmp_path: Path) -> None:
    script = json.dumps(
        {
            "experiment": {
                "name": "Replay",
                "metric": "val_auc",
                "objective": "</script><script>alert(1)</script>",
            },
            "scenes": [],
            "total_cycles": 0,
            "total_candidates": 0,
        }
    )
    output_path = tmp_path / "replay.html"
    bundle(script, output_path)

    html = output_path.read_text()
    prefix = '<script id="script-data" type="application/json">'
    start = html.index(prefix) + len(prefix)
    end = html.index("</script>", start)
    payload = html[start:end]

    assert "\\u003c/script\\u003e\\u003cscript\\u003ealert(1)\\u003c/script\\u003e" in payload
