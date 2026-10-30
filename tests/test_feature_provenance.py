from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from model_finder.feature_provenance import validate_feature_provenance


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    path = tmp_path / "all_features_ranked.csv"
    path.write_text("feature\nsignal\n")
    return path


def _write_provenance(path: Path, label_dates: list[object]) -> None:
    (path.parent / "feature_selection_provenance.json").write_text(
        json.dumps(
            {
                path.name: {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "label_dates": label_dates,
                }
            }
        )
    )


def _validate(path: Path) -> None:
    validate_feature_provenance(
        path,
        selection_train_dates={"2025-01-01", "2025-04-01"},
        validation_dates={"2025-07-01"},
        test_dates={"2025-10-01"},
    )


def test_training_only_provenance_is_accepted(artifact: Path) -> None:
    _write_provenance(artifact, ["2025-01-01", "2025-04-01"])
    _validate(artifact)


@pytest.mark.parametrize("date", ["2025-07-01", "2025-10-01", "2024-10-01"])
def test_non_training_labels_fail_closed(artifact: Path, date: str) -> None:
    _write_provenance(artifact, ["2025-01-01", date])
    with pytest.raises(ValueError, match="labels"):
        _validate(artifact)


def test_stale_digest_fails_closed(artifact: Path) -> None:
    _write_provenance(artifact, ["2025-01-01"])
    artifact.write_text("feature\nchanged_feature\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        _validate(artifact)


@pytest.mark.parametrize("payload", ["{}", "[]", "invalid json"])
def test_missing_or_malformed_provenance_fails_closed(artifact: Path, payload: str) -> None:
    (artifact.parent / "feature_selection_provenance.json").write_text(payload)
    with pytest.raises(ValueError, match="provenance"):
        _validate(artifact)


@pytest.mark.parametrize("dates", [[], ["2025-1-1"], ["2025-02-30"], [20250101]])
def test_label_date_evidence_must_be_explicit_and_valid(
    artifact: Path, dates: list[object]
) -> None:
    _write_provenance(artifact, dates)
    with pytest.raises(ValueError, match="Invalid feature-selection provenance"):
        _validate(artifact)
