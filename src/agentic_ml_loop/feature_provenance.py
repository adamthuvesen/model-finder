"""Validate supervised feature artifacts against the selection split."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from datetime import date
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


class FeatureSelectionProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    label_dates: Annotated[list[str], Field(min_length=1)]

    @field_validator("label_dates")
    @classmethod
    def validate_label_dates(cls, values: list[str]) -> list[str]:
        for value in values:
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError("Label dates must use YYYY-MM-DD.")
        return values


def validate_feature_provenance(
    artifact_path: Path,
    *,
    selection_train_dates: Collection[str],
    validation_dates: Collection[str],
    test_dates: Collection[str],
) -> None:
    """Reject missing, stale, or overlapping supervised feature-selection evidence."""
    provenance_path = artifact_path.parent / "feature_selection_provenance.json"
    if not provenance_path.is_file():
        raise ValueError(
            f"Missing feature-selection provenance for {artifact_path.name}: "
            f"{provenance_path}. Regenerate the artifact using selection training labels only."
        )
    try:
        payload = json.loads(provenance_path.read_text())
        entries = TypeAdapter(dict[str, FeatureSelectionProvenance]).validate_python(payload)
    except ValueError as error:
        raise ValueError(f"Invalid feature-selection provenance: {provenance_path}") from error
    if artifact_path.name not in entries:
        raise ValueError(f"Missing feature-selection provenance entry: {artifact_path.name}")
    provenance = entries[artifact_path.name]
    label_dates = set(provenance.label_dates)
    if label_dates & (set(validation_dates) | set(test_dates)):
        raise ValueError(
            f"Feature artifact {artifact_path.name} used validation or holdout labels. "
            "Historical comparisons are exploratory. Regenerate using selection training labels only."
        )
    if not label_dates <= set(selection_train_dates):
        raise ValueError(
            f"Feature artifact {artifact_path.name} used labels outside selection training dates."
        )
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if digest != provenance.sha256:
        raise ValueError(f"Feature artifact digest mismatch: {artifact_path}")
