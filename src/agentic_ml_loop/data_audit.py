"""Warning-level reports on the splits. Nothing here blocks a run.

Thresholds come from the removed ``lib/evaluation_review.py``: a 0.05 target-rate shift for a
binary target, 0.5 train standard deviations otherwise, and a scaled feature mean shift of 1.0.

Overlap counts use plain value comparison. A column whose type differs between frames is not
comparable, and the report says so instead of guessing. A count of zero does not prove the rows
are unrelated.
"""

from __future__ import annotations

import math

import pandas as pd
from pydantic import BaseModel, ConfigDict

from agentic_ml_loop.splits import HoldoutSplits, SearchSplits

_BINARY_TARGET_SHIFT = 0.05
_SCALED_TARGET_SHIFT = 0.5
_SCALED_FEATURE_SHIFT = 1.0


class DataAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_column: str
    train_target_mean: float
    validation_target_mean: float
    concerns: list[str]


class OverlapReport(BaseModel):
    """Rows shared between splits. ``None`` means the frames cannot be compared by plain value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key_columns: list[str]
    train_validation_exact_rows: int | None
    search_holdout_exact_rows: int | None
    train_validation_keys: int | None
    search_holdout_keys: int | None

    @property
    def shares_rows(self) -> bool:
        return any(
            (
                self.train_validation_exact_rows,
                self.search_holdout_exact_rows,
                self.train_validation_keys,
                self.search_holdout_keys,
            )
        )


def _is_binary(values: pd.Series) -> bool:
    return set(values.unique().tolist()) <= {0, 1}


def target_shift_concern(train: pd.Series, other: pd.Series, name: str) -> str | None:
    """Describe a shift in the target's mean from train to another split."""
    shift = float(other.mean() - train.mean())
    if _is_binary(train):
        if abs(shift) >= _BINARY_TARGET_SHIFT:
            return f"{name} target rate differs from train by {shift:+.3f}"
        return None
    scale = float(train.std(ddof=0))
    scaled = abs(shift) / scale if scale > 0 and math.isfinite(scale) else 0.0
    if scaled >= _SCALED_TARGET_SHIFT:
        return f"{name} target mean differs from train by {scaled:.2f} train standard deviations"
    return None


def _feature_drift_concerns(search: SearchSplits, target_column: str) -> list[str]:
    concerns: list[str] = []
    numeric = search.train.select_dtypes(include="number").columns
    for column in numeric:
        if column == target_column or column not in search.validation.columns:
            continue
        scale = float(search.train[column].std(ddof=0))
        if scale <= 0 or not math.isfinite(scale):
            continue
        delta = abs(float(search.validation[column].mean() - search.train[column].mean())) / scale
        if delta >= _SCALED_FEATURE_SHIFT:
            concerns.append(f"validation feature {column} shifted {delta:.2f} train std")
    return concerns


def audit_search_splits(search: SearchSplits, target_column: str) -> DataAudit:
    """Report target and feature drift between train and validation. Reads no holdout rows."""
    if target_column not in search.train.columns:
        raise ValueError(f"Target column is missing from search rows: {target_column}")
    train_target = pd.to_numeric(search.train[target_column], errors="coerce").dropna()
    validation_target = pd.to_numeric(search.validation[target_column], errors="coerce").dropna()
    shift = target_shift_concern(train_target, validation_target, "validation")
    return DataAudit(
        target_column=target_column,
        train_target_mean=float(train_target.mean()),
        validation_target_mean=float(validation_target.mean()),
        concerns=[*([shift] if shift else []), *_feature_drift_concerns(search, target_column)],
    )


def _shared_rows(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> int | None:
    """Count distinct left rows found in right, or None when the frames are not comparable."""
    if not columns or any(c not in left.columns or c not in right.columns for c in columns):
        return None
    if any(left[c].dtype != right[c].dtype for c in columns):
        return None
    return len(left[columns].drop_duplicates().merge(right[columns].drop_duplicates(), on=columns))


def overlap_report(
    search: SearchSplits, holdout: HoldoutSplits, key_columns: list[str]
) -> OverlapReport:
    search_rows = pd.concat([search.train, search.validation], ignore_index=True)
    all_columns = [str(column) for column in search.train.columns]
    return OverlapReport(
        key_columns=key_columns,
        train_validation_exact_rows=_shared_rows(search.train, search.validation, all_columns),
        search_holdout_exact_rows=_shared_rows(search_rows, holdout.holdout, all_columns),
        train_validation_keys=_shared_rows(search.train, search.validation, key_columns),
        search_holdout_keys=_shared_rows(search_rows, holdout.holdout, key_columns),
    )
