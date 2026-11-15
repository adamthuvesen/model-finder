"""Warning-level reports on the splits. Nothing here blocks a run.

Thresholds come from the removed ``lib/evaluation_review.py``: a 0.05 target-rate shift for a
binary target, 0.5 train standard deviations otherwise, and a scaled feature mean shift of 1.0.

Overlap counts use plain value comparison. Numeric columns compare by value across int and float.
Any other type difference makes a check not comparable, and the report names it. A count of zero
does not prove the rows are unrelated.
"""

from __future__ import annotations

import math

import pandas as pd
from pydantic import BaseModel, ConfigDict

from model_finder.splits import HoldoutSplits, SearchSplits

_BINARY_TARGET_SHIFT = 0.05
_SCALED_TARGET_SHIFT = 0.5
_SCALED_FEATURE_SHIFT = 1.0


class DataAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_column: str
    train_target_mean: float
    validation_target_mean: float
    concerns: list[str]


# No source: a judgement default. Below this share, overlap only warns.
LARGE_OVERLAP_SHARE = 0.10


class OverlapReport(BaseModel):
    """Rows shared between splits. ``None`` means undeclared keys or a listed failed comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key_columns: list[str]
    train_validation_exact_rows: int | None
    search_holdout_exact_rows: int | None
    train_validation_keys: int | None
    search_holdout_keys: int | None
    not_comparable: list[str]
    holdout_rows: int

    @property
    def needs_warning(self) -> bool:
        return bool(self.not_comparable) or any(
            (
                self.train_validation_exact_rows,
                self.search_holdout_exact_rows,
                self.train_validation_keys,
                self.search_holdout_keys,
            )
        )

    @property
    def is_large(self) -> bool:
        shared = max(self.search_holdout_exact_rows or 0, self.search_holdout_keys or 0)
        return shared >= LARGE_OVERLAP_SHARE * self.holdout_rows


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


def audit_search_splits(
    search: SearchSplits, target_column: str, key_columns: list[str]
) -> DataAudit:
    """Report target and feature drift between train and validation. Reads no holdout rows."""
    if target_column not in search.train.columns:
        raise ValueError(
            f"Target column is missing from search rows: {target_column}. "
            "Fix: set target_column in your RunnerConfig"
        )
    missing_keys = [column for column in key_columns if column not in search.train.columns]
    if missing_keys:
        raise ValueError(
            f"experiment.md: key_columns names columns that search rows lack: {missing_keys}. "
            f"Fix: use these columns: {sorted(str(column) for column in search.train.columns)}"
        )
    train_target = pd.to_numeric(search.train[target_column], errors="coerce").dropna()
    validation_target = pd.to_numeric(search.validation[target_column], errors="coerce").dropna()
    shift = target_shift_concern(train_target, validation_target, "validation")
    return DataAudit(
        target_column=target_column,
        train_target_mean=float(train_target.mean()),
        validation_target_mean=float(validation_target.mean()),
        concerns=[*([shift] if shift else []), *_feature_drift_concerns(search, target_column)],
    )


def _comparable(left: pd.Series, right: pd.Series) -> bool:
    both_numeric = pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right)
    return both_numeric or left.dtype == right.dtype


def _shared_rows(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> int | None:
    """Count left rows whose values appear in right, or None when the frames are not comparable.

    Left is not deduplicated, so the count is in left rows and compares with len(left).
    """
    if any(c not in left.columns or c not in right.columns for c in columns):
        return None
    if not all(_comparable(left[c], right[c]) for c in columns):
        return None
    return len(left[columns].merge(right[columns].drop_duplicates(), on=columns))


def _with_keys(rows: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    return rows.dropna(subset=[column for column in key_columns if column in rows.columns])


def overlap_report(
    search: SearchSplits, holdout: HoldoutSplits, key_columns: list[str]
) -> OverlapReport:
    search_rows = pd.concat([search.train, search.validation], ignore_index=True)
    all_columns = [str(column) for column in search.train.columns]
    counts = {
        "train_validation_exact_rows": _shared_rows(search.train, search.validation, all_columns),
        "search_holdout_exact_rows": _shared_rows(holdout.holdout, search_rows, all_columns),
    }
    if key_columns:
        # A missing key identifies nothing, so NaN keys never count as shared.
        counts["train_validation_keys"] = _shared_rows(
            _with_keys(search.train, key_columns),
            _with_keys(search.validation, key_columns),
            key_columns,
        )
        counts["search_holdout_keys"] = _shared_rows(
            _with_keys(holdout.holdout, key_columns),
            _with_keys(search_rows, key_columns),
            key_columns,
        )
    return OverlapReport(
        key_columns=key_columns,
        train_validation_exact_rows=counts["train_validation_exact_rows"],
        search_holdout_exact_rows=counts["search_holdout_exact_rows"],
        train_validation_keys=counts.get("train_validation_keys"),
        search_holdout_keys=counts.get("search_holdout_keys"),
        not_comparable=[name for name, count in counts.items() if count is None],
        holdout_rows=len(holdout.holdout),
    )
