"""The split types that make holdout access in candidate code a type error."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

MINIMUM_SPLIT_ROWS = 10


@dataclass(frozen=True)
class SearchSplits:
    """The only rows a candidate receives."""

    train: pd.DataFrame
    validation: pd.DataFrame


@dataclass(frozen=True)
class HoldoutSplits:
    """Rows the engine loads only after a selection is frozen."""

    holdout: pd.DataFrame


class FrameIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: Sha256
    rows: Annotated[int, Field(gt=0)]
    columns: Annotated[int, Field(gt=0)]
    algorithm: Literal["sha256-pandas-v1"] = "sha256-pandas-v1"


class SearchDataIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    train: FrameIdentity
    validation: FrameIdentity


def frame_identity(frame: pd.DataFrame) -> FrameIdentity:
    schema = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_names": [str(name) for name in frame.index.names],
        "index_dtype": str(frame.index.dtype),
    }
    digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return FrameIdentity(sha256=digest.hexdigest(), rows=len(frame), columns=len(frame.columns))


def search_data_identity(splits: SearchSplits) -> SearchDataIdentity:
    return SearchDataIdentity(
        train=frame_identity(splits.train), validation=frame_identity(splits.validation)
    )


def _time_ordered(rows: pd.DataFrame, time_column: str) -> pd.DataFrame:
    if time_column not in rows.columns:
        raise ValueError(f"Time column is missing: {time_column}")
    if len(rows) < MINIMUM_SPLIT_ROWS:
        raise ValueError(f"Dataset too small to split: {len(rows)} rows (minimum 10)")
    return rows.sort_values(time_column).reset_index(drop=True)


def temporal_search_split(
    rows: pd.DataFrame, time_column: str, *, train_fraction: float = 0.75
) -> SearchSplits:
    """Order rows by time into train and validation, and keep no holdout."""
    ordered = _time_ordered(rows, time_column)
    train_end = int(len(ordered) * train_fraction)
    return SearchSplits(
        train=ordered.iloc[:train_end].reset_index(drop=True),
        validation=ordered.iloc[train_end:].reset_index(drop=True),
    )


def temporal_split(
    rows: pd.DataFrame,
    time_column: str,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> tuple[SearchSplits, HoldoutSplits]:
    """Order rows by time. The latest rows become the holdout."""
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must be positive and leave a holdout")
    ordered = _time_ordered(rows, time_column)
    train_end = int(len(ordered) * train_fraction)
    validation_end = int(len(ordered) * (train_fraction + validation_fraction))
    return (
        SearchSplits(
            train=ordered.iloc[:train_end].reset_index(drop=True),
            validation=ordered.iloc[train_end:validation_end].reset_index(drop=True),
        ),
        HoldoutSplits(holdout=ordered.iloc[validation_end:].reset_index(drop=True)),
    )
