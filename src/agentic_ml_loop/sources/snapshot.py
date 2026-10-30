# pyarrow ships no stubs, so Arrow types are unknown to the checker in this module.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false
"""Freeze a warehouse query into a reproducible local parquet snapshot, then seal it.

``freeze_snapshot`` is the edge step: guard the query for determinism, apply the
as-of, extract Arrow, write ``snapshot.parquet`` + ``dataset_manifest.json``.
``seal_snapshot`` moves the latest rows to the holdout store. Workspace data then holds
search rows only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_ml_loop.holdout import (
    TRAIN_FILE_NAME,
    VALIDATION_FILE_NAME,
    HoldoutSeal,
    read_holdout_seal,
    seal_holdout,
    verify_sealed_holdout,
)
from agentic_ml_loop.io import file_sha256
from agentic_ml_loop.paths import DATA_DIRNAME, HOLDOUT_SEAL_FILE_NAME
from agentic_ml_loop.sources.errors import SourceError, SourceExtractionError
from agentic_ml_loop.sources.extractor import WarehouseExtractor, get_extractor
from agentic_ml_loop.sources.manifest import (
    MANIFEST_FILENAME,
    SNAPSHOT_FILENAME,
    DatasetManifest,
    dataset_manifest_from_table,
    verify_snapshot,
)
from agentic_ml_loop.sources.query import (
    apply_as_of,
    require_deterministic_query,
    require_read_only_query,
)
from agentic_ml_loop.splits import temporal_split

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa


@dataclass(frozen=True)
class SnapshotFreezeRequest:
    source_type: str
    query: str
    out_dir: Path
    config: dict[str, object] | None = None
    as_of: str | None = None
    sample_seed: int | None = None
    allow_nondeterministic: bool = False
    extractor: WarehouseExtractor | None = None


def freeze_snapshot(request: SnapshotFreezeRequest) -> DatasetManifest:
    """Extract *query* read-only and freeze it under *out_dir*.

    Returns the written :class:`DatasetManifest`. Pass *extractor* to inject a
    pre-built or fake adapter (used in tests); otherwise one is built from
    *source_type* and *config*.
    """
    require_read_only_query(request.query, request.source_type)
    require_deterministic_query(
        request.query,
        dialect=request.source_type,
        allow_nondeterministic=request.allow_nondeterministic,
    )
    rewritten_query, recorded_as_of = apply_as_of(request.query, request.source_type, request.as_of)

    extractor = request.extractor or get_extractor(request.source_type, request.config)
    try:
        table = extractor.extract(rewritten_query)
    except SourceError:
        raise
    # Warehouse drivers share no base error class.
    except Exception as exc:
        message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        raise SourceExtractionError(f"{request.source_type} query failed: {message}") from exc

    out_dir = request.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_dir / SNAPSHOT_FILENAME
    _write_parquet(table, snapshot_path)

    manifest = dataset_manifest_from_table(
        table,
        source_type=request.source_type,
        driver=extractor.driver,
        query=request.query,
        as_of=recorded_as_of,
        sample_seed=request.sample_seed,
        content_sha256=file_sha256(snapshot_path),
        allow_nondeterministic=request.allow_nondeterministic,
    )
    manifest.write(out_dir / MANIFEST_FILENAME)
    return manifest


def read_snapshot(experiment_dir: Path) -> pd.DataFrame:
    """Load a frozen snapshot, verifying it against its manifest when one is present."""
    import pyarrow.parquet as pq

    data_dir = experiment_dir / DATA_DIRNAME
    snapshot_path = data_dir / SNAPSHOT_FILENAME
    manifest_path = data_dir / MANIFEST_FILENAME
    if not snapshot_path.exists():
        raise FileNotFoundError(f"No snapshot at {snapshot_path}")
    if manifest_path.exists():
        verify_snapshot(snapshot_path, DatasetManifest.load(manifest_path))
    return pq.read_table(str(snapshot_path)).to_pandas()


def seal_snapshot(
    experiment_dir: Path,
    time_column: str,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> HoldoutSeal:
    """Split the snapshot by time, seal the latest rows, and remove the full snapshot.

    The seal record lands before the snapshot is removed. A rerun after a crash between them
    finishes the removal, because the record proves the rows are already sealed.
    """
    snapshot = experiment_dir / DATA_DIRNAME / SNAPSHOT_FILENAME
    if (experiment_dir / HOLDOUT_SEAL_FILE_NAME).exists():
        seal = read_holdout_seal(experiment_dir)
        verify_sealed_holdout(seal)
        rows_dir = experiment_dir / DATA_DIRNAME
        if (
            not (rows_dir / TRAIN_FILE_NAME).is_file()
            or not (rows_dir / VALIDATION_FILE_NAME).is_file()
        ):
            raise FileNotFoundError(f"Sealed experiment has no search rows in {rows_dir}")
        snapshot.unlink(missing_ok=True)
        return seal
    search, holdout = temporal_split(
        read_snapshot(experiment_dir),
        time_column,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    seal = seal_holdout(experiment_dir, search, holdout)
    snapshot.unlink()
    return seal


def _write_parquet(table: pa.Table, path: Path) -> None:
    import pyarrow.parquet as pq

    # store_schema=True (pyarrow's default, set explicitly) embeds the Arrow
    # schema in the parquet footer so verify_snapshot's schema hash round-trips.
    pq.write_table(table, str(path), compression="zstd", store_schema=True)
