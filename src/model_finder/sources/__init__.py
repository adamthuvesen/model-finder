# pyarrow ships no stubs, so Arrow types are unknown to the checker in this module.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false
"""Warehouse ingestion: pull a read-only query into a frozen parquet snapshot.

Edge-only by design — a one-time ``pull`` materializes a reproducible snapshot
plus a provenance manifest; experiments then run offline against that snapshot.
Importing this package is cheap and offline: pyarrow and every warehouse client
are imported lazily inside the functions that need them.
"""

from __future__ import annotations

from model_finder.sources.adapters import SOURCE_EXTRAS, SUPPORTED_SOURCE_TYPES
from model_finder.sources.errors import (
    MissingSourceExtra,
    NonDeterministicQueryError,
    SnapshotIntegrityError,
    SourceError,
)
from model_finder.sources.extractor import WarehouseExtractor, get_extractor
from model_finder.sources.manifest import (
    MANIFEST_FILENAME,
    SNAPSHOT_FILENAME,
    DatasetManifest,
    dataset_manifest_from_table,
    verify_snapshot,
)
from model_finder.sources.query import apply_as_of, require_deterministic_query
from model_finder.sources.registry import Bundle, get_bundle, list_bundles
from model_finder.sources.snapshot import (
    SnapshotFreezeRequest,
    freeze_snapshot,
    read_snapshot,
    seal_snapshot,
)

__all__ = [
    "MANIFEST_FILENAME",
    "SNAPSHOT_FILENAME",
    "SOURCE_EXTRAS",
    "SUPPORTED_SOURCE_TYPES",
    "Bundle",
    "DatasetManifest",
    "MissingSourceExtra",
    "NonDeterministicQueryError",
    "SnapshotFreezeRequest",
    "SnapshotIntegrityError",
    "SourceError",
    "WarehouseExtractor",
    "apply_as_of",
    "dataset_manifest_from_table",
    "freeze_snapshot",
    "get_bundle",
    "get_extractor",
    "list_bundles",
    "read_snapshot",
    "require_deterministic_query",
    "seal_snapshot",
    "verify_snapshot",
]
