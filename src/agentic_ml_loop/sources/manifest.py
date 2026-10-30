# pyarrow ships no stubs, so Arrow types are unknown to the checker in this module.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false
"""The dataset snapshot manifest and its integrity check.

The manifest is the reproducible recipe for a frozen snapshot: source, exact query, as-of,
a schema fingerprint, and a content hash. ``verify_snapshot`` hashes the parquet file and
fails loudly when values drift, even when the shape stays the same.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from agentic_ml_loop.io import file_sha256, read_text, utc_now, write_json
from agentic_ml_loop.paths import SNAPSHOT_FILE_NAME
from agentic_ml_loop.sources.errors import SnapshotIntegrityError
from agentic_ml_loop.splits import Sha256

if TYPE_CHECKING:
    import pyarrow as pa

SNAPSHOT_FILENAME = SNAPSHOT_FILE_NAME
MANIFEST_FILENAME = "dataset_manifest.json"


class DatasetManifest(BaseModel):
    """Provenance for one frozen snapshot. All fields are required on load."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: Literal[2]
    source_type: str
    driver: str
    query: str
    as_of: str | None
    row_count: int
    column_count: int
    columns: list[str]
    schema_hash: Sha256
    content_sha256: Sha256
    sample_seed: int | None
    allow_nondeterministic: bool
    snapshot_filename: str
    pulled_at: str
    nondeterministic_reason: str | None = None

    def write(self, path: Path) -> None:
        write_json(path, self.model_dump())

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        return cls.model_validate_json(read_text(path))


def schema_fingerprint(schema: pa.Schema) -> str:
    """A stable hash of field names + types (ignores schema metadata)."""
    parts = [f"{field.name}:{field.type}" for field in schema]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def dataset_manifest_from_table(
    table: pa.Table,
    *,
    source_type: str,
    driver: str,
    query: str,
    as_of: str | None,
    sample_seed: int | None,
    content_sha256: str,
    nondeterministic_reason: str | None = None,
    snapshot_filename: str = SNAPSHOT_FILENAME,
) -> DatasetManifest:
    """Build a manifest from an extracted Arrow *table* and its written file hash."""
    schema = table.schema
    return DatasetManifest(
        manifest_version=2,
        source_type=source_type,
        driver=driver,
        query=query,
        as_of=as_of,
        row_count=table.num_rows,
        column_count=len(schema),
        columns=[field.name for field in schema],
        schema_hash=schema_fingerprint(schema),
        content_sha256=content_sha256,
        sample_seed=sample_seed,
        allow_nondeterministic=nondeterministic_reason is not None,
        nondeterministic_reason=nondeterministic_reason,
        snapshot_filename=snapshot_filename,
        pulled_at=utc_now(),
    )


def verify_snapshot(snapshot_path: Path, manifest: DatasetManifest) -> None:
    """Raise :class:`SnapshotIntegrityError` if the parquet drifted from *manifest*.

    Hashes the whole file, then reads the footer to name what changed. Raises
    ``ModuleNotFoundError`` if pyarrow is not installed, so callers can treat "cannot check"
    differently from "check failed".
    """
    import pyarrow.parquet as pq

    metadata = pq.read_metadata(str(snapshot_path))
    schema = metadata.schema.to_arrow_schema()
    problems: list[str] = []
    if metadata.num_rows != manifest.row_count:
        problems.append(f"row count {metadata.num_rows} != manifest {manifest.row_count}")
    if schema_fingerprint(schema) != manifest.schema_hash:
        problems.append("schema hash mismatch (columns or types changed)")
    if file_sha256(snapshot_path) != manifest.content_sha256:
        problems.append("content hash mismatch (values changed)")
    if problems:
        raise SnapshotIntegrityError(f"{snapshot_path.name}: " + "; ".join(problems))


def nondeterministic_pull_label(rows_dir: Path) -> str | None:
    """Label search rows from a pull that a person allowed to be irreproducible."""
    path = rows_dir / MANIFEST_FILENAME
    if not path.is_file():
        return None
    manifest = DatasetManifest.load(path)
    if not manifest.allow_nondeterministic:
        return None
    return f"The data pull is not reproducible: {manifest.nondeterministic_reason}"
