# DuckDB source

Use DuckDB to test ingestion without a server or credentials. Provide a local database
and an existing experiment for the example below.

## Install

```bash
uv sync --extra models   # duckdb + pyarrow already live in the models extra
```

## Auth

None. DuckDB reads a local file or `.duckdb` database. The extractor opens it
**read-only** (`read_only=True`) when it is a real file.

## As-of

DuckDB has no time travel. Model an as-of with a `WHERE` predicate on a time
column; the value is recorded in the manifest for provenance.

## Caveats

- An in-memory database (`:memory:`) cannot be opened read-only.
- Types map cleanly to Arrow; decimals and timestamps round-trip to parquet.

## Example

```bash
uv run experiment pull \
  --experiment experiments/demo_warehouse \
  --source duckdb \
  --database path/to/events.duckdb \
  --query "SELECT * FROM events ORDER BY snapshot_date"
```
