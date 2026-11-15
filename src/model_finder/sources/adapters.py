# pyarrow ships no stubs, so Arrow types are unknown to the checker in this module.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false
"""Per-warehouse extractor adapters.

Every adapter exposes the same shape — ``source_type``, ``driver``, and
``extract(query) -> pyarrow.Table`` — so the rest of the package never branches
on warehouse. Warehouse client libraries are imported lazily inside ``extract``
so that ``import model_finder.sources`` stays cheap and works without any extra installed;
a missing client raises :class:`MissingSourceExtra` naming the extra to install.

Backend split (see the change design): ADBC for Postgres / Snowflake / BigQuery
(clean wheels, best Arrow type fidelity), native clients for Databricks and
Redshift (their ADBC drivers are not pip-clean yet).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from model_finder.sources.errors import MissingSourceExtra
from model_finder.sources.registry import bundle_definitions

if TYPE_CHECKING:
    import pyarrow as pa

# Every table here comes from bundles/<name>/source.json, so a new ADBC source is one bundle.
_BUNDLES = {bundle["source_type"]: bundle for bundle in bundle_definitions()}
SOURCE_EXTRAS: dict[str, tuple[str, str]] = {
    name: (bundle["extra"], bundle["package"]) for name, bundle in _BUNDLES.items()
}
ADBC_DBAPI: dict[str, str] = {
    name: bundle["adbc_module"] for name, bundle in _BUNDLES.items() if "adbc_module" in bundle
}

SUPPORTED_SOURCE_TYPES: list[str] = sorted(SOURCE_EXTRAS)


def _require(source_type: str, module: str, package: str | None = None) -> Any:
    """Import *module* or raise an actionable :class:`MissingSourceExtra`."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        extra, default_pkg = SOURCE_EXTRAS[source_type]
        raise MissingSourceExtra(source_type, extra, package or default_pkg) from exc


def _to_adbc_db_kwargs(source_type: str, config: dict[str, Any]) -> dict[str, str]:
    """Translate friendly connection keys/values into ADBC's option namespace.

    Unknown or already-namespaced keys pass through unchanged. All values are
    coerced to ``str`` because ADBC ``db_kwargs`` is a string-to-string mapping.
    """
    key_map = _BUNDLES[source_type].get("connection_keys", {})
    value_maps = _BUNDLES[source_type].get("connection_values", {})
    result: dict[str, str] = {}
    for key, value in config.items():
        adbc_key = key_map.get(key, key)
        value_map = value_maps.get(key)
        adbc_value = value_map.get(str(value), value) if value_map else value
        result[adbc_key] = str(adbc_value)
    return result


class AdbcExtractor:
    """Read-only extractor for Postgres, Snowflake, and BigQuery via ADBC.

    Read-only is enforced server-side by the connection's least-privilege role
    (documented per source in its ``SETUP.md``); ADBC exposes no portable
    read-only session flag, so the role is the real guarantee.
    """

    driver = "adbc"

    def __init__(self, source_type: str, config: dict[str, Any]) -> None:
        if source_type not in ADBC_DBAPI:
            raise ValueError(f"ADBC does not back source type {source_type!r}")
        self.source_type = source_type
        config = dict(config)
        if source_type == "postgres" and not config.get("uri"):
            raise ValueError("postgres source requires --uri.")
        # Control keys that are not connection kwargs:
        self._max_bytes_billed = config.pop("max_bytes_billed", None)
        self._config = config

    def extract(self, query: str) -> pa.Table:
        impl = _require(self.source_type, ADBC_DBAPI[self.source_type])
        if self.source_type == "postgres":
            conn = impl.connect(self._config["uri"])
        else:
            db_kwargs = _to_adbc_db_kwargs(
                self.source_type,
                {key: value for key, value in self._config.items() if key != "uri"},
            )
            conn = impl.connect(db_kwargs=db_kwargs)
        try:
            with conn.cursor() as cur:
                self._apply_statement_options(cur)
                cur.execute(query)
                return cur.fetch_arrow_table()
        finally:
            conn.close()

    def _apply_statement_options(self, cursor: Any) -> None:
        """Set per-statement ADBC options before execute (best-effort).

        BigQuery's bytes-billed ceiling is a *statement* option, not a connection
        kwarg, so it is set on the cursor here. Not live-verified in this repo.
        """
        if self.source_type == "bigquery" and self._max_bytes_billed is not None:
            cursor.adbc_statement.set_options(
                **{"adbc.bigquery.sql.query.max_bytes_billed": str(self._max_bytes_billed)}
            )


class DatabricksExtractor:
    """Read-only extractor for Databricks SQL warehouses (native client)."""

    source_type = "databricks"
    driver = "databricks-sql-connector"

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = dict(config)

    def extract(self, query: str) -> pa.Table:
        module = _require("databricks", "databricks.sql", "databricks-sql-connector")
        with module.connect(**self._config) as conn, conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall_arrow()


class RedshiftExtractor:
    """Read-only extractor for Amazon Redshift (native client).

    Redshift has no Arrow-native fetch, so results come back through a pandas
    DataFrame; see the source ``SETUP.md`` for the DECIMAL/timestamp caveats.
    """

    source_type = "redshift"
    driver = "redshift-connector"

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = dict(config)

    def extract(self, query: str) -> pa.Table:
        redshift_connector = _require("redshift", "redshift_connector")
        pa = _require("redshift", "pyarrow", "pyarrow")
        conn = redshift_connector.connect(**self._config)
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            frame = cursor.fetch_dataframe()
            return pa.Table.from_pandas(frame, preserve_index=False)
        finally:
            conn.close()


class DuckDbExtractor:
    """Read-only extractor over a local DuckDB database or file (tier-0, creds-free)."""

    source_type = "duckdb"
    driver = "duckdb"

    def __init__(self, config: dict[str, Any]) -> None:
        self._database = config.get("database", ":memory:")
        # A :memory: database cannot be opened read-only; a real file can.
        self._read_only = bool(config.get("read_only", self._database != ":memory:"))

    def extract(self, query: str) -> pa.Table:
        duckdb = _require("duckdb", "duckdb")
        conn = duckdb.connect(self._database, read_only=self._read_only)
        try:
            return conn.execute(query).to_arrow_table()
        finally:
            conn.close()
