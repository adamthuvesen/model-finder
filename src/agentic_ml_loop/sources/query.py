"""Query-level reproducibility helpers: as-of rewriting and a determinism guard.

As-of substitutes an ``{as_of}`` placeholder with the warehouse's time-travel clause.
The read-only and determinism guards parse the query with sqlglot in the warehouse's dialect.
A query they cannot parse fails closed.
"""

from __future__ import annotations

import re
import warnings

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from agentic_ml_loop.sources.errors import NonDeterministicQueryError, ReadOnlyQueryError

# Warehouses that support reading a historical version via a FROM-clause suffix.
_TIME_TRAVEL = {"snowflake", "bigquery", "databricks"}

_AS_OF_PLACEHOLDER = "{as_of}"

# Accepted --as-of forms. A Delta version must be the explicit ``v<n>`` form so a
# compact date like ``20260601`` can never be mistaken for a commit number.
_AS_OF_VERSION_RE = re.compile(r"\Av\d+\Z", re.IGNORECASE)
_AS_OF_TIMESTAMP_RE = re.compile(
    r"\A\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)? *(Z|UTC|[+-]\d{2}:?\d{2})?\Z"
)

_RANDOM_FUNCTIONS = frozenset(
    {
        "rand",
        "randn",
        "random",
        "randstr",
        "newid",
        "uuid",
        "uuid_string",
        "generate_uuid",
        "gen_random_uuid",
        "uuid_generate_v1",
        "uuid_generate_v1mc",
        "uuid_generate_v4",
        "seq1",
        "seq2",
        "seq4",
        "seq8",
        "uniform",
        "normal",
        "zipf",
    }
)
_MOVING_TIME_FUNCTIONS = frozenset(
    {
        "now",
        "getdate",
        "sysdate",
        "current_date",
        "current_datetime",
        "current_time",
        "current_timestamp",
        "localtime",
        "localtimestamp",
        "systimestamp",
    }
)
_RANKING_FUNCTIONS = (exp.RowNumber, exp.Rank, exp.DenseRank, exp.Ntile, exp.PercentRank)

# Raw-text checks, because a comment is invisible to the parser. MySQL runs /*! ... */ as SQL,
# and engines disagree on whether block comments nest.
_EXECUTABLE_COMMENT = re.compile(r"/\*M?!", re.IGNORECASE)
_NESTED_COMMENT = re.compile(r"/\*(?:(?!\*/).)*?/\*", re.DOTALL)
_WRITE_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Copy,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Grant,
    exp.Revoke,
    exp.Command,
    exp.Into,
    exp.Lock,
    exp.NextValueFor,
)
# sqlglot raises these on some malformed input instead of ParseError.
_PARSER_CRASHES = (IndexError, KeyError, AttributeError, TypeError, RecursionError)


def _validate_as_of(as_of: str, source_type: str) -> None:
    """Reject an as-of that is neither a timestamp nor (Databricks) a ``v<n>``.

    Guards against garbage being string-interpolated into time-travel SQL and
    against ambiguous all-digit tokens.
    """
    if _AS_OF_VERSION_RE.match(as_of):
        if source_type != "databricks":
            raise ValueError(
                f"as-of version form 'v<n>' is only valid for databricks, not {source_type}."
            )
        return
    if _AS_OF_TIMESTAMP_RE.match(as_of):
        return
    suffix = " or a Delta version 'v<n>'" if source_type == "databricks" else ""
    raise ValueError(
        f"Invalid --as-of {as_of!r}: expected an ISO timestamp "
        f"(e.g. 2026-06-01 or 2026-06-01T00:00:00Z){suffix}."
    )


def _time_travel_clause(source_type: str, as_of: str) -> str:
    """Return the dialect time-travel clause to substitute for ``{as_of}``."""
    if source_type == "snowflake":
        return f"AT(TIMESTAMP => '{as_of}'::timestamp_tz)"
    if source_type == "bigquery":
        return f"FOR SYSTEM_TIME AS OF TIMESTAMP '{as_of}'"
    if source_type == "databricks":
        if _AS_OF_VERSION_RE.match(as_of):
            return f"VERSION AS OF {as_of[1:]}"
        return f"TIMESTAMP AS OF '{as_of}'"
    raise ValueError(f"{source_type} has no time travel")  # pragma: no cover


def apply_as_of(query: str, source_type: str, as_of: str | None) -> tuple[str, str | None]:
    """Apply an as-of point and return ``(rewritten_query, recorded_as_of)``.

    - Time-travel warehouses (Snowflake/BigQuery/Databricks): the query MUST
      contain an ``{as_of}`` placeholder right after the table reference; it is
      replaced with the dialect clause. Asking for an as-of without the
      placeholder is an error (we will not guess where the table is).
    - Other warehouses (Redshift/Postgres/DuckDB): there is no time travel, so
      the query must carry its own time predicate; the as-of value is recorded
      for provenance but the query is otherwise unchanged.
    - A placeholder with no as-of value is an error; a malformed as-of is an error.

    The recorded as-of (the caller's value) is what lands in the manifest so a
    re-pull reproduces the same rows.
    """
    has_placeholder = _AS_OF_PLACEHOLDER in query

    if as_of is None:
        if has_placeholder:
            raise ValueError(
                "Query contains an {as_of} placeholder but no --as-of value was given."
            )
        return query, None

    _validate_as_of(as_of, source_type)

    if source_type in _TIME_TRAVEL:
        if not has_placeholder:
            raise ValueError(
                f"{source_type} as-of requires an {{as_of}} placeholder after the table, "
                f"e.g. `FROM my_table {{as_of}}`."
            )
        rewritten = query.replace(_AS_OF_PLACEHOLDER, _time_travel_clause(source_type, as_of))
        return rewritten, as_of

    # Non-time-travel: the predicate lives in the query; just record the value.
    if has_placeholder:
        rewritten = query.replace(_AS_OF_PLACEHOLDER, "")
        return rewritten, as_of
    return query, as_of


def _parse_one_query(query: str, dialect: str | None) -> exp.Query:
    try:
        statements = sqlglot.parse(query.replace(_AS_OF_PLACEHOLDER, ""), dialect=dialect)
    except (SqlglotError, *_PARSER_CRASHES) as exc:
        raise NonDeterministicQueryError(
            "Query cannot be parsed, so its safety and reproducibility are unknown. Fix the SQL."
        ) from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ReadOnlyQueryError("Pull exactly one read-only query.")
    return statements[0]


def require_read_only_query(query: str, dialect: str) -> None:
    """Refuse a write in the warehouse's own dialect. No flag turns this off.

    This is defence in depth. It cannot see a write inside ``SELECT side_effect_proc()``. The
    read-only role on the connection is the real guard.
    """
    if _EXECUTABLE_COMMENT.search(query):
        raise ReadOnlyQueryError("Query contains an executable /*! comment.")
    if _NESTED_COMMENT.search(query):
        raise ReadOnlyQueryError("Query contains a nested block comment.")
    tree = _parse_one_query(query, dialect)
    write = tree.find(*_WRITE_NODES)
    if write is not None:
        raise ReadOnlyQueryError(f"Query contains a {type(write).__name__.upper()}.")
    for function in tree.find_all(exp.Anonymous):
        name = function.name.upper()
        if name.startswith("SYSTEM$") or name in {"NEXTVAL", "SETVAL"}:
            raise ReadOnlyQueryError(f"Query calls {function.name}.")
    for column in tree.find_all(exp.Column):
        if column.table and column.name.upper() == "NEXTVAL":
            raise ReadOnlyQueryError("Query advances a sequence with NEXTVAL.")


def _function_names(node: exp.Func) -> set[str]:
    if isinstance(node, exp.Anonymous):
        return {node.name.lower()}
    return {str(name).lower() for name in type(node).sql_names()}


def _has_unordered_ranking(tree: exp.Query) -> bool:
    return any(
        isinstance(window.this, _RANKING_FUNCTIONS) and window.args.get("order") is None
        for window in tree.find_all(exp.Window)
    )


def _nondeterminism_reasons(tree: exp.Query) -> list[str]:
    reasons: list[str] = []
    if any(_function_names(node) & _RANDOM_FUNCTIONS for node in tree.find_all(exp.Func)):
        reasons.append("uses a random or sequence function")
    if _has_unordered_ranking(tree):
        reasons.append("ranks rows in a window without ORDER BY")
    if any(sample.args.get("seed") is None for sample in tree.find_all(exp.TableSample)):
        reasons.append("samples without a REPEATABLE/seed clause")
    if any(
        distinct.args.get("on") is not None
        and (scope := distinct.find_ancestor(exp.Select)) is not None
        and scope.args.get("order") is None
        for distinct in tree.find_all(exp.Distinct)
    ):
        reasons.append("uses DISTINCT ON without ORDER BY")
    if any(
        scope.args.get("order") is None
        and (scope.args.get("limit") is not None or scope.args.get("offset") is not None)
        for scope in tree.find_all(exp.Query)
    ):
        reasons.append("caps rows (LIMIT/FETCH FIRST/TOP) without ORDER BY")
    return reasons


def _uses_moving_time(tree: exp.Query) -> bool:
    return any(_function_names(node) & _MOVING_TIME_FUNCTIONS for node in tree.find_all(exp.Func))


def require_deterministic_query(
    query: str, *, dialect: str | None = None, allow_nondeterministic: bool = False
) -> None:
    """Guard a pull query's reproducibility.

    Raises :class:`NonDeterministicQueryError` for random functions, unseeded sampling, a row
    cap in any query scope without an ``ORDER BY`` in that scope, or a query that does not
    parse. Warns on a moving-time function, which freezes fine in this pull but drifts on a
    re-pull. ``ORDER BY`` on a non-unique key still passes: the guard cannot see uniqueness.
    """
    if allow_nondeterministic:
        return
    tree = _parse_one_query(query, dialect)
    reasons = _nondeterminism_reasons(tree)
    if reasons:
        raise NonDeterministicQueryError(
            "Query is not reproducible (" + "; ".join(reasons) + "). "
            "Add an ORDER BY / fixed seed, or pass --allow-nondeterministic --reason <text>, which labels the result exploratory."
        )
    if _uses_moving_time(tree):
        warnings.warn(
            "Query uses a moving-time function (now()/current_date/...): a re-pull will "
            "return different rows. Pin the point in time with --as-of (or a literal "
            "timestamp predicate) for a reproducible snapshot.",
            stacklevel=2,
        )
