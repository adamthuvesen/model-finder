# Snowflake source

Bulk extract via ADBC (Arrow-native). There is no anonymous public Snowflake, so
this is bring-your-own-account; the repo carries no credentials.

## Install

```bash
uv sync --extra snowflake   # adbc-driver-snowflake + manager + pyarrow
```

## Auth (keyless local)

Pass connection values with repeated `--conn key=value` flags. The engine reads no
connection file. Use browser SSO for local work. Nothing secret belongs in the repo:

```bash
--conn account=myorg-myacct --conn user=you@example.com \
--conn authenticator=externalbrowser --conn role=ANALYST_RO \
--conn warehouse=WH_XS --conn database=ANALYTICS --conn schema=MART
```

Use key-pair auth for unattended runs, with the private key outside the repo.

## Read-only grant (the real guarantee)

A dedicated **SELECT-only role** (`ANALYST_RO`): `USAGE` on warehouse/db/schema +
`SELECT` on the target tables/views, nothing else. Snowflake has **no** read-only
session flag, so the role grant is the boundary.

## As-of (reproducible)

Snowflake Time Travel. Put an `{as_of}` placeholder right after the table:

```sql
SELECT * FROM mart.events {as_of} ORDER BY event_id
```

`--as-of '2026-06-01 00:00:00'` rewrites it to
`AT(TIMESTAMP => '2026-06-01 00:00:00'::timestamp_tz)`. Bounded by the table's
Time Travel retention (1 day default; up to 90 on Enterprise).

## Caveats

- `NUMBER`/`DECIMAL` may arrive as float; a nullable integer column can become
  float. Cast large keys to string in SQL if exact values matter.
- Normalize `TIMESTAMP_TZ` to UTC in SQL so the snapshot is session-tz-independent.

## Example

```bash
uv run experiment pull \
  --experiment experiments/expansion \
  --source snowflake --as-of '2026-06-01 00:00:00' \
  --conn account=myorg-myacct --conn user=you@example.com \
  --conn authenticator=externalbrowser --conn role=ANALYST_RO \
  --conn warehouse=WH_XS --conn database=ANALYTICS --conn schema=MART \
  --query "SELECT * FROM mart.events {as_of} ORDER BY event_id"
```
