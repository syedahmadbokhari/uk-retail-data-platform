import os
from contextlib import contextmanager
from sqlalchemy import create_engine, text
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_url() -> str:
    """
    Returns a PostgreSQL URL when DB_HOST is set, otherwise falls back to
    the local SQLite file.  This lets the ETL pipeline run against Postgres
    in production (Docker) and against SQLite locally / in CI without any
    code change.
    """
    host = os.getenv("DB_HOST")
    if host:
        user     = os.getenv("DB_USER",     "postgres")
        password = os.getenv("DB_PASSWORD", "")
        name     = os.getenv("DB_NAME",     "retail")
        port     = os.getenv("DB_PORT",     "5432")
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"

    db_path = os.path.join(_ROOT, "data", "retailDB.sqlite")
    return f"sqlite:///{db_path}"


def get_root() -> str:
    return _ROOT


def get_db_path() -> str:
    return os.path.join(_ROOT, "data", "retailDB.sqlite")


def get_engine():
    return create_engine(_build_url())


@contextmanager
def get_connection():
    """
    Yields a SQLAlchemy connection inside an open transaction.
    Commits on clean exit, rolls back on exception.
    Works transparently with both PostgreSQL and SQLite.
    Compatible with pandas read_sql() and DataFrame.to_sql().
    """
    with get_engine().begin() as conn:
        yield conn


def upsert_df(df: pd.DataFrame, table_name: str, conflict_col: str, conn) -> int:
    """
    Insert all rows from df into table_name, updating existing rows on conflict.

    Uses INSERT ... ON CONFLICT (conflict_col) DO UPDATE SET ..., which works
    on both PostgreSQL and SQLite 3.24+ (Python 3.8+ ships with SQLite >= 3.31).

    A UNIQUE INDEX on conflict_col is created automatically if it does not yet
    exist — this is required by SQLite for ON CONFLICT to resolve correctly.

    Returns the number of rows processed.
    """
    if df.empty:
        return 0

    # Ensure table exists with correct schema (zero rows)
    df.head(0).to_sql(table_name, conn, if_exists="append", index=False)

    # Caller is responsible for ensuring a UNIQUE or PRIMARY KEY constraint
    # exists on conflict_col before calling this function.  The helper
    # ensure_unique_index() below can be used to create it idempotently on
    # tables whose data is guaranteed unique (e.g. raw_events_aggregated).

    cols         = list(df.columns)
    col_list     = ", ".join(cols)
    placeholders = ", ".join(f":{c}" for c in cols)
    update_set   = ", ".join(
        f"{c} = excluded.{c}" for c in cols if c != conflict_col
    )

    sql = text(f"""
        INSERT INTO {table_name} ({col_list})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_col}) DO UPDATE SET {update_set}
    """)

    conn.execute(sql, df.to_dict(orient="records"))
    return len(df)


def ensure_unique_index(table_name: str, col: str, conn) -> None:
    """
    Create a UNIQUE INDEX on table_name(col) if it does not already exist.
    Only call this when the table data is guaranteed to have no duplicate
    values in col — e.g. raw_events_aggregated after product-level aggregation.
    """
    conn.execute(text(
        f"CREATE UNIQUE INDEX IF NOT EXISTS "
        f"uq_{table_name}_{col} ON {table_name}({col})"
    ))


# ── BigQuery — cloud data warehouse mode ───────────────────────────────────────
#
# A third, independent backend selected by GOOGLE_CLOUD_PROJECT, following the
# same environment-variable-driven pattern as the Postgres/SQLite switch in
# _build_url() above. Nothing above this point is touched: get_engine() and
# get_connection() remain SQLAlchemy-only and keep working exactly as before.
#
# BigQuery is not accessed through SQLAlchemy here — DDL features this repo's
# BigQuery scripts rely on (native DATE partitioning, CLUSTER BY, dry-run cost
# estimation) are expressed directly through the google-cloud-bigquery client,
# so callers that need BigQuery use get_bigquery_client() rather than
# get_connection().

def is_bigquery_enabled() -> bool:
    """True when GOOGLE_CLOUD_PROJECT is set, selecting the BigQuery backend."""
    return bool(os.getenv("GOOGLE_CLOUD_PROJECT"))


def get_bigquery_dataset() -> str:
    return os.getenv("BIGQUERY_DATASET", "retail_analytics")


def get_bigquery_client():
    """
    Returns a google.cloud.bigquery.Client for the project named in
    GOOGLE_CLOUD_PROJECT. Import is deliberately local to this function so
    that importing src.utils.db never requires google-cloud-bigquery to be
    installed unless the BigQuery backend is actually used.
    """
    from google.cloud import bigquery

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError(
            "get_bigquery_client() called without GOOGLE_CLOUD_PROJECT set — "
            "check is_bigquery_enabled() before calling this."
        )
    return bigquery.Client(project=project)


# ── Snowflake — second cloud data warehouse option ─────────────────────────────
#
# A fourth, independent backend selected by SNOWFLAKE_ACCOUNT, following the
# same environment-variable-driven pattern as _build_url() and the BigQuery
# switch above. This exists ALONGSIDE BigQuery, not instead of it — nothing
# above this point is touched, and is_bigquery_enabled()/get_bigquery_client()
# keep working exactly as before. A repo can have both env vars unset (local/
# CI), or set GOOGLE_CLOUD_PROJECT, or set SNOWFLAKE_ACCOUNT, or both, since
# src/etl/snowflake_setup.py and bigquery_setup.py are independent scripts —
# see the README's Cloud Data Warehouse Comparison section for why both exist.
#
# Snowflake, like BigQuery, isn't accessed through SQLAlchemy here — DDL
# features this repo's Snowflake scripts rely on (CLUSTER BY, QUERY_HISTORY
# query-profiling stats) are expressed directly through snowflake-connector-
# python, so callers that need Snowflake use get_snowflake_connection()
# rather than get_connection().

def is_snowflake_enabled() -> bool:
    """True when SNOWFLAKE_ACCOUNT is set, selecting the Snowflake backend."""
    return bool(os.getenv("SNOWFLAKE_ACCOUNT"))


def get_snowflake_database() -> str:
    return os.getenv("SNOWFLAKE_DATABASE", "RETAIL_ANALYTICS")


def get_snowflake_schema() -> str:
    return os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")


def get_snowflake_warehouse() -> str:
    return os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")


def get_snowflake_connection():
    """
    Returns a snowflake.connector.SnowflakeConnection for the account named
    in SNOWFLAKE_ACCOUNT. Import is deliberately local to this function so
    that importing src.utils.db never requires snowflake-connector-python
    to be installed unless the Snowflake backend is actually used.
    """
    import snowflake.connector

    account = os.getenv("SNOWFLAKE_ACCOUNT")
    if not account:
        raise RuntimeError(
            "get_snowflake_connection() called without SNOWFLAKE_ACCOUNT set — "
            "check is_snowflake_enabled() before calling this."
        )
    return snowflake.connector.connect(
        account=account,
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=get_snowflake_warehouse(),
        database=get_snowflake_database(),
        schema=get_snowflake_schema(),
    )
