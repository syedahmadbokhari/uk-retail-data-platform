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

    # SQLite requires a UNIQUE or PRIMARY KEY constraint for ON CONFLICT to work.
    # Create a unique index idempotently — safe to call on every upsert.
    conn.execute(text(
        f"CREATE UNIQUE INDEX IF NOT EXISTS "
        f"uq_{table_name}_{conflict_col} ON {table_name}({conflict_col})"
    ))

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
