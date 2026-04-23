import os
from contextlib import contextmanager
from sqlalchemy import create_engine

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_url() -> str:
    """
    Returns a PostgreSQL URL when DB_HOST is set, otherwise falls back to
    the local SQLite file.  This lets the ETL pipeline run against Postgres
    in production and against SQLite locally / in CI without any code change.
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
