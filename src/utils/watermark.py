"""
Watermark tracking for incremental pipeline runs.

The pipeline_watermarks table stores the last successful timestamp and row count
for each step.  On re-runs the step can compare against these values to decide
whether processing is needed, and every write becomes an idempotent UPSERT rather
than a destructive replace.

Compatible with PostgreSQL and SQLite 3.24+ (both support the
INSERT ... ON CONFLICT ... DO UPDATE syntax).
"""

from datetime import datetime, timezone
from sqlalchemy import text
from src.utils.logger import get_logger

logger = get_logger("utils.watermark")

_CREATE_TABLE = text("""
    CREATE TABLE IF NOT EXISTS pipeline_watermarks (
        step_name          VARCHAR(100) PRIMARY KEY,
        last_successful_at TIMESTAMP,
        last_row_count     INTEGER,
        updated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")


def ensure_watermark_table(conn) -> None:
    """Create pipeline_watermarks if it does not yet exist."""
    conn.execute(_CREATE_TABLE)


def get_watermark(conn, step_name: str) -> tuple:
    """
    Return (last_successful_at, last_row_count) for step_name.
    Returns (None, None) on first run.
    """
    row = conn.execute(
        text("""
            SELECT last_successful_at, last_row_count
            FROM pipeline_watermarks
            WHERE step_name = :s
        """),
        {"s": step_name},
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def set_watermark(conn, step_name: str, row_count: int) -> None:
    """
    Upsert the watermark for step_name with the current UTC timestamp.
    Safe to call multiple times — always overwrites with the latest values.
    """
    conn.execute(
        text("""
            INSERT INTO pipeline_watermarks
                (step_name, last_successful_at, last_row_count, updated_at)
            VALUES (:s, :ts, :n, CURRENT_TIMESTAMP)
            ON CONFLICT (step_name) DO UPDATE SET
                last_successful_at = excluded.last_successful_at,
                last_row_count     = excluded.last_row_count,
                updated_at         = CURRENT_TIMESTAMP
        """),
        {"s": step_name, "ts": datetime.now(timezone.utc), "n": row_count},
    )
    logger.info(f"Watermark updated: step={step_name} rows={row_count}")
