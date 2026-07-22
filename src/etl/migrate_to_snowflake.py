"""
Migrates data from the currently active local/CI backend (PostgreSQL or
SQLite, selected by src/utils/db.py) into the Snowflake tables defined in
snowflake_setup.py. Mirrors migrate_to_bigquery.py's structure and
idempotency reasoning exactly — same source query, same "replace the whole
table on every run" strategy — only the load mechanism differs, since
Snowflake's Python connector doesn't have BigQuery's load_table_from_dataframe.

Idempotency
-----------
Each table load uses snowflake.connector.pandas_tools.write_pandas(...,
overwrite=True) — Snowflake's documented bulk-load helper, with overwrite=True
truncating the target table before loading. This is the Snowflake-idiomatic
equivalent of migrate_to_bigquery.py's WRITE_TRUNCATE: re-running this script
never duplicates rows, for the same reason WRITE_TRUNCATE doesn't — full
replacement, not append or merge.
"""
import time
import pandas as pd
from snowflake.connector.pandas_tools import write_pandas

from src.utils.db import get_connection, get_snowflake_connection, get_snowflake_database, get_snowflake_schema
from src.utils.logger import get_logger
from src.etl.snowflake_setup import setup_all, FACT_TABLE, MART_COLUMNS

logger = get_logger("etl.migrate_to_snowflake")

_FACT_QUERY = (
    "SELECT event_id, product_id, price, discount, quantity, revenue, "
    "event_timestamp, ingested_at FROM fact_sales_events"
)


def _load_dataframe(conn, table_name: str, df: pd.DataFrame) -> int:
    if df.empty:
        logger.info(f"  {table_name}: source is empty — skipped")
        return 0

    success, n_chunks, n_rows, _ = write_pandas(
        conn, df, table_name.upper(),
        database=get_snowflake_database(), schema=get_snowflake_schema(),
        overwrite=True,
    )
    if not success:
        raise RuntimeError(f"write_pandas reported failure loading {table_name}")
    logger.info(f"  {table_name}: {n_rows} rows loaded (overwrite=True)")
    return n_rows


def migrate_fact_table(conn=None) -> int:
    conn = conn or get_snowflake_connection()

    with get_connection() as source_conn:
        df = pd.read_sql(_FACT_QUERY, source_conn)

    for col in ("event_timestamp", "ingested_at"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    return _load_dataframe(conn, FACT_TABLE, df)


def migrate_mart_tables(conn=None) -> dict:
    conn = conn or get_snowflake_connection()

    counts = {}
    with get_connection() as source_conn:
        for table_name in MART_COLUMNS:
            try:
                df = pd.read_sql(f"SELECT * FROM {table_name}", source_conn)
            except Exception:
                logger.info(f"  {table_name}: not yet built in the local layer — skipped")
                counts[table_name] = 0
                continue
            counts[table_name] = _load_dataframe(conn, table_name, df)
    return counts


def migrate_all() -> dict:
    start = time.time()
    logger.info("=== Migrating clean/analytics layer -> Snowflake ===")

    conn = get_snowflake_connection()

    # Ensure the target tables (with the clustering key) exist before loading.
    setup_all()

    results = {FACT_TABLE: migrate_fact_table(conn)}
    results.update(migrate_mart_tables(conn))

    logger.info(f"Migration complete in {time.time() - start:.2f}s")
    return results


if __name__ == "__main__":
    migrate_all()
