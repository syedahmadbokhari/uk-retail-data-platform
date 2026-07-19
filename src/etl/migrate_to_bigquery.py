"""
Migrates data from the currently active local/CI backend (PostgreSQL or
SQLite, selected by src/utils/db.py) into the BigQuery tables defined in
bigquery_setup.py.

Idempotency
-----------
Each table load uses WRITE_TRUNCATE — the target table's contents are fully
replaced on every run, so re-running this script never duplicates rows. This
mirrors the same idempotent-by-replacement pattern already used elsewhere in
this repo (see src/etl/aggregate.py and src/clustering.py, both of which use
to_sql(..., if_exists="replace")), rather than introducing a separate
incremental-merge strategy just for this one script. A true incremental
load into a partitioned table would use date-partition decorators or a
MERGE statement — noted here, not implemented, since WRITE_TRUNCATE already
satisfies "safe to re-run without duplicating data" for a dataset this size.
"""
import time
import pandas as pd
from google.cloud import bigquery

from src.utils.db import get_connection, get_bigquery_client, get_bigquery_dataset
from src.utils.logger import get_logger
from src.etl.bigquery_setup import setup_all, FACT_TABLE, MART_SCHEMAS

logger = get_logger("etl.migrate_to_bigquery")

_FACT_QUERY = (
    "SELECT event_id, product_id, price, discount, quantity, revenue, "
    "event_timestamp, ingested_at FROM fact_sales_events"
)


def _load_dataframe(client, dataset_id: str, table_name: str, df: pd.DataFrame) -> int:
    if df.empty:
        logger.info(f"  {table_name}: source is empty — skipped")
        return 0

    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
    job = client.load_table_from_dataframe(df, f"{dataset_id}.{table_name}", job_config=job_config)
    job.result()  # block until the load finishes, so errors surface here
    logger.info(f"  {table_name}: {len(df)} rows loaded (WRITE_TRUNCATE)")
    return len(df)


def migrate_fact_table(client=None, dataset_id: str = None) -> int:
    client = client or get_bigquery_client()
    dataset_id = dataset_id or f"{client.project}.{get_bigquery_dataset()}"

    with get_connection() as conn:
        df = pd.read_sql(_FACT_QUERY, conn)

    for col in ("event_timestamp", "ingested_at"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    return _load_dataframe(client, dataset_id, FACT_TABLE, df)


def migrate_mart_tables(client=None, dataset_id: str = None) -> dict:
    client = client or get_bigquery_client()
    dataset_id = dataset_id or f"{client.project}.{get_bigquery_dataset()}"

    counts = {}
    with get_connection() as conn:
        for table_name in MART_SCHEMAS:
            try:
                df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            except Exception:
                logger.info(f"  {table_name}: not yet built in the local layer — skipped")
                counts[table_name] = 0
                continue
            counts[table_name] = _load_dataframe(client, dataset_id, table_name, df)
    return counts


def migrate_all() -> dict:
    start = time.time()
    logger.info("=== Migrating clean/analytics layer -> BigQuery ===")

    client = get_bigquery_client()
    dataset_id = f"{client.project}.{get_bigquery_dataset()}"

    # Ensure the target tables (with partitioning/clustering) exist before loading.
    setup_all()

    results = {FACT_TABLE: migrate_fact_table(client, dataset_id)}
    results.update(migrate_mart_tables(client, dataset_id))

    logger.info(f"Migration complete in {time.time() - start:.2f}s")
    return results


if __name__ == "__main__":
    migrate_all()
