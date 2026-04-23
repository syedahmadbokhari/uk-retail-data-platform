import time
import pandas as pd

from src.utils.db import get_connection, upsert_df
from src.utils.logger import get_logger
from src.utils.validation import validate
from src.utils.watermark import ensure_watermark_table, get_watermark, set_watermark

logger = get_logger("etl.ingest")

# Tables where product_id is a unique natural key → safe to UPSERT by product_id.
# Traffic is an append-only event log (many rows per product), so it gets a full
# snapshot replace instead of a per-product UPSERT.
_UPSERT_TABLES = {
    "finance": "product_id",
    "brands":  "product_id",
    "info":    "product_id",
    "reviews": "product_id",
}
_SNAPSHOT_TABLES = ["traffic"]


def ingest_raw():
    """
    Load source tables into the raw_* layer.

    Incremental behaviour
    ─────────────────────
    • pipeline_watermarks tracks the row count from the last successful run.
    • If the source total row count matches the stored watermark the step is
      skipped — no work needed.
    • On change (or first run) each table is UPSERTed by product_id so the
      operation is always idempotent: re-running never creates duplicates.
    """
    start = time.time()
    logger.info("=== Ingestion: source → raw layer ===")

    with get_connection() as conn:
        ensure_watermark_table(conn)
        last_ts, last_count = get_watermark(conn, "ingest")

        # ── Quick change-detection: compare source row total to watermark ─────
        source_total = sum(
            conn.execute(
                __import__("sqlalchemy").text(f"SELECT COUNT(*) FROM {t}")
            ).scalar()
            for t in list(_UPSERT_TABLES) + _SNAPSHOT_TABLES
        )

        if last_count is not None and source_total == last_count:
            logger.info(
                f"Source unchanged ({source_total} rows) since {last_ts} — skipping ingest"
            )
            return

        logger.info(
            f"Source rows: {source_total} "
            f"(previous: {last_count if last_count is not None else 'first run'})"
        )

        # ── UPSERT tables (unique product_id) ─────────────────────────────────
        total_upserted = 0
        for tbl, conflict_col in _UPSERT_TABLES.items():
            df = pd.read_sql(f"SELECT * FROM {tbl}", conn)
            validate(df, tbl)
            n = upsert_df(df, f"raw_{tbl}", conflict_col, conn)
            total_upserted += n
            logger.info(f"  raw_{tbl}: {n} rows upserted")

        # ── Snapshot tables (full replace, event-log semantics) ───────────────
        for tbl in _SNAPSHOT_TABLES:
            df = pd.read_sql(f"SELECT * FROM {tbl}", conn)
            validate(df, tbl)
            df.to_sql(f"raw_{tbl}", conn, if_exists="replace", index=False)
            logger.info(f"  raw_{tbl}: {len(df)} rows replaced (snapshot)")
            total_upserted += len(df)

        set_watermark(conn, "ingest", source_total)

    elapsed = time.time() - start
    logger.info(f"Ingestion complete — {total_upserted} rows processed in {elapsed:.2f}s")
