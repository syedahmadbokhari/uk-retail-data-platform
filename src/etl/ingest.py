import time
import pandas as pd
from sqlalchemy import text

from src.utils.db import get_connection
from src.utils.logger import get_logger
from src.utils.validation import validate
from src.utils.watermark import ensure_watermark_table, get_watermark, set_watermark

logger = get_logger("etl.ingest")

# All five source tables are full snapshots of a static catalogue.
# We use if_exists="replace" — the watermark check below skips the
# entire step when the source is unchanged, so this is still efficient.
_SOURCE_TABLES = ["finance", "brands", "info", "reviews", "traffic"]


def ingest_raw():
    """
    Load source tables into the raw_* layer.

    Incremental behaviour
    ─────────────────────
    • pipeline_watermarks tracks the total source row count from the last run.
    • If the count matches the watermark the step is skipped entirely —
      no writes needed.
    • On change (or first run) each table is fully replaced (snapshot semantics).
      This is correct because the source tables are a static product catalogue,
      not an event stream — every row represents the current state of a product.
    """
    start = time.time()
    logger.info("=== Ingestion: source → raw layer ===")

    with get_connection() as conn:
        ensure_watermark_table(conn)
        last_ts, last_count = get_watermark(conn, "ingest")

        source_total = sum(
            conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
            for t in _SOURCE_TABLES
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

        total = 0
        for tbl in _SOURCE_TABLES:
            df = pd.read_sql(f"SELECT * FROM {tbl}", conn)
            validate(df, tbl)
            df.to_sql(f"raw_{tbl}", conn, if_exists="replace", index=False)
            total += len(df)
            logger.info(f"  raw_{tbl}: {len(df)} rows loaded")

        set_watermark(conn, "ingest", source_total)

    logger.info(f"Ingestion complete — {total} rows in {time.time() - start:.2f}s")
