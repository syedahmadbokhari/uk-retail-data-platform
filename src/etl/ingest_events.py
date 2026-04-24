"""
Incremental Event Ingestion
============================
Reads only NEW rows from fact_sales_events (those whose event_timestamp exceeds
the stored watermark), aggregates them to product level, and UPSERTs the result
into raw_finance so the rest of the pipeline sees up-to-date data.

Watermark table: event_ingestion_watermark (single-row singleton, id=1)
  - max_event_ts    : the highest event_timestamp successfully processed
  - total_processed : cumulative count of events processed across all runs

Idempotency guarantee
---------------------
Even if this step is re-run, it reads the same "new" window (events after
max_event_ts) and re-UPSERTs the same product-level aggregates — the result
is always identical to running it once.
"""

import time
from datetime import timezone

import pandas as pd
from sqlalchemy import text

from src.utils.db import get_connection, upsert_df, ensure_unique_index
from src.utils.logger import get_logger
from src.utils.validation import validate, ValidationError

logger = get_logger("etl.ingest_events")

# ── Watermark DDL ─────────────────────────────────────────────────────────────
_CREATE_WATERMARK = text("""
    CREATE TABLE IF NOT EXISTS event_ingestion_watermark (
        id              INTEGER PRIMARY KEY,
        max_event_ts    TIMESTAMP,
        total_processed INTEGER  DEFAULT 0,
        updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

_UPSERT_WATERMARK = text("""
    INSERT INTO event_ingestion_watermark (id, max_event_ts, total_processed, updated_at)
    VALUES (1, :ts, :total, CURRENT_TIMESTAMP)
    ON CONFLICT (id) DO UPDATE SET
        max_event_ts    = excluded.max_event_ts,
        total_processed = excluded.total_processed,
        updated_at      = CURRENT_TIMESTAMP
""")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_watermark_table(conn) -> None:
    conn.execute(_CREATE_WATERMARK)


def _get_watermark(conn) -> tuple:
    """Return (max_event_ts, total_processed) or (None, 0) on first run."""
    row = conn.execute(
        text("SELECT max_event_ts, total_processed FROM event_ingestion_watermark WHERE id = 1")
    ).fetchone()
    return (row[0], row[1]) if row else (None, 0)


def _set_watermark(conn, max_ts, total_processed: int) -> None:
    conn.execute(_UPSERT_WATERMARK, {"ts": max_ts, "total": total_processed})
    logger.info(f"Watermark advanced to: {max_ts} | cumulative events: {total_processed}")


def _fetch_new_events(conn, last_ts) -> pd.DataFrame:
    """Fetch all events with event_timestamp strictly after last_ts."""
    if last_ts is not None:
        query = text("""
            SELECT event_id, product_id, price, discount, quantity, revenue, event_timestamp
            FROM   fact_sales_events
            WHERE  event_timestamp > :last_ts
            ORDER  BY event_timestamp
        """)
        df = pd.read_sql(query, conn, params={"last_ts": str(last_ts)})
    else:
        # First run — process everything
        df = pd.read_sql(
            "SELECT event_id, product_id, price, discount, quantity, revenue, event_timestamp "
            "FROM fact_sales_events ORDER BY event_timestamp",
            conn,
        )
    return df


def _aggregate_to_product(events: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse event-level rows into one row per product.

    Aggregation logic:
      modified_revenue        → SUM  (total sales value from new events)
      modified_sale_price     → MEAN (average selling price)
      modified_listing_price  → MAX  (highest observed price = listing price proxy)
      modified_discount       → MEAN clipped to [0, 1]
    """
    agg = (
        events
        .groupby("product_id", as_index=False)
        .agg(
            modified_revenue=("revenue",  "sum"),
            modified_sale_price=("price", "mean"),
            modified_listing_price=("price", "max"),
            modified_discount=("discount", "mean"),
        )
    )
    agg["modified_revenue"]       = agg["modified_revenue"].round(2)
    agg["modified_sale_price"]    = agg["modified_sale_price"].round(2)
    agg["modified_listing_price"] = agg["modified_listing_price"].round(2)
    agg["modified_discount"]      = agg["modified_discount"].clip(0, 1).round(4)
    return agg


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_incremental() -> int:
    """
    Incremental ingest: fact_sales_events (new rows only) → raw_finance.

    Returns
    -------
    int
        Number of new events processed (0 if nothing new).
    """
    start = time.time()
    logger.info("=== Incremental ingest: fact_sales_events → raw_finance ===")

    with get_connection() as conn:
        _ensure_watermark_table(conn)
        last_ts, prev_total = _get_watermark(conn)

        logger.info(
            f"Watermark: {last_ts or 'none (first run)'} | "
            f"previously processed: {prev_total:,} events"
        )

        # ── Check fact table exists ───────────────────────────────────────────
        try:
            total_in_table = conn.execute(
                text("SELECT COUNT(*) FROM fact_sales_events")
            ).scalar()
        except Exception:
            logger.warning("fact_sales_events does not exist — run generate_events first")
            return 0

        logger.info(f"fact_sales_events total rows: {total_in_table:,}")

        # ── Fetch only new events ─────────────────────────────────────────────
        new_events = _fetch_new_events(conn, last_ts)
        n_new = len(new_events)

        if n_new == 0:
            logger.info("No new events since last watermark — nothing to do")
            return 0

        logger.info(f"New events to process: {n_new:,}")

        # ── Validate before aggregating ───────────────────────────────────────
        validate(new_events, "new_events_batch",
                 critical_cols=["product_id", "revenue"], min_rows=1)

        # ── Aggregate to product level ────────────────────────────────────────
        product_agg = _aggregate_to_product(new_events)
        logger.info(f"Aggregated to {len(product_agg)} unique products")

        # ── UPSERT into raw_events_aggregated ────────────────────────────────
        # product_id is unique here by construction (we grouped before writing).
        # ensure_unique_index() is safe to call because we know there are no
        # duplicate product_ids in this table.
        ensure_unique_index("raw_events_aggregated", "product_id", conn)
        n_upserted = upsert_df(product_agg, "raw_events_aggregated", "product_id", conn)
        logger.info(f"raw_events_aggregated: {n_upserted} product rows upserted")

        # ── Advance watermark ─────────────────────────────────────────────────
        max_ts = new_events["event_timestamp"].max()
        _set_watermark(conn, max_ts, prev_total + n_new)

    elapsed = time.time() - start
    logger.info(
        f"Incremental ingest complete — {n_new:,} events → "
        f"{len(product_agg)} products updated in {elapsed:.2f}s"
    )
    return n_new
