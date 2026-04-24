"""
Synthetic Sales Event Generator
=================================
Generates realistic retail sales events and appends them to fact_sales_events.

Each call to generate_events() produces N new rows with:
  - a unique UUID event_id
  - a product_id drawn from the existing product catalogue
  - realistic price / discount / quantity values
  - an event_timestamp close to NOW (< 60s jitter)

Using current-time timestamps guarantees that every batch sits strictly
AFTER the previous batch's watermark, which makes the incremental ingest
step provably correct: re-running the pipeline never double-counts events.

Usage
-----
    python -m src.data_generator.generate_events          # 200 events (default)
    python -m src.data_generator.generate_events --n 500  # 500 events
"""

import argparse
import random
import time
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import text

from src.utils.db import get_connection
from src.utils.logger import get_logger

logger = get_logger("data_generator.events")

# ── Realistic ranges for athletic footwear ────────────────────────────────────
_PRICE_MIN, _PRICE_MAX       = 49.99, 249.99
_DISCOUNT_MIN, _DISCOUNT_MAX = 0.0,   0.55
_QTY_MIN, _QTY_MAX           = 1,     5

_CREATE_FACT_TABLE = text("""
    CREATE TABLE IF NOT EXISTS fact_sales_events (
        event_id        TEXT        PRIMARY KEY,
        product_id      TEXT        NOT NULL,
        price           REAL        NOT NULL,
        discount        REAL        NOT NULL,
        quantity        INTEGER     NOT NULL,
        revenue         REAL        NOT NULL,
        event_timestamp TIMESTAMP   NOT NULL,
        ingested_at     TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
    )
""")


def _ensure_schema(conn) -> None:
    conn.execute(_CREATE_FACT_TABLE)


def _load_product_ids(conn) -> list:
    """Return all known product IDs from the product catalogue."""
    rows = conn.execute(
        text("SELECT DISTINCT product_id FROM info WHERE product_id IS NOT NULL")
    ).fetchall()
    ids = [r[0] for r in rows if r[0]]
    if not ids:
        logger.warning("info table empty — using synthetic product IDs as fallback")
        ids = [f"PROD_{i:04d}" for i in range(1, 101)]
    return ids


def _count_events(conn) -> int:
    try:
        return conn.execute(text("SELECT COUNT(*) FROM fact_sales_events")).scalar()
    except Exception:
        return 0


def generate_events(n_events: int = 200, seed: int = None) -> int:
    """
    Generate n_events synthetic sales events and append to fact_sales_events.

    Parameters
    ----------
    n_events : int
        Number of events to generate per call (default 200).
    seed : int | None
        Fix the random seed for reproducibility.  None (default) uses the
        current epoch second so every run produces different data.

    Returns
    -------
    int
        Number of events inserted.
    """
    random.seed(seed if seed is not None else int(time.time()))

    start = time.time()
    logger.info(f"=== Generating {n_events} synthetic sales events ===")

    with get_connection() as conn:
        _ensure_schema(conn)
        before = _count_events(conn)

        product_ids = _load_product_ids(conn)
        now = datetime.now(timezone.utc)

        rows = []
        for _ in range(n_events):
            price    = round(random.uniform(_PRICE_MIN, _PRICE_MAX), 2)
            discount = round(random.uniform(_DISCOUNT_MIN, _DISCOUNT_MAX), 2)
            qty      = random.randint(_QTY_MIN, _QTY_MAX)
            revenue  = round(price * (1 - discount) * qty, 2)

            # Timestamps spread FORWARD from now (0–999 ms ahead per event).
            # Forward-only jitter guarantees every batch's timestamps are
            # strictly greater than the previous batch's watermark, so the
            # incremental ingest never misses or double-counts events.
            jitter_ms = random.randint(0, 999)
            event_ts = now + timedelta(milliseconds=jitter_ms)

            rows.append({
                "event_id":        str(uuid.uuid4()),
                "product_id":      random.choice(product_ids),
                "price":           price,
                "discount":        discount,
                "quantity":        qty,
                "revenue":         revenue,
                "event_timestamp": event_ts,
            })

        df = pd.DataFrame(rows)
        df.to_sql("fact_sales_events", conn, if_exists="append", index=False)

        after = _count_events(conn)

    elapsed = time.time() - start
    logger.info(
        f"Inserted {n_events} events in {elapsed:.2f}s | "
        f"fact_sales_events: {before} → {after} rows"
    )
    return n_events


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic sales events")
    parser.add_argument("--n", type=int, default=200, help="Number of events to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()
    generate_events(n_events=args.n, seed=args.seed)
