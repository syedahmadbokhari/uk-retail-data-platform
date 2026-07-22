"""
Compares query efficiency (partitions scanned, bytes scanned, elapsed time)
for the same representative query run against the clustered FACT_SALES_EVENTS
table vs. an unclustered copy of the same data — the Snowflake counterpart to
bigquery_cost_comparison.py.

WHY THIS SCRIPT WORKS DIFFERENTLY FROM bigquery_cost_comparison.py — read
this before assuming the two are interchangeable:

  BigQuery has a free dry-run mode (job_config.dry_run = True) that reports
  bytes-that-would-be-scanned with ZERO execution and zero cost. Snowflake
  has no equivalent. There is no "ask the optimizer what this would cost"
  call in Snowflake's connector or SQL surface — the only documented way to
  see how well a query pruned micro-partitions is to actually RUN it on a
  warehouse (consuming real, small credits — seconds of X-Small warehouse
  time, billed with a 60-second minimum per Snowflake's billing model) and
  then look up its stats afterward via INFORMATION_SCHEMA.QUERY_HISTORY(),
  a documented table function returning BYTES_SCANNED, PARTITIONS_SCANNED,
  PARTITIONS_TOTAL, and TOTAL_ELAPSED_TIME for recent queries in the current
  session/account (no ACCOUNT_USAGE grant required, and no data-latency
  problem the way SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY has — that view can
  lag by up to ~45 minutes, useless for "run a query, immediately check its
  own stats").

  This is a genuine cost-model difference, not a gap in this script: on
  BigQuery, comparing an optimised vs. unoptimised table is free. On
  Snowflake, it necessarily costs a small amount of real warehouse-time.
  This script is written and validated against the real Snowflake API
  surface (see tests/test_snowflake_setup.py's mocks) but has not been run
  against a live account, for exactly this reason — see the README's Cloud
  Data Warehouse Comparison section.

Cost estimate: Snowflake bills compute by WAREHOUSE SIZE x TIME, not bytes
scanned — an X-Small warehouse costs 1 credit/hour (billed per-second, 60s
minimum per session), scaling up by warehouse size (Small=2, Medium=4, ...).
On-demand credit price is commonly cited as $2-4/credit depending on cloud
provider, region, and Snowflake edition — there is no single global rate the
way BigQuery publishes one, so this script reports a RANGE, not a false-
precision single number, and the true rate should be checked against the
account's own rate sheet.
"""
import datetime

from src.utils.db import get_snowflake_connection, get_snowflake_database, get_snowflake_schema, is_snowflake_enabled
from src.utils.logger import get_logger
from src.etl.snowflake_setup import FACT_TABLE

logger = get_logger("analysis.snowflake_cost_comparison")

FLAT_TABLE = f"{FACT_TABLE}_FLAT"

# X-Small is Snowflake's smallest/cheapest warehouse size — 1 credit/hour,
# billed per-second with a 60s minimum. See module docstring for why this
# script reports a cost RANGE rather than a single dollar figure.
WAREHOUSE_CREDITS_PER_HOUR = 1.0
CREDIT_PRICE_USD_RANGE = (2.0, 4.0)

_SAMPLE_PRODUCT_ID = "G27341"
_WINDOW_DAYS = 3  # same representative window size as bigquery_cost_comparison.py

_QUERY_TEMPLATE = """
    SELECT SUM(revenue) AS total_revenue
    FROM {table}
    WHERE event_timestamp BETWEEN %(start_date)s AND %(end_date)s
      AND product_id = %(product_id)s
"""


def _representative_date_window(conn, table_fqn: str, window_days: int = _WINDOW_DAYS) -> tuple:
    """
    Anchors the demo query's date filter to the LAST `window_days` days that
    actually exist in the table, not datetime.date.today() — applying
    bigquery_cost_comparison.py's date-window fix from day one here, rather
    than re-introducing the exact bug that script caught and had to fix.
    """
    cur = conn.cursor()
    cur.execute(f"SELECT MAX(event_timestamp) FROM {table_fqn}")
    max_ts = cur.fetchone()[0]
    max_date = max_ts.date() if hasattr(max_ts, "date") else max_ts
    start_date = max_date - datetime.timedelta(days=window_days - 1)
    return start_date.isoformat(), max_date.isoformat()


def create_flat_comparison_table(conn, database: str, schema: str) -> None:
    """
    Creates FACT_SALES_EVENTS_FLAT as an unclustered copy of the real fact
    table's current data, purely so the comparison below is apples-to-apples
    on identical rows.
    """
    cur = conn.cursor()
    cur.execute(f"CREATE OR REPLACE TABLE {database}.{schema}.{FLAT_TABLE} AS "
                f"SELECT * FROM {database}.{schema}.{FACT_TABLE}")
    logger.info(f"  {FLAT_TABLE}: refreshed as an unclustered copy for comparison")


def _run_and_get_stats(conn, table_fqn: str, start_date: str, end_date: str) -> dict:
    """
    Actually executes the representative query (real, small warehouse cost —
    see module docstring for why Snowflake has no free dry-run equivalent),
    then looks up its real stats via INFORMATION_SCHEMA.QUERY_HISTORY().
    """
    cur = conn.cursor()
    cur.execute(
        _QUERY_TEMPLATE.format(table=table_fqn),
        {"start_date": start_date, "end_date": end_date, "product_id": _SAMPLE_PRODUCT_ID},
    )
    cur.fetchall()
    query_id = cur.sfqid

    stats_cur = conn.cursor()
    stats_cur.execute(
        "SELECT bytes_scanned, partitions_scanned, partitions_total, total_elapsed_time "
        "FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY(RESULT_LIMIT => 100)) "
        "WHERE QUERY_ID = %(query_id)s",
        {"query_id": query_id},
    )
    row = stats_cur.fetchone()
    if row is None:
        raise RuntimeError(f"No QUERY_HISTORY row found for query {query_id}")

    bytes_scanned, partitions_scanned, partitions_total, total_elapsed_time_ms = row
    return {
        "query_id": query_id,
        "bytes_scanned": bytes_scanned,
        "partitions_scanned": partitions_scanned,
        "partitions_total": partitions_total,
        "elapsed_ms": total_elapsed_time_ms,
    }


def estimated_cost_usd_range(elapsed_ms: float, credits_per_hour: float = WAREHOUSE_CREDITS_PER_HOUR) -> tuple:
    """
    Converts elapsed query time into an estimated (low, high) USD cost range,
    using the documented execution-time-share-of-warehouse-credits method
    (see module docstring) and the $2-4/credit on-demand range.
    """
    hours = (elapsed_ms / 1000) / 3600
    credits = hours * credits_per_hour
    low, high = CREDIT_PRICE_USD_RANGE
    return credits * low, credits * high


def compare() -> dict:
    """
    Runs the representative query against both the clustered table and the
    unclustered comparison table, and prints the partitions-scanned /
    bytes-scanned / elapsed-time difference. Returns the result dict, or
    None if Snowflake isn't available in this environment.
    """
    if not is_snowflake_enabled():
        logger.info(
            "SNOWFLAKE_ACCOUNT is not set — skipping Snowflake cost comparison "
            "(this is expected in local/CI environments with no Snowflake account)."
        )
        return None

    try:
        conn = get_snowflake_connection()
        database, schema = get_snowflake_database(), get_snowflake_schema()
        fact_fqn = f"{database}.{schema}.{FACT_TABLE}"
        flat_fqn = f"{database}.{schema}.{FLAT_TABLE}"

        create_flat_comparison_table(conn, database, schema)

        start_date, end_date = _representative_date_window(conn, fact_fqn)
        logger.info(f"  Representative date window (from real data): {start_date} to {end_date}")

        clustered_stats = _run_and_get_stats(conn, fact_fqn, start_date, end_date)
        flat_stats = _run_and_get_stats(conn, flat_fqn, start_date, end_date)
    except Exception as exc:
        logger.info(
            f"Snowflake is not reachable in this environment ({type(exc).__name__}: {exc}) "
            "— skipping cost comparison rather than failing."
        )
        return None

    result = {
        "date_window": (start_date, end_date),
        "product_id": _SAMPLE_PRODUCT_ID,
        "clustered": clustered_stats,
        "flat": flat_stats,
        "clustered_cost_usd_range": estimated_cost_usd_range(clustered_stats["elapsed_ms"]),
        "flat_cost_usd_range": estimated_cost_usd_range(flat_stats["elapsed_ms"]),
    }

    logger.info("=== Snowflake cost comparison (real execution — small warehouse cost incurred) ===")
    logger.info(
        f"  Clustered : {clustered_stats['partitions_scanned']}/{clustered_stats['partitions_total']} "
        f"partitions scanned, {clustered_stats['bytes_scanned']:,} bytes, {clustered_stats['elapsed_ms']}ms"
    )
    logger.info(
        f"  Flat      : {flat_stats['partitions_scanned']}/{flat_stats['partitions_total']} "
        f"partitions scanned, {flat_stats['bytes_scanned']:,} bytes, {flat_stats['elapsed_ms']}ms"
    )

    return result


if __name__ == "__main__":
    compare()
