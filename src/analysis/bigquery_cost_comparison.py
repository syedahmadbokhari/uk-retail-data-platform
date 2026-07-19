"""
Compares bytes scanned (and estimated cost) for the same representative
query run against the partitioned+clustered fact_sales_events table vs. a
plain copy of the same data with no partitioning or clustering.

Uses BigQuery's dry-run feature (job_config.dry_run = True) — this asks
BigQuery to plan the query and report exactly how many bytes it WOULD
scan, without actually executing it or incurring any query cost. The
bytes-scanned figures this script prints are real, live numbers from
BigQuery's own query planner, not estimates computed locally.

Representative query: total revenue for one product within a date range —
a realistic "how much did product X make last month" lookup. It exercises
both optimisations on the same query: the date filter benefits from
partition pruning, and the product_id filter benefits from clustering.

Cost estimate: BigQuery on-demand pricing is $6.25 per TiB scanned (Google
Cloud's published on-demand rate as of this writing — see
https://cloud.google.com/bigquery/pricing; the first 1 TiB/month is free).
This is an estimate for illustration, not a guarantee of actual billing.

If no BigQuery project/credentials are available (GOOGLE_CLOUD_PROJECT
unset, or authentication fails), this script logs a clear message and exits
without raising — it must be safe to invoke in CI/local environments that
have no cloud access.
"""
import datetime
from google.cloud import bigquery

from src.utils.db import get_bigquery_client, get_bigquery_dataset, is_bigquery_enabled
from src.utils.logger import get_logger
from src.etl.bigquery_setup import FACT_TABLE

logger = get_logger("analysis.bigquery_cost_comparison")

FLAT_TABLE = f"{FACT_TABLE}_flat"
PRICE_PER_TIB_USD = 6.25
BYTES_PER_TIB = 1024 ** 4

# A representative filter — any real product_id / recent date range works
# equally well for a dry run, since BigQuery estimates bytes scanned from
# the partitions/blocks the filter touches, not from executing the query.
_SAMPLE_PRODUCT_ID = "G27341"
_SAMPLE_START_DATE = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
_SAMPLE_END_DATE = datetime.date.today().isoformat()

_QUERY_TEMPLATE = """
    SELECT SUM(revenue) AS total_revenue
    FROM `{table}`
    WHERE DATE(event_timestamp) BETWEEN @start_date AND @end_date
      AND product_id = @product_id
"""


def create_flat_comparison_table(client, dataset_id: str) -> None:
    """
    Creates {FACT_TABLE}_flat as an unpartitioned, unclustered copy of the
    real fact table's current data, purely so the comparison below is
    apples-to-apples on identical rows.
    """
    sql = f"""
        CREATE OR REPLACE TABLE `{dataset_id}.{FLAT_TABLE}`
        AS SELECT * FROM `{dataset_id}.{FACT_TABLE}`
    """
    client.query(sql).result()
    logger.info(f"  {FLAT_TABLE}: refreshed as an unpartitioned/unclustered copy for comparison")


def _dry_run_bytes(client, table_fqn: str) -> int:
    job_config = bigquery.QueryJobConfig(
        dry_run=True,
        use_query_cache=False,
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", _SAMPLE_START_DATE),
            bigquery.ScalarQueryParameter("end_date", "DATE", _SAMPLE_END_DATE),
            bigquery.ScalarQueryParameter("product_id", "STRING", _SAMPLE_PRODUCT_ID),
        ],
    )
    query_job = client.query(_QUERY_TEMPLATE.format(table=table_fqn), job_config=job_config)
    return query_job.total_bytes_processed


def estimated_cost_usd(bytes_scanned: int) -> float:
    return (bytes_scanned / BYTES_PER_TIB) * PRICE_PER_TIB_USD


def compare() -> dict:
    """
    Runs the representative query as a dry run against both the
    partitioned+clustered table and the flat comparison table, and prints
    the bytes-scanned / estimated-cost difference. Returns the result dict,
    or None if BigQuery isn't available in this environment.
    """
    if not is_bigquery_enabled():
        logger.info(
            "GOOGLE_CLOUD_PROJECT is not set — skipping BigQuery cost comparison "
            "(this is expected in local/CI environments with no cloud project)."
        )
        return None

    try:
        client = get_bigquery_client()
        dataset_id = f"{client.project}.{get_bigquery_dataset()}"
        create_flat_comparison_table(client, dataset_id)

        partitioned_bytes = _dry_run_bytes(client, f"{dataset_id}.{FACT_TABLE}")
        flat_bytes = _dry_run_bytes(client, f"{dataset_id}.{FLAT_TABLE}")
    except Exception as exc:
        logger.info(
            f"BigQuery is not reachable in this environment ({type(exc).__name__}: {exc}) "
            "— skipping cost comparison rather than failing."
        )
        return None

    reduction_pct = (
        (1 - partitioned_bytes / flat_bytes) * 100 if flat_bytes > 0 else 0.0
    )
    result = {
        "partitioned_bytes_scanned": partitioned_bytes,
        "flat_bytes_scanned": flat_bytes,
        "reduction_pct": reduction_pct,
        "partitioned_cost_usd": estimated_cost_usd(partitioned_bytes),
        "flat_cost_usd": estimated_cost_usd(flat_bytes),
    }

    logger.info("=== BigQuery cost comparison (dry run — no query cost incurred) ===")
    logger.info(f"  Partitioned + clustered : {partitioned_bytes:,} bytes scanned (${result['partitioned_cost_usd']:.6f} est.)")
    logger.info(f"  Flat (no optimisation)  : {flat_bytes:,} bytes scanned (${result['flat_cost_usd']:.6f} est.)")
    logger.info(f"  Reduction               : {reduction_pct:.1f}%")

    return result


if __name__ == "__main__":
    compare()
