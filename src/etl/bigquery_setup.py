"""
BigQuery schema setup — cloud data warehouse layer.

Defines the target BigQuery dataset and table schemas, matching the existing
dbt mart structure (see dbt/models/marts/) and the fact_sales_events table
defined in src/etl/ingest_events.py.

Partitioning and clustering are applied only to fact_sales_events, the one
table that actually grows unboundedly and is queried with date-range and
per-product filters. The five analytics_* mart tables are small aggregates
(one row per brand, per product, per month, or per discount category) —
partitioning or clustering a table with a handful of rows adds metadata
overhead for no pruning benefit, so they are created as plain tables.

WHY partition fact_sales_events BY DATE(event_timestamp):
    Every representative query against this table (see the README's SQL
    Analysis section, and bigquery_cost_comparison.py) filters on a date
    range — "revenue in the last N days," "events during month X." Native
    date partitioning lets BigQuery skip reading entire partitions (days)
    that fall outside the filter, which is where the majority of bytes-
    scanned savings comes from as the table grows.

WHY cluster BY product_id (not brand):
    Clustering only pays off on high-cardinality columns used in filters —
    it lets BigQuery skip blocks within a partition. brand has exactly 2
    distinct values in this dataset (Adidas, Nike), which is far too low
    for block pruning to help. product_id has ~3,120 distinct values, is
    the fact table's own natural key, and is exactly what per-product
    lookups (e.g. the recommender, or "revenue for product X") filter on —
    the ideal shape for BigQuery clustering.
"""
from google.cloud import bigquery

from src.utils.db import get_bigquery_client, get_bigquery_dataset
from src.utils.logger import get_logger

logger = get_logger("etl.bigquery_setup")

FACT_TABLE = "fact_sales_events"

FACT_SCHEMA = [
    bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("product_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("price", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("discount", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("quantity", "INT64", mode="REQUIRED"),
    bigquery.SchemaField("revenue", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("event_timestamp", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("ingested_at", "TIMESTAMP", mode="NULLABLE"),
]

# Mart schemas — mirror the columns actually produced by src/etl/aggregate.py
# and dbt/models/marts/*.sql. No partitioning/clustering: see module docstring.
MART_SCHEMAS = {
    "analytics_brand_revenue": [
        bigquery.SchemaField("brand", "STRING"),
        bigquery.SchemaField("total_revenue", "FLOAT64"),
        bigquery.SchemaField("product_count", "INT64"),
        bigquery.SchemaField("revenue_share_pct", "FLOAT64"),
    ],
    "analytics_product_revenue": [
        bigquery.SchemaField("product_name", "STRING"),
        bigquery.SchemaField("brand", "STRING"),
        bigquery.SchemaField("total_revenue", "FLOAT64"),
        bigquery.SchemaField("revenue_rank", "INT64"),
    ],
    "analytics_monthly_traffic": [
        bigquery.SchemaField("month", "STRING"),
        bigquery.SchemaField("visit_count", "INT64"),
    ],
    "analytics_discount_impact": [
        bigquery.SchemaField("category", "STRING"),
        bigquery.SchemaField("total_revenue", "FLOAT64"),
        bigquery.SchemaField("avg_revenue", "FLOAT64"),
        bigquery.SchemaField("product_count", "INT64"),
    ],
    "analytics_event_revenue": [
        bigquery.SchemaField("product_id", "STRING"),
        bigquery.SchemaField("product_name", "STRING"),
        bigquery.SchemaField("brand", "STRING"),
        bigquery.SchemaField("event_revenue", "FLOAT64"),
        bigquery.SchemaField("avg_price", "FLOAT64"),
        bigquery.SchemaField("avg_discount", "FLOAT64"),
        bigquery.SchemaField("revenue_rank", "INT64"),
    ],
}


def build_fact_table(dataset_ref: str) -> bigquery.Table:
    """
    Constructs (but does not create) the fact_sales_events Table definition:
    DAY partitioning on event_timestamp, clustered on product_id.
    """
    table = bigquery.Table(f"{dataset_ref}.{FACT_TABLE}", schema=FACT_SCHEMA)
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="event_timestamp",
    )
    table.clustering_fields = ["product_id"]
    return table


def create_dataset(client=None) -> str:
    """Creates the analytics dataset if it doesn't already exist. Idempotent."""
    client = client or get_bigquery_client()
    dataset_id = f"{client.project}.{get_bigquery_dataset()}"
    dataset = bigquery.Dataset(dataset_id)
    client.create_dataset(dataset, exists_ok=True)
    logger.info(f"Dataset ready: {dataset_id}")
    return dataset_id


def create_fact_table(client=None, dataset_id: str = None) -> bigquery.Table:
    """Creates the partitioned + clustered fact_sales_events table. Idempotent."""
    client = client or get_bigquery_client()
    dataset_id = dataset_id or f"{client.project}.{get_bigquery_dataset()}"
    table = build_fact_table(dataset_id)
    created = client.create_table(table, exists_ok=True)
    logger.info(
        f"  {FACT_TABLE}: partitioned DAY on event_timestamp, "
        f"clustered on product_id"
    )
    return created


def create_mart_tables(client=None, dataset_id: str = None) -> list:
    """Creates the five analytics_* mart tables (plain, no partition/cluster). Idempotent."""
    client = client or get_bigquery_client()
    dataset_id = dataset_id or f"{client.project}.{get_bigquery_dataset()}"

    created = []
    for name, schema in MART_SCHEMAS.items():
        table = bigquery.Table(f"{dataset_id}.{name}", schema=schema)
        created.append(client.create_table(table, exists_ok=True))
        logger.info(f"  {name}: {len(schema)} columns")
    return created


def setup_all() -> None:
    logger.info("=== BigQuery setup: dataset + fact table + marts ===")
    client = get_bigquery_client()
    dataset_id = create_dataset(client)
    create_fact_table(client, dataset_id)
    create_mart_tables(client, dataset_id)
    logger.info("BigQuery setup complete")


if __name__ == "__main__":
    setup_all()
