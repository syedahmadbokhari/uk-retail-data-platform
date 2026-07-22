"""
Snowflake schema setup — second cloud data warehouse option, alongside (not
replacing) BigQuery. Mirrors bigquery_setup.py's structure: same fact table
and mart tables, same "cluster the one table that grows and gets filtered by
date + product, leave the small aggregate marts unclustered" reasoning — but
the actual clustering mechanism is genuinely different, not just a syntax
swap. See the module-level note below before assuming BigQuery's model maps
1:1 onto Snowflake's.

HOW SNOWFLAKE'S MODEL DIFFERS FROM BIGQUERY'S (read this before comparing
the two scripts line-by-line):

  BigQuery has TWO separate, independent mechanisms: native DAY partitioning
  on a column (PARTITION BY — must be a DATE/TIMESTAMP column, prunes whole
  partitions) and clustering (CLUSTER BY — sorts data within partitions,
  prunes blocks). bigquery_setup.py uses both: PARTITION BY DATE(event_
  timestamp), CLUSTER BY product_id.

  Snowflake has ONE mechanism: every table is automatically divided into
  micro-partitions (contiguous, compressed, ~50-500MB uncompressed chunks)
  regardless of anything the user configures. There is no separate "PARTITION
  BY" DDL at all. An optional CLUSTER BY clustering key tells Snowflake how
  to keep co-locating similar values across micro-partitions as data is
  loaded/changed — it's the same mechanism whether you're trying to achieve
  what BigQuery would call "partitioning" (date-range pruning) or
  "clustering" (per-entity pruning). That's why this table clusters on
  BOTH columns at once — CLUSTER BY (event_timestamp, product_id) — rather
  than splitting them into two separate features the way BigQuery does.

  A genuinely different cost consideration, not just a naming difference:
  BigQuery's clustering maintenance is free, folded into background storage
  optimization. Snowflake's Automatic Clustering — the background service
  that keeps a clustered table's micro-partitions well-organized as new rows
  arrive — itself consumes credits ("To control cost, you can suspend
  automatic reclustering" per Snowflake's own docs). Clustering a small,
  rarely-changing table in Snowflake can cost more in reclustering credits
  than it ever saves in query pruning — reinforced by the same reasoning
  bigquery_setup.py already applies to the small mart tables here (not
  clustered, for exactly this kind of cost-benefit reason).
"""
from src.utils.db import get_snowflake_connection, get_snowflake_database, get_snowflake_schema
from src.utils.logger import get_logger

logger = get_logger("etl.snowflake_setup")

FACT_TABLE = "FACT_SALES_EVENTS"

# Mirrors bigquery_setup.py's FACT_SCHEMA columns — Snowflake-native types:
# STRING -> VARCHAR, FLOAT64 -> FLOAT, INT64 -> NUMBER, TIMESTAMP -> TIMESTAMP_NTZ
# (no timezone — matches this repo's naive-timestamp data, same as SQLite/BigQuery).
_FACT_COLUMNS = """
    event_id        VARCHAR      NOT NULL,
    product_id      VARCHAR      NOT NULL,
    price           FLOAT        NOT NULL,
    discount        FLOAT        NOT NULL,
    quantity        NUMBER       NOT NULL,
    revenue         FLOAT        NOT NULL,
    event_timestamp TIMESTAMP_NTZ NOT NULL,
    ingested_at     TIMESTAMP_NTZ
"""

# Mirrors bigquery_setup.py's MART_SCHEMAS exactly (same columns, same tables).
MART_COLUMNS = {
    "analytics_brand_revenue": """
        brand              VARCHAR,
        total_revenue       FLOAT,
        product_count       NUMBER,
        revenue_share_pct   FLOAT
    """,
    "analytics_product_revenue": """
        product_name    VARCHAR,
        brand           VARCHAR,
        total_revenue   FLOAT,
        revenue_rank    NUMBER
    """,
    "analytics_monthly_traffic": """
        month           VARCHAR,
        visit_count     NUMBER
    """,
    "analytics_discount_impact": """
        category        VARCHAR,
        total_revenue   FLOAT,
        avg_revenue     FLOAT,
        product_count   NUMBER
    """,
    "analytics_event_revenue": """
        product_id      VARCHAR,
        product_name    VARCHAR,
        brand           VARCHAR,
        event_revenue   FLOAT,
        avg_price       FLOAT,
        avg_discount    FLOAT,
        revenue_rank    NUMBER
    """,
}


def build_fact_table_ddl(database: str, schema: str) -> str:
    """
    Returns (does not execute) the CREATE TABLE statement for
    FACT_SALES_EVENTS: clustered on (event_timestamp, product_id).

    WHY cluster on (event_timestamp, product_id) together, not one or the
    other: every representative query (see snowflake_cost_comparison.py)
    filters on a date range AND a specific product_id, same as BigQuery's
    equivalent query. Snowflake has no separate partitioning mechanism to
    hand the date filter to, so the same clustering key has to help both
    filters at once — date first (the coarser, more selective filter across
    a table this size), product_id second.
    """
    return f"""
        CREATE TABLE IF NOT EXISTS {database}.{schema}.{FACT_TABLE} (
            {_FACT_COLUMNS}
        )
        CLUSTER BY (event_timestamp, product_id)
    """


def build_mart_table_ddl(database: str, schema: str, table_name: str) -> str:
    """Plain (unclustered) mart table DDL — see module docstring for why."""
    columns = MART_COLUMNS[table_name]
    return f"CREATE TABLE IF NOT EXISTS {database}.{schema}.{table_name} ({columns})"


def create_database_and_schema(conn=None) -> None:
    """Creates the database + schema if they don't already exist. Idempotent."""
    conn = conn or get_snowflake_connection()
    database, schema = get_snowflake_database(), get_snowflake_schema()
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {database}.{schema}")
    logger.info(f"Database/schema ready: {database}.{schema}")


def create_fact_table(conn=None) -> None:
    """Creates the clustered FACT_SALES_EVENTS table. Idempotent (IF NOT EXISTS)."""
    conn = conn or get_snowflake_connection()
    database, schema = get_snowflake_database(), get_snowflake_schema()
    conn.cursor().execute(build_fact_table_ddl(database, schema))
    logger.info(f"  {FACT_TABLE}: clustered on (event_timestamp, product_id)")


def create_mart_tables(conn=None) -> None:
    """Creates the five analytics_* mart tables (plain, no clustering key). Idempotent."""
    conn = conn or get_snowflake_connection()
    database, schema = get_snowflake_database(), get_snowflake_schema()
    for table_name in MART_COLUMNS:
        conn.cursor().execute(build_mart_table_ddl(database, schema, table_name))
        logger.info(f"  {table_name}: created")


def setup_all() -> None:
    logger.info("=== Snowflake setup: database/schema + fact table + marts ===")
    conn = get_snowflake_connection()
    create_database_and_schema(conn)
    create_fact_table(conn)
    create_mart_tables(conn)
    logger.info("Snowflake setup complete")


if __name__ == "__main__":
    setup_all()
