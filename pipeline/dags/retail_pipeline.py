"""
Retail Data Platform — Airflow DAG
====================================
Production-grade pipeline with incremental loading, dbt transformations,
and explicit data-quality gates at every layer boundary.

Task chain
──────────
  ingest_raw
      └─► validate_raw_layer
              └─► clean_tables
                      └─► build_analytics      (Python, SQLite-compatible)
                              └─► dbt_run       (Postgres only; skipped on SQLite)
                                      └─► validate_marts
                                              └─► build_features
                                                      └─► build_similarity_matrix

Run locally (Docker):
    docker compose up --build

Run locally (SQLite, no Docker):
    python pipeline/run_pipeline.py
"""

import os
import sys
import shutil
import subprocess
from datetime import datetime, timedelta

from sqlalchemy import text

# ── Make project root importable ──────────────────────────────────────────────
_DAG_DIR      = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_DAG_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.etl.ingest              import ingest_raw
from src.etl.clean               import clean_tables
from src.etl.aggregate           import build_analytics
from src.features.build_features import build_features
from src.recommender             import build_similarity_matrix
from src.utils.db                import get_connection
from src.utils.logger            import get_logger
from src.utils.validation        import ValidationError

logger = get_logger("dag.retail_pipeline")

# ── Default task arguments ────────────────────────────────────────────────────
_default_args = {
    "owner":            "ahmad",
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

# ── Quality-gate helpers ──────────────────────────────────────────────────────

def _check_table(conn, table: str, min_rows: int, critical_col: str, max_null_rate: float = 0.05):
    """Assert row count and null rate for a single table."""
    row = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    if row < min_rows:
        raise ValidationError(f"{table}: {row} rows — expected >= {min_rows}")

    nulls = conn.execute(
        text(f"SELECT COUNT(*) FROM {table} WHERE {critical_col} IS NULL")
    ).scalar()
    null_rate = nulls / row if row else 0
    if null_rate > max_null_rate:
        raise ValidationError(
            f"{table}.{critical_col}: null rate {null_rate:.1%} exceeds threshold {max_null_rate:.0%}"
        )

    logger.info(f"  ✓ {table}: {row} rows | {critical_col} null rate {null_rate:.1%}")


# ── DAG task functions ────────────────────────────────────────────────────────

def validate_raw_layer():
    """
    Quality gate after ingest.
    Checks: minimum row counts and null rate on product_id for every raw_* table.
    """
    logger.info("=== Quality gate: raw layer ===")
    checks = {
        "raw_finance": ("product_id", 100),
        "raw_brands":  ("product_id", 100),
        "raw_info":    ("product_id", 100),
        "raw_reviews": ("product_id", 100),
        "raw_traffic": ("product_id", 100),
    }
    with get_connection() as conn:
        for table, (col, min_rows) in checks.items():
            _check_table(conn, table, min_rows, col)

        # Extra: warn if revenue nulls exceed 1 %
        row = conn.execute(text("SELECT COUNT(*) FROM raw_finance")).scalar()
        rev_nulls = conn.execute(
            text("SELECT COUNT(*) FROM raw_finance WHERE modified_revenue IS NULL")
        ).scalar()
        rev_null_rate = rev_nulls / row if row else 0
        if rev_null_rate > 0.01:
            logger.warning(f"raw_finance: {rev_null_rate:.1%} NULL revenue rows")

    logger.info("Raw layer quality gate PASSED")


def dbt_run():
    """
    Execute dbt run + dbt test against the staging and mart models.

    Behaviour
    ─────────
    • PostgreSQL mode (DB_HOST set): runs dbt — this is the production path.
    • SQLite / CI mode: skips gracefully — dbt-postgres does not support SQLite.
    • If dbt binary not found: logs a warning and skips (useful during local dev
      before installing the full requirements).
    """
    if not os.getenv("DB_HOST"):
        logger.info("SQLite mode — dbt step skipped (PostgreSQL required for dbt)")
        return

    if not shutil.which("dbt"):
        logger.warning("dbt binary not found in PATH — skipping dbt step")
        return

    dbt_dir = os.path.join(_PROJECT_ROOT, "dbt")
    base_cmd = [
        "dbt", "--no-use-colors",
        "--project-dir", dbt_dir,
        "--profiles-dir", dbt_dir,
    ]

    for sub in ("run", "test"):
        cmd = base_cmd + [sub]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"dbt {sub} failed (exit {result.returncode})")

    logger.info("dbt run + test completed successfully")


def validate_marts():
    """
    Quality gate after dbt (or aggregate step in SQLite mode).
    Confirms analytics tables exist and contain data.
    """
    logger.info("=== Quality gate: mart/analytics layer ===")

    # In Postgres mode analytics are dbt mart tables; in SQLite mode they are
    # the Python-generated analytics_* tables.  Check whichever exists.
    postgres_mode = bool(os.getenv("DB_HOST"))
    tables = (
        {
            "marts.mart_brand_revenue":   ("brand",   1),
            "marts.mart_product_revenue": ("product_name", 10),
            "marts.mart_monthly_traffic": ("month",   5),
        }
        if postgres_mode
        else {
            "analytics_brand_revenue":   ("brand",   1),
            "analytics_product_revenue": ("product_name", 10),
            "analytics_monthly_traffic": ("month",   5),
        }
    )

    with get_connection() as conn:
        for table, (col, min_rows) in tables.items():
            _check_table(conn, table, min_rows, col)

    logger.info("Mart/analytics quality gate PASSED")


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id           = "retail_pipeline",
    description      = (
        "Retail data platform: incremental ingest → quality gates → "
        "clean → dbt transformations → feature engineering → recommendation model"
    ),
    schedule_interval = "@daily",
    start_date        = datetime(2024, 1, 1),
    catchup           = False,
    default_args      = _default_args,
    tags              = ["retail", "etl", "dbt", "ml", "recommendations"],
) as dag:

    t_ingest = PythonOperator(
        task_id         = "ingest_raw",
        python_callable = ingest_raw,
    )

    t_validate_raw = PythonOperator(
        task_id         = "validate_raw_layer",
        python_callable = validate_raw_layer,
    )

    t_clean = PythonOperator(
        task_id         = "clean_tables",
        python_callable = clean_tables,
    )

    t_aggregate = PythonOperator(
        task_id         = "build_analytics",
        python_callable = build_analytics,
    )

    t_dbt = PythonOperator(
        task_id         = "dbt_run",
        python_callable = dbt_run,
    )

    t_validate_marts = PythonOperator(
        task_id         = "validate_marts",
        python_callable = validate_marts,
    )

    t_features = PythonOperator(
        task_id         = "build_features",
        python_callable = build_features,
    )

    t_similarity = PythonOperator(
        task_id         = "build_similarity_matrix",
        python_callable = build_similarity_matrix,
    )

    # ── Pipeline dependency chain ─────────────────────────────────────────────
    (
        t_ingest
        >> t_validate_raw
        >> t_clean
        >> t_aggregate
        >> t_dbt
        >> t_validate_marts
        >> t_features
        >> t_similarity
    )
