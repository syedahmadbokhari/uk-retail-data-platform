"""
Retail Data Platform — Airflow DAG
===================================
Wraps the existing ETL functions into a daily Airflow pipeline.

DAG structure:
    ingest_raw → clean_tables → build_analytics → build_features → build_similarity_matrix

Prerequisites:
    pip install apache-airflow
    export AIRFLOW_HOME=~/airflow
    airflow db init
    airflow scheduler &
    airflow webserver &

Then place (or symlink) this file into $AIRFLOW_HOME/dags/.
Set DB_HOST / DB_USER / DB_PASSWORD / DB_NAME if using PostgreSQL,
otherwise the pipeline falls back to the local SQLite file.
"""

import os
import sys
from datetime import datetime, timedelta

# ── Make sure project root is on sys.path so src.* imports resolve ──────────
_DAG_DIR      = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_DAG_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.etl.ingest            import ingest_raw
from src.etl.clean             import clean_tables
from src.etl.aggregate         import build_analytics
from src.features.build_features import build_features
from src.recommender           import build_similarity_matrix

# ── Default task arguments ───────────────────────────────────────────────────
default_args = {
    "owner":          "ahmad",
    "retries":        1,
    "retry_delay":    timedelta(minutes=5),
    "email_on_failure": False,
}

# ── DAG definition ───────────────────────────────────────────────────────────
with DAG(
    dag_id          = "retail_pipeline",
    description     = "End-to-end retail data platform: ETL → features → recommendation model",
    schedule_interval = "@daily",
    start_date      = datetime(2024, 1, 1),
    catchup         = False,
    default_args    = default_args,
    tags            = ["retail", "etl", "ml", "recommendations"],
) as dag:

    t_ingest = PythonOperator(
        task_id         = "ingest_raw",
        python_callable = ingest_raw,
    )

    t_clean = PythonOperator(
        task_id         = "clean_tables",
        python_callable = clean_tables,
    )

    t_aggregate = PythonOperator(
        task_id         = "build_analytics",
        python_callable = build_analytics,
    )

    t_features = PythonOperator(
        task_id         = "build_features",
        python_callable = build_features,
    )

    t_similarity = PythonOperator(
        task_id         = "build_similarity_matrix",
        python_callable = build_similarity_matrix,
    )

    # ── Pipeline dependency chain ────────────────────────────────────────────
    t_ingest >> t_clean >> t_aggregate >> t_features >> t_similarity
