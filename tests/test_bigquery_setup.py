"""
Unit tests for the BigQuery backend: src/etl/bigquery_setup.py,
src/etl/migrate_to_bigquery.py, and src/analysis/bigquery_cost_comparison.py.

No live BigQuery connection required. The google-cloud-bigquery client is
mocked throughout — this repo has no existing AWS/moto-style mocking
precedent to follow, so these tests use the same unittest.mock.patch +
MagicMock approach already used for the SQLAlchemy connection everywhere
else in tests/ (see test_clustering.py, test_recommender.py).

Table/SchemaField/TimePartitioning objects from build_fact_table() are real
google-cloud-bigquery objects, not mocks — only the network-calling Client
methods (create_table, create_dataset, query, load_table_from_dataframe)
are mocked.
"""
import datetime
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock, ANY
from google.cloud import bigquery

from src.etl.bigquery_setup import (
    build_fact_table,
    create_dataset,
    create_fact_table,
    create_mart_tables,
    FACT_TABLE,
    MART_SCHEMAS,
)
from src.etl.migrate_to_bigquery import migrate_fact_table, migrate_mart_tables, _load_dataframe
from src.analysis.bigquery_cost_comparison import (
    compare, estimated_cost_usd, BYTES_PER_TIB, _representative_date_window,
)
from src.utils.db import get_bigquery_client, is_bigquery_enabled

_DATASET_ID = "test-project.retail_analytics"


# ── Partitioning + clustering configuration (no mocking needed — these are
#    plain local objects, not API calls) ─────────────────────────────────────

def test_fact_table_is_partitioned_by_day_on_event_timestamp():
    table = build_fact_table(_DATASET_ID)
    assert table.time_partitioning.type_ == bigquery.TimePartitioningType.DAY
    assert table.time_partitioning.field == "event_timestamp"


def test_fact_table_is_clustered_on_product_id():
    table = build_fact_table(_DATASET_ID)
    assert table.clustering_fields == ["product_id"]


def test_fact_table_schema_matches_source_columns():
    table = build_fact_table(_DATASET_ID)
    field_names = {f.name for f in table.schema}
    assert field_names == {
        "event_id", "product_id", "price", "discount",
        "quantity", "revenue", "event_timestamp", "ingested_at",
    }


# ── create_* functions call the (mocked) client correctly, idempotently ──────

def test_create_dataset_uses_exists_ok_for_idempotency():
    mock_client = MagicMock(project="test-project")
    create_dataset(client=mock_client)
    _, kwargs = mock_client.create_dataset.call_args
    assert kwargs.get("exists_ok") is True


def test_create_fact_table_passes_partitioned_clustered_table_to_client():
    mock_client = MagicMock(project="test-project")
    create_fact_table(client=mock_client, dataset_id=_DATASET_ID)

    args, kwargs = mock_client.create_table.call_args
    created_table = args[0]
    assert created_table.time_partitioning.field == "event_timestamp"
    assert created_table.clustering_fields == ["product_id"]
    assert kwargs.get("exists_ok") is True


def test_create_mart_tables_creates_all_five_marts():
    mock_client = MagicMock(project="test-project")
    create_mart_tables(client=mock_client, dataset_id=_DATASET_ID)
    assert mock_client.create_table.call_count == len(MART_SCHEMAS)


def test_mart_tables_have_no_partitioning_or_clustering():
    mock_client = MagicMock(project="test-project")
    create_mart_tables(client=mock_client, dataset_id=_DATASET_ID)
    for call in mock_client.create_table.call_args_list:
        table = call.args[0]
        assert table.time_partitioning is None
        assert table.clustering_fields is None


# ── Migration — write disposition + idempotency ──────────────────────────────

def test_load_dataframe_uses_write_truncate_for_idempotency():
    mock_client = MagicMock(project="test-project")
    df = pd.DataFrame({"a": [1, 2]})

    _load_dataframe(mock_client, _DATASET_ID, "some_table", df)

    _, kwargs = mock_client.load_table_from_dataframe.call_args
    job_config = kwargs["job_config"]
    assert job_config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE


def test_load_dataframe_skips_empty_dataframe_without_calling_client():
    mock_client = MagicMock(project="test-project")
    _load_dataframe(mock_client, _DATASET_ID, "some_table", pd.DataFrame())
    mock_client.load_table_from_dataframe.assert_not_called()


def test_migrate_fact_table_is_idempotent_across_repeated_runs():
    """Running the same load twice must not accumulate rows — both calls use WRITE_TRUNCATE."""
    mock_client = MagicMock(project="test-project")
    fact_df = pd.DataFrame({
        "event_id": ["e1"], "product_id": ["P1"], "price": [10.0], "discount": [0.0],
        "quantity": [1], "revenue": [10.0], "event_timestamp": ["2026-01-01"], "ingested_at": ["2026-01-01"],
    })
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    with patch("src.etl.migrate_to_bigquery.get_connection", return_value=mock_conn), \
         patch("src.etl.migrate_to_bigquery.pd.read_sql", return_value=fact_df):
        migrate_fact_table(client=mock_client, dataset_id=_DATASET_ID)
        migrate_fact_table(client=mock_client, dataset_id=_DATASET_ID)

    assert mock_client.load_table_from_dataframe.call_count == 2
    for call in mock_client.load_table_from_dataframe.call_args_list:
        assert call.kwargs["job_config"].write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE


def test_migrate_mart_tables_skips_marts_not_yet_built_locally():
    mock_client = MagicMock(project="test-project")
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    with patch("src.etl.migrate_to_bigquery.get_connection", return_value=mock_conn), \
         patch("src.etl.migrate_to_bigquery.pd.read_sql", side_effect=Exception("no such table")):
        counts = migrate_mart_tables(client=mock_client, dataset_id=_DATASET_ID)

    assert all(count == 0 for count in counts.values())
    mock_client.load_table_from_dataframe.assert_not_called()


# ── db.py BigQuery helpers ────────────────────────────────────────────────────

def test_is_bigquery_enabled_false_without_env_var(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    assert is_bigquery_enabled() is False


def test_is_bigquery_enabled_true_with_env_var(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    assert is_bigquery_enabled() is True


def test_get_bigquery_client_raises_clear_error_without_project(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(RuntimeError):
        get_bigquery_client()


# ── Cost comparison — graceful skip, and correct cost math ───────────────────

def test_compare_skips_cleanly_when_bigquery_not_enabled(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    assert compare() is None


def test_compare_skips_cleanly_when_credentials_unavailable(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    with patch("src.analysis.bigquery_cost_comparison.get_bigquery_client", side_effect=Exception("no credentials")):
        assert compare() is None


def test_estimated_cost_usd_matches_published_rate_at_one_tib():
    assert estimated_cost_usd(BYTES_PER_TIB) == pytest.approx(6.25)


def test_estimated_cost_usd_is_zero_for_zero_bytes():
    assert estimated_cost_usd(0) == 0.0


# ── Representative date window — anchored to real data, not datetime.today() ─
#
# Regression coverage for a real bug: the window used to be computed from
# datetime.date.today(), which silently stopped overlapping this static
# dataset's actual dates once enough real time had passed, producing a
# 0-bytes-scanned result that looked like a clean partitioning win but was
# actually just a date filter matching zero partitions.

def test_representative_date_window_ends_at_the_tables_actual_max_date():
    mock_client = MagicMock()
    mock_client.query.return_value.result.return_value = [
        MagicMock(max_date=datetime.date(2026, 4, 30))
    ]
    start, end = _representative_date_window(mock_client, "proj.ds.fact_sales_events")
    assert end == "2026-04-30"


def test_representative_date_window_spans_requested_number_of_days():
    mock_client = MagicMock()
    mock_client.query.return_value.result.return_value = [
        MagicMock(max_date=datetime.date(2026, 4, 30))
    ]
    start, end = _representative_date_window(mock_client, "proj.ds.fact_sales_events", window_days=3)
    assert start == "2026-04-28"   # 28, 29, 30 -> 3 days inclusive
    assert end == "2026-04-30"


def test_representative_date_window_never_uses_todays_date(monkeypatch):
    """The window must come from the table's own data, not wall-clock time."""
    monkeypatch.setattr(
        "src.analysis.bigquery_cost_comparison.datetime",
        MagicMock(date=MagicMock(today=MagicMock(side_effect=AssertionError("must not call datetime.date.today()"))),
                  timedelta=datetime.timedelta),
    )
    mock_client = MagicMock()
    mock_client.query.return_value.result.return_value = [
        MagicMock(max_date=datetime.date(2020, 1, 10))
    ]
    start, end = _representative_date_window(mock_client, "proj.ds.fact_sales_events")
    assert end == "2020-01-10"
