"""
Unit tests for the Snowflake backend: src/etl/snowflake_setup.py,
src/etl/migrate_to_snowflake.py, and src/analysis/snowflake_cost_comparison.py.

No live Snowflake account required — mirrors tests/test_bigquery_setup.py's
approach exactly: the snowflake.connector client is mocked (unittest.mock,
same convention as the rest of tests/), while build_fact_table_ddl()/
build_mart_table_ddl() are pure string-building functions tested directly,
with no mocking needed, same spirit as BigQuery's build_fact_table() tests
(just adapted to Snowflake's DDL-string API instead of BigQuery's
Table/SchemaField objects — see snowflake_setup.py's module docstring for
why the two platforms' APIs differ here).
"""
import datetime
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.etl.snowflake_setup import (
    build_fact_table_ddl,
    build_mart_table_ddl,
    create_database_and_schema,
    create_fact_table,
    create_mart_tables,
    FACT_TABLE,
    MART_COLUMNS,
)
from src.etl.migrate_to_snowflake import migrate_fact_table, migrate_mart_tables, _load_dataframe
from src.analysis.snowflake_cost_comparison import (
    compare, estimated_cost_usd_range, _representative_date_window, WAREHOUSE_CREDITS_PER_HOUR,
)
from src.utils.db import get_snowflake_connection, is_snowflake_enabled

_DATABASE = "TEST_DB"
_SCHEMA = "PUBLIC"


# ── DDL string construction (no mocking needed — pure functions) ─────────────

def test_fact_table_ddl_clusters_on_event_timestamp_and_product_id():
    ddl = build_fact_table_ddl(_DATABASE, _SCHEMA)
    assert "CLUSTER BY (event_timestamp, product_id)" in ddl


def test_fact_table_ddl_contains_all_expected_columns():
    ddl = build_fact_table_ddl(_DATABASE, _SCHEMA)
    for col in ("event_id", "product_id", "price", "discount", "quantity", "revenue",
                "event_timestamp", "ingested_at"):
        assert col in ddl


def test_fact_table_ddl_is_idempotent_create():
    ddl = build_fact_table_ddl(_DATABASE, _SCHEMA)
    assert "CREATE TABLE IF NOT EXISTS" in ddl
    assert f"{_DATABASE}.{_SCHEMA}.{FACT_TABLE}" in ddl


def test_mart_table_ddl_has_no_clustering_key():
    for table_name in MART_COLUMNS:
        ddl = build_mart_table_ddl(_DATABASE, _SCHEMA, table_name)
        assert "CLUSTER BY" not in ddl


# ── create_* functions call the (mocked) connection correctly ───────────────

def test_create_database_and_schema_uses_if_not_exists():
    mock_conn = MagicMock()
    create_database_and_schema(conn=mock_conn)

    executed = [call.args[0] for call in mock_conn.cursor.return_value.execute.call_args_list]
    assert any("CREATE DATABASE IF NOT EXISTS" in sql for sql in executed)
    assert any("CREATE SCHEMA IF NOT EXISTS" in sql for sql in executed)


def test_create_fact_table_executes_clustered_ddl():
    mock_conn = MagicMock()
    create_fact_table(conn=mock_conn)

    executed_sql = mock_conn.cursor.return_value.execute.call_args.args[0]
    assert "CLUSTER BY (event_timestamp, product_id)" in executed_sql


def test_create_mart_tables_creates_all_five_marts():
    mock_conn = MagicMock()
    create_mart_tables(conn=mock_conn)
    assert mock_conn.cursor.return_value.execute.call_count == len(MART_COLUMNS)


# ── Migration — overwrite semantics + idempotency ────────────────────────────

def test_load_dataframe_uses_overwrite_true_for_idempotency():
    mock_conn = MagicMock()
    df = pd.DataFrame({"a": [1, 2]})

    with patch("src.etl.migrate_to_snowflake.write_pandas", return_value=(True, 1, 2, [])) as mock_write:
        _load_dataframe(mock_conn, "some_table", df)

    _, kwargs = mock_write.call_args
    assert kwargs.get("overwrite") is True


def test_load_dataframe_skips_empty_dataframe_without_calling_write_pandas():
    mock_conn = MagicMock()
    with patch("src.etl.migrate_to_snowflake.write_pandas") as mock_write:
        _load_dataframe(mock_conn, "some_table", pd.DataFrame())
    mock_write.assert_not_called()


def test_migrate_fact_table_is_idempotent_across_repeated_runs():
    """Running the same load twice must not accumulate rows — both calls use overwrite=True."""
    mock_conn = MagicMock()
    fact_df = pd.DataFrame({
        "event_id": ["e1"], "product_id": ["P1"], "price": [10.0], "discount": [0.0],
        "quantity": [1], "revenue": [10.0], "event_timestamp": ["2026-01-01"], "ingested_at": ["2026-01-01"],
    })
    mock_source_conn = MagicMock()
    mock_source_conn.__enter__ = MagicMock(return_value=mock_source_conn)
    mock_source_conn.__exit__ = MagicMock(return_value=False)

    with patch("src.etl.migrate_to_snowflake.get_connection", return_value=mock_source_conn), \
         patch("src.etl.migrate_to_snowflake.pd.read_sql", return_value=fact_df), \
         patch("src.etl.migrate_to_snowflake.write_pandas", return_value=(True, 1, 1, [])) as mock_write:
        migrate_fact_table(conn=mock_conn)
        migrate_fact_table(conn=mock_conn)

    assert mock_write.call_count == 2
    for call in mock_write.call_args_list:
        assert call.kwargs["overwrite"] is True


def test_migrate_mart_tables_skips_marts_not_yet_built_locally():
    mock_conn = MagicMock()
    mock_source_conn = MagicMock()
    mock_source_conn.__enter__ = MagicMock(return_value=mock_source_conn)
    mock_source_conn.__exit__ = MagicMock(return_value=False)

    with patch("src.etl.migrate_to_snowflake.get_connection", return_value=mock_source_conn), \
         patch("src.etl.migrate_to_snowflake.pd.read_sql", side_effect=Exception("no such table")), \
         patch("src.etl.migrate_to_snowflake.write_pandas") as mock_write:
        counts = migrate_mart_tables(conn=mock_conn)

    assert all(count == 0 for count in counts.values())
    mock_write.assert_not_called()


# ── db.py Snowflake helpers ───────────────────────────────────────────────────

def test_is_snowflake_enabled_false_without_env_var(monkeypatch):
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)
    assert is_snowflake_enabled() is False


def test_is_snowflake_enabled_true_with_env_var(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "test-account")
    assert is_snowflake_enabled() is True


def test_get_snowflake_connection_raises_clear_error_without_account(monkeypatch):
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)
    with pytest.raises(RuntimeError):
        get_snowflake_connection()


# ── Cost comparison — graceful skip, and correct cost math ───────────────────

def test_compare_skips_cleanly_when_snowflake_not_enabled(monkeypatch):
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)
    assert compare() is None


def test_compare_skips_cleanly_when_credentials_unavailable(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "test-account")
    with patch("src.analysis.snowflake_cost_comparison.get_snowflake_connection", side_effect=Exception("no credentials")):
        assert compare() is None


def test_estimated_cost_usd_range_is_ordered_low_to_high():
    low, high = estimated_cost_usd_range(elapsed_ms=3_600_000)  # exactly 1 hour
    assert low < high


def test_estimated_cost_usd_range_matches_known_one_hour_x_small_cost():
    # 1 hour at X-Small (1 credit/hour) = 1 credit -> $2 to $4 at the cited on-demand range.
    low, high = estimated_cost_usd_range(elapsed_ms=3_600_000, credits_per_hour=WAREHOUSE_CREDITS_PER_HOUR)
    assert low == pytest.approx(2.0)
    assert high == pytest.approx(4.0)


def test_estimated_cost_usd_range_is_zero_for_zero_elapsed_time():
    low, high = estimated_cost_usd_range(elapsed_ms=0)
    assert low == 0.0
    assert high == 0.0


# ── Representative date window — anchored to real data, not datetime.today() ─
#
# Same regression class as bigquery_cost_comparison.py's equivalent test —
# applying that fix from day one here rather than reintroducing the bug.

def test_representative_date_window_ends_at_the_tables_actual_max_date():
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.fetchone.return_value = (datetime.datetime(2026, 4, 30, 12, 0, 0),)
    start, end = _representative_date_window(mock_conn, "db.schema.fact_sales_events")
    assert end == "2026-04-30"


def test_representative_date_window_spans_requested_number_of_days():
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.fetchone.return_value = (datetime.date(2026, 4, 30),)
    start, end = _representative_date_window(mock_conn, "db.schema.fact_sales_events", window_days=3)
    assert start == "2026-04-28"
    assert end == "2026-04-30"
