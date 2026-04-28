"""
Unit tests for src/clustering.py  build_clusters()
Uses mocks so no database connection is required.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# ── Synthetic data — three well-separated groups so k=3 is unambiguous ────────

_SYNTHETIC = pd.DataFrame({
    "product_id":    [f"P{i:03d}" for i in range(1, 13)],
    "product_name":  [f"Product {i}" for i in range(1, 13)],
    "brand":         ["Adidas", "Nike"] * 6,
    "brand_encoded": [0, 1] * 6,
    # Low cluster: low price + low revenue
    # Mid cluster: mid price + mid revenue
    # Premium cluster: high price + high revenue
    "listing_price": [50, 55, 52, 105, 110, 108, 200, 205, 210, 195, 198, 202],
    "discount":      [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    "revenue":       [2.0, 2.1, 1.9, 5.0, 5.1, 4.9, 9.0, 9.1, 8.9, 8.8, 9.2, 9.0],
    "rating":        [3.0, 3.1, 2.9, 4.0, 4.1, 3.9, 4.9, 5.0, 4.8, 4.7, 5.0, 4.9],
    "review_count":  [10,  11,  9,   50,  51,  49,  100, 101, 99,  98,  102, 100],
})


def _run_build_clusters(source_df=None):
    """Run build_clusters() with all DB and IO calls mocked out."""
    from src.clustering import build_clusters

    if source_df is None:
        source_df = _SYNTHETIC.copy()

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__  = MagicMock(return_value=False)

    with patch("src.clustering.get_connection", return_value=mock_conn), \
         patch("src.clustering.pd.read_sql", return_value=source_df), \
         patch("pandas.DataFrame.to_sql"):
        return build_clusters()


# ── Output schema ─────────────────────────────────────────────────────────────

def test_output_has_required_columns():
    result = _run_build_clusters()
    required = {"product_id", "cluster_label", "cluster_id", "log_revenue"}
    assert required.issubset(set(result.columns))


def test_output_has_no_extra_columns():
    result = _run_build_clusters()
    assert set(result.columns) == {"product_id", "cluster_label", "cluster_id", "log_revenue"}


# ── Cluster label correctness ─────────────────────────────────────────────────

def test_exactly_three_unique_cluster_labels():
    result = _run_build_clusters()
    assert result["cluster_label"].nunique() == 3


def test_cluster_labels_are_valid_business_names():
    result = _run_build_clusters()
    valid  = {"Low Performer", "Mid Tier", "Premium"}
    assert set(result["cluster_label"].unique()).issubset(valid)


def test_cluster_id_values_are_zero_one_two():
    result = _run_build_clusters()
    assert set(result["cluster_id"].unique()) == {0, 1, 2}


def test_cluster_id_zero_has_lowest_mean_log_revenue():
    """cluster_id=0 must map to 'Low Performer' — lowest mean revenue."""
    result = _run_build_clusters()
    mean_by_id = result.groupby("cluster_id")["log_revenue"].mean()
    assert mean_by_id[0] < mean_by_id[1] < mean_by_id[2]


def test_cluster_id_two_has_highest_mean_log_revenue():
    result = _run_build_clusters()
    mean_by_id = result.groupby("cluster_id")["log_revenue"].mean()
    assert mean_by_id[2] == mean_by_id.max()


# ── Row integrity ─────────────────────────────────────────────────────────────

def test_no_duplicate_product_ids():
    result = _run_build_clusters()
    assert result["product_id"].duplicated().sum() == 0


def test_output_row_count_matches_input():
    result = _run_build_clusters()
    assert len(result) == len(_SYNTHETIC)


def test_no_nulls_in_cluster_label():
    result = _run_build_clusters()
    assert result["cluster_label"].isnull().sum() == 0


# ── Empty input handling ──────────────────────────────────────────────────────

def test_empty_input_returns_empty_dataframe():
    empty_df = pd.DataFrame(columns=_SYNTHETIC.columns)
    result   = _run_build_clusters(empty_df)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_empty_input_does_not_write_to_db():
    from src.clustering import build_clusters

    empty_df   = pd.DataFrame(columns=_SYNTHETIC.columns)
    mock_conn  = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__  = MagicMock(return_value=False)

    with patch("src.clustering.get_connection", return_value=mock_conn), \
         patch("src.clustering.pd.read_sql", return_value=empty_df), \
         patch("pandas.DataFrame.to_sql") as mock_to_sql:
        build_clusters()

    mock_to_sql.assert_not_called()


# ── Determinism ───────────────────────────────────────────────────────────────

def test_clustering_is_deterministic():
    """Same input with random_state=42 must produce identical cluster assignments."""
    result1 = _run_build_clusters(_SYNTHETIC.copy())
    result2 = _run_build_clusters(_SYNTHETIC.copy())
    merged = result1.sort_values("product_id").reset_index(drop=True).merge(
        result2.sort_values("product_id").reset_index(drop=True),
        on="product_id", suffixes=("_a", "_b"),
    )
    assert (merged["cluster_label_a"] == merged["cluster_label_b"]).all()
