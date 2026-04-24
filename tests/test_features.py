"""
Unit tests for src/features/build_features.py
Uses mocks so no database connection is required.
"""
import numpy as np
import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# ── Synthetic feature data ────────────────────────────────────────────────────

_SYNTHETIC = pd.DataFrame({
    "product_id":    ["P001", "P002", "P003", "P004"],
    "product_name":  ["Shoe A", "Shoe B", "Shoe C", "Shoe D"],
    "brand":         ["Adidas", "Nike", "Adidas", "Nike"],
    "listing_price": [100.0,    150.0,   np.nan,   80.0],
    "discount":      [0.2,      0.0,     0.3,      np.nan],
    "revenue":       [5000.0,   3000.0,  7000.0,   4000.0],
    "rating":        [4.5,      np.nan,  3.8,      4.0],
    "review_count":  [50.0,     30.0,    np.nan,   80.0],
})

# Synthetic data with one corrupted zero-price row (P001)
_SYNTHETIC_WITH_ZERO_PRICE = pd.DataFrame({
    "product_id":    ["P001", "P002", "P003", "P004"],
    "product_name":  ["Shoe A", "Shoe B", "Shoe C", "Shoe D"],
    "brand":         ["Adidas", "Nike", "Adidas", "Nike"],
    "listing_price": [0.0,      150.0,   120.0,    80.0],
    "discount":      [0.2,      0.0,     0.3,      0.1],
    "revenue":       [5000.0,   3000.0,  7000.0,   4000.0],
    "rating":        [4.5,      4.0,     3.8,      4.0],
    "review_count":  [50.0,     30.0,    40.0,     80.0],
})


def _run_build_features(source_df=None):
    """
    Run build_features() with all DB and IO calls mocked out.
    Accepts an optional source DataFrame to feed into pd.read_sql.
    """
    from src.features.build_features import build_features

    if source_df is None:
        source_df = _SYNTHETIC.copy()

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    with patch("src.features.build_features.get_connection", return_value=mock_conn), \
         patch("src.features.build_features.pd.read_sql", return_value=source_df), \
         patch("pandas.DataFrame.to_sql"), \
         patch("pandas.DataFrame.to_csv"), \
         patch("os.makedirs"):
        return build_features()


# ── Column structure ──────────────────────────────────────────────────────────

def test_all_required_columns_present():
    result = _run_build_features()
    required = {
        "product_id", "product_name", "brand", "brand_encoded",
        "listing_price", "discount", "revenue", "rating", "review_count",
    }
    assert required.issubset(set(result.columns))


def test_no_extra_unexpected_columns():
    result = _run_build_features()
    allowed = {
        "product_id", "product_name", "brand", "brand_encoded",
        "listing_price", "discount", "revenue", "rating", "review_count",
    }
    assert set(result.columns) == allowed


# ── Null imputation ───────────────────────────────────────────────────────────

def test_rating_nulls_imputed():
    result = _run_build_features()
    assert result["rating"].isnull().sum() == 0


def test_review_count_nulls_imputed_with_median():
    # NaN review_count should be filled with the column median, not 0
    result = _run_build_features()
    assert result["review_count"].isnull().sum() == 0
    # median of [50, 30, 80] = 50; NaN row (P003) should become 50, not 0
    p003_reviews = result.loc[result["product_id"] == "P003", "review_count"].iloc[0]
    assert p003_reviews == pytest.approx(50.0)


def test_listing_price_nulls_imputed_with_median():
    # NaN listing_price should be filled with the column median, not 0
    result = _run_build_features()
    assert result["listing_price"].isnull().sum() == 0
    # median of [100, 150, 80] = 100; NaN row (P003) should become 100, not 0
    p003_price = result.loc[result["product_id"] == "P003", "listing_price"].iloc[0]
    assert p003_price == pytest.approx(100.0)


def test_discount_nulls_filled_with_zero():
    result = _run_build_features()
    assert result["discount"].isnull().sum() == 0


# ── Zero-price row removal ────────────────────────────────────────────────────

def test_listing_price_zero_rows_are_dropped():
    # Rows with listing_price=0 are corrupted and must not appear in output
    result = _run_build_features(_SYNTHETIC_WITH_ZERO_PRICE.copy())
    assert len(result) == len(_SYNTHETIC_WITH_ZERO_PRICE) - 1
    assert (result["listing_price"] > 0).all()


def test_listing_price_never_zero_in_normal_data():
    result = _run_build_features()
    assert (result["listing_price"] > 0).all()


# ── Brand encoding ────────────────────────────────────────────────────────────

def test_brand_encoded_column_is_integer():
    result = _run_build_features()
    assert np.issubdtype(result["brand_encoded"].dtype, np.integer)


def test_brand_encoded_has_two_distinct_values():
    result = _run_build_features()
    assert result["brand_encoded"].nunique() == 2


def test_same_brand_gets_same_encoding():
    result = _run_build_features()
    adidas_codes = result[result["brand"] == "Adidas"]["brand_encoded"].unique()
    assert len(adidas_codes) == 1


# ── Row integrity ─────────────────────────────────────────────────────────────

def test_row_count_matches_input_when_no_zero_prices():
    # All rows in _SYNTHETIC have valid prices — none should be dropped
    result = _run_build_features()
    assert len(result) == len(_SYNTHETIC)


def test_no_nulls_in_product_id():
    result = _run_build_features()
    assert result["product_id"].isnull().sum() == 0


def test_no_nulls_in_product_name():
    result = _run_build_features()
    assert result["product_name"].isnull().sum() == 0
