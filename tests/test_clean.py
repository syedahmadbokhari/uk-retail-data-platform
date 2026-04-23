"""
Unit tests for src/etl/clean.py
Tests pure transformation functions — no database required.
"""
import pandas as pd
import numpy as np
import pytest

from src.etl.clean import (
    _clean_finance,
    _clean_brands,
    _clean_info,
    _clean_reviews,
    _clean_traffic,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _finance_row(**overrides):
    base = {
        "product_id":            "P001",
        "modified_listing_price": 100.0,
        "modified_sale_price":    80.0,
        "modified_discount":      0.2,
        "modified_revenue":       500.0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def _reviews_row(**overrides):
    base = {"product_id": "P001", "real_rating": "4,0", "real_reviews": 10.0}
    base.update(overrides)
    return pd.DataFrame([base])


# ── Finance tests ─────────────────────────────────────────────────────────────

def test_negative_revenue_is_dropped():
    df = _finance_row(modified_revenue=-100.0)
    assert len(_clean_finance(df)) == 0


def test_zero_revenue_is_kept():
    df = _finance_row(modified_revenue=0.0)
    assert len(_clean_finance(df)) == 1


def test_null_product_id_dropped_in_finance():
    df = pd.concat([_finance_row(product_id=None), _finance_row()], ignore_index=True)
    result = _clean_finance(df)
    assert len(result) == 1
    assert result["product_id"].iloc[0] == "P001"


def test_null_revenue_dropped():
    df = _finance_row(modified_revenue=None)
    assert len(_clean_finance(df)) == 0


def test_discount_above_one_is_clipped():
    df = _finance_row(modified_discount=1.8)
    result = _clean_finance(df)
    assert result["modified_discount"].iloc[0] == pytest.approx(1.0)


def test_discount_below_zero_is_clipped():
    df = _finance_row(modified_discount=-0.5)
    result = _clean_finance(df)
    assert result["modified_discount"].iloc[0] == pytest.approx(0.0)


def test_null_listing_price_filled_with_zero():
    df = _finance_row(modified_listing_price=None)
    result = _clean_finance(df)
    assert result["modified_listing_price"].iloc[0] == pytest.approx(0.0)


def test_finance_output_columns():
    result = _clean_finance(_finance_row())
    expected = {
        "product_id", "modified_listing_price", "modified_sale_price",
        "modified_discount", "modified_revenue",
    }
    assert expected == set(result.columns)


# ── Reviews tests ─────────────────────────────────────────────────────────────

def test_european_decimal_converted_to_float():
    """Core data-quality bug fix: "3,3" must become 3.3, not 3.0."""
    result = _clean_reviews(_reviews_row(real_rating="3,3"))
    assert result["real_rating"].iloc[0] == pytest.approx(3.3)


def test_dot_decimal_rating_unchanged():
    result = _clean_reviews(_reviews_row(real_rating="4.5"))
    assert result["real_rating"].iloc[0] == pytest.approx(4.5)


def test_integer_string_rating_parsed():
    result = _clean_reviews(_reviews_row(real_rating="4"))
    assert result["real_rating"].iloc[0] == pytest.approx(4.0)


def test_rating_above_five_clipped():
    result = _clean_reviews(_reviews_row(real_rating="6.0"))
    assert result["real_rating"].iloc[0] == pytest.approx(5.0)


def test_rating_below_zero_clipped():
    result = _clean_reviews(_reviews_row(real_rating="-1.0"))
    assert result["real_rating"].iloc[0] == pytest.approx(0.0)


def test_empty_product_id_rows_removed():
    df = pd.DataFrame({
        "product_id":    ["P1", "", "  "],
        "real_rating":   ["3,5", "4,0", "2,0"],
        "real_reviews":  [10.0, 5.0, 3.0],
    })
    result = _clean_reviews(df)
    assert len(result) == 1
    assert result["product_id"].iloc[0] == "P1"


def test_null_review_product_id_dropped():
    df = pd.DataFrame({
        "product_id":   [None],
        "real_rating":  ["3,5"],
        "real_reviews": [10.0],
    })
    assert len(_clean_reviews(df)) == 0


def test_negative_review_count_clipped():
    result = _clean_reviews(_reviews_row(real_reviews=-5.0))
    assert result["real_reviews"].iloc[0] == pytest.approx(0.0)


# ── Brands tests ──────────────────────────────────────────────────────────────

def test_null_brand_is_dropped():
    df = pd.DataFrame({"product_id": ["P1", "P2"], "modified_brand": [None, "Adidas"]})
    result = _clean_brands(df)
    assert len(result) == 1
    assert result["modified_brand"].iloc[0] == "Adidas"


def test_brand_name_is_title_cased():
    df = pd.DataFrame({"product_id": ["P1"], "modified_brand": ["  adidas  "]})
    result = _clean_brands(df)
    assert result["modified_brand"].iloc[0] == "Adidas"


def test_brand_whitespace_stripped():
    df = pd.DataFrame({"product_id": ["P1"], "modified_brand": ["  Nike  "]})
    result = _clean_brands(df)
    assert result["modified_brand"].iloc[0] == "Nike"


# ── Info tests ────────────────────────────────────────────────────────────────

def test_null_product_name_dropped():
    df = pd.DataFrame({
        "product_id":            ["P1", "P2"],
        "modified_product_name": [None, "Shoe A"],
        "modified_description":  ["", "desc"],
    })
    result = _clean_info(df)
    assert len(result) == 1
    assert result["modified_product_name"].iloc[0] == "Shoe A"


def test_product_name_whitespace_stripped():
    df = pd.DataFrame({
        "product_id":            ["P1"],
        "modified_product_name": ["  Ultraboost  "],
        "modified_description":  ["desc"],
    })
    result = _clean_info(df)
    assert result["modified_product_name"].iloc[0] == "Ultraboost"


def test_null_description_filled_with_empty_string():
    df = pd.DataFrame({
        "product_id":            ["P1"],
        "modified_product_name": ["Shoe"],
        "modified_description":  [None],
    })
    result = _clean_info(df)
    assert result["modified_description"].iloc[0] == ""


# ── Traffic tests ─────────────────────────────────────────────────────────────

def test_null_traffic_date_dropped():
    df = pd.DataFrame({
        "product_id":            ["P1", "P2", "P3"],
        "modified_last_visited": [None, "2019-01-01 10:00:00", ""],
    })
    result = _clean_traffic(df)
    assert len(result) == 1
    assert result["product_id"].iloc[0] == "P2"


def test_valid_traffic_row_kept():
    df = pd.DataFrame({
        "product_id":            ["P1"],
        "modified_last_visited": ["2019-06-15 14:30:00"],
    })
    result = _clean_traffic(df)
    assert len(result) == 1
