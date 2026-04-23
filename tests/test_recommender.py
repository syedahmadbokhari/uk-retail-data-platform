"""
Unit tests for src/recommender.py  get_recommendations()
Pure function — no database or file I/O required.
"""
import numpy as np
import pandas as pd
import pytest

from src.recommender import get_recommendations

# ── Synthetic fixtures ────────────────────────────────────────────────────────

FEAT_DF = pd.DataFrame({
    "product_id":    ["P001", "P002", "P003", "P004", "P005"],
    "product_name":  ["Shoe A", "Shoe B", "Shoe C", "Shoe D", "Shoe E"],
    "brand":         ["Adidas", "Nike",   "Adidas", "Nike",   "Adidas"],
    "listing_price": [100.0,    150.0,    105.0,    200.0,    90.0],
    "rating":        [4.5,      3.8,      4.4,      3.0,      4.6],
    "revenue":       [5000.0,   3000.0,   4800.0,   2000.0,   5200.0],
    "review_count":  [50.0,     30.0,     48.0,     20.0,     55.0],
})

# Hand-crafted similarity matrix:
#   P001 is most similar to P003 (0.90), then P005 (0.85)
#   P002 is most similar to P004 (0.70)
SIM = np.array([
    [1.00, 0.50, 0.90, 0.20, 0.85],   # P001
    [0.50, 1.00, 0.40, 0.70, 0.30],   # P002
    [0.90, 0.40, 1.00, 0.10, 0.88],   # P003
    [0.20, 0.70, 0.10, 1.00, 0.15],   # P004
    [0.85, 0.30, 0.88, 0.15, 1.00],   # P005
])


# ── Shape and structure ───────────────────────────────────────────────────────

def test_returns_correct_number_of_results():
    result = get_recommendations("P001", FEAT_DF, SIM, top_n=3)
    assert len(result) == 3


def test_returns_dataframe():
    result = get_recommendations("P001", FEAT_DF, SIM, top_n=2)
    assert isinstance(result, pd.DataFrame)


def test_similarity_score_column_present():
    result = get_recommendations("P001", FEAT_DF, SIM, top_n=3)
    assert "similarity_score" in result.columns


def test_product_name_column_present():
    result = get_recommendations("P001", FEAT_DF, SIM, top_n=3)
    assert "product_name" in result.columns


def test_brand_column_present():
    result = get_recommendations("P001", FEAT_DF, SIM, top_n=3)
    assert "brand" in result.columns


# ── Self-exclusion ────────────────────────────────────────────────────────────

def test_query_product_not_in_results():
    result = get_recommendations("P001", FEAT_DF, SIM, top_n=4)
    assert "Shoe A" not in result["product_name"].values


def test_query_product_excluded_even_when_top_n_equals_all():
    """With 5 products and top_n=4, all 4 *other* products returned — not self."""
    result = get_recommendations("P002", FEAT_DF, SIM, top_n=4)
    assert len(result) == 4
    assert "Shoe B" not in result["product_name"].values


# ── Ordering ──────────────────────────────────────────────────────────────────

def test_results_sorted_by_similarity_descending():
    result = get_recommendations("P001", FEAT_DF, SIM, top_n=4)
    scores = result["similarity_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_top_recommendation_is_most_similar():
    """P003 has similarity 0.90 to P001 — should be ranked first."""
    result = get_recommendations("P001", FEAT_DF, SIM, top_n=1)
    assert result.iloc[0]["product_name"] == "Shoe C"
    assert result.iloc[0]["similarity_score"] == pytest.approx(0.90)


def test_second_recommendation_correct():
    """P005 has similarity 0.85 to P001 — should be ranked second."""
    result = get_recommendations("P001", FEAT_DF, SIM, top_n=2)
    assert result.iloc[1]["product_name"] == "Shoe E"
    assert result.iloc[1]["similarity_score"] == pytest.approx(0.85)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_unknown_product_returns_empty_dataframe():
    result = get_recommendations("DOES_NOT_EXIST", FEAT_DF, SIM, top_n=5)
    assert result.empty


def test_similarity_scores_in_valid_range():
    result = get_recommendations("P001", FEAT_DF, SIM, top_n=4)
    assert (result["similarity_score"] >= 0).all()
    assert (result["similarity_score"] <= 1).all()


def test_top_n_larger_than_available_returns_all_others():
    """top_n=100 should return only the 4 other products, not 100."""
    result = get_recommendations("P001", FEAT_DF, SIM, top_n=100)
    assert len(result) == 4
