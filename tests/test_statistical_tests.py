"""
Unit tests for src/analysis/statistical_tests.py
Uses mocks and synthetic data — no database connection required.
"""
import os
import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.analysis.statistical_tests import (
    get_revenue_groups,
    check_normality,
    select_test,
    run_hypothesis_test,
    summarize_result,
    _cohens_d,
    _rank_biserial,
)

# ── Synthetic fixtures ────────────────────────────────────────────────────────

_FINANCE_DF = pd.DataFrame({
    "modified_discount": [0.0, 0.0, 0.3, 0.5, 0.0, 0.4],
    "modified_revenue":  [100.0, 200.0, 500.0, 600.0, 300.0, 450.0],
})

# Reasonably normal, symmetric group (Shapiro p > 0.05, |skew| < 0.5)
_NORMAL_A = pd.Series(np.random.default_rng(42).normal(loc=100, scale=10, size=300))
_NORMAL_B = pd.Series(np.random.default_rng(7).normal(loc=105, scale=10, size=300))

# Heavily right-skewed group (Shapiro p << 0.05, |skew| >> 0.5) — mirrors the
# real revenue distributions this module actually runs against.
_SKEWED_A = pd.Series(np.random.default_rng(42).exponential(scale=50, size=300))
_SKEWED_B = pd.Series(np.random.default_rng(7).exponential(scale=70, size=300))

# Manufactured dataset with a known Cohen's d: equal spread, shifted by a
# fixed amount, so the pooled std is identical to each group's own std.
_D_GROUP_A = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])   # mean=30, std=15.8113883
_D_GROUP_B = pd.Series([20.0, 30.0, 40.0, 50.0, 60.0])   # mean=40, std=15.8113883
_EXPECTED_COHENS_D = (30.0 - 40.0) / math.sqrt(250.0)     # -0.6324555...

# Manufactured dataset where group A completely dominates group B in rank —
# rank-biserial r must come out to exactly +1.0.
_RANK_GROUP_A = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
_RANK_GROUP_B = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])


# ── get_revenue_groups ────────────────────────────────────────────────────────

def test_get_revenue_groups_splits_by_discount():
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    with patch("src.analysis.statistical_tests.get_connection", return_value=mock_conn), \
         patch("src.analysis.statistical_tests.pd.read_sql", return_value=_FINANCE_DF):
        discounted, full_price = get_revenue_groups()

    assert sorted(discounted.tolist()) == [450.0, 500.0, 600.0]
    assert sorted(full_price.tolist()) == [100.0, 200.0, 300.0]


# ── check_normality — structure and classification ───────────────────────────

def test_check_normality_returns_expected_structure():
    result = check_normality(_NORMAL_A, _NORMAL_B, save_plots=False)
    assert set(result.keys()) == {"discounted", "full_price"}
    for group in result.values():
        assert set(group.keys()) == {"statistic", "p_value", "skew", "is_normal"}
        assert isinstance(group["statistic"], float)
        assert isinstance(group["p_value"], float)
        assert isinstance(group["is_normal"], bool)


def test_check_normality_flags_normal_data_as_normal():
    result = check_normality(_NORMAL_A, _NORMAL_B, save_plots=False)
    assert result["discounted"]["is_normal"] is True
    assert result["full_price"]["is_normal"] is True


def test_check_normality_flags_skewed_data_as_not_normal():
    result = check_normality(_SKEWED_A, _SKEWED_B, save_plots=False)
    assert result["discounted"]["is_normal"] is False
    assert result["full_price"]["is_normal"] is False


def test_check_normality_saves_plots_to_assets_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("src.analysis.statistical_tests._ASSETS_DIR", str(tmp_path))
    check_normality(_NORMAL_A, _NORMAL_B, save_plots=True)
    assert (tmp_path / "discount_revenue_histograms.png").exists()
    assert (tmp_path / "discount_revenue_qqplots.png").exists()


def test_check_normality_no_plots_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr("src.analysis.statistical_tests._ASSETS_DIR", str(tmp_path))
    check_normality(_NORMAL_A, _NORMAL_B, save_plots=False)
    assert list(tmp_path.iterdir()) == []


# ── select_test — decision logic ─────────────────────────────────────────────

def test_select_test_uses_welch_when_both_normal():
    normality = {
        "discounted": {"is_normal": True},
        "full_price": {"is_normal": True},
    }
    assert select_test(normality) == "welch_t"


@pytest.mark.parametrize("discounted_normal,full_price_normal", [
    (False, True),
    (True, False),
    (False, False),
])
def test_select_test_uses_mann_whitney_unless_both_normal(discounted_normal, full_price_normal):
    normality = {
        "discounted": {"is_normal": discounted_normal},
        "full_price": {"is_normal": full_price_normal},
    }
    assert select_test(normality) == "mann_whitney"


# ── run_hypothesis_test — correct test selected + structure ──────────────────

def test_run_hypothesis_test_selects_welch_t_for_normal_groups():
    result = run_hypothesis_test(_NORMAL_A, _NORMAL_B)
    assert result["test"] == "welch_t"
    assert result["effect_size_type"] == "cohens_d"
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["significant"] == (result["p_value"] < 0.05)


def test_run_hypothesis_test_selects_mann_whitney_for_skewed_groups():
    result = run_hypothesis_test(_SKEWED_A, _SKEWED_B)
    assert result["test"] == "mann_whitney"
    assert result["effect_size_type"] == "rank_biserial_r"
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["significant"] == (result["p_value"] < 0.05)


def test_run_hypothesis_test_returns_full_structure():
    result = run_hypothesis_test(_NORMAL_A, _NORMAL_B)
    expected_keys = {
        "test", "statistic", "p_value", "significant", "effect_size",
        "effect_size_type", "n_discounted", "n_full_price",
        "mean_discounted", "mean_full_price", "median_discounted",
        "median_full_price", "normality",
    }
    assert expected_keys.issubset(result.keys())
    assert result["n_discounted"] == len(_NORMAL_A)
    assert result["n_full_price"] == len(_NORMAL_B)


# ── Effect size correctness on known synthetic datasets ──────────────────────

def test_cohens_d_matches_hand_calculated_value():
    d = _cohens_d(_D_GROUP_A, _D_GROUP_B)
    assert d == pytest.approx(_EXPECTED_COHENS_D, rel=1e-9)


def test_run_hypothesis_test_cohens_d_end_to_end():
    normality = {
        "discounted": {"is_normal": True},
        "full_price": {"is_normal": True},
    }
    result = run_hypothesis_test(_D_GROUP_A, _D_GROUP_B, normality=normality)
    assert result["effect_size"] == pytest.approx(_EXPECTED_COHENS_D, rel=1e-9)


def test_rank_biserial_is_plus_one_when_group_a_fully_dominates():
    from scipy import stats as scipy_stats
    u_stat, _ = scipy_stats.mannwhitneyu(_RANK_GROUP_A, _RANK_GROUP_B, alternative="two-sided")
    r = _rank_biserial(u_stat, len(_RANK_GROUP_A), len(_RANK_GROUP_B))
    assert r == pytest.approx(1.0)


def test_rank_biserial_is_minus_one_when_group_b_fully_dominates():
    from scipy import stats as scipy_stats
    u_stat, _ = scipy_stats.mannwhitneyu(_RANK_GROUP_B, _RANK_GROUP_A, alternative="two-sided")
    r = _rank_biserial(u_stat, len(_RANK_GROUP_B), len(_RANK_GROUP_A))
    assert r == pytest.approx(-1.0)


# ── summarize_result — honest phrasing ────────────────────────────────────────

def test_summarize_result_reports_significant_result_plainly():
    normality = {
        "discounted": {"is_normal": False},
        "full_price": {"is_normal": False},
    }
    result = run_hypothesis_test(_RANK_GROUP_A, _RANK_GROUP_B, normality=normality)
    summary = summarize_result(result)
    assert "statistically significant" in summary
    assert "no statistically significant" not in summary
    assert f"p = {result['p_value']:.4g}" in summary


def test_summarize_result_reports_non_significant_result_plainly():
    identical = pd.Series([50.0, 51.0, 49.0, 50.0, 52.0, 48.0])
    normality = {
        "discounted": {"is_normal": False},
        "full_price": {"is_normal": False},
    }
    result = run_hypothesis_test(identical, identical.copy(), normality=normality)
    summary = summarize_result(result)
    assert result["significant"] is False
    assert "no statistically significant difference" in summary
