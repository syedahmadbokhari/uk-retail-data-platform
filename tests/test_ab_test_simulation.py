"""
Unit tests for src/analysis/ab_test_simulation.py
Uses mocked data and synthetic distributions — no database connection required.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.analysis.ab_test_simulation import (
    get_product_revenue,
    assign_treatment_control,
    calculate_required_sample_size,
    run_ab_test,
    summarize_ab_test,
    RANDOM_SEED,
)

_FINANCE_DF = pd.DataFrame({"modified_revenue": np.arange(1.0, 21.0)})

# A large synthetic revenue series for balance/reproducibility checks —
# values don't matter here, only the group assignment does.
_LARGE_REVENUE = pd.Series(np.arange(1.0, 2001.0))

# Manufactured groups with a known, large, unmistakable effect (d far above
# any threshold), so significance is not a matter of chance.
_TREATMENT_BIG_EFFECT = pd.Series(np.random.default_rng(1).normal(loc=100, scale=5, size=50))
_CONTROL_BIG_EFFECT = pd.Series(np.random.default_rng(2).normal(loc=60, scale=5, size=50))


# ── get_product_revenue ───────────────────────────────────────────────────────

def test_get_product_revenue_returns_the_revenue_column():
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    with patch("src.analysis.ab_test_simulation.get_connection", return_value=mock_conn), \
         patch("src.analysis.ab_test_simulation.pd.read_sql", return_value=_FINANCE_DF):
        revenue = get_product_revenue()

    assert list(revenue) == list(_FINANCE_DF["modified_revenue"])


# ── assign_treatment_control — randomization ─────────────────────────────────

def test_assign_treatment_control_produces_roughly_balanced_groups():
    assigned = assign_treatment_control(_LARGE_REVENUE, seed=RANDOM_SEED)
    counts = assigned["group"].value_counts()
    total = len(_LARGE_REVENUE)
    # A 50/50 coin flip over 2000 rows should land well within 5% of even.
    assert abs(counts["Treatment"] - counts["Control"]) < 0.05 * total


def test_assign_treatment_control_only_produces_two_labels():
    assigned = assign_treatment_control(_LARGE_REVENUE, seed=RANDOM_SEED)
    assert set(assigned["group"].unique()) == {"Treatment", "Control"}


def test_assign_treatment_control_preserves_row_count():
    assigned = assign_treatment_control(_LARGE_REVENUE, seed=RANDOM_SEED)
    assert len(assigned) == len(_LARGE_REVENUE)


def test_assign_treatment_control_is_reproducible_with_fixed_seed():
    first = assign_treatment_control(_LARGE_REVENUE, seed=RANDOM_SEED)
    second = assign_treatment_control(_LARGE_REVENUE, seed=RANDOM_SEED)
    assert list(first["group"]) == list(second["group"])


def test_assign_treatment_control_differs_across_seeds():
    first = assign_treatment_control(_LARGE_REVENUE, seed=1)
    second = assign_treatment_control(_LARGE_REVENUE, seed=2)
    assert list(first["group"]) != list(second["group"])


# ── calculate_required_sample_size — cross-checked against a known reference ──

def test_required_sample_size_matches_cohen_1988_reference_value():
    # Cohen (1988): d=0.5 (medium), alpha=0.05, power=0.8 -> n ~= 63.77/group,
    # the textbook reference value for a two-sample t-test power analysis.
    n = calculate_required_sample_size(effect_size=0.5, alpha=0.05, power=0.8)
    assert n == 64  # ceil(63.77)


def test_required_sample_size_decreases_as_effect_size_grows():
    n_small_effect = calculate_required_sample_size(effect_size=0.2)
    n_large_effect = calculate_required_sample_size(effect_size=0.8)
    assert n_large_effect < n_small_effect


def test_required_sample_size_increases_as_power_requirement_grows():
    n_low_power = calculate_required_sample_size(effect_size=0.5, power=0.7)
    n_high_power = calculate_required_sample_size(effect_size=0.5, power=0.9)
    assert n_high_power > n_low_power


# ── run_ab_test — correctness on a known, manufactured effect ────────────────

def test_run_ab_test_detects_a_large_manufactured_effect():
    result = run_ab_test(_TREATMENT_BIG_EFFECT, _CONTROL_BIG_EFFECT)
    assert result["significant"] is True
    assert result["p_value"] < 0.001
    assert result["effect_size"] > 0.8  # large effect, per Cohen's conventions


def test_run_ab_test_effect_direction_matches_which_group_is_higher():
    result = run_ab_test(_TREATMENT_BIG_EFFECT, _CONTROL_BIG_EFFECT)
    assert result["mean_treatment"] > result["mean_control"]
    assert result["effect_size"] > 0


def test_run_ab_test_confidence_interval_is_correctly_ordered():
    result = run_ab_test(_TREATMENT_BIG_EFFECT, _CONTROL_BIG_EFFECT)
    assert result["ci_low"] < result["ci_high"]


def test_run_ab_test_returns_expected_structure():
    result = run_ab_test(_TREATMENT_BIG_EFFECT, _CONTROL_BIG_EFFECT)
    expected_keys = {
        "statistic", "p_value", "significant", "ci_low", "ci_high",
        "effect_size", "n_treatment", "n_control", "mean_treatment", "mean_control",
    }
    assert expected_keys.issubset(result.keys())
    assert result["n_treatment"] == len(_TREATMENT_BIG_EFFECT)
    assert result["n_control"] == len(_CONTROL_BIG_EFFECT)


# ── summarize_ab_test — honest, explicitly-labeled-as-simulated output ───────

def test_summarize_ab_test_explicitly_labels_result_as_simulated():
    result = run_ab_test(_TREATMENT_BIG_EFFECT, _CONTROL_BIG_EFFECT)
    summary = summarize_ab_test(result, required_n=10)
    assert "SIMULATED" in summary
    assert "not a live experiment" in summary or "not a live" in summary


def test_summarize_ab_test_calls_significant_result_a_false_positive():
    """
    Since assignment is pure random chance, a 'significant' result on the
    simulated groups must be described as a false positive / Type I error,
    never as a genuine treatment effect.
    """
    result = run_ab_test(_TREATMENT_BIG_EFFECT, _CONTROL_BIG_EFFECT)
    assert result["significant"] is True
    summary = summarize_ab_test(result, required_n=10)
    assert "false positive" in summary.lower() or "type i error" in summary.lower()


def test_summarize_ab_test_reports_power_met_when_sample_size_sufficient():
    result = run_ab_test(_TREATMENT_BIG_EFFECT, _CONTROL_BIG_EFFECT)
    summary = summarize_ab_test(result, required_n=10)  # well below the 50/group achieved
    assert "meets the 10 per group required" in summary


def test_summarize_ab_test_reports_power_not_met_when_sample_size_insufficient():
    result = run_ab_test(_TREATMENT_BIG_EFFECT, _CONTROL_BIG_EFFECT)
    summary = summarize_ab_test(result, required_n=10_000)  # far above the 50/group achieved
    assert "falls short of the 10000 per group required" in summary


def test_summarize_ab_test_no_effect_case_calls_it_the_expected_outcome():
    identical_a = pd.Series([50.0, 51.0, 49.0, 50.0, 52.0, 48.0])
    identical_b = identical_a.copy()
    result = run_ab_test(identical_a, identical_b)
    assert result["significant"] is False
    summary = summarize_ab_test(result, required_n=10)
    assert "no real effect to detect" in summary or "correct expectation" in summary
