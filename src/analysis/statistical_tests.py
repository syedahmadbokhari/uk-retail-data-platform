import os
import time
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.db import get_connection, get_root
from src.utils.logger import get_logger

logger = get_logger("analysis.statistical_tests")

ALPHA = 0.05

# Shapiro-Wilk grows unreliable (near-certain rejection) above ~5000 samples,
# so each group is capped to a random sample before testing.
_SHAPIRO_SAMPLE_CAP = 5000

# abs(skew) below this is treated as "reasonably symmetric" — a numeric proxy
# for the by-eye symmetry check normally done against the saved histograms.
_SYMMETRY_SKEW_THRESHOLD = 0.5

_ASSETS_DIR = os.path.join(get_root(), "assets")

_TEST_NAMES = {
    "welch_t": "Welch's t-test",
    "mann_whitney": "Mann-Whitney U test",
}


def get_revenue_groups() -> tuple:
    """
    Pull per-product revenue from the clean layer (clean_finance — one row
    per product, pre-aggregation), split into discounted vs. full-price
    groups using the same rule as the analytics_discount_impact mart
    (modified_discount > 0 -> Discounted).
    """
    with get_connection() as conn:
        df = pd.read_sql(
            "SELECT modified_discount, modified_revenue FROM clean_finance", conn
        )

    discounted = df.loc[df["modified_discount"] > 0, "modified_revenue"].reset_index(drop=True)
    full_price = df.loc[df["modified_discount"] == 0, "modified_revenue"].reset_index(drop=True)
    return discounted, full_price


def check_normality(discounted: pd.Series, full_price: pd.Series, save_plots: bool = True) -> dict:
    """
    Shapiro-Wilk normality check on each group. Also saves a histogram and a
    Q-Q plot per group to assets/ for visual inspection, since Shapiro-Wilk
    alone can flag large-but-mildly-non-normal samples as non-normal.
    """
    result = {}
    for label, series in (("discounted", discounted), ("full_price", full_price)):
        sample = (
            series.sample(min(len(series), _SHAPIRO_SAMPLE_CAP), random_state=42)
            if len(series) > 0 else series
        )
        stat, p = stats.shapiro(sample)
        skew = float(stats.skew(series)) if len(series) > 0 else 0.0
        result[label] = {
            "statistic": float(stat),
            "p_value": float(p),
            "skew": skew,
            "is_normal": bool(p > ALPHA and abs(skew) < _SYMMETRY_SKEW_THRESHOLD),
        }

    if save_plots:
        _save_distribution_plots(discounted, full_price)

    return result


def _save_distribution_plots(discounted: pd.Series, full_price: pd.Series) -> None:
    os.makedirs(_ASSETS_DIR, exist_ok=True)
    groups = (("Discounted", discounted), ("Full Price", full_price))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (label, series) in zip(axes, groups):
        ax.hist(series, bins=30)
        ax.set_title(f"{label} — Revenue Distribution")
        ax.set_xlabel("Revenue")
        ax.set_ylabel("Count")
    plt.tight_layout()
    hist_path = os.path.join(_ASSETS_DIR, "discount_revenue_histograms.png")
    plt.savefig(hist_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved histogram: {hist_path}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (label, series) in zip(axes, groups):
        stats.probplot(series, dist="norm", plot=ax)
        ax.set_title(f"{label} — Q-Q Plot")
    plt.tight_layout()
    qq_path = os.path.join(_ASSETS_DIR, "discount_revenue_qqplots.png")
    plt.savefig(qq_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Q-Q plot: {qq_path}")


def select_test(normality: dict) -> str:
    """
    'welch_t' if both groups pass the normality check, else 'mann_whitney'.
    """
    both_normal = normality["discounted"]["is_normal"] and normality["full_price"]["is_normal"]
    return "welch_t" if both_normal else "mann_whitney"


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Cohen's d for two independent samples, using pooled standard deviation."""
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1 - 1) * a.std() ** 2 + (n2 - 1) * b.std() ** 2) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled_std


def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """
    Rank-biserial correlation for Mann-Whitney U. Positive means the first
    sample (discounted) tends to rank higher than the second (full_price).
    """
    return (2 * u_stat) / (n1 * n2) - 1


def _effect_magnitude(abs_effect: float, test: str) -> str:
    thresholds = (
        [(0.2, "negligible"), (0.5, "small"), (0.8, "medium")]
        if test == "welch_t"
        else [(0.1, "negligible"), (0.3, "small"), (0.5, "medium")]
    )
    for cutoff, label in thresholds:
        if abs_effect < cutoff:
            return label
    return "large"


def run_hypothesis_test(discounted: pd.Series, full_price: pd.Series, normality: dict = None) -> dict:
    """
    Runs Welch's t-test if both groups are reasonably normal, otherwise
    Mann-Whitney U. Returns test name, statistic, p-value, significance,
    effect size, group sizes, and group means/medians.
    """
    if normality is None:
        normality = check_normality(discounted, full_price, save_plots=False)

    test_name = select_test(normality)

    if test_name == "welch_t":
        stat, p = stats.ttest_ind(discounted, full_price, equal_var=False)
        effect_size = _cohens_d(discounted, full_price)
        effect_size_type = "cohens_d"
    else:
        stat, p = stats.mannwhitneyu(discounted, full_price, alternative="two-sided")
        effect_size = _rank_biserial(stat, len(discounted), len(full_price))
        effect_size_type = "rank_biserial_r"

    return {
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < ALPHA),
        "effect_size": float(effect_size),
        "effect_size_type": effect_size_type,
        "n_discounted": len(discounted),
        "n_full_price": len(full_price),
        "mean_discounted": float(discounted.mean()),
        "mean_full_price": float(full_price.mean()),
        "median_discounted": float(discounted.median()),
        "median_full_price": float(full_price.median()),
        "normality": normality,
    }


def summarize_result(result: dict) -> str:
    """
    Human-readable summary of run_hypothesis_test()'s output. Phrased
    honestly in both directions — it does not force a significant-sounding
    conclusion when the data doesn't support one.
    """
    test_label = _TEST_NAMES[result["test"]]
    p = result["p_value"]
    stat_desc = f"t = {result['statistic']:.3f}" if result["test"] == "welch_t" else f"U = {result['statistic']:.1f}"

    if not result["significant"]:
        return (
            f"Result: {test_label} found no statistically significant difference in revenue "
            f"between discounted and full-price products (p = {p:.4g}, {stat_desc}). The raw-totals "
            f"gap is best explained by there being more discounted products, not a genuine "
            f"per-product revenue difference."
        )

    effect = result["effect_size"]
    magnitude = _effect_magnitude(abs(effect), result["test"])
    favors_discounted = effect > 0

    mean_favors_discounted = result["mean_discounted"] > result["mean_full_price"]
    median_favors_discounted = result["median_discounted"] > result["median_full_price"]

    if favors_discounted:
        direction = (
            "discounted products have a significantly higher typical per-product revenue than "
            "full-price products, so the discount-led revenue pattern is a real distributional "
            "effect and not just a byproduct of raw totals"
        )
    else:
        direction = (
            "full-price products actually have a significantly higher typical per-product revenue "
            "than discounted products — the raw-totals lead for discounted products is a volume "
            "effect (more discounted products sold), not a per-product revenue advantage"
        )

    caveat = ""
    if mean_favors_discounted != median_favors_discounted:
        caveat = (
            " Note: mean revenue is skewed by a small number of high-value outlier products "
            f"(mean favors {'discounted' if mean_favors_discounted else 'full-price'} products, "
            f"median favors {'discounted' if median_favors_discounted else 'full-price'} products) "
            "— exactly why a rank-based test rather than a raw mean comparison is used here."
        )

    return (
        f"Result: {test_label} shows a statistically significant ({magnitude} effect) difference "
        f"in revenue between discounted and full-price products (p = {p:.4g}, {stat_desc}). This "
        f"confirms the discount-led revenue pattern observed in raw totals is not a distributional "
        f"artifact — {direction}.{caveat}"
    )


def run_statistical_validation() -> dict:
    start = time.time()
    logger.info("=== Statistical validation: discounted vs full-price revenue ===")

    discounted, full_price = get_revenue_groups()
    logger.info(f"Loaded {len(discounted)} discounted, {len(full_price)} full-price products")

    normality = check_normality(discounted, full_price, save_plots=True)
    for label, group_stats in normality.items():
        logger.info(
            f"  Shapiro-Wilk ({label}): W={group_stats['statistic']:.4f}, "
            f"p={group_stats['p_value']:.4g}, skew={group_stats['skew']:.3f}, "
            f"normal={group_stats['is_normal']}"
        )

    result = run_hypothesis_test(discounted, full_price, normality=normality)
    logger.info(f"  Test used: {result['test']} (p={result['p_value']:.4g})")

    summary = summarize_result(result)
    logger.info(summary)

    logger.info(f"Statistical validation complete ({time.time() - start:.2f}s)")
    return {**result, "summary": summary}


if __name__ == "__main__":
    print(run_statistical_validation()["summary"])
