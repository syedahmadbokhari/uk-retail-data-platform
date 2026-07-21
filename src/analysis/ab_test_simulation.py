"""
Simulated A/B test methodology demonstration.

This is deliberately distinct from src/analysis/statistical_tests.py:
that module compares REAL historical discount vs. full-price revenue —
the discount status reflects whatever pricing decisions were actually
made, with no randomization, so it can only show association in
observational data, not a controlled experimental result.

This module demonstrates the three pillars a real A/B test needs that the
observational analysis doesn't and can't provide: randomization, an a
priori power analysis, and a significance test on the randomly-assigned
groups. Real per-product revenue values are kept, but each product is
independently, randomly re-assigned to a SIMULATED "Treatment" or
"Control" group — this is a methodology demonstration built on real
data, NOT a claim that a live experiment was run on real customers,
website traffic, or transactions. Every function and log line below
says so explicitly rather than leaving it implied.

Because Treatment/Control assignment here is pure chance, unconnected to
any real intervention, the statistically correct expectation is that the
two groups usually show NO significant difference — precisely because
there is no real "treatment" behind the label. If a run does come back
significant, that is a false positive (a Type I error), which is expected
to happen approximately alpha of the time by construction — the summary
function says so plainly rather than dressing it up as a discovery.
"""
import time
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower

from src.utils.db import get_connection
from src.utils.logger import get_logger
from src.analysis.statistical_tests import _cohens_d

logger = get_logger("analysis.ab_test_simulation")

ALPHA = 0.05
POWER = 0.8
RANDOM_SEED = 42

# Cohen's (1988) conventional "medium" effect size — used here as the a priori
# minimum detectable effect for the power analysis, since there is no real
# business-specified minimum practically significant effect for a simulation.
DEFAULT_MIN_DETECTABLE_EFFECT = 0.5


def get_product_revenue() -> pd.Series:
    """
    Pulls per-product revenue from the same clean layer used by
    statistical_tests.py (clean_finance) — but this module ignores the
    real modified_discount column entirely and applies its own simulated
    random assignment instead (see assign_treatment_control()).
    """
    with get_connection() as conn:
        df = pd.read_sql("SELECT modified_revenue FROM clean_finance", conn)
    return df["modified_revenue"].reset_index(drop=True)


def assign_treatment_control(revenue: pd.Series, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    SIMULATED random assignment — an independent 50/50 coin flip per row,
    not the real historical discount status. Reproducible via a fixed seed.
    """
    rng = np.random.default_rng(seed)
    group = np.where(rng.random(len(revenue)) < 0.5, "Treatment", "Control")
    return pd.DataFrame({"revenue": revenue.reset_index(drop=True), "group": group})


def calculate_required_sample_size(
    effect_size: float = DEFAULT_MIN_DETECTABLE_EFFECT,
    alpha: float = ALPHA,
    power: float = POWER,
) -> int:
    """
    Minimum sample size per group (rounded up) needed to detect
    `effect_size` (Cohen's d) at the given significance level and power,
    using statsmodels' analytical power solver for an independent
    two-sample t-test. This is an a priori calculation — done before
    looking at results, as real experimental design requires.
    """
    n = TTestIndPower().solve_power(
        effect_size=effect_size, alpha=alpha, power=power, ratio=1.0, alternative="two-sided",
    )
    return int(np.ceil(n))


def run_ab_test(treatment: pd.Series, control: pd.Series) -> dict:
    """
    Independent two-sample t-test (Welch's — unequal variance assumed,
    consistent with statistical_tests.py's approach) on the simulated
    groups. Returns statistic, p-value, significance, a 95% confidence
    interval for the mean difference, and Cohen's d.
    """
    res = stats.ttest_ind(treatment, control, equal_var=False)
    ci = res.confidence_interval(confidence_level=0.95)
    effect_size = _cohens_d(treatment, control)

    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "significant": bool(res.pvalue < ALPHA),
        "ci_low": float(ci.low),
        "ci_high": float(ci.high),
        "effect_size": float(effect_size),
        "n_treatment": len(treatment),
        "n_control": len(control),
        "mean_treatment": float(treatment.mean()),
        "mean_control": float(control.mean()),
    }


def summarize_ab_test(result: dict, required_n: int) -> str:
    """
    Human-readable summary — explicit that this is a SIMULATED experiment,
    and honest that a significant result here is a Type I error (there is
    no real treatment effect to detect by construction), not a finding.
    """
    achieved_n = min(result["n_treatment"], result["n_control"])
    power_met = achieved_n >= required_n

    power_note = (
        f"the achieved sample size ({achieved_n} per group) meets the {required_n} per group "
        f"required for {POWER:.0%} power to detect an effect of this size"
        if power_met else
        f"the achieved sample size ({achieved_n} per group) falls short of the {required_n} per group "
        f"required for {POWER:.0%} power — this run would be under-powered to detect an effect of this size"
    )

    if not result["significant"]:
        return (
            f"Result (SIMULATED A/B test — randomized assignment on real product data, not a live "
            f"experiment): no statistically significant difference in revenue between the randomly-"
            f"assigned Treatment and Control groups (p = {result['p_value']:.4g}, "
            f"t = {result['statistic']:.3f}, 95% CI [{result['ci_low']:.2f}, {result['ci_high']:.2f}]). "
            f"This is the statistically correct expectation: since assignment was random and unconnected "
            f"to any real intervention, there is no real effect to detect. {power_note}."
        )

    return (
        f"Result (SIMULATED A/B test — randomized assignment on real product data, not a live "
        f"experiment): a statistically significant difference was found between the randomly-assigned "
        f"Treatment and Control groups (p = {result['p_value']:.4g}, t = {result['statistic']:.3f}, "
        f"95% CI [{result['ci_low']:.2f}, {result['ci_high']:.2f}]). Because group assignment was pure "
        f"chance with no real intervention behind it, this is a false positive (a Type I error) rather "
        f"than a genuine treatment effect — expected to happen in roughly {ALPHA:.0%} of runs by "
        f"construction, which is exactly why a pre-registered alpha threshold matters. {power_note}."
    )


def run_ab_test_simulation(effect_size: float = DEFAULT_MIN_DETECTABLE_EFFECT) -> dict:
    start = time.time()
    logger.info("=== SIMULATED A/B test: randomized Treatment/Control on real product revenue ===")
    logger.info("    (This is a methodology demonstration — not a live experiment on real customers.)")

    revenue = get_product_revenue()
    assigned = assign_treatment_control(revenue)
    treatment = assigned.loc[assigned["group"] == "Treatment", "revenue"]
    control = assigned.loc[assigned["group"] == "Control", "revenue"]
    logger.info(f"Simulated assignment (seed={RANDOM_SEED}): {len(treatment)} Treatment, {len(control)} Control")

    required_n = calculate_required_sample_size(effect_size)
    logger.info(
        f"Required sample size per group for d={effect_size} at alpha={ALPHA}, power={POWER:.0%}: {required_n}"
    )

    result = run_ab_test(treatment, control)
    logger.info(
        f"  t={result['statistic']:.4f}, p={result['p_value']:.4g}, "
        f"significant={result['significant']}, cohen_d={result['effect_size']:.4f}"
    )

    summary = summarize_ab_test(result, required_n)
    logger.info(summary)

    logger.info(f"Simulated A/B test complete ({time.time() - start:.2f}s)")
    return {**result, "required_n": required_n, "effect_size_target": effect_size, "summary": summary}


if __name__ == "__main__":
    print(run_ab_test_simulation()["summary"])
