"""Sample-size and minimum-detectable-effect (MDE) calculations.

Two-sided normal-approximation formulas, the same ones used by every major
experimentation platform (Optimizely, Statsig, GrowthBook). Kept dependency-free
(only scipy.stats.norm for the quantile function) so the formulas are auditable
line by line instead of hidden behind a library call.
"""
from __future__ import annotations

import math

from scipy.stats import norm


def _z(alpha_two_sided: float, power: float) -> tuple[float, float]:
    z_alpha = norm.ppf(1 - alpha_two_sided / 2)
    z_beta = norm.ppf(power)
    return z_alpha, z_beta


def sample_size_two_proportion(
    baseline_rate: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Required users **per arm** to detect an absolute lift of `mde_absolute`
    over a `baseline_rate` conversion rate, with a two-sided test.

    Uses the pooled-variance normal approximation:
        n = (z_a + z_b)^2 * (p1(1-p1) + p2(1-p2)) / (p1 - p2)^2
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be in (0, 1)")
    if mde_absolute <= 0:
        raise ValueError("mde_absolute must be positive")

    p1 = baseline_rate
    p2 = baseline_rate + mde_absolute
    z_alpha, z_beta = _z(alpha, power)
    numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    denominator = mde_absolute ** 2
    return math.ceil(numerator / denominator)


def sample_size_two_mean(
    std: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Required users **per arm** to detect an absolute lift of `mde_absolute`
    in a continuous metric with standard deviation `std` (equal variance assumed
    in both arms), two-sided test.

        n = 2 * (z_a + z_b)^2 * sigma^2 / mde^2
    """
    if std <= 0:
        raise ValueError("std must be positive")
    if mde_absolute <= 0:
        raise ValueError("mde_absolute must be positive")

    z_alpha, z_beta = _z(alpha, power)
    numerator = 2 * (z_alpha + z_beta) ** 2 * std ** 2
    return math.ceil(numerator / mde_absolute ** 2)


def minimum_detectable_effect_two_proportion(
    baseline_rate: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Inverse of `sample_size_two_proportion`: given a fixed sample size per
    arm, the smallest absolute lift the test can detect at the requested power.
    Solved numerically (bisection) since the closed form is not linear in mde.
    """
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")

    lo, hi = 1e-6, 1 - baseline_rate - 1e-6
    for _ in range(60):
        mid = (lo + hi) / 2
        n_needed = sample_size_two_proportion(baseline_rate, mid, alpha, power)
        if n_needed > n_per_arm:
            lo = mid
        else:
            hi = mid
    return hi
