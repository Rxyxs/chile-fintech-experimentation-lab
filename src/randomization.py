"""Unit randomization and Sample Ratio Mismatch (SRM) detection.

SRM is the single most common silent failure in a real experimentation
pipeline: the observed split between arms (e.g. 49.1% / 50.9%) deviates from
the intended one (50/50) by more than chance, almost always because of a bug
in the assignment or logging path (bot filtering, redirect leakage,
event-loss asymmetry between arms) rather than in the treatment effect
itself. Any result read off an experiment with a failed SRM check is
unreliable, independent of what the effect estimate says.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chisquare


def assign_variant(
    n: int,
    ratio: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Deterministic block randomization: exactly `round(n * ratio)` units get
    variant 1 (treatment), shuffled. Avoids the SRM that a per-unit coin flip
    can introduce purely from finite-sample noise, isolating SRM checks below
    to genuine pipeline bugs rather than expected binomial variation.
    """
    if not 0 < ratio < 1:
        raise ValueError("ratio must be in (0, 1)")
    n_treatment = round(n * ratio)
    assignment = np.zeros(n, dtype=int)
    assignment[:n_treatment] = 1
    rng = np.random.default_rng(seed)
    rng.shuffle(assignment)
    return assignment


def srm_check(
    observed_counts: dict[str, int],
    expected_ratio: dict[str, float],
    alpha: float = 0.005,
) -> dict:
    """Chi-square goodness-of-fit test of observed arm sizes against the
    intended ratio. `alpha` defaults to 0.005 (not 0.05) — the standard
    recommendation for SRM checks, since falsely flagging SRM on a healthy
    experiment is far more disruptive (it blocks the whole readout) than a
    slightly conservative threshold.

    Returns a dict with the chi-square statistic, p-value, and `srm_detected`.
    """
    labels = list(observed_counts.keys())
    if set(labels) != set(expected_ratio.keys()):
        raise ValueError("observed_counts and expected_ratio must share keys")

    total = sum(observed_counts.values())
    observed = np.array([observed_counts[k] for k in labels], dtype=float)
    expected = np.array([expected_ratio[k] * total for k in labels], dtype=float)

    stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    return {
        "labels": labels,
        "observed": observed.tolist(),
        "expected": expected.tolist(),
        "chi2_statistic": float(stat),
        "p_value": float(p_value),
        "srm_detected": bool(p_value < alpha),
    }
