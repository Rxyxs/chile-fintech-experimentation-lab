"""Data-generating processes with a known ground truth.

Every technique in this repo is validated against a *simulated* experiment
because that is the only way to check whether a statistical procedure is
correct: with real data you never know the true effect, so you cannot tell
a well-calibrated test from a lucky one. This mirrors the same approach used
in chile-mining-fleet-causal-impact — a known, deliberately-injected effect
is the only way any estimator here can be checked against a real answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_daily_conversion_experiment(
    daily_n: int,
    days: int,
    baseline_rate: float,
    true_effect_absolute: float,
    novelty_decay: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulates a daily-cohort conversion experiment (e.g. "did the user
    activate a new savings product within 24h of seeing the nudge").

    `true_effect_absolute` is the steady-state treatment lift. `novelty_decay`
    optionally shrinks the effect geometrically day over day (0.0 = constant
    effect, 0.1 = effect multiplied by 0.9 each day), to give the novelty
    check in `novelty.py` something real to detect.

    Returns one row per user: columns `day`, `variant` (0=control,
    1=treatment), `converted`.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(days):
        variant = rng.integers(0, 2, size=daily_n)
        effect_today = true_effect_absolute * ((1 - novelty_decay) ** day)
        p = np.where(variant == 1, baseline_rate + effect_today, baseline_rate)
        p = np.clip(p, 0.0, 1.0)
        converted = rng.binomial(1, p)
        rows.append(pd.DataFrame({"day": day, "variant": variant, "converted": converted}))
    return pd.concat(rows, ignore_index=True)


def simulate_revenue_experiment_with_covariate(
    n_per_arm: int,
    true_effect_absolute: float,
    covariate_correlation: float = 0.6,
    outcome_std: float = 20.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulates a single-snapshot continuous-outcome experiment (e.g. 30-day
    post-period revenue per user) together with a pre-period covariate (e.g.
    30-day pre-period revenue) correlated with it — the setup CUPED needs.

    `covariate_correlation` is the *exact* population correlation between
    `x_pre` and the outcome absent the treatment effect, built from two
    independent standard normals `z1, z2` as `x_pre = f(z1)` and
    `y_base = f(rho*z1 + sqrt(1-rho^2)*z2)` — not diluted by any further
    independent noise added downstream, so `cuped.cuped_adjust`'s measured
    variance reduction can be checked directly against `rho**2`, the
    theoretical CUPED reduction under a linear covariate relationship.
    """
    rng = np.random.default_rng(seed)
    n = n_per_arm * 2
    variant = np.array([0] * n_per_arm + [1] * n_per_arm)
    rng.shuffle(variant)

    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    z_y = covariate_correlation * z1 + np.sqrt(1 - covariate_correlation ** 2) * z2

    x_pre = 50 + 15 * z1
    y = 50 + outcome_std * z_y + variant * true_effect_absolute

    return pd.DataFrame({"variant": variant, "x_pre": x_pre, "y": y})
