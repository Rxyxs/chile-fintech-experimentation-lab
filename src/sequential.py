"""Two ways of looking at an experiment before it reaches its planned sample
size, and why only one of them is safe.

`naive_peek_p_value` is what a fixed-horizon two-proportion z-test gives you
if you compute it every day and stop the first time p < 0.05 — this is
"peeking", and `calibration.calibrate_naive_peeking` shows empirically how
badly it inflates the false-positive rate.

`bayesian_probability_treatment_better` is a Beta-Bernoulli posterior
comparison, commonly assumed to be "safe to peek" because it answers a
different question than a p-value ("what is the current probability the
treatment is better" rather than "would a once-only test reject"). Whether
that assumption actually holds — i.e. whether dichotomizing it into a stop/
continue rule and checking it daily still inflates the long-run false-decision
rate — is not asserted here; `calibration.py` measures it empirically instead
of taking the claim on faith.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import norm


def naive_peek_p_value(successes_t: int, n_t: int, successes_c: int, n_c: int) -> float:
    """Two-sided two-proportion z-test p-value on whatever data has
    accumulated so far. Statistically valid only if computed once, at a
    pre-registered sample size — see calibration.py for what happens when
    it is instead checked daily.
    """
    if n_t == 0 or n_c == 0:
        return 1.0
    p_t, p_c = successes_t / n_t, successes_c / n_c
    p_pool = (successes_t + successes_c) / (n_t + n_c)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    if se == 0:
        return 1.0
    z = (p_t - p_c) / se
    return float(2 * (1 - norm.cdf(abs(z))))


def bayesian_probability_treatment_better(
    successes_t: int,
    n_t: int,
    successes_c: int,
    n_c: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 20_000,
    seed: int | None = None,
) -> float:
    """P(conversion_rate_treatment > conversion_rate_control) under
    independent Beta(prior) posteriors, estimated by Monte Carlo sampling.
    Safe to compute after every new observation: it is a statement about the
    current posterior, not a hypothesis test with a fixed rejection region,
    so there is no "spending of alpha" to inflate by looking often.
    """
    rng = np.random.default_rng(seed)
    post_t = beta_dist(prior_alpha + successes_t, prior_beta + n_t - successes_t)
    post_c = beta_dist(prior_alpha + successes_c, prior_beta + n_c - successes_c)
    draws_t = post_t.rvs(n_samples, random_state=rng)
    draws_c = post_c.rvs(n_samples, random_state=rng)
    return float(np.mean(draws_t > draws_c))
