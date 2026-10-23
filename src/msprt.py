"""Always-valid monitoring: the mixture sequential probability ratio test.

`calibration.py` shows that checking a fixed-horizon z-test every day takes
its false-positive rate from 5% to about 24%, and that a Bayesian
stop-at-95% rule does not fix it. The mixture SPRT (mSPRT) of Johari,
Koomen, Pekelis and Walsh, "Peeking at A/B Tests" (KDD 2017), the procedure
behind Optimizely's Stats Engine, is built to be looked at: under the null,
the probability that its p-value *ever* drops below alpha, however often it
is checked, is at most alpha.

It compares how likely the observed difference is under a normal mixture of
alternatives (true difference ~ N(0, tau^2)) with how likely it is under no
difference. With the estimated difference `theta_hat` and its variance `V`:

    Lambda = sqrt(V / (V + tau^2)) * exp(tau^2 * theta_hat^2 / (2 V (V + tau^2)))

For normal data with known variance, Lambda is a nonnegative martingale under
the null, so P(sup Lambda >= 1/alpha) <= alpha (Ville's inequality), and the
always-valid p-value is the running minimum of 1/Lambda. Two-proportion data
is only approximately normal and its variance is estimated (pooled under the
null, as in `sequential.naive_peek_p_value`), so the guarantee is checked
empirically in `calibration.calibrate_msprt` and, on real data, in
`real_criteo`. `tau` sets the effect size the test is quickest to pick up; a
plausible absolute effect for the metric is the usual choice.
"""
from __future__ import annotations

import numpy as np


def msprt_statistic(successes_t, n_t, successes_c, n_c, tau: float) -> np.ndarray:
    """Mixture likelihood ratio Lambda for the difference of two proportions.

    Accepts scalars or arrays of cumulative counts (one entry per look) and
    returns Lambda for each. Lambda = 1 (no evidence either way) when an arm is
    still empty or the pooled variance is zero.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    st, nt, sc, nc = (np.asarray(a, dtype=float) for a in (successes_t, n_t, successes_c, n_c))
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = st / nt - sc / nc
        p_pool = (st + sc) / (nt + nc)
        v = p_pool * (1 - p_pool) * (1 / nt + 1 / nc)
        t2 = tau ** 2
        lam = np.sqrt(v / (v + t2)) * np.exp(t2 * theta ** 2 / (2 * v * (v + t2)))
    ok = (nt > 0) & (nc > 0) & (v > 0)
    return np.where(ok, lam, 1.0)


def always_valid_p_values(successes_t, n_t, successes_c, n_c, tau: float) -> np.ndarray:
    """Always-valid p-value at each look: the running minimum of min(1, 1/Lambda).

    The inputs are cumulative counts in the order the looks were taken. Stopping
    the first time this drops below alpha keeps the false-positive rate at or
    below alpha no matter how many looks there are.
    """
    lam = np.atleast_1d(msprt_statistic(successes_t, n_t, successes_c, n_c, tau))
    return np.minimum.accumulate(np.minimum(1.0, 1.0 / lam))
