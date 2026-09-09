"""CUPED (Controlled-experiment Using Pre-Existing Data), Microsoft's
variance-reduction technique for online experiments (Deng et al., 2013).

Uses a pre-period covariate correlated with the outcome but unaffected by the
treatment (measured before assignment) to subtract out predictable
between-user noise, shrinking the confidence interval without touching the
randomization or spending any alpha.
"""
from __future__ import annotations

import numpy as np


def cuped_adjust(y: np.ndarray, x_pre: np.ndarray) -> dict:
    """Returns the CUPED-adjusted outcome `y_cuped = y - theta * (x_pre - mean(x_pre))`,
    where `theta = Cov(y, x_pre) / Var(x_pre)` is the OLS coefficient that
    minimizes the adjusted outcome's variance. `theta` must be estimated once
    on pooled pre-experiment data (or here, on the full pre-period sample) —
    never separately per arm, or the adjustment itself becomes a source of bias.
    """
    y = np.asarray(y, dtype=float)
    x_pre = np.asarray(x_pre, dtype=float)
    if y.shape != x_pre.shape:
        raise ValueError("y and x_pre must have the same shape")

    var_x = np.var(x_pre, ddof=1)
    if var_x == 0:
        theta = 0.0
    else:
        theta = np.cov(y, x_pre, ddof=1)[0, 1] / var_x

    y_cuped = y - theta * (x_pre - np.mean(x_pre))
    var_before = np.var(y, ddof=1)
    var_after = np.var(y_cuped, ddof=1)
    variance_reduction_pct = 0.0 if var_before == 0 else 100 * (1 - var_after / var_before)

    return {
        "theta": float(theta),
        "y_cuped": y_cuped,
        "var_before": float(var_before),
        "var_after": float(var_after),
        "variance_reduction_pct": float(variance_reduction_pct),
    }
