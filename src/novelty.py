"""Novelty / primacy effect check: does the treatment effect change over the
course of the experiment?

A treatment*day interaction that is significant means the headline "average"
effect is a mix of two different regimes (e.g. a UI change that spikes
curiosity clicks in week 1 and fades, or a discount offer whose effect grows
as word-of-mouth compounds) and the pooled estimate should not be trusted as
a forecast of the steady-state effect.
"""
from __future__ import annotations

import numpy as np
import statsmodels.api as sm


def detect_novelty_effect(day: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> dict:
    """OLS of `outcome ~ treatment + day + treatment:day`, with HC1
    heteroskedasticity-robust standard errors. The coefficient and p-value on
    the interaction term are the novelty-effect diagnostic.

    Robust errors are not optional here: `outcome` is typically binary
    (converted / not), which makes this a linear probability model with
    variance Var(outcome|X) = p(X)(1-p(X)) that changes with X by
    construction — textbook heteroskedasticity. Classical OLS standard
    errors assume constant variance and were found, in this repo's own test
    suite, to produce false novelty-effect alarms on a constant true effect
    at a noticeably higher rate than the nominal 5% until this was fixed.

    Named `detect_*` rather than `test_*` on purpose: pytest auto-collects
    any `test_*`-named callable it can import, including one imported into a
    test module under a different alias — pulling this function into
    tests/test_novelty.py under its old name made pytest try to run it
    directly as a test and fail on missing fixtures.
    """
    day = np.asarray(day, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    outcome = np.asarray(outcome, dtype=float)

    day_c = day - day.mean()
    interaction = treatment * day_c
    X = np.column_stack([np.ones_like(day_c), treatment, day_c, interaction])
    model = sm.OLS(outcome, X).fit(cov_type="HC1")

    return {
        "interaction_coefficient": float(model.params[3]),
        "interaction_p_value": float(model.pvalues[3]),
        "novelty_effect_detected": bool(model.pvalues[3] < 0.05),
        "model": model,
    }
