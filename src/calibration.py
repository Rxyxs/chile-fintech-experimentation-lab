"""The honest-finding module: does each stopping rule actually control the
error rate it claims to?

A power/alpha calculation is a promise about long-run behaviour. The only way
to check the promise is kept is to run the whole procedure thousands of times
under a *known* truth and count how often it gets the answer wrong — exactly
the same logic as `drift.target_shift` reproducing a baseline's R² in
Limpieza_Datos: the diagnostic is only trustworthy once it has been checked
against a case where the right answer is known in advance.
"""
from __future__ import annotations

import numpy as np

from src.sequential import bayesian_probability_treatment_better, naive_peek_p_value
from src.simulate import simulate_daily_conversion_experiment


def calibrate_naive_peeking(
    n_simulations: int,
    daily_n: int,
    days: int,
    baseline_rate: float,
    true_effect_absolute: float = 0.0,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict:
    """Runs `n_simulations` independent experiments, checking the naive
    fixed-alpha p-value every day and stopping at the first day it drops
    below `alpha`. Reports the fraction of simulations that stopped early
    with a "significant" result — under `true_effect_absolute=0.0` this is
    the empirical false-positive rate, and a fixed-horizon test's whole
    guarantee is that this number should equal `alpha`.
    """
    rng = np.random.default_rng(seed)
    rejections = 0
    stopping_days = []

    for i in range(n_simulations):
        df = simulate_daily_conversion_experiment(
            daily_n, days, baseline_rate, true_effect_absolute,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        cum_t_s = cum_t_n = cum_c_s = cum_c_n = 0
        stopped = False
        for day in range(days):
            day_df = df[df["day"] == day]
            cum_t_s += int(day_df.loc[day_df.variant == 1, "converted"].sum())
            cum_t_n += int((day_df.variant == 1).sum())
            cum_c_s += int(day_df.loc[day_df.variant == 0, "converted"].sum())
            cum_c_n += int((day_df.variant == 0).sum())

            p = naive_peek_p_value(cum_t_s, cum_t_n, cum_c_s, cum_c_n)
            if p < alpha:
                rejections += 1
                stopping_days.append(day)
                stopped = True
                break
        if not stopped:
            stopping_days.append(None)

    return {
        "n_simulations": n_simulations,
        "nominal_alpha": alpha,
        "empirical_false_positive_rate": rejections / n_simulations,
        "early_stops": rejections,
        "median_stopping_day": float(np.median([d for d in stopping_days if d is not None]))
        if rejections
        else None,
    }


def calibrate_bayesian_sequential(
    n_simulations: int,
    daily_n: int,
    days: int,
    baseline_rate: float,
    true_effect_absolute: float = 0.0,
    upper_threshold: float = 0.95,
    lower_threshold: float = 0.05,
    seed: int | None = None,
) -> dict:
    """Same protocol as `calibrate_naive_peeking`, but stopping on the
    Bayesian posterior probability crossing `upper_threshold` (declare
    treatment better) or `lower_threshold` (declare control better), checked
    daily. Under `true_effect_absolute=0.0`, the "declare treatment better"
    rate is the quantity to compare against the naive method's false-positive
    rate — this is the empirical check of whether daily Bayesian peeking is
    actually as safe as it is commonly assumed to be, not an assertion that
    it is.
    """
    rng = np.random.default_rng(seed)
    declared_treatment_better = 0
    stopping_days = []

    for i in range(n_simulations):
        df = simulate_daily_conversion_experiment(
            daily_n, days, baseline_rate, true_effect_absolute,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        cum_t_s = cum_t_n = cum_c_s = cum_c_n = 0
        stopped = False
        for day in range(days):
            day_df = df[df["day"] == day]
            cum_t_s += int(day_df.loc[day_df.variant == 1, "converted"].sum())
            cum_t_n += int((day_df.variant == 1).sum())
            cum_c_s += int(day_df.loc[day_df.variant == 0, "converted"].sum())
            cum_c_n += int((day_df.variant == 0).sum())

            prob = bayesian_probability_treatment_better(
                cum_t_s, cum_t_n, cum_c_s, cum_c_n,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            if prob > upper_threshold:
                declared_treatment_better += 1
                stopping_days.append(day)
                stopped = True
                break
            if prob < lower_threshold:
                stopping_days.append(day)
                stopped = True
                break
        if not stopped:
            stopping_days.append(None)

    return {
        "n_simulations": n_simulations,
        "empirical_false_positive_rate": declared_treatment_better / n_simulations,
        "declared_treatment_better": declared_treatment_better,
        "median_stopping_day": float(np.median([d for d in stopping_days if d is not None]))
        if any(d is not None for d in stopping_days)
        else None,
    }
