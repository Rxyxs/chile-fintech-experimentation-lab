import numpy as np
import pytest

from src.calibration import calibrate_msprt, calibrate_naive_peeking
from src.msprt import always_valid_p_values, msprt_statistic


def test_no_difference_gives_no_evidence():
    """theta_hat = 0 leaves only the sqrt(V / (V + tau^2)) factor, below 1."""
    lam = msprt_statistic(50, 1000, 50, 1000, tau=0.02)
    v = 0.05 * 0.95 * (2 / 1000)
    assert lam == pytest.approx(np.sqrt(v / (v + 0.02 ** 2)))
    assert lam < 1


def test_matches_the_closed_form():
    st, nt, sc, nc, tau = 130, 1000, 100, 1000, 0.03
    theta = st / nt - sc / nc
    p = (st + sc) / (nt + nc)
    v = p * (1 - p) * (1 / nt + 1 / nc)
    expected = np.sqrt(v / (v + tau ** 2)) * np.exp(tau ** 2 * theta ** 2 / (2 * v * (v + tau ** 2)))
    assert msprt_statistic(st, nt, sc, nc, tau) == pytest.approx(expected)


def test_empty_arm_or_zero_variance_is_neutral():
    assert msprt_statistic(0, 0, 5, 100, tau=0.01) == 1.0
    assert msprt_statistic(0, 100, 0, 100, tau=0.01) == 1.0


def test_p_value_is_a_running_minimum():
    st = np.array([10, 30, 45, 50])
    nt = np.array([100, 200, 300, 400])
    sc = np.array([5, 10, 30, 50])
    nc = np.array([100, 200, 300, 400])
    p = always_valid_p_values(st, nt, sc, nc, tau=0.05)
    assert np.all(np.diff(p) <= 0)
    assert np.all((p > 0) & (p <= 1))


def test_tau_must_be_positive():
    with pytest.raises(ValueError):
        msprt_statistic(1, 10, 1, 10, tau=0.0)


def test_msprt_controls_the_false_positive_rate_under_daily_peeking():
    """The property naive peeking lacks: same null simulations, daily looks."""
    kw = dict(n_simulations=300, daily_n=150, days=21, baseline_rate=0.12, seed=7)
    msprt = calibrate_msprt(tau=0.015, **kw)
    naive = calibrate_naive_peeking(**kw)
    assert msprt["empirical_false_positive_rate"] <= 0.05 + 0.02
    assert naive["empirical_false_positive_rate"] > 0.10


def test_msprt_detects_a_real_effect():
    res = calibrate_msprt(n_simulations=100, daily_n=400, days=21, baseline_rate=0.12,
                          tau=0.03, true_effect_absolute=0.04, seed=3)
    assert res["rejection_rate"] > 0.8
