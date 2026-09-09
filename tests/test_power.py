import pytest

from src.power import (
    minimum_detectable_effect_two_proportion,
    sample_size_two_mean,
    sample_size_two_proportion,
)


def test_sample_size_matches_known_textbook_value():
    # Classic example: baseline 20%, MDE +5pp absolute, alpha=0.05, power=0.80.
    # Matches Evan Miller's A/B testing sample size calculator (1,091/arm) for
    # this exact input, verified by hand from the pooled-variance formula:
    # n = (z_a/2 + z_b)^2 * (p1(1-p1) + p2(1-p2)) / (p1-p2)^2
    #   = (1.95996 + 0.84162)^2 * (0.16 + 0.1875) / 0.0025 = 1090.9 -> 1091
    n = sample_size_two_proportion(baseline_rate=0.20, mde_absolute=0.05, alpha=0.05, power=0.80)
    assert n == 1091


def test_sample_size_decreases_with_larger_effect():
    small = sample_size_two_proportion(0.10, 0.01)
    large = sample_size_two_proportion(0.10, 0.05)
    assert large < small


def test_sample_size_increases_with_higher_power():
    n_80 = sample_size_two_proportion(0.10, 0.02, power=0.80)
    n_95 = sample_size_two_proportion(0.10, 0.02, power=0.95)
    assert n_95 > n_80


def test_invalid_baseline_rate_raises():
    with pytest.raises(ValueError):
        sample_size_two_proportion(baseline_rate=1.5, mde_absolute=0.05)


def test_sample_size_two_mean_positive():
    n = sample_size_two_mean(std=20.0, mde_absolute=3.0)
    assert n > 0


def test_mde_inverts_sample_size():
    baseline = 0.15
    n = sample_size_two_proportion(baseline, mde_absolute=0.03)
    mde_recovered = minimum_detectable_effect_two_proportion(baseline, n_per_arm=n)
    assert mde_recovered == pytest.approx(0.03, abs=0.002)
