from src.calibration import calibrate_bayesian_sequential, calibrate_naive_peeking


def test_naive_peeking_inflates_false_positive_rate_above_nominal_alpha():
    result = calibrate_naive_peeking(
        n_simulations=150, daily_n=100, days=14, baseline_rate=0.15,
        true_effect_absolute=0.0, alpha=0.05, seed=1,
    )
    # this is the headline honest finding of this repo: checking a fixed-alpha
    # test every day inflates the false-positive rate well past the nominal 5%
    assert result["empirical_false_positive_rate"] > 0.10


def test_calibration_reports_required_fields():
    result = calibrate_naive_peeking(
        n_simulations=30, daily_n=50, days=7, baseline_rate=0.15, seed=2,
    )
    assert "empirical_false_positive_rate" in result
    assert 0.0 <= result["empirical_false_positive_rate"] <= 1.0


def test_bayesian_sequential_calibration_runs_end_to_end():
    result = calibrate_bayesian_sequential(
        n_simulations=30, daily_n=50, days=7, baseline_rate=0.15,
        true_effect_absolute=0.0, seed=3,
    )
    assert 0.0 <= result["empirical_false_positive_rate"] <= 1.0
