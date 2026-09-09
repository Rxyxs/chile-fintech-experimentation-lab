import numpy as np

from src.cuped import cuped_adjust
from src.simulate import simulate_revenue_experiment_with_covariate


def test_cuped_reduces_variance_when_covariate_is_correlated():
    df = simulate_revenue_experiment_with_covariate(
        n_per_arm=2000, true_effect_absolute=2.0, covariate_correlation=0.7, seed=1,
    )
    result = cuped_adjust(df["y"].to_numpy(), df["x_pre"].to_numpy())
    assert result["var_after"] < result["var_before"]
    assert result["variance_reduction_pct"] > 20


def test_cuped_preserves_the_mean_difference():
    df = simulate_revenue_experiment_with_covariate(
        n_per_arm=3000, true_effect_absolute=5.0, covariate_correlation=0.6, seed=2,
    )
    result = cuped_adjust(df["y"].to_numpy(), df["x_pre"].to_numpy())
    y_cuped = result["y_cuped"]

    raw_diff = df.loc[df.variant == 1, "y"].mean() - df.loc[df.variant == 0, "y"].mean()
    cuped_diff = y_cuped[df.variant == 1].mean() - y_cuped[df.variant == 0].mean()
    # CUPED must not shift the *point estimate* of the treatment effect,
    # only shrink its variance
    assert abs(raw_diff - cuped_diff) < 0.5


def test_cuped_no_op_when_covariate_is_uncorrelated():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, 5000)
    x_uncorrelated = rng.normal(0, 1, 5000)
    result = cuped_adjust(y, x_uncorrelated)
    assert abs(result["theta"]) < 0.1
    assert result["variance_reduction_pct"] < 5


def test_cuped_handles_zero_variance_covariate():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    x_constant = np.array([5.0, 5.0, 5.0, 5.0])
    result = cuped_adjust(y, x_constant)
    assert result["theta"] == 0.0
