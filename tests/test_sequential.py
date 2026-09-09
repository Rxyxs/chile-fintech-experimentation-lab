from src.sequential import bayesian_probability_treatment_better, naive_peek_p_value


def test_naive_p_value_is_one_with_no_data():
    assert naive_peek_p_value(0, 0, 0, 0) == 1.0


def test_naive_p_value_small_when_arms_clearly_differ():
    p = naive_peek_p_value(successes_t=600, n_t=1000, successes_c=400, n_c=1000)
    assert p < 0.001


def test_naive_p_value_large_when_arms_are_identical():
    p = naive_peek_p_value(successes_t=500, n_t=1000, successes_c=500, n_c=1000)
    assert p > 0.9


def test_bayesian_probability_near_one_when_treatment_clearly_better():
    prob = bayesian_probability_treatment_better(
        successes_t=600, n_t=1000, successes_c=400, n_c=1000, seed=1,
    )
    assert prob > 0.99


def test_bayesian_probability_near_half_with_no_data():
    prob = bayesian_probability_treatment_better(0, 0, 0, 0, seed=1)
    assert 0.4 < prob < 0.6
