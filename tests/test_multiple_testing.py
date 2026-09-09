import numpy as np

from src.multiple_testing import benjamini_hochberg


def test_bh_rejects_more_than_bonferroni_would():
    p_values = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.74])
    result = benjamini_hochberg(p_values, alpha=0.05)
    # Bonferroni at alpha=0.05/7 =~ 0.0071 would only reject the first p-value;
    # BH is less conservative and should reject at least that many
    assert result["reject"].sum() >= 1
    assert result["reject"][0] == True  # noqa: E712


def test_bh_q_values_are_monotonic_with_sorted_p_values():
    p_values = np.array([0.5, 0.01, 0.2, 0.001, 0.3])
    result = benjamini_hochberg(p_values)
    order = np.argsort(p_values)
    q_sorted = result["q_values"][order]
    assert np.all(np.diff(q_sorted) >= -1e-9)


def test_bh_rejects_nothing_when_all_p_values_are_large():
    p_values = np.array([0.4, 0.5, 0.6, 0.7])
    result = benjamini_hochberg(p_values, alpha=0.05)
    assert not result["reject"].any()


def test_bh_empty_input():
    result = benjamini_hochberg(np.array([]))
    assert len(result["reject"]) == 0
