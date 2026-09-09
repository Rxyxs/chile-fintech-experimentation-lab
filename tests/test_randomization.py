import numpy as np
import pytest

from src.randomization import assign_variant, srm_check


def test_assign_variant_hits_exact_ratio():
    assignment = assign_variant(10_000, ratio=0.5, seed=1)
    assert assignment.sum() == 5000


def test_assign_variant_is_shuffled_not_blocked():
    assignment = assign_variant(1000, ratio=0.5, seed=1)
    # if it were left as [0]*500 + [1]*500, the first 10 values would all be 0
    assert assignment[:10].sum() > 0


def test_srm_check_passes_on_balanced_split():
    result = srm_check({"control": 5001, "treatment": 4999}, {"control": 0.5, "treatment": 0.5})
    assert result["srm_detected"] is False


def test_srm_check_flags_real_mismatch():
    # a 47/53 split on 20k units is not plausible chance variation from 50/50
    result = srm_check({"control": 9400, "treatment": 10600}, {"control": 0.5, "treatment": 0.5})
    assert result["srm_detected"] is True


def test_srm_check_rejects_mismatched_keys():
    with pytest.raises(ValueError):
        srm_check({"a": 100}, {"b": 1.0})
