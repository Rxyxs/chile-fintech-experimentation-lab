"""The real-experiment study, on a small frame shaped like Criteo's file."""
import numpy as np
import pandas as pd
import pytest

from src import real_criteo as rc


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    n = 60_000
    X = rng.normal(size=(n, 12)).astype("float32")
    t = (rng.random(n) < 0.85).astype("int8")
    base = 1 / (1 + np.exp(-(-3.0 + 1.2 * X[:, 3])))          # f3 drives the outcome
    exposure = (t == 1) & (rng.random(n) < 0.3)
    visit = (rng.random(n) < np.clip(base + 0.05 * exposure, 0, 1)).astype("int8")
    conversion = (visit == 1) & (rng.random(n) < 0.1)
    df = pd.DataFrame(X, columns=rc.FEATURES)
    df["treatment"] = t
    df["visit"] = visit
    df["conversion"] = conversion.astype("int8")
    df["exposure"] = exposure.astype("int8")
    return df


def test_randomization_checks_on_a_random_split(frame):
    r = rc.randomization_checks(frame, auc_sample=40_000)
    assert r["max_abs_smd"] < 0.05
    assert abs(r["treatment_from_features_auc"] - 0.5) < 4 * r["auc_null_se"] + 0.01
    assert 0 <= r["probability_of_landing_this_close"] <= 1


def test_effect_per_exposed_user_is_the_wald_ratio(frame):
    e = rc.effects(frame)
    v = e["visit"]
    assert v["effect_per_exposed_user"] == pytest.approx(v["difference"] / e["exposed_share_of_treated"])
    assert e["exposed_share_of_control"] == 0.0
    assert v["ci95"][0] < v["difference"] < v["ci95"][1]


def test_variance_reduction_finds_the_predictive_feature(frame):
    v = rc.variance_reduction(frame)
    assert v["visit"]["best_feature"] == "f3"
    for name in ("best_single_feature", "linear_12_features", "cupac_gbm"):
        assert v["visit"][name]["variance_reduction_pct"] > 0
        assert v["visit"][name]["ci_width_ratio"] < 1


def test_fixed_horizon_users_shrinks_with_the_effect():
    small = rc.fixed_horizon_users(0.045, 0.040)
    large = rc.fixed_horizon_users(0.060, 0.040)
    assert small > large > 0


def test_aa_calibration_and_detection_run(frame):
    aa = rc.aa_calibration(frame, "visit", tau=0.005, reps=20, looks=5, per_arm_per_look=1_000)
    assert set(aa["false_positive_rate"]) >= {"naive_peeking", "msprt", "fixed_horizon"}
    assert all(0 <= v <= 1 for v in aa["false_positive_rate"].values())
    d = rc.detection(frame, "visit", tau=0.005, orders=5, horizon_multiple=2, looks_per_fixed=5)
    assert d["fixed_horizon_users"] > 0
    assert 0 <= d["msprt_detected_by_max"] <= 1


def test_wilson_interval_contains_the_rate():
    lo, hi = rc.wilson(20, 400)
    assert lo < 0.05 < hi
