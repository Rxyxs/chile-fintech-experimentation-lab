"""End-to-end pipeline: run with `python -m src.pipeline`.

Scenario: a Chilean fintech app is testing an in-app nudge that offers to
auto-enroll users in a savings product, aiming to lift 30-day activation.
Every number below comes from an actual run of this script on simulated
data with a known, deliberately injected ground truth — the only reason any
of these techniques can be checked against a real answer at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.calibration import calibrate_bayesian_sequential, calibrate_naive_peeking
from src.cuped import cuped_adjust
from src.multiple_testing import benjamini_hochberg
from src.novelty import detect_novelty_effect
from src.power import sample_size_two_proportion
from src.randomization import assign_variant, srm_check
from src.simulate import simulate_daily_conversion_experiment, simulate_revenue_experiment_with_covariate

BASELINE_RATE = 0.12
MDE_ABSOLUTE = 0.015
ALPHA = 0.05
POWER = 0.80
SEED = 42

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def run() -> dict:
    report: dict = {}

    section("1. Power analysis")
    n_per_arm = sample_size_two_proportion(BASELINE_RATE, MDE_ABSOLUTE, ALPHA, POWER)
    print(f"Baseline activation rate: {BASELINE_RATE:.1%}")
    print(f"Minimum detectable effect: +{MDE_ABSOLUTE:.1%} absolute")
    print(f"Required sample size per arm: {n_per_arm:,}")
    report["power_analysis"] = {
        "baseline_rate": BASELINE_RATE,
        "mde_absolute": MDE_ABSOLUTE,
        "n_per_arm": n_per_arm,
    }

    section("2. Randomization + Sample Ratio Mismatch check")
    assignment = assign_variant(n_per_arm * 2, ratio=0.5, seed=SEED)
    counts = {"control": int((assignment == 0).sum()), "treatment": int((assignment == 1).sum())}
    srm_result = srm_check(counts, {"control": 0.5, "treatment": 0.5})
    print(f"Observed split: {counts}")
    print(f"SRM check p-value: {srm_result['p_value']:.4f} -> SRM detected: {srm_result['srm_detected']}")
    report["srm_check"] = {k: v for k, v in srm_result.items() if k != "labels"}

    section("3. CUPED variance reduction (revenue metric)")
    rev_df = simulate_revenue_experiment_with_covariate(
        n_per_arm=n_per_arm, true_effect_absolute=3.0, covariate_correlation=0.65, seed=SEED,
    )
    cuped_result = cuped_adjust(rev_df["y"].to_numpy(), rev_df["x_pre"].to_numpy())
    print(f"theta = {cuped_result['theta']:.3f}")
    print(f"Variance before CUPED: {cuped_result['var_before']:.2f}")
    print(f"Variance after CUPED:  {cuped_result['var_after']:.2f}")
    print(f"Variance reduction: {cuped_result['variance_reduction_pct']:.1f}%")
    report["cuped"] = {k: v for k, v in cuped_result.items() if k != "y_cuped"}

    section("4. Multiple-testing correction across guardrail metrics")
    guardrail_p_values = np.array([0.001, 0.032, 0.041, 0.048, 0.29, 0.61, 0.77])
    guardrail_names = [
        "primary_activation", "latency_p95", "unsubscribe_rate",
        "support_tickets", "app_crashes", "revenue_per_user", "nps_score",
    ]
    bh_result = benjamini_hochberg(guardrail_p_values, alpha=ALPHA)
    for name, p, q, rej in zip(guardrail_names, guardrail_p_values, bh_result["q_values"], bh_result["reject"]):
        print(f"  {name:22s} p={p:.3f}  q={q:.3f}  reject={rej}")
    report["multiple_testing"] = {
        "metrics": guardrail_names,
        "p_values": guardrail_p_values.tolist(),
        "q_values": bh_result["q_values"].tolist(),
        "reject": bh_result["reject"].tolist(),
    }

    section("5. Novelty-effect check (decaying true effect)")
    # daily_n=2000 and a larger injected effect on purpose: this diagnostic's
    # own test suite showed a linear treatment:day interaction term needs a
    # meaningfully powered sample to separate a real exponential decay from
    # per-day binomial noise over just 21-30 days — see tests/test_novelty.py.
    novelty_df = simulate_daily_conversion_experiment(
        daily_n=2000, days=21, baseline_rate=BASELINE_RATE,
        true_effect_absolute=MDE_ABSOLUTE * 6, novelty_decay=0.10, seed=SEED,
    )
    novelty_result = detect_novelty_effect(
        novelty_df["day"].to_numpy(), novelty_df["variant"].to_numpy(), novelty_df["converted"].to_numpy(),
    )
    print(f"treatment:day interaction coefficient: {novelty_result['interaction_coefficient']:.5f}")
    print(f"interaction p-value: {novelty_result['interaction_p_value']:.4f}")
    print(f"Novelty effect detected: {novelty_result['novelty_effect_detected']}")
    report["novelty"] = {
        "interaction_coefficient": novelty_result["interaction_coefficient"],
        "interaction_p_value": novelty_result["interaction_p_value"],
        "novelty_effect_detected": novelty_result["novelty_effect_detected"],
    }

    section("6. Calibration: naive peeking vs. Bayesian sequential (under the null)")
    n_sims = 400
    naive_cal = calibrate_naive_peeking(
        n_simulations=n_sims, daily_n=150, days=21, baseline_rate=BASELINE_RATE,
        true_effect_absolute=0.0, alpha=ALPHA, seed=SEED,
    )
    bayes_cal = calibrate_bayesian_sequential(
        n_simulations=n_sims, daily_n=150, days=21, baseline_rate=BASELINE_RATE,
        true_effect_absolute=0.0, seed=SEED,
    )
    print(f"Naive daily peeking, nominal alpha={ALPHA}:")
    print(f"  empirical false-positive rate: {naive_cal['empirical_false_positive_rate']:.1%}"
          f"  ({naive_cal['early_stops']}/{n_sims} simulations)")
    print(f"Bayesian sequential (stop at P>0.95), same {n_sims} simulations:")
    print(f"  empirical false-positive rate: {bayes_cal['empirical_false_positive_rate']:.1%}"
          f"  ({bayes_cal['declared_treatment_better']}/{n_sims} simulations)")
    report["calibration"] = {"naive_peeking": naive_cal, "bayesian_sequential": bayes_cal}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "pipeline_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report written to {OUTPUT_DIR / 'pipeline_report.json'}")

    return report


if __name__ == "__main__":
    run()
