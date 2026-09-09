from src.novelty import detect_novelty_effect
from src.simulate import simulate_daily_conversion_experiment


def test_detects_injected_novelty_decay():
    # daily_n=500 (the repo's realistic pipeline scale) is not enough signal
    # for a LINEAR interaction term to reliably pick up an *exponential* decay
    # over only 30 days against per-day binomial noise (~0.03 sd) — a real,
    # measured power limitation of this diagnostic, not a bug: see the
    # power-vs-n sweep this test's parameters were chosen from.
    df = simulate_daily_conversion_experiment(
        daily_n=2000, days=30, baseline_rate=0.15,
        true_effect_absolute=0.10, novelty_decay=0.15, seed=0,
    )
    result = detect_novelty_effect(df["day"].to_numpy(), df["variant"].to_numpy(), df["converted"].to_numpy())
    assert result["novelty_effect_detected"] is True
    assert result["interaction_coefficient"] < 0  # effect shrinks over time


def test_no_false_novelty_alarm_on_constant_effect():
    df = simulate_daily_conversion_experiment(
        daily_n=800, days=30, baseline_rate=0.15,
        true_effect_absolute=0.03, novelty_decay=0.0, seed=0,
    )
    result = detect_novelty_effect(df["day"].to_numpy(), df["variant"].to_numpy(), df["converted"].to_numpy())
    assert result["novelty_effect_detected"] is False
