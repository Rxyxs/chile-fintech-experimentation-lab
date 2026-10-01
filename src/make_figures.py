"""Figures for the README: run with `python -m src.make_figures`.

Every figure is drawn from the same functions and the same constants
`src/pipeline.py` uses, with the same seed, so the numbers annotated on the
charts are the numbers the pipeline prints. Nothing here is illustrative.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.calibration import calibrate_bayesian_sequential, calibrate_naive_peeking
from src.cuped import cuped_adjust
from src.multiple_testing import benjamini_hochberg
from src.novelty import detect_novelty_effect
from src.power import sample_size_two_proportion
from src.simulate import simulate_daily_conversion_experiment, simulate_revenue_experiment_with_covariate

BASELINE_RATE = 0.12
MDE_ABSOLUTE = 0.015
ALPHA = 0.05
POWER = 0.80
SEED = 42

FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"

INK = "#2B2B2B"
GRID = "#D9D9D9"
CONTROL = "#6E8CA0"
TREAT = "#B5553D"
ACCENT = "#8A5A2C"
OK = "#4C7A3E"


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)
    if title:
        ax.set_title(title, fontsize=11.5, color=INK, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
    return ax


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(FIG_DIR.parent.parent)}")


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval. Used instead of the normal approximation because
    these are proportions near the small end (5%) on n=400, where the normal
    interval can run below zero."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------
# 1. The headline: what each stopping rule claims vs. what it delivers
# --------------------------------------------------------------------------
def figure_calibration(n_sims=400):
    print("1/6 calibration_false_positive_rate ...")
    naive = calibrate_naive_peeking(
        n_simulations=n_sims, daily_n=150, days=21, baseline_rate=BASELINE_RATE,
        true_effect_absolute=0.0, alpha=ALPHA, seed=SEED,
    )
    bayes = calibrate_bayesian_sequential(
        n_simulations=n_sims, daily_n=150, days=21, baseline_rate=BASELINE_RATE,
        true_effect_absolute=0.0, seed=SEED,
    )

    labels = ["Nominal\n(what both claim)", "Naive daily peeking\n(fixed-α z-test)",
              "Bayesian sequential\nP(treat. better) > 0.95"]
    rates = [ALPHA, naive["empirical_false_positive_rate"], bayes["empirical_false_positive_rate"]]
    hits = [None, naive["early_stops"], bayes["declared_treatment_better"]]
    colors = [OK, TREAT, ACCENT]

    fig, ax = plt.subplots(figsize=(8.2, 5))
    bars = ax.bar(labels, rates, color=colors, width=0.58, edgecolor="white", linewidth=1.2)

    for bar, rate, hit in zip(bars, rates, hits):
        if hit is None:
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.008, f"{rate:.1%}",
                    ha="center", fontsize=11, fontweight="bold", color=OK)
            continue
        lo, hi = wilson_interval(hit, n_sims)
        ax.errorbar(bar.get_x() + bar.get_width() / 2, rate, yerr=[[rate - lo], [hi - rate]],
                    fmt="none", ecolor=INK, elinewidth=1.3, capsize=5)
        ax.text(bar.get_x() + bar.get_width() / 2, hi + 0.009,
                f"{rate:.1%}\n{hit}/{n_sims}\n95% CI [{lo:.1%}, {hi:.1%}]",
                ha="center", fontsize=9, color=INK)

    ax.axhline(ALPHA, color=OK, linestyle="--", linewidth=1.2, zorder=0)
    ax.text(-0.46, ALPHA + 0.006, "nominal α = 5%", fontsize=8.5, color=OK, ha="left")

    _style(ax, ylabel="Empirical false-positive rate")
    ax.set_title("Both arms drawn from the SAME 12% conversion rate:\nevery rejection below is a false positive",
                 fontsize=11.5, color=INK, pad=14)
    ax.set_ylim(0, 0.345)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    fig.text(0.5, -0.07,
             f"{n_sims} simulated experiments per rule, 21 days x 150 users/arm/day, seed {SEED}.\n"
             "Intervals are Wilson score intervals. Note they overlap: at this number of simulations\n"
             "the two naive rules are not distinguishable from each other, only from the 5% they claim.",
             ha="center", fontsize=8.5, color="#666666")
    _save(fig, "calibration_false_positive_rate.png")
    return naive, bayes


# --------------------------------------------------------------------------
# 2. Where the inflation comes from: one look is fine, twenty-one are not
# --------------------------------------------------------------------------
def figure_peeking_inflation(n_sims=400):
    print("2/6 peeking_inflation_curve ...")
    day_grid = [1, 2, 3, 5, 8, 11, 14, 17, 21]
    rates, los, his = [], [], []
    for d in day_grid:
        res = calibrate_naive_peeking(
            n_simulations=n_sims, daily_n=150, days=d, baseline_rate=BASELINE_RATE,
            true_effect_absolute=0.0, alpha=ALPHA, seed=SEED,
        )
        r = res["empirical_false_positive_rate"]
        lo, hi = wilson_interval(res["early_stops"], n_sims)
        rates.append(r), los.append(lo), his.append(hi)

    fig, ax = plt.subplots(figsize=(8.4, 5))
    ax.fill_between(day_grid, los, his, color=TREAT, alpha=0.16, linewidth=0)
    ax.plot(day_grid, rates, color=TREAT, linewidth=2, marker="o", markersize=5,
            markerfacecolor="white", markeredgewidth=1.6, label="Fixed-α z-test, checked once per day")
    ax.axhline(ALPHA, color=OK, linestyle="--", linewidth=1.3,
               label="Nominal α = 5% (what the test promises)")

    ax.annotate(f"{rates[0]:.1%}\none look only:\nthe promise holds",
                xy=(day_grid[0], rates[0]), xytext=(2.2, 0.105), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.9))
    ax.annotate(f"{rates[-1]:.1%}\n{day_grid[-1]} looks",
                xy=(day_grid[-1], rates[-1]), xytext=(16.2, 0.135), fontsize=9.5,
                color=TREAT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=TREAT, linewidth=0.9))

    _style(ax, xlabel="Days the experiment runs (checked every day)",
           ylabel="Empirical false-positive rate")
    ax.set_title("The false-positive rate is not inflated by a point or two.\nIt compounds with every look.",
                 fontsize=11.5, color=INK, pad=12)
    ax.set_ylim(0, 0.30)
    ax.set_xticks(day_grid)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.text(0.5, -0.06,
             f"{n_sims} null simulations per point (both arms at 12%). Each point is a separate experiment\n"
             "of that length, checked daily. Band is the Wilson 95% interval.",
             ha="center", fontsize=8.5, color="#666666")
    _save(fig, "peeking_inflation_curve.png")


# --------------------------------------------------------------------------
# 3. Power: the cost of wanting to detect a smaller effect
# --------------------------------------------------------------------------
def figure_power_curve():
    print("3/6 power_curve_mde ...")
    mdes = np.linspace(0.005, 0.05, 120)
    n_12 = [sample_size_two_proportion(BASELINE_RATE, m, ALPHA, POWER) for m in mdes]
    n_20 = [sample_size_two_proportion(0.20, m, ALPHA, POWER) for m in mdes]

    fig, ax = plt.subplots(figsize=(8.4, 5))
    ax.plot(mdes * 100, n_12, color=TREAT, linewidth=2,
            label="Baseline 12% — this experiment's activation rate")
    ax.plot(mdes * 100, n_20, color=CONTROL, linewidth=1.8, linestyle="--",
            label="Baseline 20% — the textbook case used to validate the formula")

    n_op = sample_size_two_proportion(BASELINE_RATE, MDE_ABSOLUTE, ALPHA, POWER)
    ax.scatter([MDE_ABSOLUTE * 100], [n_op], s=70, color=TREAT, zorder=5,
               edgecolor="white", linewidth=1.5)
    ax.annotate(f"+{MDE_ABSOLUTE * 100:.1f}pp -> {n_op:,} per arm\n(this experiment)",
                xy=(MDE_ABSOLUTE * 100, n_op), xytext=(2.6, 22000), fontsize=9.5,
                color=TREAT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=TREAT, linewidth=0.9))

    n_val = sample_size_two_proportion(0.20, 0.05, ALPHA, POWER)
    ax.scatter([5.0], [n_val], s=55, color=CONTROL, zorder=5, edgecolor="white", linewidth=1.5)
    ax.annotate(f"+5pp on a 20% baseline -> {n_val:,}\nmatches Evan Miller's calculator",
                xy=(5.0, n_val), xytext=(2.45, 620), fontsize=9, color=CONTROL,
                arrowprops=dict(arrowstyle="->", color=CONTROL, linewidth=0.9))

    # The title's claim, computed rather than asserted.
    n_double = sample_size_two_proportion(BASELINE_RATE, MDE_ABSOLUTE * 2, ALPHA, POWER)
    ax.annotate("", xy=(MDE_ABSOLUTE * 200, n_double), xytext=(MDE_ABSOLUTE * 100, n_op),
                arrowprops=dict(arrowstyle="<->", color="#9A9A9A", linewidth=1, linestyle=":"))
    ax.text(4.25, 26000, f"double the MDE,\n{n_op / n_double:.1f}x fewer users\n({n_op:,} -> {n_double:,})",
            fontsize=8.5, color="#777777", ha="center")

    _style(ax, xlabel="Minimum detectable effect (percentage points, absolute)",
           ylabel="Required sample size per arm (log scale)")
    ax.set_title(f"Halving the effect you want to detect roughly quadruples the sample\n"
                 f"(α = {ALPHA}, power = {POWER:.0%}, two-sided)", fontsize=11.5, color=INK, pad=12)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(frameon=False, fontsize=9)
    _save(fig, "power_curve_mde.png")


# --------------------------------------------------------------------------
# 4. CUPED: a tighter interval around the same answer
# --------------------------------------------------------------------------
def figure_cuped():
    print("4/6 cuped_variance_reduction ...")
    n_arm = sample_size_two_proportion(BASELINE_RATE, MDE_ABSOLUTE, ALPHA, POWER)
    df = simulate_revenue_experiment_with_covariate(
        n_per_arm=n_arm, true_effect_absolute=3.0, covariate_correlation=0.65, seed=SEED,
    )
    y = df["y"].to_numpy()
    x_pre = df["x_pre"].to_numpy()
    variant = df["variant"].to_numpy()
    res = cuped_adjust(y, x_pre)
    y_c = res["y_cuped"]
    rho = float(np.corrcoef(x_pre, y)[0, 1])

    def effect_ci(values):
        t, c = values[variant == 1], values[variant == 0]
        diff = t.mean() - c.mean()
        se = np.sqrt(t.var(ddof=1) / t.size + c.var(ddof=1) / c.size)
        return diff, 1.96 * se

    d_raw, h_raw = effect_ci(y)
    d_cup, h_cup = effect_ci(y_c)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))

    bins = np.linspace(min(y.min(), y_c.min()), max(y.max(), y_c.max()), 60)
    ax1.hist(y, bins=bins, color=CONTROL, alpha=0.62, label=f"Raw outcome (var {res['var_before']:.1f})")
    ax1.hist(y_c, bins=bins, color=TREAT, alpha=0.62, label=f"CUPED-adjusted (var {res['var_after']:.1f})")
    _style(ax1, xlabel="30-day revenue per user", ylabel="Users")
    ax1.set_title(f"CUPED removes the variance the pre-period already explained\n"
                  f"{res['variance_reduction_pct']:.1f}% reduction", fontsize=11, color=INK, pad=10)
    ax1.legend(frameon=False, fontsize=8.5)

    ax2.errorbar([0], [d_raw], yerr=[h_raw], fmt="o", color=CONTROL, markersize=9,
                 capsize=7, elinewidth=1.8, label="Raw")
    ax2.errorbar([1], [d_cup], yerr=[h_cup], fmt="o", color=TREAT, markersize=9,
                 capsize=7, elinewidth=1.8, label="CUPED")
    ax2.axhline(3.0, color=OK, linestyle="--", linewidth=1.2)
    ax2.text(1.42, 3.06, "true injected\neffect = 3.0", fontsize=8.5, color=OK, ha="right")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([f"Raw\n{d_raw:.3f} ± {h_raw:.3f}", f"CUPED\n{d_cup:.3f} ± {h_cup:.3f}"])
    ax2.set_xlim(-0.5, 1.5)
    _style(ax2, ylabel="Estimated treatment effect (95% CI)")
    ax2.set_title(f"Effect estimate barely moves ({abs(d_cup - d_raw):.3f}, {abs(d_cup - d_raw) / 3.0:.1%} of it)\n"
                  f"while the interval shrinks {1 - h_cup / h_raw:.1%}",
                  fontsize=11, color=INK, pad=10)

    fig.text(0.5, -0.055,
             f"n = {n_arm:,} per arm. Realized pre/post correlation ρ = {rho:.4f}, so CUPED's theoretical ceiling\n"
             f"is ρ² = {rho ** 2:.2%} — the measured {res['variance_reduction_pct']:.2f}% sits exactly on it, "
             "not short of it.",
             ha="center", fontsize=8.5, color="#666666")
    _save(fig, "cuped_variance_reduction.png")
    return res, rho


# --------------------------------------------------------------------------
# 5. BH-FDR: four metrics look significant, one survives
# --------------------------------------------------------------------------
def figure_bh_fdr():
    print("5/6 bh_fdr_guardrails ...")
    p = np.array([0.001, 0.032, 0.041, 0.048, 0.29, 0.61, 0.77])
    names = ["primary_activation", "latency_p95", "unsubscribe_rate", "support_tickets",
             "app_crashes", "revenue_per_user", "nps_score"]
    res = benjamini_hochberg(p, alpha=ALPHA)

    order = np.argsort(p)
    m = p.size
    ranks = np.arange(1, m + 1)
    crit = ranks / m * ALPHA

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(ranks, crit, color=TREAT, linewidth=1.8, marker="s", markersize=4,
            label=f"BH critical value  (i/m)·α = (i/7)·{ALPHA}")
    ax.axhline(ALPHA, color=CONTROL, linestyle="--", linewidth=1.4,
               label=f"Raw α = {ALPHA}  (no correction)")

    for i, idx in enumerate(order):
        rejected = bool(res["reject"][idx])
        ax.scatter(i + 1, p[idx], s=95, zorder=5,
                   color=OK if rejected else "#9A9A9A",
                   edgecolor="white", linewidth=1.5)
        label = names[idx] + ("\n✓ survives BH" if rejected else "")
        # Centred above each point: ranks 3 and 4 have p-values 0.041 and 0.048,
        # so side-placed labels run into each other.
        ax.annotate(label, xy=(i + 1, p[idx]), xytext=(0, 11),
                    textcoords="offset points", fontsize=8.6, ha="center",
                    color=OK if rejected else INK,
                    fontweight="bold" if rejected else "normal")

    n_raw = int((p <= ALPHA).sum())
    n_bh = int(res["reject"].sum())
    _style(ax, xlabel="Metric, ranked by p-value (i)", ylabel="p-value (log scale)")
    ax.set_title(f"At a raw α of 5%, {n_raw} of {m} guardrail metrics look significant.\n"
                 f"Under BH-FDR control, {n_bh} does.", fontsize=11.5, color=INK, pad=12)
    ax.set_yscale("log")
    ax.set_xlim(0.4, m + 0.7)
    ax.set_xticks(ranks)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.text(0.5, -0.07,
             "A metric is rejected only where it falls below the BH line at its own rank — not merely\n"
             "below the raw α line. The three near-α metrics (latency, unsubscribes, support tickets)\n"
             "are exactly the guardrails a launch decision should not overreact to.",
             ha="center", fontsize=8.5, color="#666666")
    _save(fig, "bh_fdr_guardrails.png")
    return n_raw, n_bh


# --------------------------------------------------------------------------
# 6. The honest limitation: a straight line through a curved decay
# --------------------------------------------------------------------------
def figure_novelty():
    print("6/6 novelty_decay_vs_linear_fit ...")
    days, daily_n = 21, 2000
    true_effect, decay = MDE_ABSOLUTE * 6, 0.10
    df = simulate_daily_conversion_experiment(
        daily_n=daily_n, days=days, baseline_rate=BASELINE_RATE,
        true_effect_absolute=true_effect, novelty_decay=decay, seed=SEED,
    )
    res = detect_novelty_effect(df["day"].to_numpy(), df["variant"].to_numpy(),
                                df["converted"].to_numpy())

    per_day = df.groupby(["day", "variant"])["converted"].mean().unstack()
    observed_lift = (per_day[1] - per_day[0]).to_numpy()
    day_idx = per_day.index.to_numpy()

    injected = true_effect * np.exp(-decay * day_idx)
    coef = res["interaction_coefficient"]
    fitted = coef * day_idx + (observed_lift.mean() - coef * day_idx.mean())

    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    ax.scatter(day_idx, observed_lift * 100, s=42, color=CONTROL, alpha=0.8, zorder=3,
               edgecolor="white", linewidth=1, label="Observed daily lift (treatment − control)")
    ax.plot(day_idx, injected * 100, color=OK, linewidth=2.2,
            label=f"True injected effect: {true_effect:.1%}·e^(−{decay}·day)")
    ax.plot(day_idx, fitted * 100, color=TREAT, linewidth=2, linestyle="--",
            label=f"Fitted linear treatment×day term (coef {coef:.5f})")
    ax.axhline(0, color=GRID, linewidth=1)

    _style(ax, xlabel="Day of experiment", ylabel="Lift in activation rate (pp)")
    ax.set_title("The diagnostic fires correctly (p < 0.0001) — but a straight line is\n"
                 "the wrong shape for an exponential decay",
                 fontsize=11.5, color=INK, pad=12)
    ax.legend(frameon=False, fontsize=8.8)
    fig.text(0.5, -0.10,
             f"{daily_n:,} users/arm/day over {days} days. A straight line through a convex decay crosses it twice:\n"
             "it understates the lift at both ends and overstates it in the middle. By the last days the fitted line\n"
             f"has gone negative — claiming the nudge now hurts activation, when the true effect is still "
             f"+{injected[-1] * 100:.1f}pp and positive.\n"
             "The direction and the p-value are right; the shape is not. At the pipeline's more realistic daily\n"
             "volume it stops separating decay from binomial noise at all — see tests/test_novelty.py.",
             ha="center", fontsize=8.5, color="#666666")
    _save(fig, "novelty_decay_vs_linear_fit.png")
    return res


if __name__ == "__main__":
    print(f"Writing figures to {FIG_DIR}\n")
    naive, bayes = figure_calibration()
    figure_peeking_inflation()
    figure_power_curve()
    cuped, rho = figure_cuped()
    n_raw, n_bh = figure_bh_fdr()
    nov = figure_novelty()

    print("\nNumbers annotated on the figures:")
    print(f"  naive peeking FPR      : {naive['empirical_false_positive_rate']:.1%} "
          f"({naive['early_stops']}/400)")
    print(f"  bayesian sequential FPR: {bayes['empirical_false_positive_rate']:.1%} "
          f"({bayes['declared_treatment_better']}/400)")
    print(f"  CUPED reduction        : {cuped['variance_reduction_pct']:.4f}%  "
          f"(realized rho={rho:.4f}, rho^2={rho ** 2:.4%})")
    print(f"  BH guardrails          : {n_raw} below raw alpha -> {n_bh} survive BH")
    print(f"  novelty interaction    : {nov['interaction_coefficient']:.5f} "
          f"(p={nov['interaction_p_value']:.4f})")
