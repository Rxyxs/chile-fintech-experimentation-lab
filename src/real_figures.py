"""Figures for the real-experiment section: run with `python -m src.real_figures`.

Drawn from `outputs/tables/real_criteo.json`, so they can be redrawn without
downloading the Criteo file again.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.make_figures import ACCENT, CONTROL, INK, OK, TREAT, _save, _style
from src.real_criteo import OUT

METRIC_LABEL = {"visit": "Visit (base 4.9%)", "conversion": "Conversion (base 0.31%)"}


def figure_aa(r: dict) -> None:
    rules = [("naive_peeking", "Naive daily peeking", TREAT),
             ("bayesian_treatment_better", "Bayesian P > 0.95", ACCENT),
             ("fixed_horizon", "Fixed horizon (one look)", OK),
             ("msprt", "mSPRT", CONTROL)]
    fig, ax = plt.subplots(figsize=(9.4, 5))
    width = 0.2
    for j, (key, label, color) in enumerate(rules):
        for i, m in enumerate(("visit", "conversion")):
            a = r["aa_calibration"][m]
            rate = a["false_positive_rate"][key]
            lo, hi = a["wilson95"][key]
            x = i + (j - 1.5) * width
            ax.bar(x, rate, width * 0.92, color=color, label=label if i == 0 else None)
            ax.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], fmt="none", ecolor=INK, capsize=3, linewidth=1)
            ax.text(x, hi + 0.008, f"{rate:.1%}", ha="center", fontsize=8.5, color=INK)
    ax.axhline(0.05, color=OK, linestyle="--", linewidth=1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([METRIC_LABEL["visit"], METRIC_LABEL["conversion"]])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 0.4)
    _style(ax, ylabel="False-positive rate over 30 looks")
    ax.set_title("A/A tests on real Criteo users: two random halves of the treated group,\n"
                 "so every rejection is a false positive", fontsize=11.5, color=INK, pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    a = r["aa_calibration"]["visit"]
    fig.text(0.5, -0.05, f"{a['reps']} A/A tests per metric, {a['looks']} looks of {a['per_arm_per_look']:,} "
             "users per arm each. Wilson 95% intervals. Dashed line: the 5% every rule promises.",
             ha="center", fontsize=8.5, color="#666666")
    _save(fig, "real_aa_false_positive_rates.png")


def figure_adjustment(r: dict) -> None:
    names = [("unadjusted", "Unadjusted"), ("best_single_feature", "CUPED, best feature"),
             ("linear_12_features", "Linear, 12 features"), ("cupac_gbm", "CUPAC (boosting)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, m in zip(axes, ("visit", "conversion")):
        v = r["variance_reduction"][m]
        for k, (key, label) in enumerate(names):
            d, (lo, hi) = 100 * v[key]["difference"], (100 * c for c in v[key]["ci95"])
            color = INK if key == "unadjusted" else CONTROL
            ax.errorbar(d, k, xerr=[[d - lo], [hi - d]], fmt="o", color=color, capsize=4, markersize=6)
            note = "" if key == "unadjusted" else f"  variance −{v[key]['variance_reduction_pct']:.0f}%"
            ax.text(hi, k, f"  {d:.3f} pp{note}", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n[1] for n in names])
        ax.invert_yaxis()
        lo_all = min(100 * v[k]["ci95"][0] for k, _ in names)
        hi_all = max(100 * v[k]["ci95"][1] for k, _ in names)
        ax.set_xlim(lo_all - 0.15 * (hi_all - lo_all), hi_all + 0.9 * (hi_all - lo_all))
        _style(ax, title=METRIC_LABEL[m], xlabel="Treatment effect, percentage points (95% CI)")
    fig.suptitle("On this file, adjusting for pre-test features moves the estimate, not only the interval:\n"
                 "the arms differ on those features", fontsize=11.5, color=INK, y=1.04)
    fig.tight_layout()
    _save(fig, "real_adjustment_moves_the_estimate.png")


def figure_msprt_stopping(r: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for m, color in (("visit", TREAT), ("conversion", CONTROL)):
        d = r["detection"][m]
        stops = np.array([s / d["fixed_horizon_users"] if s is not None else np.inf for s in d["msprt_stopping_users"]])
        grid = np.linspace(0, d["max_users"] / d["fixed_horizon_users"], 200)
        ax.plot(grid, [(stops <= g).mean() for g in grid], color=color, linewidth=2,
                label=f"{m}: fixed design needs {d['fixed_horizon_users']:,} users")
    ax.axvline(1, color=OK, linestyle="--", linewidth=1.2)
    ax.axhline(0.8, color="#999999", linestyle=":", linewidth=1)
    ax.text(1.03, 0.62, "fixed-horizon sample\n(80% power, one look)", fontsize=8.5, color=OK)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    _style(ax, xlabel="Users seen, as a multiple of the fixed-horizon sample",
           ylabel="Share of random orders where the mSPRT has stopped")
    ax.set_title("The price of being allowed to look: on the real effects, the mSPRT\n"
                 "(tau = 10% of the base rate) needs more users than a fixed design", fontsize=11.5, color=INK, pad=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _save(fig, "real_msprt_stopping.png")


def main() -> None:
    r = json.loads(OUT.read_text(encoding="utf-8"))
    figure_aa(r)
    figure_adjustment(r)
    figure_msprt_stopping(r)


if __name__ == "__main__":
    main()
