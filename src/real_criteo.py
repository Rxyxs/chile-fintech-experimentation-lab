"""The toolkit on a real randomized experiment: Criteo's uplift benchmark.

Everything else in this repository runs on simulated data, because a known
truth is the only way to check that a method does what it claims. This module
asks the complementary question: what do the same checks look like on a real
experiment, where nobody wrote the truth down?

Data: Criteo AI Lab's uplift dataset v2.1 (Diemert et al., "A Large Scale
Benchmark for Uplift Modeling", AdKDD 2018; CC BY-NC-SA 4.0), here from the
Kaggle mirror `arashnic/uplift-modeling` because the original download link no
longer resolves. 13,979,592 users of an incrementality test: 85% shown the
advertiser's ads, 15% held out, 12 anonymised features measured before the
test, and two outcomes over the next two weeks (visit, conversion).

    python -m src.real_criteo            # ~15 minutes on a laptop CPU

Four questions:

1. Randomization: does the split match the 85/15 design (SRM), and can the
   pre-treatment features tell the arms apart?
2. Effects: the intention-to-treat lift on visits and conversions, and what it
   implies per user actually exposed to an ad.
3. Variance reduction: CUPED with the best single feature, with all twelve
   (linear regression adjustment), and with a cross-fitted gradient-boosting
   prediction (CUPAC), against the 41.9% of the simulated module.
4. Sequential monitoring on real rows: A/A tests drawn from the treated users
   (no true difference by construction) checked 30 times, for naive peeking,
   the Bayesian rule, the fixed-horizon test and the mSPRT; then how many
   users the mSPRT needs to find the real effects, against a fixed design.
"""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.cuped import cuped_adjust
from src.msprt import always_valid_p_values
from src.randomization import srm_check
from src.sequential import bayesian_probability_treatment_better

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "uplift-modeling.zip"
OUT = ROOT / "outputs" / "tables" / "real_criteo.json"

FEATURES = [f"f{i}" for i in range(12)]
METRICS = ("visit", "conversion")
DESIGN_TREATED_SHARE = 0.85
ALPHA = 0.05
TAU_RELATIVE = 0.10    # mSPRT mixing scale: effects of about 10% of the control rate
TAU_SENSITIVITY = (0.3, 1.0)
SEED = 42


def load(path: Path = DATA) -> pd.DataFrame:
    dtypes = {**{f: "float32" for f in FEATURES},
              "treatment": "int8", "conversion": "int8", "visit": "int8", "exposure": "int8"}
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
            return pd.read_csv(f, dtype=dtypes)
    return pd.read_csv(path, dtype=dtypes)


def wilson(successes: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


# ---------------------------------------------------------------- 1. randomization
def randomization_checks(df: pd.DataFrame, seed: int = SEED, auc_sample: int = 2_000_000) -> dict:
    n = len(df)
    n_t = int(df["treatment"].sum())
    srm = srm_check({"control": n - n_t, "treatment": n_t},
                    {"control": 1 - DESIGN_TREATED_SHARE, "treatment": DESIGN_TREATED_SHARE})
    expected = DESIGN_TREATED_SHARE * n
    sd = np.sqrt(n * DESIGN_TREATED_SHARE * (1 - DESIGN_TREATED_SHARE))
    deviation = n_t - expected
    # How often a fair 85% coin lands at least this close to its expectation.
    p_this_close = float(2 * norm.cdf((abs(deviation) + 0.5) / sd) - 1)

    t = df["treatment"].to_numpy() == 1
    smd = {}
    for f in FEATURES:
        x = df[f].to_numpy(dtype=np.float64)
        smd[f] = float((x[t].mean() - x[~t].mean()) / np.sqrt((x[t].var() + x[~t].var()) / 2))

    # Can the features predict the arm? Two-fold cross-fitted, on a random sample.
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(auc_sample, n), replace=False)
    X = df[FEATURES].to_numpy()[idx]
    y = df["treatment"].to_numpy()[idx]
    fold = rng.random(len(idx)) < 0.5
    score = np.empty(len(idx))
    for train in (fold, ~fold):
        m = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(X[train], y[train])
        score[~train] = m.predict_proba(X[~train])[:, 1]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    auc_null_se = float(np.sqrt((n0 + n1 + 1) / (12 * n0 * n1)))
    return {
        "users": n, "treated": n_t, "treated_share": n_t / n,
        "srm_p_value": srm["p_value"], "srm_detected": srm["srm_detected"],
        "deviation_from_design_users": float(deviation), "binomial_sd_users": float(sd),
        "probability_of_landing_this_close": p_this_close,
        "standardized_mean_differences": smd,
        "max_abs_smd": float(max(abs(v) for v in smd.values())),
        "treatment_from_features_auc": float(roc_auc_score(y, score)),
        "auc_sample": int(len(idx)), "auc_null_se": auc_null_se,
    }


# ---------------------------------------------------------------- 2. effects
def _diff(y: np.ndarray, t: np.ndarray) -> dict:
    a, b = y[t], y[~t]
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"treated_mean": float(a.mean()), "control_mean": float(b.mean()), "difference": float(d),
            "se": float(se), "ci95": [float(d - 1.96 * se), float(d + 1.96 * se)],
            "p_value": float(2 * norm.sf(abs(d / se)))}


def effects(df: pd.DataFrame) -> dict:
    t = df["treatment"].to_numpy() == 1
    exposed_share = float(df.loc[t, "exposure"].mean())
    out = {"exposed_share_of_treated": exposed_share,
           "exposed_share_of_control": float(df.loc[~t, "exposure"].mean())}
    for m in METRICS:
        r = _diff(df[m].to_numpy(dtype=np.float64), t)
        r["relative_lift"] = r["difference"] / r["control_mean"]
        # Only treated users can be exposed, so the effect per exposed user is
        # the intention-to-treat effect divided by the exposure rate (Wald / IV).
        r["effect_per_exposed_user"] = r["difference"] / exposed_share
        r["effect_per_exposed_ci95"] = [c / exposed_share for c in r["ci95"]]
        out[m] = r
    return out


# ---------------------------------------------------------------- 3. variance reduction
def _adjusted(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Regression adjustment with a covariate built before the test (CUPED form)."""
    return cuped_adjust(y, pred)["y_cuped"]


def variance_reduction(df: pd.DataFrame, seed: int = SEED) -> dict:
    t = df["treatment"].to_numpy() == 1
    X = df[FEATURES].to_numpy()
    Xd = np.column_stack([np.ones(len(df)), X.astype(np.float64)])
    rng = np.random.default_rng(seed)
    fold = rng.random(len(df)) < 0.5
    out = {}
    for m in METRICS:
        y = df[m].to_numpy(dtype=np.float64)
        base = _diff(y, t)
        corr = {f: float(np.corrcoef(y, X[:, i])[0, 1]) for i, f in enumerate(FEATURES)}
        best = max(corr, key=lambda f: abs(corr[f]))

        single = cuped_adjust(y, df[best].to_numpy(dtype=np.float64))["y_cuped"]
        beta = np.linalg.solve(Xd.T @ Xd, Xd.T @ y)
        linear = _adjusted(y, Xd @ beta)

        # CUPAC: a model's prediction from pre-treatment features as the
        # covariate, cross-fitted so no user's covariate saw its own outcome.
        pred = np.empty(len(y))
        for train in (fold, ~fold):
            clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed)
            clf.fit(X[train], y[train].astype(np.int8))
            pred[~train] = clf.predict_proba(X[~train])[:, 1]
        cupac = _adjusted(y, pred)

        rows = {}
        for name, ya in (("best_single_feature", single), ("linear_12_features", linear), ("cupac_gbm", cupac)):
            r = _diff(ya, t)
            rows[name] = {"variance_reduction_pct": float(100 * (1 - ya.var() / y.var())),
                          "ci_width_ratio": float(r["se"] / base["se"]),
                          "difference": r["difference"], "ci95": r["ci95"]}
        out[m] = {"unadjusted": {"difference": base["difference"], "ci95": base["ci95"]},
                  "best_feature": best, "best_feature_corr": corr[best], **rows}
    return out


# ---------------------------------------------------------------- 4. sequential monitoring
def _z_p(st, nt, sc, nc):
    p = (st + sc) / (nt + nc)
    se = np.sqrt(p * (1 - p) * (1 / nt + 1 / nc))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (st / nt - sc / nc) / se, 0.0)
    return 2 * norm.sf(np.abs(z))


def aa_calibration(df: pd.DataFrame, metric: str, tau: float, reps: int = 400, looks: int = 30,
                   per_arm_per_look: int = 10_000, seed: int = SEED) -> dict:
    """A/A tests on real rows: two random halves of treated users, so there is
    no true difference, checked `looks` times with every rule."""
    pool = df.loc[df["treatment"] == 1, metric].to_numpy(dtype=np.int8)
    rng = np.random.default_rng(seed)
    k = looks * per_arm_per_look
    hits = {"naive_peeking": 0, "bayesian_treatment_better": 0, "bayesian_either_side": 0,
            "msprt": 0, "fixed_horizon": 0}
    for _ in range(reps):
        y = pool[rng.choice(len(pool), size=2 * k, replace=False)]
        a = y[:k].reshape(looks, per_arm_per_look).sum(axis=1).cumsum()
        b = y[k:].reshape(looks, per_arm_per_look).sum(axis=1).cumsum()
        n = per_arm_per_look * np.arange(1, looks + 1)
        p = _z_p(a, n, b, n)
        hits["naive_peeking"] += bool((p < ALPHA).any())
        hits["fixed_horizon"] += bool(p[-1] < ALPHA)
        hits["msprt"] += bool((always_valid_p_values(a, n, b, n, tau) < ALPHA).any())
        for j in range(looks):   # technique 6's Bayesian rule, unchanged
            prob = bayesian_probability_treatment_better(int(a[j]), int(n[j]), int(b[j]), int(n[j]),
                                                         seed=int(rng.integers(0, 2**31 - 1)))
            if prob > 0.95 or prob < 0.05:
                hits["bayesian_treatment_better"] += prob > 0.95
                hits["bayesian_either_side"] += 1
                break
    return {"metric": metric, "reps": reps, "looks": looks, "per_arm_per_look": per_arm_per_look,
            "base_rate": float(pool.mean()), "tau": tau,
            "false_positive_rate": {k_: v / reps for k_, v in hits.items()},
            "wilson95": {k_: wilson(v, reps) for k_, v in hits.items()}}


def fixed_horizon_users(p_t: float, p_c: float, share_t: float = DESIGN_TREATED_SHARE,
                        alpha: float = ALPHA, power: float = 0.80) -> int:
    """Total users a fixed-horizon test needs at this allocation to detect p_t - p_c."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    var = p_t * (1 - p_t) / share_t + p_c * (1 - p_c) / (1 - share_t)
    return int(np.ceil(z ** 2 * var / (p_t - p_c) ** 2))


def detection(df: pd.DataFrame, metric: str, tau: float, orders: int = 200, horizon_multiple: int = 4,
              looks_per_fixed: int = 20, seed: int = SEED) -> dict:
    """The real effect, streamed: users arrive in a random order, and every
    rule is checked every `fixed / looks_per_fixed` users."""
    y_all = df[metric].to_numpy(dtype=np.int8)
    t_all = df["treatment"].to_numpy(dtype=np.int8)
    p_t, p_c = y_all[t_all == 1].mean(), y_all[t_all == 0].mean()
    fixed = fixed_horizon_users(p_t, p_c)
    step = max(fixed // looks_per_fixed, 1)
    max_n = horizon_multiple * fixed
    n_looks = max_n // step
    rng = np.random.default_rng(seed)
    stop_msprt, stop_naive, fixed_rejects = [], [], 0
    for _ in range(orders):
        idx = rng.choice(len(y_all), size=n_looks * step, replace=False)
        y, t = y_all[idx].reshape(n_looks, step), t_all[idx].reshape(n_looks, step)
        st = (y * t).sum(axis=1).cumsum()
        nt = t.sum(axis=1).cumsum()
        sc = (y * (1 - t)).sum(axis=1).cumsum()
        nc = (1 - t).sum(axis=1).cumsum()
        users = step * np.arange(1, n_looks + 1)
        pv = always_valid_p_values(st, nt, sc, nc, tau)
        hit = np.flatnonzero(pv < ALPHA)
        stop_msprt.append(int(users[hit[0]]) if hit.size else None)
        naive = np.flatnonzero(_z_p(st, nt, sc, nc) < ALPHA)
        stop_naive.append(int(users[naive[0]]) if naive.size else None)
        j = looks_per_fixed - 1          # the look at the fixed-horizon sample size
        fixed_rejects += bool(_z_p(st[j], nt[j], sc[j], nc[j]) < ALPHA)
    ms = np.array([s if s is not None else np.inf for s in stop_msprt], dtype=float)
    return {
        "metric": metric, "tau": tau, "orders": orders, "true_rates": [float(p_t), float(p_c)],
        "fixed_horizon_users": fixed, "look_every_users": int(step), "max_users": int(n_looks * step),
        "fixed_horizon_power": fixed_rejects / orders,
        "msprt_detected_by_fixed": float(np.mean(ms <= fixed)),
        "msprt_detected_by_2x_fixed": float(np.mean(ms <= 2 * fixed)),
        "msprt_detected_by_max": float(np.mean(np.isfinite(ms))),
        "msprt_median_users_over_fixed": float(np.median(ms) / fixed) if np.isfinite(np.median(ms)) else None,
        "msprt_stopping_users": [s for s in stop_msprt],
        "naive_detected_by_fixed": float(np.mean([s is not None and s <= fixed for s in stop_naive])),
    }


def main() -> dict:
    start = time.perf_counter()
    df = load()
    print(f"Criteo uplift v2.1: {len(df):,} users")
    report = {"source": "Criteo AI Lab uplift v2.1 (CC BY-NC-SA 4.0), Kaggle mirror arashnic/uplift-modeling"}

    report["randomization"] = randomization_checks(df)
    r = report["randomization"]
    print(f"treated share {r['treated_share']:.7f}, SRM p {r['srm_p_value']:.4f}, "
          f"P(this close) {r['probability_of_landing_this_close']:.4f}, max |SMD| {r['max_abs_smd']:.4f}, "
          f"AUC {r['treatment_from_features_auc']:.4f} (null SE {r['auc_null_se']:.4f})")

    report["effects"] = effects(df)
    for m in METRICS:
        e = report["effects"][m]
        print(f"{m}: {e['control_mean']:.5f} -> {e['treated_mean']:.5f} ({e['relative_lift']:+.1%}), "
              f"per exposed user {e['effect_per_exposed_user']:+.4f}")

    report["variance_reduction"] = variance_reduction(df)
    for m in METRICS:
        v = report["variance_reduction"][m]
        print(m, {k: round(v[k]["variance_reduction_pct"], 2)
                  for k in ("best_single_feature", "linear_12_features", "cupac_gbm")})

    report["aa_calibration"], report["detection"] = {}, {}
    for m in METRICS:
        tau = TAU_RELATIVE * report["effects"][m]["control_mean"]
        report["aa_calibration"][m] = aa_calibration(df, m, tau)
        print("A/A", m, report["aa_calibration"][m]["false_positive_rate"])
        report["detection"][m] = detection(df, m, tau)
        d = report["detection"][m]
        print("detect", m, {k: d[k] for k in ("fixed_horizon_users", "fixed_horizon_power", "msprt_detected_by_fixed",
                                              "msprt_detected_by_2x_fixed", "msprt_median_users_over_fixed")})

    # How much the mSPRT's speed depends on tau, and whether its error control does.
    report["tau_sensitivity"] = {}
    for m in METRICS:
        report["tau_sensitivity"][m] = {}
        for rel in TAU_SENSITIVITY:
            tau = rel * report["effects"][m]["control_mean"]
            d = detection(df, m, tau)
            aa = aa_calibration(df, m, tau, reps=200)
            report["tau_sensitivity"][m][str(rel)] = {
                "tau": tau, "aa_false_positive_rate": aa["false_positive_rate"]["msprt"],
                **{k: d[k] for k in ("msprt_detected_by_fixed", "msprt_detected_by_2x_fixed",
                                     "msprt_detected_by_max", "msprt_median_users_over_fixed")}}
            print("tau", m, rel, report["tau_sensitivity"][m][str(rel)])

    report["elapsed_seconds"] = round(time.perf_counter() - start, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} in {report['elapsed_seconds']} s")
    return report


if __name__ == "__main__":
    main()
