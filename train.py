"""
ASTRAM Theme 2 - Training pipeline.

Trains, evaluates (temporal holdout), and saves:
  - a point model (median resolution time)
  - global quantile models P10 / P50 / P90  (prediction intervals)
  - cause-specific quantile models for the high-variance causes
  - historical lookup tables used for feature construction at serve time

Run:  .venv/bin/python train.py
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

import common

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = common.find_csv()
ART = os.path.join(HERE, "artifacts")
os.makedirs(ART, exist_ok=True)

QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
OUTLIER_Q = 0.995          # drop clear data-entry tails above this
MIN_CAUSE_SAMPLES = 80     # min rows to train a dedicated cause model


def lgb_point():
    return lgb.LGBMRegressor(
        objective="regression", n_estimators=400, learning_rate=0.05,
        num_leaves=31, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1,
    )


def lgb_quantile(alpha):
    return lgb.LGBMRegressor(
        objective="quantile", alpha=alpha, n_estimators=400, learning_rate=0.05,
        num_leaves=31, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1,
    )


def pinball(y_true, y_pred, alpha):
    d = y_true - y_pred
    return np.mean(np.maximum(alpha * d, (alpha - 1) * d))


def conformal_delta(qlo_log, qhi_log, y_log, target=0.80):
    """Conformalized Quantile Regression width (in log space).

    Romano et al. 2019: on a calibration set, the conformity score is how far
    the true value falls outside the predicted [qlo, qhi] band. Widening the
    band by the (target)-quantile of those scores guarantees ~target coverage
    on exchangeable future data. Returns a scalar added to qhi / subtracted
    from qlo at inference.
    """
    if len(y_log) < 10:
        return 0.0
    scores = np.maximum(qlo_log - y_log, y_log - qhi_log)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * target) / n)
    return float(np.quantile(scores, level))


def prep(df):
    """Filter to modellable rows and cap the extreme tail."""
    d = df[df["resolution_time_hours"].notna() & (df["resolution_time_hours"] > 0)].copy()
    cap = d["resolution_time_hours"].quantile(OUTLIER_Q)
    d = d[d["resolution_time_hours"] <= cap]
    d = d.sort_values("start_datetime").reset_index(drop=True)
    # temporal fields used by the feature builder
    d["start_hour"] = d["start_datetime"].dt.hour
    d["start_day"] = d["start_datetime"].dt.dayofweek
    d["start_month"] = d["start_datetime"].dt.month
    d["is_weekend"] = d["start_day"].isin([5, 6]).astype(int)
    d["is_planned"] = (d["event_type"] == "planned").astype(int)
    return d, cap


def main():
    print("=" * 64)
    print("ASTRAM Theme 2 - Resolution-time model training")
    print("=" * 64)

    raw = common.load_raw(CSV)
    print(f"Loaded {len(raw):,} events "
          f"({raw['start_datetime'].min().date()} -> {raw['start_datetime'].max().date()})")

    d, cap = prep(raw)
    print(f"Modellable rows: {len(d):,}  (capped at {cap:.1f}h, {OUTLIER_Q:.1%})")

    # ---- 3-way temporal split: train / calibrate / test -----------------
    # train -> fit models, calib -> conformal width, test -> honest metrics.
    i1, i2 = int(len(d) * 0.64), int(len(d) * 0.80)
    train, calib, test = d.iloc[:i1], d.iloc[i1:i2], d.iloc[i2:]
    print(f"Temporal split -> train {len(train):,} / calib {len(calib):,} / test {len(test):,}")

    lk = common.fit_lookups(train)                 # leakage-safe: train only
    Xtr = common.build_feature_frame(train, lk)
    Xca = common.build_feature_frame(calib, lk)
    Xte = common.build_feature_frame(test, lk)
    ytr = np.log1p(train["resolution_time_hours"].values)
    yca_log = np.log1p(calib["resolution_time_hours"].values)
    yte = test["resolution_time_hours"].values

    # ---- point model -----------------------------------------------------
    print("\n[1/3] Point model (median resolution time)")
    point = lgb_point().fit(Xtr, ytr)
    pred_point = np.clip(np.expm1(point.predict(Xte)), 0, None)
    mae = np.mean(np.abs(yte - pred_point))
    medae = np.median(np.abs(yte - pred_point))
    print(f"      Test MAE   : {mae:7.2f} h  (mean - tail-dominated)")
    print(f"      Test MedAE : {medae:7.2f} h  (median abs error - typical event)")

    # ---- global quantile models + conformal calibration -----------------
    print("\n[2/3] Global quantile models  P10 / P50 / P90  (+ CQR calibration)")
    qm = {name: lgb_quantile(a).fit(Xtr, ytr) for name, a in QUANTILES.items()}
    delta = conformal_delta(qm["p10"].predict(Xca), qm["p90"].predict(Xca), yca_log)

    def q_eval(models, d_adj, X, y):
        lo = models["p10"].predict(X) - d_adj
        mid = models["p50"].predict(X)
        hi = models["p90"].predict(X) + d_adj
        s = np.sort(np.vstack([lo, mid, hi]).T, axis=1)
        p10, p50, p90 = (np.clip(np.expm1(s[:, i]), 0, None) for i in range(3))
        cov = np.mean((y >= p10) & (y <= p90))
        return p10, p50, p90, cov

    p10, p50c, p90, coverage = q_eval(qm, delta, Xte, yte)
    raw10, _, raw90, raw_cov = q_eval(qm, 0.0, Xte, yte)
    p50_mae = np.mean(np.abs(yte - p50c))
    print(f"      P50 MAE              : {p50_mae:7.2f} h")
    print(f"      P10-P90 coverage     : raw {raw_cov:5.1%}  ->  calibrated {coverage:5.1%}  (target 80%)")
    print(f"      conformal width (log): {delta:.3f}")

    # ---- cause-specific quantile models ---------------------------------
    print("\n[3/3] Cause-specific quantile models (high-variance causes)")
    cause_models, cause_delta, cause_eval = {}, {}, {}
    for cause in common.HIGH_VARIANCE_CAUSES:
        ctr = train[train["event_cause"] == cause]
        cca = calib[calib["event_cause"] == cause]
        cte = test[test["event_cause"] == cause]
        if len(ctr) < MIN_CAUSE_SAMPLES:
            print(f"      {cause:16s}: skipped (only {len(ctr)} train rows)")
            continue
        cXtr = common.build_feature_frame(ctr, lk)
        cytr = np.log1p(ctr["resolution_time_hours"].values)
        trip = {name: lgb_quantile(a).fit(cXtr, cytr) for name, a in QUANTILES.items()}
        cause_models[cause] = trip
        cd = 0.0
        if len(cca) >= 10:
            cXca = common.build_feature_frame(cca, lk)
            cd = conformal_delta(trip["p10"].predict(cXca), trip["p90"].predict(cXca),
                                 np.log1p(cca["resolution_time_hours"].values))
        cause_delta[cause] = cd

        msg = f"      {cause:16s}: n_train={len(ctr):4d}"
        if len(cte) >= 10:
            cXte = common.build_feature_frame(cte, lk)
            cyte = cte["resolution_time_hours"].values
            _, _, _, cov = q_eval(trip, cd, cXte, cyte)
            _, cp50, _, _ = q_eval(trip, cd, cXte, cyte)
            cmae = np.mean(np.abs(cyte - cp50))
            cause_eval[cause] = {"n_test": int(len(cte)), "p50_mae": float(cmae),
                                 "coverage": float(cov)}
            msg += f"  test_n={len(cte):3d}  P50 MAE={cmae:7.2f}h  cov(cal)={cov:5.1%}"
        print(msg)

    # ---- refit everything on FULL data for deployment -------------------
    # Models fit on all data; conformal deltas carried over from the calibration
    # split (the full-fit band is slightly tighter, so deltas stay conservative).
    print("\nRefitting on full dataset for deployment artifacts...")
    lk_full = common.fit_lookups(d)
    Xall = common.build_feature_frame(d, lk_full)
    yall = np.log1p(d["resolution_time_hours"].values)

    point_full = lgb_point().fit(Xall, yall)
    qfull = {name: lgb_quantile(a).fit(Xall, yall) for name, a in QUANTILES.items()}
    cause_full = {}
    for cause in cause_models:
        cdf = d[d["event_cause"] == cause]
        cX = common.build_feature_frame(cdf, lk_full)
        cy = np.log1p(cdf["resolution_time_hours"].values)
        cause_full[cause] = {name: lgb_quantile(a).fit(cX, cy)
                             for name, a in QUANTILES.items()}

    # ---- save ------------------------------------------------------------
    joblib.dump({
        "point": point_full,
        "quantile": qfull,
        "cause_quantile": cause_full,
        "conformal_global": delta,
        "conformal_cause": cause_delta,
        "features": common.FEATURES,
    }, os.path.join(ART, "models.joblib"))

    with open(os.path.join(ART, "lookups.json"), "w") as f:
        json.dump(lk_full, f)

    metrics = {
        "n_events": int(len(raw)),
        "n_modelled": int(len(d)),
        "outlier_cap_h": float(cap),
        "point_mae_h": float(mae),
        "point_medae_h": float(medae),
        "p50_mae_h": float(p50_mae),
        "p10_p90_coverage_raw": float(raw_cov),
        "p10_p90_coverage": float(coverage),
        "cause_models": list(cause_full.keys()),
        "cause_eval": cause_eval,
        "date_range": [str(raw["start_datetime"].min().date()),
                       str(raw["start_datetime"].max().date())],
    }
    with open(os.path.join(ART, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved artifacts to", ART)
    print("  - models.joblib   (point + quantile + cause-specific)")
    print("  - lookups.json    (historical feature tables)")
    print("  - metrics.json    (evaluation summary)")
    print("\nDone.")


if __name__ == "__main__":
    main()
