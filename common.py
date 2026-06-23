"""
ASTRAM Theme 2 - Common feature engineering, inference, and operational logic.

This module is the SINGLE SOURCE OF TRUTH used by both training (train.py) and
serving (app.py). Every feature the model sees at inference time is reconstructed
here from (a) the event attributes a dispatcher actually knows and (b) historical
lookup tables fitted at train time. This guarantees train/serve parity - no
feature is used in training that cannot be reproduced for a brand-new event.
"""

import json
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CSV_NAME = "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"


def find_csv():
    """Locate the dataset CSV across common layouts (env var wins)."""
    env = os.environ.get("ASTRAM_CSV")
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "data", CSV_NAME),
        os.path.join(here, "data", CSV_NAME),
        os.path.join(here, "..", "work-done", CSV_NAME),
        os.path.join(here, CSV_NAME),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        f"Could not find '{CSV_NAME}'. Set ASTRAM_CSV=/path/to/file.csv "
        f"or place it in a ./data/ folder next to the prototype.")

MORNING_PEAK = [5, 6]
EVENING_PEAK = [20, 21]
MIDDAY_TROUGH = list(range(10, 17))

# Causes with high resolution-time variance -> get dedicated quantile models.
HIGH_VARIANCE_CAUSES = ["pot_holes", "water_logging", "construction", "others"]

# Model feature set. Every entry is reproducible by build_feature_frame().
FEATURES = [
    # temporal
    "start_hour", "start_day", "start_month", "is_weekend",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
    "is_morning_peak", "is_evening_peak", "is_midday_trough",
    # event attributes
    "is_corridor", "is_high_priority", "requires_road_closure", "is_planned",
    "risk_of_cascade", "weighted_impact_score",
    # encoded categoricals
    "event_cause_encoded", "corridor_encoded", "priority_encoded",
    # cause-level historical aggregates
    "cause_median_resolution", "cause_mean_resolution", "cause_std_resolution",
    "cause_closure_rate", "cause_priority_high_rate",
    # corridor-level historical aggregates
    "corridor_count", "corridor_median_resolution", "corridor_breakdown_rate",
    "corridor_water_rate", "corridor_pothole_rate", "corridor_construction_rate",
    # corridor x cause aggregates
    "cc_median_resolution", "cc_count",
    # real-time signal (current concurrent load on the corridor)
    "current_active_load",
]


# ---------------------------------------------------------------------------
# Loading & target
# ---------------------------------------------------------------------------

def load_raw(filepath):
    """Load CSV and compute resolution_time_hours + base temporal/categorical cols."""
    df = pd.read_csv(filepath, low_memory=False)

    for col in ["start_datetime", "end_datetime", "modified_datetime",
                "created_date", "closed_datetime", "resolved_datetime"]:
        df[col] = pd.to_datetime(df[col], utc=True, format="mixed", errors="coerce")

    # Resolution time: prefer resolved_datetime, fall back to closed_datetime.
    df["resolution_time_hours"] = (
        df["resolved_datetime"] - df["start_datetime"]
    ).dt.total_seconds() / 3600
    mask = df["resolution_time_hours"].isna() & df["closed_datetime"].notna()
    df.loc[mask, "resolution_time_hours"] = (
        df.loc[mask, "closed_datetime"] - df.loc[mask, "start_datetime"]
    ).dt.total_seconds() / 3600

    df["corridor"] = df["corridor"].fillna("Non-corridor")
    df["priority"] = df["priority"].fillna("Low")
    df["event_cause"] = (
        df["event_cause"].fillna("others").str.lower().str.replace(" ", "_")
    )
    return df


# ---------------------------------------------------------------------------
# Lookup tables (fitted on training data, saved, reused at inference)
# ---------------------------------------------------------------------------

def fit_lookups(df):
    """Build historical aggregate lookups + category vocabularies from a frame.

    Pass only the TRAIN slice here during evaluation to avoid leakage; pass the
    full frame when fitting the final deployment artifacts.
    """
    d = df[df["resolution_time_hours"].notna() & (df["resolution_time_hours"] > 0)].copy()
    d["is_high_priority"] = (d["priority"] == "High").astype(int)
    d["requires_road_closure"] = _to_int(d["requires_road_closure"])

    cause = d.groupby("event_cause").agg(
        cause_median_resolution=("resolution_time_hours", "median"),
        cause_mean_resolution=("resolution_time_hours", "mean"),
        cause_std_resolution=("resolution_time_hours", "std"),
        cause_closure_rate=("requires_road_closure", "mean"),
        cause_priority_high_rate=("is_high_priority", "mean"),
    )

    corridor = d.groupby("corridor").agg(
        corridor_count=("id", "count"),
        corridor_median_resolution=("resolution_time_hours", "median"),
        corridor_breakdown_rate=("event_cause", lambda x: (x == "vehicle_breakdown").mean()),
        corridor_water_rate=("event_cause", lambda x: (x == "water_logging").mean()),
        corridor_pothole_rate=("event_cause", lambda x: (x == "pot_holes").mean()),
        corridor_construction_rate=("event_cause", lambda x: (x == "construction").mean()),
    )

    cc = d.groupby(["corridor", "event_cause"]).agg(
        cc_median_resolution=("resolution_time_hours", "median"),
        cc_count=("id", "count"),
    )

    lookups = {
        "cause": cause.to_dict("index"),
        "corridor": corridor.to_dict("index"),
        "cc": {f"{k[0]}||{k[1]}": v for k, v in cc.to_dict("index").items()},
        "global": {
            "median_resolution": float(d["resolution_time_hours"].median()),
            "mean_resolution": float(d["resolution_time_hours"].mean()),
            "std_resolution": float(d["resolution_time_hours"].std()),
            "closure_rate": float(d["requires_road_closure"].mean()),
            "priority_high_rate": float(d["is_high_priority"].mean()),
            "corridor_count": float(corridor["corridor_count"].median()),
        },
        "vocab": {
            "event_cause": sorted(d["event_cause"].unique().tolist()),
            "corridor": sorted(d["corridor"].unique().tolist()),
            "priority": sorted(d["priority"].unique().tolist()),
        },
        # typical concurrent load per corridor (median events in a 24h window),
        # used as the default real-time signal when none is supplied.
        "typical_load": _typical_load(d),
    }
    return lookups


def _typical_load(d):
    """Median number of events per active day per corridor (real-time proxy)."""
    d = d.copy()
    d["date"] = d["start_datetime"].dt.date
    daily = d.groupby(["corridor", "date"]).size().reset_index(name="n")
    med = daily.groupby("corridor")["n"].median()
    return {k: float(v) for k, v in med.items()}


def _to_int(s):
    """Coerce TRUE/FALSE/1/0/strings to 0/1 int."""
    return (
        s.astype(str).str.upper().isin(["TRUE", "1", "1.0", "YES"]).astype(int)
    )


def _vocab_index(vocab):
    """Build {value: code} maps from saved vocabularies (unseen -> -1)."""
    return {col: {v: i for i, v in enumerate(vals)} for col, vals in vocab.items()}


# ---------------------------------------------------------------------------
# Feature construction (used identically for training rows and live events)
# ---------------------------------------------------------------------------

def build_feature_frame(events, lookups):
    """Turn a frame of events into the model feature matrix using saved lookups.

    `events` must contain: event_cause, corridor, priority, requires_road_closure,
    is_planned, start_hour, start_day, start_month, is_weekend, and optionally
    current_active_load. Returns a DataFrame with exactly FEATURES columns.
    """
    e = events.copy().reset_index(drop=True)
    g = lookups["global"]
    idx = _vocab_index(lookups["vocab"])

    # temporal cyclical
    e["hour_sin"] = np.sin(2 * np.pi * e["start_hour"] / 24)
    e["hour_cos"] = np.cos(2 * np.pi * e["start_hour"] / 24)
    e["day_sin"] = np.sin(2 * np.pi * e["start_day"] / 7)
    e["day_cos"] = np.cos(2 * np.pi * e["start_day"] / 7)
    e["month_sin"] = np.sin(2 * np.pi * e["start_month"] / 12)
    e["month_cos"] = np.cos(2 * np.pi * e["start_month"] / 12)
    e["is_morning_peak"] = e["start_hour"].isin(MORNING_PEAK).astype(int)
    e["is_evening_peak"] = e["start_hour"].isin(EVENING_PEAK).astype(int)
    e["is_midday_trough"] = e["start_hour"].isin(MIDDAY_TROUGH).astype(int)

    # event flags
    e["is_corridor"] = (e["corridor"] != "Non-corridor").astype(int)
    e["is_high_priority"] = (e["priority"] == "High").astype(int)
    e["requires_road_closure"] = _to_int(e["requires_road_closure"])
    e["is_planned"] = _to_int(e.get("is_planned", pd.Series(0, index=e.index)))
    e["risk_of_cascade"] = e["event_cause"].apply(
        lambda x: 1.25 if x in ["construction", "pot_holes"] else 1.00
    )
    e["weighted_impact_score"] = (
        e["is_corridor"] * 0.5 + e["requires_road_closure"] * 0.3 + e["risk_of_cascade"] * 0.2
    )

    # encoded categoricals
    e["event_cause_encoded"] = e["event_cause"].map(idx["event_cause"]).fillna(-1).astype(int)
    e["corridor_encoded"] = e["corridor"].map(idx["corridor"]).fillna(-1).astype(int)
    e["priority_encoded"] = e["priority"].map(idx["priority"]).fillna(-1).astype(int)

    # cause aggregates
    e["cause_median_resolution"] = e["event_cause"].map(
        lambda c: lookups["cause"].get(c, {}).get("cause_median_resolution", g["median_resolution"]))
    e["cause_mean_resolution"] = e["event_cause"].map(
        lambda c: lookups["cause"].get(c, {}).get("cause_mean_resolution", g["mean_resolution"]))
    e["cause_std_resolution"] = e["event_cause"].map(
        lambda c: lookups["cause"].get(c, {}).get("cause_std_resolution", g["std_resolution"]))
    e["cause_closure_rate"] = e["event_cause"].map(
        lambda c: lookups["cause"].get(c, {}).get("cause_closure_rate", g["closure_rate"]))
    e["cause_priority_high_rate"] = e["event_cause"].map(
        lambda c: lookups["cause"].get(c, {}).get("cause_priority_high_rate", g["priority_high_rate"]))

    # corridor aggregates
    e["corridor_count"] = e["corridor"].map(
        lambda c: lookups["corridor"].get(c, {}).get("corridor_count", g["corridor_count"]))
    e["corridor_median_resolution"] = e["corridor"].map(
        lambda c: lookups["corridor"].get(c, {}).get("corridor_median_resolution", g["median_resolution"]))
    for rate in ["breakdown", "water", "pothole", "construction"]:
        col = f"corridor_{rate}_rate"
        e[col] = e["corridor"].map(lambda c, col=col: lookups["corridor"].get(c, {}).get(col, 0.0))

    # corridor x cause aggregates
    def cc_get(row, field, default):
        return lookups["cc"].get(f"{row['corridor']}||{row['event_cause']}", {}).get(field, default)
    e["cc_median_resolution"] = e.apply(
        lambda r: cc_get(r, "cc_median_resolution", r["cause_median_resolution"]), axis=1)
    e["cc_count"] = e.apply(lambda r: cc_get(r, "cc_count", 0), axis=1)

    # real-time load
    if "current_active_load" not in e.columns:
        e["current_active_load"] = e["corridor"].map(
            lambda c: lookups["typical_load"].get(c, 1.0))
    e["current_active_load"] = e["current_active_load"].fillna(1.0)

    return e[FEATURES].astype(float)


def predict_intervals(events, models, lookups):
    """Predict P10/P50/P90 resolution hours for a frame of events.

    Uses the cause-specific quantile model when one exists for the event's
    cause (high-variance causes), else the global model. Applies the stored
    conformal width and enforces P10 <= P50 <= P90. Returns a DataFrame with
    columns p10, p50, p90, model_used.
    """
    X = build_feature_frame(events, lookups)
    causes = events["event_cause"].reset_index(drop=True)
    n = len(events)
    out = {"p10": np.zeros(n), "p50": np.zeros(n), "p90": np.zeros(n)}
    model_used = ["global"] * n

    def apply(idx, qmods, delta):
        sub = X.iloc[idx]
        lo = qmods["p10"].predict(sub) - delta
        mid = qmods["p50"].predict(sub)
        hi = qmods["p90"].predict(sub) + delta
        s = np.sort(np.vstack([lo, mid, hi]).T, axis=1)
        out["p10"][idx] = np.clip(np.expm1(s[:, 0]), 0, None)
        out["p50"][idx] = np.clip(np.expm1(s[:, 1]), 0, None)
        out["p90"][idx] = np.clip(np.expm1(s[:, 2]), 0, None)

    cause_models = models.get("cause_quantile", {})
    cause_delta = models.get("conformal_cause", {})
    handled = np.zeros(n, dtype=bool)
    for cause, qmods in cause_models.items():
        idx = np.where(causes.values == cause)[0]
        if len(idx):
            apply(idx, qmods, cause_delta.get(cause, 0.0))
            for i in idx:
                model_used[i] = f"cause:{cause}"
            handled[idx] = True
    rest = np.where(~handled)[0]
    if len(rest):
        apply(rest, models["quantile"], models.get("conformal_global", 0.0))

    res = pd.DataFrame(out)
    res["model_used"] = model_used
    return res


def base_temporal(dt):
    """Extract the temporal fields build_feature_frame expects from a Timestamp."""
    return {
        "start_hour": dt.hour,
        "start_day": dt.weekday(),
        "start_month": dt.month,
        "is_weekend": int(dt.weekday() in (5, 6)),
    }


# ---------------------------------------------------------------------------
# Operational recommendation logic (rule-based, grounded in the data findings)
# ---------------------------------------------------------------------------

# Per-cause response templates. Equipment/crew scaled by severity downstream.
CAUSE_PLAYBOOK = {
    "vehicle_breakdown": {
        "team": "Quick-response towing unit",
        "equipment": ["Tow truck / crane (sized to vehicle)"],
        "barricade": "Lane-taper cones around the stalled vehicle",
    },
    "water_logging": {
        "team": "Drainage / pumping crew",
        "equipment": ["High-capacity water pump", "Warning signage"],
        "barricade": "Hard barricades around the flooded stretch",
    },
    "pot_holes": {
        "team": "Road-repair / patching crew",
        "equipment": ["Cold-mix / patching material", "Compactor"],
        "barricade": "Cone off the damaged lane",
    },
    "construction": {
        "team": "Traffic-management + works crew",
        "equipment": ["Lane-closure barricades", "Variable message sign"],
        "barricade": "Full work-zone barricading with taper",
    },
    "tree_fall": {
        "team": "Clearance crew",
        "equipment": ["Chainsaw / cutter", "Recovery vehicle"],
        "barricade": "Cone off the obstructed lane",
    },
    "others": {
        "team": "General response team",
        "equipment": ["Standard response kit"],
        "barricade": "Cones as required",
    },
}

# Heuristic parallel-route hints for the major corridors (diversion labels only -
# ASTRAM would resolve the exact route from its live network graph).
DIVERSION_HINTS = {
    "Tumkur Road": "Divert via Mathikere / CV Raman Nagar service roads",
    "ORR East 1": "Divert via Marathahalli ORR service road / inner ring",
    "ORR East 2": "Divert via Bellandur - Iblur service road",
    "ORR South": "Divert via BTM / Bannerghatta Road",
    "Mysore Road": "Divert via Magadi Road / Nayandahalli underpass",
    "Hosur Road": "Divert via Koramangala 80ft / inner ring",
    "Old Madras Road": "Divert via Indiranagar 100ft / CMH Road",
    "Ballari Road": "Divert via Hebbal service road / Bellary feeder",
}


def severity_band(p50, requires_closure, is_corridor):
    """Classify operational severity from the median prediction and context."""
    score = 0
    if p50 >= 24: score += 2
    elif p50 >= 6: score += 1
    if requires_closure: score += 1
    if is_corridor: score += 1
    if score >= 3:
        return "CRITICAL"
    if score >= 2:
        return "HIGH"
    if score >= 1:
        return "MEDIUM"
    return "LOW"


def recommend_resources(cause, corridor, requires_closure, is_corridor, p50, p90):
    """Return a concrete operational recommendation dict for one event."""
    play = CAUSE_PLAYBOOK.get(cause, CAUSE_PLAYBOOK["others"])
    band = severity_band(p50, requires_closure, is_corridor)

    crew_by_band = {"LOW": 1, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    personnel = crew_by_band[band]
    teams = max(1, personnel)

    equipment = list(play["equipment"])
    if band in ("HIGH", "CRITICAL"):
        equipment.append("Backup unit on standby")

    # Diversion: only meaningful when a closure is needed or impact is high.
    if requires_closure or band in ("HIGH", "CRITICAL"):
        diversion = DIVERSION_HINTS.get(
            corridor, "Set up contraflow / divert to nearest service road")
    else:
        diversion = "No diversion needed - manage in-lane"

    return {
        "severity": band,
        "response_team": play["team"],
        "teams_to_dispatch": teams,
        "field_personnel": personnel * 2,  # ~2 per team
        "equipment": equipment,
        "barricading": play["barricade"] if (requires_closure or band != "LOW")
                        else "Minimal - cones only",
        "diversion_plan": diversion,
        "expected_clearance_h": round(p50, 1),
        "worst_case_clearance_h": round(p90, 1),
    }
