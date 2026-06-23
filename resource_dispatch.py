"""
ASTRAM Theme 2 - Resource dispatch artifact generator.

Turns historical demand + the trained resolution model into a concrete,
shift-ready pre-positioning plan. Produces three artifacts:

  artifacts/demand_matrix.csv   - events by hour x corridor (the raw heatmap)
  artifacts/dispatch_plan.csv   - per corridor x time-window staging plan
  artifacts/dispatch_plan.json  - same plan as structured JSON for the app/API

Run:  .venv/bin/python resource_dispatch.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

import common

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = common.find_csv()
ART = os.path.join(HERE, "artifacts")

# Time windows used for shift planning.
WINDOWS = {
    "Morning Peak (05-07)": [5, 6, 7],
    "Daytime (08-16)": list(range(8, 17)),
    "Evening Peak (17-22)": [17, 18, 19, 20, 21, 22],
    "Overnight (23-04)": [23, 0, 1, 2, 3, 4],
}

EQUIPMENT_BY_CAUSE = {
    "vehicle_breakdown": "tow_trucks",
    "water_logging": "water_pumps",
    "pot_holes": "patching_crews",
    "construction": "tm_crews",
    "tree_fall": "clearance_crews",
    "others": "general_units",
}


def staffing_for(n_events):
    """Map a window's expected event count to a crew count (simple, auditable)."""
    if n_events >= 40:
        return 3
    if n_events >= 15:
        return 2
    if n_events >= 4:
        return 1
    return 0


def main():
    print("Building resource dispatch artifacts...")
    raw = common.load_raw(CSV)
    bundle = joblib.load(os.path.join(ART, "models.joblib"))
    with open(os.path.join(ART, "lookups.json")) as f:
        lookups = json.load(f)

    df = raw.copy()
    df["start_hour"] = df["start_datetime"].dt.hour

    # ---- 1. demand matrix: events by hour x corridor --------------------
    demand = (df.groupby(["corridor", "start_hour"]).size()
                .reset_index(name="event_count"))
    demand_wide = demand.pivot(index="corridor", columns="start_hour",
                               values="event_count").fillna(0).astype(int)
    demand_wide = demand_wide.loc[demand_wide.sum(axis=1).sort_values(ascending=False).index]
    demand_wide.to_csv(os.path.join(ART, "demand_matrix.csv"))
    print(f"  demand_matrix.csv     ({demand_wide.shape[0]} corridors x 24h)")

    # ---- 2. dispatch plan: corridor x window ----------------------------
    # Focus on real corridors with meaningful volume.
    corridor_totals = df[df["corridor"] != "Non-corridor"]["corridor"].value_counts()
    top_corridors = corridor_totals[corridor_totals >= 20].index.tolist()

    rows = []
    for corridor in top_corridors:
        cdf = df[df["corridor"] == corridor]
        for wname, hours in WINDOWS.items():
            w = cdf[cdf["start_hour"].isin(hours)]
            n = len(w)
            if n == 0:
                continue
            # per-day rate over the ~5 month window (≈150 days)
            days = max(1, (df["start_datetime"].max() - df["start_datetime"].min()).days)
            per_day = n / days
            dominant = w["event_cause"].mode().iloc[0] if len(w) else "others"
            crews = staffing_for(n)
            if crews == 0:
                continue

            # predicted typical clearance for the dominant cause here
            ev = pd.DataFrame([{
                "event_cause": dominant, "corridor": corridor,
                "priority": "High" if corridor != "Non-corridor" else "Low",
                "requires_road_closure": int(w["requires_road_closure"]
                                             .pipe(common._to_int).mean() > 0.5),
                "is_planned": 0,
                "start_hour": hours[len(hours)//2], "start_day": 2,
                "start_month": 6, "is_weekend": 0,
            }])
            pi = common.predict_intervals(ev, bundle, lookups).iloc[0]

            equip_kind = EQUIPMENT_BY_CAUSE.get(dominant, "general_units")
            rows.append({
                "corridor": corridor,
                "time_window": wname,
                "hist_events": n,
                "events_per_day": round(per_day, 2),
                "dominant_cause": dominant,
                "recommended_crews": crews,
                "primary_equipment": equip_kind,
                "equipment_units": crews,
                "median_clearance_h": round(float(pi["p50"]), 1),
                "p90_clearance_h": round(float(pi["p90"]), 1),
            })

    plan = pd.DataFrame(rows).sort_values(
        ["corridor", "time_window"]).reset_index(drop=True)
    plan.to_csv(os.path.join(ART, "dispatch_plan.csv"), index=False)
    print(f"  dispatch_plan.csv     ({len(plan)} corridor x window rows)")

    # JSON nested by corridor for the app / an API response
    nested = {}
    for _, r in plan.iterrows():
        nested.setdefault(r["corridor"], []).append({
            k: (int(v) if isinstance(v, (np.integer,)) else
                float(v) if isinstance(v, (np.floating,)) else v)
            for k, v in r.items() if k != "corridor"
        })
    with open(os.path.join(ART, "dispatch_plan.json"), "w") as f:
        json.dump(nested, f, indent=2)
    print(f"  dispatch_plan.json    ({len(nested)} corridors)")

    # ---- 3. infrastructure hotspots (for preventive maintenance) --------
    hotspots = {}
    for cause in ["water_logging", "pot_holes", "construction"]:
        c = df[df["event_cause"] == cause]
        if len(c) == 0:
            continue
        clusters = (c.groupby(["junction", "corridor"])
                      .agg(count=("id", "count"),
                           median_resolution_h=("resolution_time_hours", "median"))
                      .reset_index()
                      .sort_values("count", ascending=False).head(15))
        clusters["median_resolution_h"] = clusters["median_resolution_h"].round(1)
        hotspots[cause] = clusters.to_dict("records")
    with open(os.path.join(ART, "hotspots.json"), "w") as f:
        json.dump(hotspots, f, indent=2, default=str)
    print(f"  hotspots.json         ({len(hotspots)} cause categories)")

    # ---- console summary ------------------------------------------------
    print("\nTop staging recommendations (by historical volume):")
    top = plan.sort_values("hist_events", ascending=False).head(8)
    for _, r in top.iterrows():
        print(f"  {r['corridor']:14s} | {r['time_window']:22s} | "
              f"{r['hist_events']:3d} events | {r['recommended_crews']} x "
              f"{r['primary_equipment']:14s} | median clear {r['median_clearance_h']}h")
    print("\nDone.")


if __name__ == "__main__":
    main()
