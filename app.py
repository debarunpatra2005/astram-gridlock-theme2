"""
ASTRAM Theme 2 - Event-Driven Congestion: forecast + dispatch prototype.

Working demo for the problem statement:
  "forecast event-related traffic impact and recommend optimal manpower,
   barricading, and diversion plans."

Run:  .venv/bin/streamlit run app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import common

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
# The app runs entirely off the precomputed artifacts; the raw CSV is only
# needed if you choose to regenerate them. Locate it lazily, never at import.

st.set_page_config(page_title="ASTRAM Congestion Forecast", layout="wide",
                   page_icon="🚦")


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    return joblib.load(os.path.join(ART, "models.joblib"))


@st.cache_data
def load_lookups():
    with open(os.path.join(ART, "lookups.json")) as f:
        return json.load(f)


@st.cache_data
def load_metrics():
    with open(os.path.join(ART, "metrics.json")) as f:
        return json.load(f)


@st.cache_data
def load_dispatch():
    plan = pd.read_csv(os.path.join(ART, "dispatch_plan.csv"))
    demand = pd.read_csv(os.path.join(ART, "demand_matrix.csv"), index_col=0)
    with open(os.path.join(ART, "hotspots.json")) as f:
        hotspots = json.load(f)
    return plan, demand, hotspots


@st.cache_data
def load_raw():
    return common.load_raw(common.find_csv())


models = load_models()
lookups = load_lookups()
metrics = load_metrics()
plan, demand, hotspots = load_dispatch()

CAUSES = lookups["vocab"]["event_cause"]
CORRIDORS = lookups["vocab"]["corridor"]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🚦 ASTRAM — Event-Driven Congestion Forecast & Dispatch")
st.caption(
    "Theme 2 prototype · forecasts incident clearance time with confidence "
    "intervals and recommends manpower, barricading & diversion. "
    f"Trained on {metrics['n_modelled']:,} resolved events "
    f"({metrics['date_range'][0]} → {metrics['date_range'][1]})."
)

tab_predict, tab_dispatch, tab_hotspots, tab_model = st.tabs(
    ["🔮 Forecast & Recommend", "🚒 Resource Dispatch", "📍 Hotspots", "📊 Model Card"])


# ===========================================================================
# TAB 1 — Forecast & Recommend
# ===========================================================================

with tab_predict:
    st.subheader("Incident → forecast → operational plan")
    left, right = st.columns([1, 1.4])

    with left:
        cause = st.selectbox("Event cause", CAUSES,
                             index=CAUSES.index("vehicle_breakdown")
                             if "vehicle_breakdown" in CAUSES else 0)
        corridor = st.selectbox("Corridor", CORRIDORS,
                                index=CORRIDORS.index("Mysore Road")
                                if "Mysore Road" in CORRIDORS else 0)
        is_corr = corridor != "Non-corridor"
        # Corridor priority rule: corridors are High priority (EDA finding).
        priority = "High" if is_corr else "Low"
        st.text_input("Priority (auto from corridor rule)", priority, disabled=True)

        c1, c2 = st.columns(2)
        hour = c1.slider("Hour of day", 0, 23, 6)
        dow = c2.selectbox("Day of week",
                           ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=2)
        month = c1.slider("Month", 1, 12, 6)
        closure = c2.checkbox("Requires road closure", value=False)
        load = st.slider("Current active incidents on corridor (real-time)", 0, 20,
                         int(lookups["typical_load"].get(corridor, 1)))
        go_btn = st.button("Forecast & recommend", type="primary",
                           use_container_width=True)

    with right:
        if go_btn or True:
            dow_idx = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(dow)
            ev = pd.DataFrame([{
                "event_cause": cause, "corridor": corridor, "priority": priority,
                "requires_road_closure": int(closure), "is_planned": 0,
                "start_hour": hour, "start_day": dow_idx, "start_month": month,
                "is_weekend": int(dow_idx in (5, 6)),
                "current_active_load": load,
            }])
            pi = common.predict_intervals(ev, models, lookups).iloc[0]
            p10, p50, p90 = pi["p10"], pi["p50"], pi["p90"]

            rec = common.recommend_resources(cause, corridor, int(closure),
                                             int(is_corr), p50, p90)

            sev_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠",
                         "CRITICAL": "🔴"}[rec["severity"]]
            st.markdown(f"### {sev_color} Severity: **{rec['severity']}**")

            m1, m2, m3 = st.columns(3)
            m1.metric("Likely clearance (P50)", f"{p50:.1f} h")
            m2.metric("Best case (P10)", f"{p10:.1f} h")
            m3.metric("Worst case (P90)", f"{p90:.1f} h")

            # interval visual
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[p10, p90], y=["clearance", "clearance"], mode="lines",
                line=dict(color="#f59e0b", width=18), name="P10–P90 range"))
            fig.add_trace(go.Scatter(
                x=[p50], y=["clearance"], mode="markers",
                marker=dict(color="#dc2626", size=18, symbol="diamond"),
                name="P50 (median)"))
            fig.update_layout(height=140, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="hours to clear", showlegend=True,
                              yaxis=dict(showticklabels=False))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Model used: `{pi['model_used']}` · interval calibrated "
                       f"via conformal prediction (~80% target coverage).")

            st.markdown("#### 🚒 Recommended operational plan")
            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"**Response team:** {rec['response_team']}")
                st.markdown(f"**Teams to dispatch:** {rec['teams_to_dispatch']}")
                st.markdown(f"**Field personnel (≈):** {rec['field_personnel']}")
                st.markdown("**Equipment:**")
                for e in rec["equipment"]:
                    st.markdown(f"- {e}")
            with r2:
                st.markdown(f"**Barricading:** {rec['barricading']}")
                st.markdown(f"**Diversion plan:** {rec['diversion_plan']}")
                st.info(f"Plan for clearance within **{rec['expected_clearance_h']} h**; "
                        f"hold resources / escalate if not cleared by "
                        f"**{rec['worst_case_clearance_h']} h** (P90).")


# ===========================================================================
# TAB 2 — Resource Dispatch
# ===========================================================================

with tab_dispatch:
    st.subheader("Shift-level pre-positioning plan")
    st.caption("Where to stage crews and equipment by corridor and time window, "
               "derived from historical demand + model-predicted clearance.")

    # demand heatmap
    st.markdown("#### Incident demand — corridor × hour")
    top_demand = demand.head(15)
    fig = px.imshow(top_demand, aspect="auto", color_continuous_scale="YlOrRd",
                    labels=dict(x="hour of day", y="corridor", color="events"))
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Staging plan")
    corr_sel = st.selectbox("Filter by corridor", ["All"] + sorted(plan["corridor"].unique()))
    view = plan if corr_sel == "All" else plan[plan["corridor"] == corr_sel]
    view = view.sort_values("hist_events", ascending=False)
    st.dataframe(
        view.rename(columns={
            "corridor": "Corridor", "time_window": "Window",
            "hist_events": "Hist. events", "events_per_day": "Events/day",
            "dominant_cause": "Dominant cause", "recommended_crews": "Crews",
            "primary_equipment": "Equipment", "equipment_units": "Units",
            "median_clearance_h": "Median clear (h)", "p90_clearance_h": "P90 clear (h)",
        }),
        use_container_width=True, hide_index=True)

    st.download_button("⬇ Download dispatch plan (CSV)",
                       plan.to_csv(index=False), "dispatch_plan.csv", "text/csv")


# ===========================================================================
# TAB 3 — Hotspots
# ===========================================================================

with tab_hotspots:
    st.subheader("Infrastructure hotspots — preventive maintenance targets")
    st.caption("Recurring waterlogging / pothole / construction locations. "
               "Fix the location, remove the recurring incident.")
    for cause, label in [("water_logging", "💧 Waterlogging"),
                         ("pot_holes", "🕳 Potholes"),
                         ("construction", "🚧 Construction")]:
        if cause in hotspots and hotspots[cause]:
            st.markdown(f"#### {label} — top recurring junctions")
            h = pd.DataFrame(hotspots[cause])
            st.dataframe(h.rename(columns={
                "junction": "Junction", "corridor": "Corridor",
                "count": "Incidents", "median_resolution_h": "Median resolution (h)"}),
                use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 4 — Model Card
# ===========================================================================

with tab_model:
    st.subheader("Model card — honest performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Typical error (MedAE)", f"{metrics['point_medae_h']:.1f} h")
    c2.metric("Mean error (MAE)", f"{metrics['point_mae_h']:.0f} h")
    c3.metric("P10–P90 coverage", f"{metrics['p10_p90_coverage']:.0%}",
              help="Calibrated via conformal prediction; target 80%.")
    c4.metric("Coverage before CQR", f"{metrics['p10_p90_coverage_raw']:.0%}")

    st.markdown(f"""
**How to read this.** The *median* absolute error is **{metrics['point_medae_h']:.1f} h** —
a typical incident's clearance time is predicted within about that. The *mean*
error ({metrics['point_mae_h']:.0f} h) is much larger because a handful of
multi-day infrastructure incidents (potholes, waterlogging) dominate the average.
That is exactly why this model reports a **P10–P90 interval** instead of a single
number: for high-variance causes the honest answer is a range, not a point.

**Method.** LightGBM gradient boosting on log-resolution, with separate quantile
models (P10/P50/P90). Intervals are calibrated with **Conformalized Quantile
Regression** on a temporal hold-out, lifting coverage from
{metrics['p10_p90_coverage_raw']:.0%} → {metrics['p10_p90_coverage']:.0%}.
Evaluation uses a strict **temporal split** (train on earlier events, test on
later) — no look-ahead leakage.
""")

    if metrics.get("cause_eval"):
        st.markdown("#### Cause-specific quantile models")
        ce = pd.DataFrame(metrics["cause_eval"]).T.reset_index()
        ce.columns = ["Cause", "Test n", "P50 MAE (h)", "P10–P90 coverage"]
        ce["P50 MAE (h)"] = ce["P50 MAE (h)"].round(1)
        ce["P10–P90 coverage"] = (ce["P10–P90 coverage"] * 100).round(0).astype(str) + "%"
        st.dataframe(ce, use_container_width=True, hide_index=True)
        st.caption("Potholes remain the hardest: ~108 training rows and a strong "
                   "period-to-period distribution shift. Reported, not hidden.")
