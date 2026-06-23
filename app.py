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

st.set_page_config(page_title="ASTRAM Congestion Copilot", layout="wide",
                   page_icon="🚦", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Design system (palette mirrors the pitch deck)
# ---------------------------------------------------------------------------

NAVY = "#0b2545"
BLUE = "#2563eb"
AMBER = "#f59e0b"
INK = "#0f172a"
MUTED = "#64748b"
LINE = "#e2e8f0"

SEV = {
    "LOW":      ("#16a34a", "#dcfce7"),
    "MEDIUM":   ("#b45309", "#fef3c7"),
    "HIGH":     ("#c2410c", "#ffedd5"),
    "CRITICAL": ("#b91c1c", "#fee2e2"),
}

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

/* hide default streamlit chrome */
#MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stHeader"] { background: transparent; height: 0; }

html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, sans-serif;
    color: #0f172a;
}
.stApp { background: #f5f7fb; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1180px; }

h1, h2, h3, h4 { font-family: 'Plus Jakarta Sans', 'Inter', sans-serif; letter-spacing:-.01em; }

/* hero banner */
.hero {
    background: linear-gradient(135deg, #0b2545 0%, #163a66 60%, #1d4ed8 140%);
    border-radius: 18px; padding: 26px 30px; color: #fff;
    box-shadow: 0 12px 30px rgba(11,37,69,.18); margin-bottom: 6px;
}
.hero-top { display:flex; align-items:center; gap:12px; }
.hero-badge { background: rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.25);
    font-size:11px; font-weight:700; letter-spacing:.08em; padding:5px 11px;
    border-radius:999px; text-transform:uppercase; color:#e0ecff; }
.hero h1 { color:#fff; font-size:30px; margin:12px 0 4px; }
.hero p { color:#c7d6ee; font-size:14.5px; margin:0; max-width:760px; line-height:1.5; }
.hero-stats { display:flex; gap:26px; margin-top:18px; flex-wrap:wrap; }
.hero-stat { }
.hero-stat .v { font-size:22px; font-weight:800; color:#fff; font-family:'Plus Jakarta Sans'; }
.hero-stat .k { font-size:11.5px; color:#9fb6d8; letter-spacing:.03em; }
.hero-rule { width:42px; height:4px; background:#f59e0b; border-radius:2px; margin-top:14px; }

/* tabs as a segmented control */
[data-baseweb="tab-list"] { gap:6px; background:#eef2f9; padding:6px; border-radius:12px;
    border:1px solid #e2e8f0; }
[data-baseweb="tab"] { height:40px; border-radius:9px; padding:0 16px; font-weight:600;
    color:#475569; background:transparent; }
[data-baseweb="tab"][aria-selected="true"] { background:#fff; color:#0b2545;
    box-shadow:0 1px 4px rgba(15,23,42,.08); }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display:none; }

/* generic card */
.card { background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:20px 22px;
    box-shadow:0 1px 3px rgba(15,23,42,.04); }
.card-title { font-size:12px; font-weight:700; letter-spacing:.07em; text-transform:uppercase;
    color:#2563eb; margin-bottom:14px; }

/* severity pill */
.sev { display:inline-flex; align-items:center; gap:8px; font-weight:700; font-size:13px;
    padding:7px 15px; border-radius:999px; letter-spacing:.03em; }
.sev .dot { width:8px; height:8px; border-radius:50%; background:currentColor; }

/* interval tiles */
.tiles { display:grid; grid-template-columns:1fr 1.25fr 1fr; gap:12px; margin:16px 0 4px; }
.tile { background:#f8fafc; border:1px solid #e9eef6; border-radius:12px; padding:14px 16px; }
.tile .k { font-size:11.5px; color:#64748b; font-weight:600; letter-spacing:.02em; }
.tile .v { font-size:26px; font-weight:800; color:#0f172a; font-family:'Plus Jakarta Sans';
    margin-top:2px; }
.tile.main { background:linear-gradient(180deg,#eff5ff,#fff); border-color:#bfd4ff; }
.tile.main .v { color:#1d4ed8; }

/* plan rows + chips */
.plan-row { display:flex; justify-content:space-between; gap:14px; padding:10px 0;
    border-bottom:1px dashed #e9eef6; font-size:14px; }
.plan-row:last-child { border-bottom:none; }
.plan-row .l { color:#64748b; font-weight:500; }
.plan-row .r { color:#0f172a; font-weight:600; text-align:right; }
.chips { display:flex; flex-wrap:wrap; gap:7px; margin-top:6px; }
.chip { background:#eef2f9; color:#334155; font-size:12.5px; font-weight:600;
    padding:5px 11px; border-radius:8px; border:1px solid #e2e8f0; }
.callout { background:#fff7ed; border:1px solid #fed7aa; border-left:4px solid #f59e0b;
    border-radius:10px; padding:12px 15px; font-size:13.5px; color:#7c2d12; margin-top:14px; }
.muted { color:#64748b; font-size:12.5px; }

/* metric strip (model card) */
.mrow { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }
.mcard { background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:18px 18px; }
.mcard .v { font-size:28px; font-weight:800; color:#1d4ed8; font-family:'Plus Jakarta Sans'; }
.mcard .k { font-size:12.5px; color:#64748b; font-weight:600; margin-top:2px; }
.section-h { font-size:18px; font-weight:700; color:#0b2545; margin:6px 0 2px; }
.section-sub { color:#64748b; font-size:13.5px; margin-bottom:14px; }

/* buttons */
.stButton>button, .stDownloadButton>button { border-radius:10px; font-weight:700;
    border:1px solid transparent; }
.stButton>button[kind="primary"] { background:#2563eb; }

/* dataframe corners */
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid #e2e8f0; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


def brand_fig(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Inter, sans-serif", color=INK, size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(255,255,255,.7)", bordercolor=LINE, borderwidth=1),
    )
    return fig


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


models = load_models()
lookups = load_lookups()
metrics = load_metrics()
plan, demand, hotspots = load_dispatch()

CAUSES = lookups["vocab"]["event_cause"]
CORRIDORS = lookups["vocab"]["corridor"]


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(f"""
<div class="hero">
  <div class="hero-top">
    <span style="font-size:26px">🚦</span>
    <span class="hero-badge">Gridlock 2.0 · Theme 2 · Prototype</span>
  </div>
  <h1>ASTRAM Congestion Copilot</h1>
  <p>Forecasts how long a traffic incident will block a corridor — as a calibrated
  confidence interval — and recommends the manpower, barricading and diversion plan
  to clear it.</p>
  <div class="hero-rule"></div>
  <div class="hero-stats">
    <div class="hero-stat"><div class="v">{metrics['n_modelled']:,}</div>
      <div class="k">resolved events trained</div></div>
    <div class="hero-stat"><div class="v">{metrics['point_medae_h']:.1f} h</div>
      <div class="k">typical (median) error</div></div>
    <div class="hero-stat"><div class="v">{metrics['p10_p90_coverage']:.0%}</div>
      <div class="k">P10–P90 coverage</div></div>
    <div class="hero-stat"><div class="v">{metrics['date_range'][0][:7]} → {metrics['date_range'][1][:7]}</div>
      <div class="k">data window</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

tab_predict, tab_dispatch, tab_hotspots, tab_model = st.tabs(
    ["🔮  Forecast & Recommend", "🚒  Resource Dispatch", "📍  Hotspots", "📊  Model Card"])


# ===========================================================================
# TAB 1 — Forecast & Recommend
# ===========================================================================

with tab_predict:
    left, right = st.columns([1, 1.45], gap="large")

    with left:
        st.markdown('<div class="card-title">Incident details</div>', unsafe_allow_html=True)
        cause = st.selectbox("Event cause", CAUSES,
                             index=CAUSES.index("vehicle_breakdown")
                             if "vehicle_breakdown" in CAUSES else 0)
        corridor = st.selectbox("Corridor", CORRIDORS,
                                index=CORRIDORS.index("Mysore Road")
                                if "Mysore Road" in CORRIDORS else 0)
        is_corr = corridor != "Non-corridor"
        priority = "High" if is_corr else "Low"  # corridor priority rule (EDA)

        c1, c2 = st.columns(2)
        hour = c1.slider("Hour of day", 0, 23, 6)
        dow = c2.selectbox("Day of week",
                           ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=2)
        month = c1.slider("Month", 1, 12, 6)
        closure = c2.checkbox("Road closure", value=False)
        load = st.slider("Active incidents on corridor now (real-time)", 0, 20,
                         int(lookups["typical_load"].get(corridor, 1)))
        st.caption(f"Priority auto-set to **{priority}** by the corridor rule.")
        st.button("Forecast & recommend", type="primary", use_container_width=True)

    with right:
        dow_idx = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(dow)
        ev = pd.DataFrame([{
            "event_cause": cause, "corridor": corridor, "priority": priority,
            "requires_road_closure": int(closure), "is_planned": 0,
            "start_hour": hour, "start_day": dow_idx, "start_month": month,
            "is_weekend": int(dow_idx in (5, 6)), "current_active_load": load,
        }])
        pi = common.predict_intervals(ev, models, lookups).iloc[0]
        p10, p50, p90 = pi["p10"], pi["p50"], pi["p90"]
        rec = common.recommend_resources(cause, corridor, int(closure),
                                         int(is_corr), p50, p90)
        fg, bg = SEV[rec["severity"]]

        # severity + interval tiles
        st.markdown(f"""
        <div class="card">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <div class="card-title" style="margin:0">Forecast</div>
            <span class="sev" style="color:{fg};background:{bg}">
              <span class="dot"></span>{rec['severity']} severity</span>
          </div>
          <div class="tiles">
            <div class="tile"><div class="k">Best case · P10</div><div class="v">{p10:.1f}<span style="font-size:14px"> h</span></div></div>
            <div class="tile main"><div class="k">Likely · P50</div><div class="v">{p50:.1f}<span style="font-size:14px"> h</span></div></div>
            <div class="tile"><div class="k">Worst case · P90</div><div class="v">{p90:.1f}<span style="font-size:14px"> h</span></div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # interval bar
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[p10, p90], y=["", ""], mode="lines",
                                 line=dict(color=AMBER, width=20), name="P10–P90 range"))
        fig.add_trace(go.Scatter(x=[p50], y=[""], mode="markers",
                                 marker=dict(color="#dc2626", size=20, symbol="diamond",
                                             line=dict(color="#fff", width=2)),
                                 name="P50 (median)"))
        brand_fig(fig, 130)
        fig.update_layout(xaxis_title="hours to clear",
                          yaxis=dict(showticklabels=False, showgrid=False),
                          xaxis=dict(gridcolor=LINE))
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(f'<div class="muted">Model: <code>{pi["model_used"]}</code> · '
                    f'interval calibrated via conformal prediction (~80% target).</div>',
                    unsafe_allow_html=True)

        # operational plan
        chips = "".join(f'<span class="chip">{e}</span>' for e in rec["equipment"])
        st.write("")
        st.markdown(f"""
        <div class="card">
          <div class="card-title">🚒 Recommended operational plan</div>
          <div class="plan-row"><span class="l">Response team</span><span class="r">{rec['response_team']}</span></div>
          <div class="plan-row"><span class="l">Teams to dispatch</span><span class="r">{rec['teams_to_dispatch']}</span></div>
          <div class="plan-row"><span class="l">Field personnel (≈)</span><span class="r">{rec['field_personnel']}</span></div>
          <div class="plan-row"><span class="l">Barricading</span><span class="r">{rec['barricading']}</span></div>
          <div class="plan-row"><span class="l">Diversion plan</span><span class="r">{rec['diversion_plan']}</span></div>
          <div class="plan-row" style="border-bottom:none"><span class="l">Equipment</span></div>
          <div class="chips">{chips}</div>
          <div class="callout">Plan for clearance within <b>{rec['expected_clearance_h']} h</b>;
          hold resources / escalate if not cleared by <b>{rec['worst_case_clearance_h']} h</b> (P90).</div>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# TAB 2 — Resource Dispatch
# ===========================================================================

with tab_dispatch:
    st.markdown('<div class="section-h">Shift-level pre-positioning plan</div>'
                '<div class="section-sub">Where to stage crews and equipment by corridor '
                'and time window, from historical demand + model-predicted clearance.</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="card-title">Incident demand · corridor × hour</div>',
                unsafe_allow_html=True)
    scale = [[0, "#f8fafc"], [0.25, "#fde68a"], [0.6, "#f59e0b"], [1, "#7f1d1d"]]
    fig = px.imshow(demand.head(15), aspect="auto", color_continuous_scale=scale,
                    labels=dict(x="hour of day", y="corridor", color="events"))
    brand_fig(fig, 470)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="card-title" style="margin-top:8px">Staging plan</div>',
                unsafe_allow_html=True)
    corr_sel = st.selectbox("Filter by corridor",
                            ["All"] + sorted(plan["corridor"].unique()))
    view = plan if corr_sel == "All" else plan[plan["corridor"] == corr_sel]
    view = view.sort_values("hist_events", ascending=False)
    st.dataframe(
        view.rename(columns={
            "corridor": "Corridor", "time_window": "Window",
            "hist_events": "Hist. events", "events_per_day": "Events/day",
            "dominant_cause": "Dominant cause", "recommended_crews": "Crews",
            "primary_equipment": "Equipment", "equipment_units": "Units",
            "median_clearance_h": "Median clear (h)", "p90_clearance_h": "P90 clear (h)",
        }), use_container_width=True, hide_index=True)
    st.download_button("⬇  Download dispatch plan (CSV)",
                       plan.to_csv(index=False), "dispatch_plan.csv", "text/csv")


# ===========================================================================
# TAB 3 — Hotspots
# ===========================================================================

with tab_hotspots:
    st.markdown('<div class="section-h">Infrastructure hotspots</div>'
                '<div class="section-sub">Recurring waterlogging / pothole / construction '
                'locations — fix the location, remove the recurring incident.</div>',
                unsafe_allow_html=True)
    for cause, label in [("water_logging", "💧 Waterlogging"),
                         ("pot_holes", "🕳 Potholes"),
                         ("construction", "🚧 Construction")]:
        if cause in hotspots and hotspots[cause]:
            st.markdown(f'<div class="card-title">{label} · top recurring junctions</div>',
                        unsafe_allow_html=True)
            h = pd.DataFrame(hotspots[cause])
            st.dataframe(h.rename(columns={
                "junction": "Junction", "corridor": "Corridor",
                "count": "Incidents", "median_resolution_h": "Median resolution (h)"}),
                use_container_width=True, hide_index=True)


# ===========================================================================
# TAB 4 — Model Card
# ===========================================================================

with tab_model:
    st.markdown('<div class="section-h">Model card — honest performance</div>'
                '<div class="section-sub">Evaluated on a strict temporal hold-out '
                '(train on earlier events, test on later) — no look-ahead leakage.</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="mrow">
      <div class="mcard"><div class="v">{metrics['point_medae_h']:.1f} h</div><div class="k">Typical error (MedAE)</div></div>
      <div class="mcard"><div class="v">{metrics['point_mae_h']:.0f} h</div><div class="k">Mean error (MAE)</div></div>
      <div class="mcard"><div class="v">{metrics['p10_p90_coverage']:.0%}</div><div class="k">P10–P90 coverage</div></div>
      <div class="mcard"><div class="v">{metrics['p10_p90_coverage_raw']:.0%}</div><div class="k">Coverage before CQR</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
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
""")

    if metrics.get("cause_eval"):
        st.markdown('<div class="card-title" style="margin-top:8px">'
                    'Cause-specific quantile models</div>', unsafe_allow_html=True)
        ce = pd.DataFrame(metrics["cause_eval"]).T.reset_index()
        ce.columns = ["Cause", "Test n", "P50 MAE (h)", "P10–P90 coverage"]
        ce["P50 MAE (h)"] = ce["P50 MAE (h)"].round(1)
        ce["P10–P90 coverage"] = (ce["P10–P90 coverage"] * 100).round(0).astype(str) + "%"
        st.dataframe(ce, use_container_width=True, hide_index=True)
        st.caption("Potholes remain the hardest: ~108 training rows and a strong "
                   "period-to-period distribution shift. Reported, not hidden.")
