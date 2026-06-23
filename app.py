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

BASE = "#0a0d18"   # deep blue-black
PANEL = "#0f1630"  # panel surface
INK = "#cdd6f4"    # light text
MUTED = "#6c7aa6"
CYAN = "#22d3ee"   # primary neon
MAGENTA = "#ff2bd6"  # secondary neon
GRID = "#1d2747"   # chart gridlines
DARK = GRID

# severity = neon text + translucent glow tint
SEV = {
    "LOW":      ("#39ff9e", "rgba(57,255,158,.12)"),
    "MEDIUM":   ("#ffd23d", "rgba(255,210,61,.12)"),
    "HIGH":     ("#ff9a3d", "rgba(255,154,61,.13)"),
    "CRITICAL": ("#ff3d6e", "rgba(255,61,110,.15)"),
}

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@600;700;800&display=swap');

:root {
  --base:#0a0d18; --panel:#0f1630; --ink:#cdd6f4; --muted:#6c7aa6;
  --cyan:#22d3ee; --magenta:#ff2bd6; --line:rgba(34,211,238,.22);
  --glow: 0 0 22px rgba(34,211,238,.10);
}

/* hide default streamlit chrome */
#MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stHeader"] { background: transparent; height: 0; }

html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, sans-serif; color: var(--ink);
}
.stApp {
  background:
    radial-gradient(circle at 12% -5%, rgba(34,211,238,.10), transparent 42%),
    radial-gradient(circle at 92% 8%, rgba(255,43,214,.08), transparent 40%),
    linear-gradient(rgba(110,140,220,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(110,140,220,.045) 1px, transparent 1px),
    #0a0d18;
  background-size: auto, auto, 34px 34px, 34px 34px, auto;
}
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1180px; }
h1,h2,h3,h4 { font-family:'Chakra Petch','Inter',sans-serif; letter-spacing:.01em; color:var(--ink); }
::selection { background:rgba(34,211,238,.3); }

/* hero — dark glass panel with neon edge */
.hero { position:relative; background:linear-gradient(135deg,#0d1430,#0b1126);
    border:1px solid var(--line); border-radius:18px; padding:28px 32px; margin-bottom:6px;
    box-shadow:0 0 30px rgba(34,211,238,.08), inset 0 1px 0 rgba(255,255,255,.04); overflow:hidden; }
.hero::before { content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,transparent,var(--cyan),var(--magenta),transparent); }
.hero-top { display:flex; align-items:center; gap:12px; }
.hero-badge { background:rgba(34,211,238,.08); border:1px solid var(--line); color:var(--cyan);
    font-family:'JetBrains Mono',monospace; font-size:10.5px; font-weight:700; letter-spacing:.12em;
    padding:6px 12px; border-radius:6px; text-transform:uppercase; }
.hero h1 { font-size:32px; font-weight:700; margin:14px 0 4px; color:#eaf2ff;
    text-shadow:0 0 16px rgba(34,211,238,.35); }
.hero p { color:#93a4cc; font-size:14.5px; margin:0; max-width:760px; line-height:1.55; }
.hero-rule { width:54px; height:3px; background:linear-gradient(90deg,var(--cyan),var(--magenta));
    border-radius:2px; margin-top:16px; box-shadow:0 0 12px rgba(34,211,238,.6); }
.hero-stats { display:flex; gap:14px; margin-top:18px; flex-wrap:wrap; }
.hero-stat { background:rgba(13,20,46,.6); border:1px solid var(--line); border-radius:11px;
    padding:11px 17px; min-width:120px; }
.hero-stat .v { font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:800;
    color:var(--cyan); text-shadow:0 0 12px rgba(34,211,238,.45); }
.hero-stat .k { font-size:10.5px; color:var(--muted); letter-spacing:.04em; margin-top:2px;
    text-transform:uppercase; }

/* tabs */
[data-baseweb="tab-list"] { gap:6px; background:rgba(13,20,46,.5); padding:7px;
    border-radius:12px; border:1px solid var(--line); }
[data-baseweb="tab"] { height:40px; border-radius:8px; padding:0 18px; font-weight:600;
    color:var(--muted); background:transparent; transition:all .15s ease; }
[data-baseweb="tab"]:hover { color:var(--ink); }
[data-baseweb="tab"][aria-selected="true"] { background:rgba(34,211,238,.10); color:var(--cyan);
    border:1px solid var(--line); box-shadow:0 0 14px rgba(34,211,238,.2); }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display:none; }

/* card */
.card { background:var(--panel); border:1px solid var(--line); border-radius:16px;
    padding:22px 24px; box-shadow:var(--glow), inset 0 1px 0 rgba(255,255,255,.03); }
.card-title { font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700;
    letter-spacing:.12em; text-transform:uppercase; color:var(--cyan); margin-bottom:16px; }

/* severity pill */
.sev { display:inline-flex; align-items:center; gap:8px; font-family:'JetBrains Mono',monospace;
    font-weight:700; font-size:12.5px; padding:8px 16px; border-radius:8px; letter-spacing:.06em;
    border:1px solid currentColor; }
.sev .dot { width:8px; height:8px; border-radius:50%; background:currentColor;
    box-shadow:0 0 10px currentColor; }

/* interval tiles */
.tiles { display:grid; grid-template-columns:1fr 1.25fr 1fr; gap:14px; margin:18px 0 6px; }
.tile { background:rgba(13,20,46,.55); border:1px solid var(--line); border-radius:13px;
    padding:15px 17px; }
.tile .k { font-family:'JetBrains Mono',monospace; font-size:10.5px; color:var(--muted);
    font-weight:600; letter-spacing:.06em; text-transform:uppercase; }
.tile .v { font-family:'JetBrains Mono',monospace; font-size:27px; font-weight:800;
    color:var(--ink); margin-top:4px; }
.tile.main { border-color:rgba(34,211,238,.5); background:rgba(34,211,238,.07);
    box-shadow:0 0 20px rgba(34,211,238,.15); }
.tile.main .v { color:var(--cyan); text-shadow:0 0 14px rgba(34,211,238,.5); }

/* plan rows + chips */
.plan-row { display:flex; justify-content:space-between; gap:14px; padding:11px 2px;
    border-bottom:1px solid rgba(110,140,220,.13); font-size:14px; }
.plan-row:last-child { border-bottom:none; }
.plan-row .l { color:var(--muted); font-weight:500; }
.plan-row .r { color:var(--ink); font-weight:600; text-align:right; }
.chips { display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }
.chip { background:rgba(34,211,238,.07); color:#a8c5e8; font-family:'JetBrains Mono',monospace;
    font-size:11.5px; font-weight:600; padding:6px 11px; border-radius:7px; border:1px solid var(--line); }
.callout { background:rgba(255,43,214,.06); border:1px solid rgba(255,43,214,.25);
    border-left:3px solid var(--magenta); border-radius:10px; padding:12px 15px;
    font-size:13.5px; color:#e6b8e0; margin-top:16px; }
.callout b { color:#fff; }
.muted { color:var(--muted); font-size:12.5px; }
.muted code { background:rgba(34,211,238,.1); padding:2px 7px; border-radius:5px; color:var(--cyan);
    font-family:'JetBrains Mono',monospace; }

/* metric strip (model card) */
.mrow { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }
.mcard { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:20px;
    box-shadow:var(--glow); }
.mcard .v { font-family:'JetBrains Mono',monospace; font-size:28px; font-weight:800;
    color:var(--cyan); text-shadow:0 0 14px rgba(34,211,238,.4); }
.mcard .k { font-size:12px; color:var(--muted); font-weight:600; margin-top:3px; }
.section-h { font-size:19px; font-weight:700; margin:6px 0 2px; color:#eaf2ff; }
.section-sub { color:var(--muted); font-size:13.5px; margin-bottom:16px; }

/* inputs — dark fields with neon focus */
[data-baseweb="select"] > div, .stTextInput input, [data-baseweb="input"] {
    background:rgba(13,20,46,.7) !important; border:1px solid var(--line) !important;
    border-radius:9px !important; color:var(--ink) !important; }
.stTextInput input { color:var(--ink) !important; }

/* buttons */
.stButton>button, .stDownloadButton>button { background:rgba(13,20,46,.7); color:var(--ink);
    border:1px solid var(--line); border-radius:10px; font-weight:700; transition:all .15s ease; }
.stButton>button:hover, .stDownloadButton>button:hover { color:var(--cyan);
    box-shadow:0 0 16px rgba(34,211,238,.25); border-color:var(--cyan); }
.stButton>button[kind="primary"] {
    background:linear-gradient(135deg,#22d3ee,#0e9bbf); color:#04121a; border:none;
    box-shadow:0 0 20px rgba(34,211,238,.4); }
.stButton>button[kind="primary"]:hover { color:#04121a; box-shadow:0 0 28px rgba(34,211,238,.6); }

/* dataframe */
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid var(--line); }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


def brand_fig(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="JetBrains Mono, monospace", color=INK, size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(15,22,48,.85)", bordercolor=GRID, borderwidth=1),
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
                                 line=dict(color=CYAN, width=20), name="P10–P90 range"))
        fig.add_trace(go.Scatter(x=[p50], y=[""], mode="markers",
                                 marker=dict(color=MAGENTA, size=20, symbol="diamond",
                                             line=dict(color="#0a0d18", width=2)),
                                 name="P50 (median)"))
        brand_fig(fig, 130)
        fig.update_layout(xaxis_title="hours to clear",
                          yaxis=dict(showticklabels=False, showgrid=False),
                          xaxis=dict(gridcolor=DARK))
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
    scale = [[0, "#0a0d18"], [0.25, "#0e6d8a"], [0.6, "#22d3ee"], [1, "#ff2bd6"]]
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
