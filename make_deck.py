"""
Generate a pitch-deck PDF for the ASTRAM Theme 2 submission.

Satisfies the "Presentation" field on the HackerEarth form (.pdf accepted).
Writes ASTRAM_Theme2_Pitch_Deck.pdf (16:9 slides).

Run:  .venv/bin/python make_deck.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "ASTRAM_Theme2_Pitch_Deck.pdf")

NAVY = "#0b2545"
BLUE = "#2563eb"
AMBER = "#f59e0b"
GREY = "#475569"

with open(os.path.join(ART, "metrics.json")) as f:
    M = json.load(f)


def new_slide():
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    return fig, ax


def footer(ax, n):
    ax.text(0.5, 0.03, "ASTRAM Congestion Copilot · Gridlock Hackathon 2.0 · Theme 2",
            ha="center", color=GREY, fontsize=9, transform=ax.transAxes)
    ax.text(0.96, 0.03, str(n), ha="right", color=GREY, fontsize=9,
            transform=ax.transAxes)


def bullets(ax, items, y0=0.72, dy=0.105, x=0.10, size=15):
    for i, t in enumerate(items):
        ax.text(x, y0 - i * dy, "▸", color=BLUE, fontsize=size, weight="bold",
                transform=ax.transAxes)
        ax.text(x + 0.035, y0 - i * dy, t, color="#1e293b", fontsize=size,
                va="baseline", transform=ax.transAxes, wrap=True)


def title_block(ax, kicker, title):
    ax.add_patch(plt.Rectangle((0.10, 0.86), 0.045, 0.012, color=AMBER,
                               transform=ax.transAxes))
    ax.text(0.10, 0.90, kicker, color=BLUE, fontsize=13, weight="bold",
            transform=ax.transAxes)
    ax.text(0.10, 0.80, title, color=NAVY, fontsize=26, weight="bold",
            transform=ax.transAxes)


def add_image(fig, path, rect):
    img = mpimg.imread(path)
    a = fig.add_axes(rect); a.axis("off"); a.imshow(img)


pages = []

# ---- Slide 1: Title -------------------------------------------------------
fig, ax = new_slide()
ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=NAVY, transform=ax.transAxes))
ax.text(0.5, 0.62, "ASTRAM Congestion Copilot", ha="center", color="white",
        fontsize=40, weight="bold", transform=ax.transAxes)
ax.text(0.5, 0.52, "Forecasting event-driven congestion — and the plan to clear it",
        ha="center", color="#cbd5e1", fontsize=18, transform=ax.transAxes)
ax.add_patch(plt.Rectangle((0.34, 0.45), 0.32, 0.006, color=AMBER,
                           transform=ax.transAxes))
ax.text(0.5, 0.36, "Gridlock Hackathon 2.0  ·  Round 2 (Prototype)  ·  Theme 2",
        ha="center", color="#e2e8f0", fontsize=14, transform=ax.transAxes)
ax.text(0.5, 0.30, "Event-Driven Congestion (Planned & Unplanned)",
        ha="center", color="#94a3b8", fontsize=13, transform=ax.transAxes)
pages.append(fig)

# ---- Slide 2: Problem -----------------------------------------------------
fig, ax = new_slide()
title_block(ax, "THE PROBLEM", "Events break traffic faster than crews can react")
bullets(ax, [
    "Rallies, festivals, sports, construction & sudden gatherings cause localized breakdowns.",
    "Today: event impact isn't quantified in advance.",
    "Resource deployment is experience-driven, not data-driven.",
    "No post-event learning system to improve next time.",
])
ax.text(0.10, 0.20, "Theme 2 asks two things — forecast impact AND recommend\n"
        "manpower, barricading & diversion. Most stop at prediction. We do both.",
        color=NAVY, fontsize=15, weight="bold", transform=ax.transAxes)
footer(ax, 2); pages.append(fig)

# ---- Slide 3: Solution ----------------------------------------------------
fig, ax = new_slide()
title_block(ax, "OUR SOLUTION", "A working forecast-to-dispatch prototype")
bullets(ax, [
    "FORECAST  —  clearance time as a P10 / P50 / P90 interval, not a fragile guess.",
    "RECOMMEND  —  response team, crew count, equipment, barricading & diversion.",
    "DISPATCH  —  shift-level pre-positioning plan by corridor × time window (CSV/JSON).",
    "PREVENT  —  recurring waterlogging / pothole / construction hotspots.",
], y0=0.70, dy=0.12)
ax.text(0.10, 0.14, "Live Streamlit app · trained on real ASTRAM / BTP event data.",
        color=GREY, fontsize=14, transform=ax.transAxes)
footer(ax, 3); pages.append(fig)

# ---- Slide 4: Forecast intervals figure -----------------------------------
fig, ax = new_slide()
title_block(ax, "FORECAST", "Why a range beats a single number")
add_image(fig, os.path.join(FIG, "fig1_forecast_intervals.png"),
          [0.08, 0.08, 0.84, 0.62])
footer(ax, 4); pages.append(fig)

# ---- Slide 5: Model + CQR -------------------------------------------------
fig, ax = new_slide()
title_block(ax, "THE MODEL", "Calibrated, leakage-safe, honest")
bullets(ax, [
    "LightGBM on log-resolution + quantile models (P10/P50/P90).",
    "Dedicated models for high-variance causes (potholes, construction).",
    "Conformalized Quantile Regression calibrates the intervals.",
    "Strict temporal validation — train on earlier events, test on later.",
], y0=0.70, dy=0.105, x=0.07, size=14)
add_image(fig, os.path.join(FIG, "fig3_cqr_coverage.png"),
          [0.58, 0.10, 0.36, 0.52])
footer(ax, 5); pages.append(fig)

# ---- Slide 6: Dispatch heatmap --------------------------------------------
fig, ax = new_slide()
title_block(ax, "DISPATCH", "Where & when to pre-position crews")
add_image(fig, os.path.join(FIG, "fig2_demand_heatmap.png"),
          [0.07, 0.06, 0.86, 0.64])
footer(ax, 6); pages.append(fig)

# ---- Slide 7: Honest performance ------------------------------------------
fig, ax = new_slide()
title_block(ax, "PERFORMANCE", "Honest numbers (temporal hold-out)")
cards = [
    (f"{M['point_medae_h']:.1f} h", "Typical (median) error"),
    ("~0.7 h", "Vehicle breakdown P50\n(60% of all events)"),
    (f"{M['p10_p90_coverage']:.0%}", "P10–P90 coverage\n(60% → after CQR)"),
]
for i, (big, small) in enumerate(cards):
    x = 0.10 + i * 0.28
    ax.add_patch(plt.Rectangle((x, 0.42), 0.24, 0.26, color="#f1f5f9",
                               transform=ax.transAxes))
    ax.text(x + 0.12, 0.58, big, ha="center", color=BLUE, fontsize=30,
            weight="bold", transform=ax.transAxes)
    ax.text(x + 0.12, 0.47, small, ha="center", color=GREY, fontsize=12,
            transform=ax.transAxes)
ax.text(0.10, 0.30, "Mean error looks large only because rare multi-day infrastructure\n"
        "incidents dominate the average — which is exactly why we report intervals.\n"
        "Surfaced in the app's Model Card, not hidden.",
        color="#1e293b", fontsize=14, transform=ax.transAxes)
footer(ax, 7); pages.append(fig)

# ---- Slide 8: Demo + impact -----------------------------------------------
fig, ax = new_slide()
title_block(ax, "DEMO & IMPACT", "From incident to action in one screen")
bullets(ax, [
    "Breakdown, Mysore Rd, 6 AM → ~0.7 h, HIGH, 2 towing units.",
    "Waterlogging + closure → ~24 h (P90 ~170 h), CRITICAL, pumps + diversion.",
    "Dispatch tab → demand heatmap + downloadable staging plan.",
    "Hotspots tab → preventive-maintenance targets.",
], y0=0.70, dy=0.10, size=14)
ax.text(0.10, 0.22, "Built on real BTP data · deployable today · scales to live ASTRAM feeds.",
        color=NAVY, fontsize=15, weight="bold", transform=ax.transAxes)
ax.text(0.10, 0.14, "Thank you.", color=BLUE, fontsize=18, weight="bold",
        transform=ax.transAxes)
footer(ax, 8); pages.append(fig)

with PdfPages(OUT) as pdf:
    for fig in pages:
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

print(f"Wrote {OUT}  ({len(pages)} slides)")
