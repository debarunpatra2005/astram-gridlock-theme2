"""
Generate deck-ready figures for the ASTRAM Theme 2 submission.

Writes PNGs to figures/ :
  fig1_forecast_intervals.png  - P10/P50/P90 clearance per cause (the headline)
  fig2_demand_heatmap.png      - corridor x hour incident demand
  fig3_cqr_coverage.png        - interval calibration before/after CQR

Run:  .venv/bin/python make_screenshots.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import common

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 11,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def load():
    models = joblib.load(os.path.join(ART, "models.joblib"))
    with open(os.path.join(ART, "lookups.json")) as f:
        lookups = json.load(f)
    with open(os.path.join(ART, "metrics.json")) as f:
        metrics = json.load(f)
    demand = pd.read_csv(os.path.join(ART, "demand_matrix.csv"), index_col=0)
    return models, lookups, metrics, demand


def fig_intervals(models, lookups):
    """Horizontal P10-P90 bars with P50 marker, one row per cause."""
    causes = ["vehicle_breakdown", "tree_fall", "others", "construction",
              "water_logging", "pot_holes"]
    rows = []
    for c in causes:
        if c not in lookups["vocab"]["event_cause"]:
            continue
        ev = pd.DataFrame([{
            "event_cause": c, "corridor": "Mysore Road", "priority": "High",
            "requires_road_closure": 1, "is_planned": 0, "start_hour": 6,
            "start_day": 2, "start_month": 6, "is_weekend": 0,
            "current_active_load": 3,
        }])
        pi = common.predict_intervals(ev, models, lookups).iloc[0]
        rows.append((c, pi["p10"], pi["p50"], pi["p90"]))

    rows.sort(key=lambda r: r[2])
    labels = [r[0].replace("_", " ") for r in rows]
    p10 = np.array([r[1] for r in rows])
    p50 = np.array([r[2] for r in rows])
    p90 = np.array([r[3] for r in rows])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hlines(y, p10, p90, color="#f59e0b", linewidth=12, alpha=0.7,
              label="P10–P90 range")
    ax.scatter(p50, y, color="#dc2626", s=90, zorder=3, marker="D",
               label="P50 (median)")
    for yi, (a, m, b) in enumerate(zip(p10, p50, p90)):
        ax.text(b, yi, f"  P90={b:.0f}h", va="center", fontsize=9, color="#444")
        ax.text(m, yi + 0.28, f"{m:.1f}h", va="center", ha="center",
                fontsize=9, color="#dc2626")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("symlog")
    ax.set_xlabel("Predicted clearance time (hours, log scale)")
    ax.set_title("Clearance-time forecast with confidence intervals\n"
                 "(Mysore Road, 6 AM, road closure)", fontsize=12, weight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_forecast_intervals.png"), dpi=160)
    plt.close(fig)
    print("  fig1_forecast_intervals.png")


def fig_heatmap(demand):
    top = demand.head(15)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = LinearSegmentedColormap.from_list("yor", ["#fffbe6", "#fb923c", "#7f1d1d"])
    im = ax.imshow(top.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(top.columns)))
    ax.set_xticklabels(top.columns, fontsize=8)
    ax.set_yticks(range(len(top.index)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.set_xlabel("Hour of day")
    ax.set_title("Incident demand by corridor × hour\n(where & when to pre-position crews)",
                 fontsize=12, weight="bold")
    fig.colorbar(im, ax=ax, label="events")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_demand_heatmap.png"), dpi=160)
    plt.close(fig)
    print("  fig2_demand_heatmap.png")


def fig_coverage(metrics):
    raw = metrics["p10_p90_coverage_raw"] * 100
    cal = metrics["p10_p90_coverage"] * 100
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(["Raw quantiles", "After CQR\ncalibration"], [raw, cal],
                  color=["#94a3b8", "#2563eb"], width=0.55)
    ax.axhline(80, ls="--", color="#16a34a", linewidth=2)
    ax.text(1.45, 81, "target 80%", color="#16a34a", fontsize=10, ha="right")
    for b, v in zip(bars, [raw, cal]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", fontsize=12, weight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("P10–P90 interval coverage")
    ax.set_title("Conformal calibration tightens reliability",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_cqr_coverage.png"), dpi=160)
    plt.close(fig)
    print("  fig3_cqr_coverage.png")


def main():
    print("Generating deck figures...")
    models, lookups, metrics, demand = load()
    fig_intervals(models, lookups)
    fig_heatmap(demand)
    fig_coverage(metrics)
    print(f"\nSaved to {FIG}")


if __name__ == "__main__":
    main()
