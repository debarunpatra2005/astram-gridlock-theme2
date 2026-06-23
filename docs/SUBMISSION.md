# HackerEarth submission text — copy/paste

Paste this into the submission "description" field, trim to fit the form.
Replace `<...>` placeholders.

---

## Title
ASTRAM Congestion Copilot — Event-Driven Congestion Forecast & Dispatch (Theme 2)

## One-line summary
A working prototype that forecasts how long a traffic incident will block a
corridor — with calibrated confidence intervals — and recommends the manpower,
barricading and diversion plan to clear it.

## Problem addressed
Theme 2 — *How can historical and real-time data be used to forecast
event-related traffic impact and recommend optimal manpower, barricading and
diversion plans?* We answer **both** halves: prediction **and** the operational
plan that acts on it.

## What it does
1. **Forecast** — for any incident (cause, corridor, time, road closure, live
   load) it predicts clearance time as a **P10 / P50 / P90 interval**, not a
   fragile single number.
2. **Recommend** — converts that forecast into a concrete plan: response team,
   crew count, equipment, barricading and a diversion route, scaled by severity.
3. **Dispatch** — a shift-level pre-positioning plan (where to stage tow trucks
   / pumps / crews by corridor × time window), exportable as CSV/JSON.
4. **Prevent** — recurring waterlogging / pothole / construction hotspots for
   preventive maintenance.

## Approach / model
- LightGBM gradient boosting on log-resolution, trained on 3,188 resolved
  incidents (Nov 2023 → Apr 2024), 36 leakage-safe features reconstructable at
  serve time.
- Separate **quantile models (P10/P50/P90)**, plus dedicated models for the
  high-variance causes (potholes, construction).
- Intervals calibrated with **Conformalized Quantile Regression** — P10–P90
  coverage improved 60% → 75% (target 80%).
- **Strict temporal validation** (train on earlier events, test on later) — no
  look-ahead leakage.

## Honest performance
- Typical (median) error ≈ **1.4 h** per incident.
- Vehicle breakdowns (60% of all events) predicted within ~0.7 h.
- Mean error is large only because rare multi-day infrastructure incidents
  dominate the average — which is exactly why we report intervals. Reported in
  the app's Model Card, not hidden.

## Real-world fit
Built on the actual ASTRAM / Bengaluru Traffic Police event dataset. The
corridor-priority rule, bimodal peak timing and cause mix all match BTP field
reality. The dispatch plan maps directly onto how control rooms pre-position
resources per shift.

## How to run
```
bash prototype/setup.sh          # sets up env + launches the app
bash prototype/setup.sh retrain  # also regenerates models from the CSV
```
App opens at http://localhost:8501. See `prototype/README.md` for details.

## Links
- Code repo: <GitHub link or "attached zip">
- Demo video: <YouTube/Drive link>
- Tech writeup: <attached docx>

## Tech stack
Python 3.11 · LightGBM · scikit-learn · Conformal Prediction · Streamlit · Plotly
