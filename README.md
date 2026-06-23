# ASTRAM — Event-Driven Congestion Forecast & Dispatch (Theme 2)

A working prototype that answers Theme 2 end-to-end:

> *"How can historical and real-time data be used to forecast event-related
> traffic impact and recommend optimal manpower, barricading, and diversion
> plans?"*

It does both halves:

1. **Forecast** — predicts how long an incident will take to clear, as a
   **P10 / P50 / P90 interval** (not a fragile point estimate).
2. **Recommend** — turns that forecast into a concrete operational plan:
   crews, equipment, barricading and a diversion label, plus a shift-level
   **pre-positioning dispatch plan** by corridor × time window.

## Quick start

```bash
# one command: sets up the venv, installs deps, launches the app
bash setup.sh

# to also retrain models + rebuild dispatch artifacts from the dataset:
bash setup.sh retrain
```

Or manually:
```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train.py              # train + evaluate models
.venv/bin/python resource_dispatch.py  # build dispatch artifacts
.venv/bin/streamlit run app.py         # launch the demo
```

The app runs entirely off the precomputed files in `artifacts/`, so it works
without retraining. It opens at http://localhost:8501.

> **Deployed demo:** this repo is Streamlit-Cloud ready — `app.py` and
> `requirements.txt` are at the root. Point share.streamlit.io at this repo with
> main file `app.py`.

## What's in the app

| Tab | What it shows |
|-----|---------------|
| 🔮 Forecast & Recommend | Enter an incident (cause, corridor, time, closure, live load) → get the clearance interval **and** a full manpower / barricade / diversion plan. |
| 🚒 Resource Dispatch | Corridor × hour demand heatmap + a downloadable shift staging plan. |
| 📍 Hotspots | Recurring waterlogging / pothole / construction junctions for preventive maintenance. |
| 📊 Model Card | Honest performance, incl. why a single number isn't enough and how intervals are calibrated. |

## How it works

- **`common.py`** — single source of truth for feature engineering. The same
  function builds features for training rows and live events, so there is no
  train/serve skew. Every feature is reconstructable from what a dispatcher
  actually knows + historical lookup tables.
- **`train.py`** — LightGBM on log-resolution. A point model plus quantile
  models (P10/P50/P90), and dedicated quantile models for the high-variance
  causes. Intervals are calibrated with **Conformalized Quantile Regression
  (CQR)** on a temporal hold-out. Strict **time-based split** — train on
  earlier events, test on later — so reported metrics have no look-ahead leak.
- **`resource_dispatch.py`** — joins historical demand with model-predicted
  clearance into `dispatch_plan.csv` / `.json` (staging by corridor × window)
  and `hotspots.json`.

## Honest performance (temporal hold-out)

- **Typical (median) error ≈ 1.4 h** — most incidents are predicted tightly.
- **Mean error is large** because a few multi-day infra incidents dominate the
  average — which is exactly why the model reports an **interval**, not a point.
- **P10–P90 coverage: ~60% raw → ~75% after CQR** (target 80%).
- Potholes stay the hardest case (~108 training rows + strong period-to-period
  shift); this is surfaced in the Model Card, not hidden.

## Artifacts (`artifacts/`)

| File | Contents |
|------|----------|
| `models.joblib` | point + quantile + cause-specific models, conformal widths |
| `lookups.json` | historical feature tables used at serve time |
| `metrics.json` | evaluation summary |
| `demand_matrix.csv` | events by corridor × hour |
| `dispatch_plan.csv` / `.json` | staging plan by corridor × time window |
| `hotspots.json` | recurring infrastructure hotspots |
