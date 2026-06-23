# ASTRAM — Slide-Ready Summary (Theme 2)

Copy these straight into the deck. Numbers are from the temporal hold-out
(`artifacts/metrics.json`), trained on 3,188 resolved incidents,
Nov 2023 → Apr 2024.

---

## Slide 1 — The Problem (one line)

> Theme 2 asks two things: **forecast** event-driven traffic impact **and
> recommend** manpower, barricading & diversion. Most solutions stop at
> prediction. **Ours does both, in a working app.**

---

## Slide 2 — What We Built (4 bullets)

- **Forecasts clearance time as a range, not a guess** — every incident gets a
  P10 / P50 / P90 interval, so dispatchers plan for the worst case, not just the
  average.
- **Turns each forecast into an action plan** — crews, equipment, barricading
  and a diversion route, generated from the prediction + cause playbook.
- **Shift-level dispatch plan** — where to pre-position tow trucks / pumps /
  crews by corridor × time window, as a downloadable artifact (CSV/JSON).
- **Preventive-maintenance hotspots** — recurring waterlogging / pothole /
  construction junctions, so the city fixes the *location*, not the symptom.

---

## Slide 3 — The Model (credibility)

- **LightGBM** gradient boosting on log-resolution + dedicated **quantile
  models** (P10/P50/P90).
- **Conformalized Quantile Regression (CQR)** calibrates the intervals —
  coverage **60% → 75%** (Romano et al., NeurIPS 2019).
- **Strict temporal validation** — train on earlier events, test on later. No
  look-ahead leakage.

| Metric | Value | Read it as |
|--------|-------|------------|
| Median error (MedAE) | **1.4 h** | A typical incident is predicted within ~1.4 h |
| P10–P90 coverage | **75%** | Calibrated interval (target 80%) |
| Vehicle breakdown (60% of all events) P50 | **~0.7 h** | Predicted tightly |

**The key insight for judges:** mean error looks large only because a handful
of multi-day infrastructure incidents dominate the average — which is *exactly
why we report intervals*. We surface this honestly instead of hiding behind one
number.

---

## Slide 4 — Demo Flow (live walk-through)

1. Pick **vehicle breakdown, Mysore Road, 6 AM** → ~0.7 h clearance, HIGH
   severity, 2 towing units, in-lane management.
2. Switch to **waterlogging, road closure ON** → ~24 h median but P90 ~170 h,
   CRITICAL → pumping crew + hard barricades + diversion via Magadi Road.
3. Open **Resource Dispatch** → heatmap shows Mysore Rd / Bellary Rd peak load;
   download the staging plan.
4. Open **Hotspots** → recurring junctions for preventive maintenance.

---

## One-liner for the title slide

> **ASTRAM Congestion Copilot** — forecasts how long an incident will block a
> corridor, with confidence, and tells the control room exactly what to send.

---

### Suggested visuals
- Screenshot of the **Forecast & Recommend** tab (the interval bar + plan card).
- Screenshot of the **corridor × hour heatmap**.
- The CQR coverage table (60% → 75%).

Generate the screenshots with:
```bash
prototype/.venv/bin/python prototype/make_screenshots.py
```
(or just screenshot the live app — `streamlit run prototype/app.py`).
