"""Weather-driven fruiting score (0..1) per species and day."""
from __future__ import annotations

import datetime as dt

import numpy as np

from .habitat import trapezoid


def dry_run_length(P, wet_mm=1.0):
    """Consecutive dry days ending on each day: (T, H, W)."""
    out = np.zeros_like(P)
    run = np.zeros(P.shape[1:], np.float32)
    for t in range(P.shape[0]):
        run = np.where(P[t] < wet_mm, run + 1, 0)
        out[t] = run
    return out


def weather_scores(sp, P, Tmin, Tmax, dates, day_idx, dry=None):
    """Return list of (H, W) arrays, one per index in day_idx."""
    w = sp["weather"]
    lag0, lag1 = w["lag"]
    need = float(w["rain_need"])
    Tmean = (Tmin + Tmax) / 2
    if dry is None:
        dry = dry_run_length(P)
    out = []
    for t in day_idx:
        a, b = max(t - lag1, 0), max(t - lag0 + 1, 0)
        rain = P[a:b].sum(axis=0)
        rain_f = np.clip(rain / need, 0, 1)
        # top-up: most species also need some moisture in the last week
        recent = P[max(t - 6, 0):t + 1].sum(axis=0)
        t7max = Tmax[max(t - 6, 0):t + 1].mean(axis=0)
        drying = np.maximum(dry[t] - 4, 0) * 0.08 * (1 + np.maximum(t7max - 20, 0) / 10)
        moist_f = np.where(recent >= 5, 1.0, np.clip(1 - drying, 0.2, 1))
        t7 = Tmean[max(t - 6, 0):t + 1].mean(axis=0)
        temp_f = trapezoid(t7, *w["temp"])
        frost_days = (Tmin[max(t - 3, 0):t + 1] < -1.5).sum(axis=0)
        frost_f = 1 - w["frost"] * np.clip(frost_days / 3, 0, 1)
        month = dt.date.fromisoformat(dates[t]).month
        season = w["season"][month - 1]
        out.append((rain_f * moist_f * temp_f * frost_f * season).astype(np.float32))
    return out


def rain_window(P, t, days=14):
    return P[max(t - days + 1, 0):t + 1].sum(axis=0)
