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


def lag_weights(lag0, lag1, ramp_in=4, ramp_out=6):
    """Weight of rain that fell k days ago (k = 0..lag1+ramp_out): rises over the
    `ramp_in` days before lag0, full inside [lag0, lag1], fades over `ramp_out` days after."""
    k = np.arange(lag1 + ramp_out + 1, dtype=np.float32)
    up = np.clip((k - (lag0 - ramp_in)) / ramp_in, 0, 1)
    down = np.clip(1 - (k - lag1) / ramp_out, 0, 1)
    return np.minimum(up, down)


def weather_scores(sp, P, Tmin, Tmax, dates, day_idx, dry=None):
    """Return list of (H, W) arrays, one per index in day_idx."""
    w = sp["weather"]
    lag0, lag1 = w["lag"]
    need = float(w["rain_need"])
    wts = lag_weights(lag0, lag1)
    Tmean = (Tmin + Tmax) / 2
    if dry is None:
        dry = dry_run_length(P)
    out = []
    for t in day_idx:
        rain = np.zeros(P.shape[1:], np.float32)
        for k, wk in enumerate(wts):
            if wk > 0 and t - k >= 0:
                rain += wk * P[t - k]
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
