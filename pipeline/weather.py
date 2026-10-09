"""Daily weather (past 31 days + 9-day forecast) from Open-Meteo on a coarse
lattice, interpolated to the model grid with an elevation lapse-rate correction."""
from __future__ import annotations

import time

import numpy as np
import requests
from scipy.interpolate import RegularGridInterpolator

from . import log

API = "https://api.open-meteo.com/v1/forecast"
PAST_DAYS = 31
FORECAST_DAYS = 9
LAPSE = -0.0065  # degC per metre
VARS = ["precipitation_sum", "temperature_2m_max", "temperature_2m_min", "et0_fao_evapotranspiration"]


def lattice(bbox, spacing):
    w, s, e, n = bbox
    lons = np.round(np.arange(w, e + spacing / 2, spacing), 4)
    lats = np.round(np.arange(s, n + spacing / 2, spacing), 4)
    return lats, lons


def _get_batch(chunk, timezone, past_days=PAST_DAYS):
    params = {
        "latitude": ",".join(f"{p[0]:.4f}" for p in chunk),
        "longitude": ",".join(f"{p[1]:.4f}" for p in chunk),
        "daily": ",".join(VARS),
        "past_days": past_days, "forecast_days": FORECAST_DAYS,
        "timezone": timezone,
    }
    for attempt in range(6):
        try:
            r = requests.get(API, params=params, timeout=(20, 90))
            if r.status_code == 200:
                data = r.json()
                return data if isinstance(data, list) else [data]
            wait = 65 if r.status_code == 429 else 10 * (attempt + 1)
            log.warn(f"open-meteo HTTP {r.status_code} ({r.text[:120]}); retry in {wait}s")
        except requests.RequestException as exc:
            wait = 15 * (attempt + 1)
            log.warn(f"open-meteo {exc.__class__.__name__}; retry in {wait}s")
        time.sleep(wait)
    return None


def fetch(bbox, spacing, timezone, batch=25, past_days=PAST_DAYS):
    lats, lons = lattice(bbox, spacing)
    ny, nx = len(lats), len(lons)
    pts = [(la, lo) for la in lats for lo in lons]
    results = [None] * len(pts)
    for i in range(0, len(pts), batch):
        got = _get_batch(pts[i:i + batch], timezone, past_days)
        if got is None:
            log.warn(f"open-meteo: batch {i // batch + 1} failed; filling from neighbours")
        else:
            results[i:i + len(got)] = got
        # stay under the free per-minute limit (long look-backs count as several calls)
        time.sleep(3 if past_days <= PAST_DAYS else 20)
    ok = [k for k, r in enumerate(results) if r is not None]
    if len(ok) < 0.7 * len(pts):
        raise RuntimeError(f"open-meteo: only {len(ok)}/{len(pts)} points fetched")
    dates = results[ok[0]]["daily"]["time"]
    out = {v: np.full((len(dates), ny, nx), np.nan, np.float32) for v in VARS}
    elev = np.full((ny, nx), np.nan, np.float32)
    for k in ok:
        res = results[k]
        iy, ix = divmod(k, nx)
        elev[iy, ix] = res.get("elevation") or 0
        for v in VARS:
            out[v][:, iy, ix] = [np.nan if x is None else x for x in res["daily"][v]]
    # points from failed batches: copy the nearest fetched point
    missing = np.isnan(elev)
    if missing.any():
        from scipy.ndimage import distance_transform_edt
        _, (iy, ix) = distance_transform_edt(missing, return_indices=True)
        elev = elev[iy, ix]
        for v in VARS:
            out[v] = out[v][:, iy, ix]
    for v in VARS:  # fill occasional gaps along time
        a = out[v]
        if np.isnan(a).any():
            mean = np.nanmean(a, axis=0, keepdims=True)
            a[:] = np.where(np.isnan(a), np.nan_to_num(mean), a)
    log.info(f"weather: {len(ok)}/{len(pts)} points fetched, {dates[0]}..{dates[-1]}")
    return {"dates": dates, "lats": lats, "lons": lons, "elev": elev,
            "P": out["precipitation_sum"], "Tmax": out["temperature_2m_max"],
            "Tmin": out["temperature_2m_min"], "ET0": out["et0_fao_evapotranspiration"]}


def to_grid(wx, lon, lat, elev):
    """Interpolate every day to target lon/lat arrays (same shape as elev)."""
    pts = np.stack([lat.ravel(), lon.ravel()], axis=-1)
    lats, lons = wx["lats"], wx["lons"]

    def interp(field2d):
        f = RegularGridInterpolator((lats, lons), field2d, bounds_error=False, fill_value=None)
        return f(pts).reshape(lat.shape).astype(np.float32)

    pt_elev = interp(wx["elev"])
    e = np.where(np.isnan(elev), pt_elev, elev)
    dT = LAPSE * (e - pt_elev)
    T = len(wx["dates"])
    P = np.empty((T,) + lat.shape, np.float32)
    Tmin = np.empty_like(P)
    Tmax = np.empty_like(P)
    for t in range(T):
        P[t] = np.maximum(interp(wx["P"][t]), 0)
        Tmin[t] = interp(wx["Tmin"][t]) + dT
        Tmax[t] = interp(wx["Tmax"][t]) + dT
    return P, Tmin, Tmax


def field_to_grid(wx, key, lon, lat):
    """Interpolate another daily field (e.g. ET0) to the target grid, no elevation correction."""
    pts = np.stack([lat.ravel(), lon.ravel()], axis=-1)
    out = np.empty((len(wx["dates"]),) + lat.shape, np.float32)
    for t in range(len(wx["dates"])):
        f = RegularGridInterpolator((wx["lats"], wx["lons"]), wx[key][t], bounds_error=False, fill_value=None)
        out[t] = np.maximum(f(pts).reshape(lat.shape), 0)
    return out
