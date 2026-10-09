"""Root-zone soil-water bucket (grass-root layer, ~7-28 cm).

Deficit D (mm below full) is updated daily from rain P and FAO reference evaporation ET0;
evaporation slows as the soil dries and the deficit is capped at 150 mm:

    D_t = min(150, max(0, D_{t-1} + ET0_t * (1 - D_{t-1}/150) - P_t))

It matches ERA5-Land 7-28 cm soil moisture at r ~ -0.9 at the four French reference places
(see the "2026 drought and the liberty cap forecast" analysis). Species with a `root_zone_d0`
setting only count rain towards their fruiting trigger once the bucket is refilled to within
D0 mm of full:

    P_eff_t = max(0, P_t - max(0, D_{t-1} - D0))

The live forecast only sees ~31 past days, so the starting deficit comes from half-degree
cells of the Open-Meteo historical archive (ERA5), run from 1 Jan 2026 (soils full after
winter) and kept up to date in the weather archive.
"""
from __future__ import annotations

import datetime as dt
import os
import time

import numpy as np
import requests
from scipy.interpolate import RegularGridInterpolator

from . import log

CAP = 150.0
SEED_FROM = "2026-01-01"   # soils were full after a very wet Jan-Feb 2026
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
ERA5_LAG_DAYS = 6


def step(d_prev, p, et0):
    return np.minimum(CAP, np.maximum(0.0, d_prev + et0 * (1.0 - d_prev / CAP) - p))


def run(P, ET0, d_init):
    """D after each day, shape like P (T, ...)."""
    D = np.empty_like(P, dtype=np.float32)
    d = np.asarray(d_init, np.float32)
    for t in range(P.shape[0]):
        d = step(d, P[t], ET0[t])
        D[t] = d
    return D


def before_each_day(D, d_init):
    """Deficit at the start of each day (the previous day's end state)."""
    return np.concatenate([np.asarray(d_init, np.float32)[None], D[:-1]], axis=0)


def effective_rain(P, D_before, d0):
    return np.maximum(0.0, P - np.maximum(0.0, D_before - d0)).astype(np.float32)


def refill_factor(D_before, d0):
    """1 when the root zone is within d0 of full, falling to 0 at the cap."""
    return np.clip(1.0 - (D_before - d0) / max(CAP - d0, 1e-6), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------- seeding from ERA5 cells
def cell_axes(bbox):
    w, s, e, n = bbox
    lats = np.arange(np.floor(s * 2) / 2, np.ceil(n * 2) / 2 + 0.01, 0.5)
    lons = np.arange(np.floor(w * 2) / 2, np.ceil(e * 2) / 2 + 0.01, 0.5)
    return np.round(lats, 1), np.round(lons, 1)


def _key(lat, lon):
    return f"{lat:.1f}_{lon:.1f}"


def _fetch(points, start, end):
    params = {"latitude": ",".join(f"{p[0]:.2f}" for p in points),
              "longitude": ",".join(f"{p[1]:.2f}" for p in points),
              "start_date": start, "end_date": end, "timezone": "UTC",
              "daily": "precipitation_sum,et0_fao_evapotranspiration"}
    for attempt in range(5):
        try:
            r = requests.get(ARCHIVE_API, params=params, timeout=(20, 180))
            if r.status_code == 200:
                d = r.json()
                return d if isinstance(d, list) else [d]
            wait = 70 if r.status_code == 429 else 20
            log.warn(f"bucket seed HTTP {r.status_code}: {r.text[:100]}; retry in {wait}s")
        except requests.RequestException as exc:
            wait = 20
            log.warn(f"bucket seed {exc.__class__.__name__}; retry in {wait}s")
        time.sleep(wait)
    return None


def load_cells(bbox, store_dir, today):
    """Daily P and ET0 per half-degree cell from SEED_FROM to ~today-6, cached and extended."""
    os.makedirs(store_dir, exist_ok=True)
    end = (dt.date.fromisoformat(today) - dt.timedelta(days=ERA5_LAG_DAYS)).isoformat()
    lats, lons = cell_axes(bbox)
    cells, todo = {}, {}
    for la in lats:
        for lo in lons:
            k = _key(la, lo)
            p = os.path.join(store_dir, f"c_{k}.npz")
            if os.path.exists(p):
                with np.load(p) as z:
                    cells[k] = {"start": str(z["start"]), "P": z["P"], "ET0": z["ET0"]}
                last = (dt.date.fromisoformat(cells[k]["start"]) + dt.timedelta(days=len(cells[k]["P"]) - 1)).isoformat()
                if last >= end:
                    continue
                todo.setdefault((dt.date.fromisoformat(last) + dt.timedelta(days=1)).isoformat(), []).append((la, lo, k))
            else:
                todo.setdefault(SEED_FROM, []).append((la, lo, k))
    for start, pts in todo.items():
        for i in range(0, len(pts), 6):
            batch = pts[i:i + 6]
            got = _fetch([(la, lo) for la, lo, _ in batch], start, end)
            if got is None:
                continue
            for (la, lo, k), d in zip(batch, got):
                dd = d["daily"]
                P = np.array([np.nan if x is None else x for x in dd["precipitation_sum"]], np.float32)
                E = np.array([np.nan if x is None else x for x in dd["et0_fao_evapotranspiration"]], np.float32)
                ok = np.isfinite(P) & np.isfinite(E)
                n = int(np.max(np.nonzero(ok)[0]) + 1) if ok.any() else 0   # drop trailing days not yet published
                P, E = np.nan_to_num(P[:n]), np.nan_to_num(E[:n])
                if k in cells:
                    cells[k]["P"] = np.concatenate([cells[k]["P"], P])
                    cells[k]["ET0"] = np.concatenate([cells[k]["ET0"], E])
                else:
                    cells[k] = {"start": start, "P": P, "ET0": E}
                np.savez_compressed(os.path.join(store_dir, f"c_{k}.npz"), **cells[k])
            time.sleep(3)
    return lats, lons, cells


def deficit_on(cells, lats, lons, date):
    """Deficit per cell at the END of `date` (2D over lats x lons); NaN where unknown."""
    out = np.full((len(lats), len(lons)), np.nan, np.float32)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            c = cells.get(_key(la, lo))
            if c is None or len(c["P"]) == 0:
                continue
            D = run(c["P"][:, None], c["ET0"][:, None], np.zeros(1, np.float32))[:, 0]
            idx = (dt.date.fromisoformat(date) - dt.date.fromisoformat(c["start"])).days
            if idx < 0:
                continue
            out[i, j] = D[min(idx, len(D) - 1)]
    return out


def initial_deficit(bbox, store_dir, today, first_date, lon, lat):
    """Deficit on each target pixel at the start of `first_date` (the first day of the live
    weather series), interpolated between half-degree cells."""
    lats, lons, cells = load_cells(bbox, store_dir, today)
    day_before = (dt.date.fromisoformat(first_date) - dt.timedelta(days=1)).isoformat()
    grid = deficit_on(cells, lats, lons, day_before)
    if not np.isfinite(grid).any():
        raise RuntimeError("no historical cells available to seed the bucket")
    grid = np.where(np.isfinite(grid), grid, np.nanmedian(grid))
    f = RegularGridInterpolator((lats, lons), grid, bounds_error=False, fill_value=None)
    d = f(np.stack([lat.ravel(), lon.ravel()], axis=-1)).reshape(lat.shape)
    return np.clip(d, 0, CAP).astype(np.float32), {k: round(float(v), 1) for k, v in zip(
        [_key(la, lo) for la in lats for lo in lons], grid.ravel())}
