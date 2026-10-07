"""Rolling archive of observed daily weather on each region's weather lattice.

Kept on the repo's `weather-archive` branch (restored and saved by the workflow).
It was seeded from Open-Meteo's last ~3 months and grows by a day each run, so a
later switch to bulk forecast files (DWD / ECMWF / Meteo-France) only has to supply
the forecast days - the past weeks of rain come from here.
"""
from __future__ import annotations

import os

import numpy as np

from . import log

SEED_DAYS = 92      # Open-Meteo's maximum look-back
KEEP_DAYS = 400


def path(archive_dir, rid):
    return os.path.join(archive_dir, f"{rid}.npz")


def load(archive_dir, rid, lats, lons):
    p = path(archive_dir, rid)
    if not os.path.exists(p):
        return None
    with np.load(p) as z:
        if z["lats"].shape != lats.shape or z["lons"].shape != lons.shape or \
                not (np.allclose(z["lats"], lats) and np.allclose(z["lons"], lons)):
            log.warn("weather archive lattice changed; starting a new archive")
            return None
        return {"dates": [str(d) for d in z["dates"]], "P": z["P"], "Tmin": z["Tmin"], "Tmax": z["Tmax"]}


def days_held(archive_dir, rid, lats, lons):
    a = load(archive_dir, rid, lats, lons)
    return 0 if a is None else len(a["dates"])


def update(archive_dir, rid, wx, today):
    """Merge the past days of a fresh fetch (dates before today) into the archive."""
    os.makedirs(archive_dir, exist_ok=True)
    lats, lons = wx["lats"], wx["lons"]
    old = load(archive_dir, rid, lats, lons)
    rows = {}
    if old:
        for i, d in enumerate(old["dates"]):
            rows[d] = (old["P"][i], old["Tmin"][i], old["Tmax"][i])
    for i, d in enumerate(wx["dates"]):
        if d < today:  # newest values win: recent days get revised as data settles
            rows[d] = (wx["P"][i], wx["Tmin"][i], wx["Tmax"][i])
    dates = sorted(rows)[-KEEP_DAYS:]
    np.savez_compressed(
        path(archive_dir, rid), lats=lats, lons=lons, dates=np.array(dates),
        P=np.stack([rows[d][0] for d in dates]).astype(np.float32),
        Tmin=np.stack([rows[d][1] for d in dates]).astype(np.float32),
        Tmax=np.stack([rows[d][2] for d in dates]).astype(np.float32))
    log.info(f"weather archive: {len(dates)} days held ({dates[0]}..{dates[-1]})")
