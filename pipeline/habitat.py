"""Static habitat suitability (0..1) per species, from the cached layers."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from .grid import Grid
from .static_layers import FOREST_TYPES, PASTURE_TYPES


def trapezoid(x, a, b, c, d):
    x = np.asarray(x, dtype=np.float32)
    up = np.clip((x - a) / max(b - a, 1e-6), 0, 1)
    down = np.clip((d - x) / max(d - c, 1e-6), 0, 1)
    return np.minimum(up, down)


def terrain(elev, grid: Grid):
    """Slope (deg) and a 0..1 dampness index (hollows and north-facing slopes)."""
    _, lat = grid.cell_lonlat()
    filled = np.where(np.isnan(elev), np.nanmean(elev) if np.isfinite(elev).any() else 0, elev)
    true_res = grid.res * np.cos(np.radians(lat))           # metres per pixel on the ground
    dz_row, dz_col = np.gradient(filled)
    dz_row = dz_row / true_res                               # +ve = rising southwards
    dz_col = dz_col / true_res
    grad = np.hypot(dz_row, dz_col)
    slope = np.degrees(np.arctan(grad))
    northness = np.where(grad > 1e-6, dz_row / np.maximum(grad, 1e-6), 0.0)  # +1 = faces north
    northness *= np.clip(slope / 15.0, 0, 1)
    tpi = filled - uniform_filter(filled, size=9)            # ~1 km neighbourhood
    hollow = np.clip(0.5 - tpi / 30.0, 0, 1)
    damp = 0.6 * hollow + 0.4 * (northness + 1) / 2
    return slope.astype(np.float32), damp.astype(np.float32)


def host_score(sp, layers, shape):
    hab = sp["habitat"]
    if hab["host"] == "grass":
        return np.clip(_grass_host(hab, layers, shape) * _plough_factor(hab, layers), 0, 1)
    return _forest_host(hab, layers, shape)


def _plough_factor(hab, layers):
    """Copernicus ploughing indicator (Europe-wide): grassland ploughed in the last
    0-2 years, or 3-6 years, or whose cover changed, keeps only part of its score."""
    pk = hab.get("ploughed")
    if not pk or "pl_recent" not in layers:
        return 1.0
    f = 1.0
    f = f - (1 - pk.get("recent", 1.0)) * layers["pl_recent"].astype(np.float32)
    f = f - (1 - pk.get("mid", 1.0)) * layers["pl_mid"].astype(np.float32)
    f = f - (1 - pk.get("changed", 1.0)) * layers["pl_changed"].astype(np.float32)
    if "mw_many" in layers:  # cut 3+ times a year = intensive silage grass
        f = f - (1 - pk.get("mown_twice", 1.0)) * layers["mw_two"].astype(np.float32)
        f = f - (1 - pk.get("mown_3plus", 1.0)) * layers["mw_many"].astype(np.float32)
    return np.clip(f, 0, 1)


def _grass_host(hab, layers, shape):
    grass = layers.get("lc_grass")
    if grass is None:
        return np.full(shape, 0.3, np.float32)
    open_ = np.clip(grass.astype(np.float32) + 0.4 * layers["lc_shrub"].astype(np.float32), 0, 1)
    pw = hab.get("pasture")
    if not pw or not all(("pa_" + k) in layers for k in PASTURE_TYPES):
        return open_
    # France: farm-parcel register says which grass is old pasture vs re-sown
    pa = {k: layers["pa_" + k].astype(np.float32) for k in PASTURE_TYPES}
    registered = sum(pa.values())
    long_term = pa["permanent"] + pa["rough"]
    lt_factor = 1.0
    if "pa_hist_perm" in layers:
        # permanent today but not in the oldest register -> possibly ploughed within ~10 years
        old = layers["pa_hist_perm"].astype(np.float32)
        recent = np.clip(long_term - old, 0, None) / np.maximum(long_term, 1e-6)
        lt_factor = 1.0 - (1.0 - pw.get("recent_permanent", 1.0)) * recent
    host = lt_factor * (pw["permanent"] * pa["permanent"] + pw["rough"] * pa["rough"])
    host = host + pw["temporary"] * pa["temporary"]
    if "pa_organic" in layers:  # organic farms: no synthetic fertiliser
        host = host * (1.0 + pw.get("organic_bonus", 0.0) * layers["pa_organic"].astype(np.float32))
    tilled = layers["pa_tilled"].astype(np.float32) if "pa_tilled" in layers else 0.0
    # grass the satellite sees but no parcel declares: commons, verges, paddocks.
    # Declared ploughed/cultivated land is removed even if it looks green.
    host += pw.get("other", 0.5) * np.clip(open_ - registered - tilled, 0, 1)
    return np.clip(host, 0, 1)


def _forest_host(hab, layers, shape):
    weights = hab["host"]
    unknown_w = hab.get("unknown_forest", 0.5)
    tree = layers.get("lc_tree")
    tree = tree.astype(np.float32) if tree is not None else np.full(shape, 0.3, np.float32)
    have_ft = all(("ft_" + k) in layers for k in FOREST_TYPES)
    if not have_ft:
        return np.clip(unknown_w * tree, 0, 1)
    total = np.zeros(shape, np.float32)
    mapped = np.zeros(shape, np.float32)
    for k in FOREST_TYPES:
        f = layers["ft_" + k].astype(np.float32)
        total += weights.get(k, 0.0) * f
        mapped += f
    # small woods missing from BD Foret but seen as trees by WorldCover
    total += unknown_w * 0.8 * np.clip(tree - mapped, 0, 1)
    return np.clip(total, 0, 1)


def habitat(sp, layers, grid: Grid, terr=None, sat_wet=None):
    shape = grid.shape
    slope, damp = terr if terr is not None else terrain(layers["elev"], grid)
    hab = sp["habitat"]
    host = host_score(sp, layers, shape)
    ph = layers.get("ph")
    ph_f = np.full(shape, 0.8, np.float32) if ph is None else \
        np.where(np.isnan(ph), 0.8, trapezoid(ph, *hab["ph"]))
    elev = layers["elev"]
    elev_f = np.where(np.isnan(elev), 1.0, trapezoid(elev, *hab["elev"]))
    if sat_wet is not None:
        # blend terrain dampness with satellite-measured relative wetness where available
        damp = np.where(np.isfinite(sat_wet), 0.5 * damp + 0.5 * np.nan_to_num(sat_wet), damp)
    w = hab.get("damp", 0.3)
    damp_f = 1.0 - w * (1.0 - damp)
    steep = (22, 40) if hab["host"] == "grass" else (30, 45)
    slope_f = trapezoid(slope, -1, 0, *steep)
    h = host * ph_f * elev_f * damp_f * slope_f
    return np.clip(h, 0, 1).astype(np.float32)
