"""Field-scale wetness from Sentinel-2 (free public archive on AWS, no account).

NDMI = (NIR - SWIR) / (NIR + SWIR) from bands B8A and B11 measures how much water
the vegetation holds. A cloud-free composite of the last ~30 days is made, then each
pixel is ranked against similar ground (woodland vs woodland, grass vs grass), giving
"wetter than X% of comparable ground". Refreshed every few days.
"""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import requests

from . import log
from .grid import Grid

STAC = "https://earth-search.aws.element84.com/v1/search"
UA = {"User-Agent": "mushroom-forecast (github.com/rkmvzm9nmd-cpu/mushroom-forecast)"}
GOOD_SCL = (4, 5)          # Sentinel-2 scene classes: vegetation, bare soil (no cloud/shadow/water/snow)
DAYS_BACK = 30
MAX_ITEMS = 40


def _search(bbox, start, end):
    body = {"collections": ["sentinel-2-l2a"], "bbox": list(bbox),
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": 70}}, "limit": 100,
            "sortby": [{"field": "properties.datetime", "direction": "desc"}]}
    feats, url = [], STAC
    for _ in range(5):
        r = requests.post(url, json=body, headers=UA, timeout=120)
        r.raise_for_status()
        d = r.json()
        feats += d.get("features", [])
        nxt = next((l for l in d.get("links", []) if l.get("rel") == "next"), None)
        if not nxt or len(feats) >= 300:
            break
        body = nxt.get("body", body)
        url = nxt.get("href", STAC)
    feats.sort(key=lambda f: f["properties"].get("datetime", ""), reverse=True)
    return feats


def _read(href, grid: Grid, resampling, dtype, nodata, overview=1):
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import reproject
    dst = np.full(grid.shape, nodata, dtype=dtype)
    with rasterio.open(href, overview_level=overview) as src:
        reproject(source=rasterio.band(src, 1), destination=dst, src_nodata=src.nodata or 0,
                  dst_nodata=nodata, dst_transform=grid.transform, dst_crs=CRS.from_epsg(3857),
                  resampling=resampling, init_dest_nodata=False, num_threads=4)
    return dst


def ndmi_composite(region, grid: Grid, workdir):
    from rasterio.enums import Resampling
    today = dt.date.today()
    start = (today - dt.timedelta(days=DAYS_BACK)).isoformat()
    items = _search(region["bbox"], start, today.isoformat())
    log.info(f"Sentinel-2: {len(items)} scenes with <70% cloud in the last {DAYS_BACK} days")
    if not items:
        raise RuntimeError("no Sentinel-2 scenes found")
    ndmi = np.full(grid.shape, np.nan, np.float32)
    age = np.full(grid.shape, 255, np.uint8)
    used, t0 = 0, time.time()
    for it in items[:MAX_ITEMS]:
        a, p = it["assets"], it["properties"]
        try:
            scl = _read(a["scl"]["href"], grid, Resampling.mode, np.uint8, 0)
            ok = np.isin(scl, GOOD_SCL) & np.isnan(ndmi)
            if ok.mean() < 0.002:
                continue
            nir = _read(a["nir08"]["href"], grid, Resampling.average, np.float32, 0).astype(np.float32)
            swir = _read(a["swir16"]["href"], grid, Resampling.average, np.float32, 0).astype(np.float32)
        except Exception as exc:
            log.warn(f"Sentinel-2 {it['id']}: {exc.__class__.__name__}: {str(exc)[:140]}")
            continue
        # processing baseline 04.00+ adds a +1000 offset unless the archive already removed it
        if str(p.get("s2:processing_baseline", "0")) >= "04.00" and not p.get("earthsearch:boa_offset_applied", False):
            nir, swir = nir - 1000, swir - 1000
        good = ok & (nir > 0) & (swir > 0)
        val = (nir - swir) / np.maximum(nir + swir, 1)
        ndmi[good] = val[good]
        days = (today - dt.date.fromisoformat(p["datetime"][:10])).days
        age[good] = min(days, 254)
        used += 1
        cover = np.isfinite(ndmi).mean()
        if cover > 0.97:
            break
    cover = float(np.isfinite(ndmi).mean())
    log.info(f"Sentinel-2 wetness composite: {used} scenes, {cover:.0%} of region covered "
             f"in {time.time() - t0:.0f}s; median age {np.median(age[age < 255]) if (age < 255).any() else '-'} days")
    if cover < 0.3:
        raise RuntimeError(f"too little cloud-free ground ({cover:.0%})")
    return {"ndmi": ndmi.astype(np.float16), "ndmi_age": age, "ndmi_built": np.array(today.isoformat())}


def relative_wetness(ndmi, layers):
    """Percentile of NDMI among comparable ground (woodland / grassland / other)."""
    from scipy.stats import rankdata
    nd = ndmi.astype(np.float32)
    out = np.full(nd.shape, np.nan, np.float32)
    tree = layers.get("lc_tree", np.zeros(nd.shape)).astype(np.float32)
    grass = (layers.get("lc_grass", np.zeros(nd.shape)).astype(np.float32)
             + layers.get("lc_shrub", np.zeros(nd.shape)).astype(np.float32))
    classes = [tree >= 0.5, (grass >= 0.5) & (tree < 0.5), (tree < 0.5) & (grass < 0.5)]
    for m in classes:
        sel = m & np.isfinite(nd)
        if sel.sum() > 50:
            out[sel] = (rankdata(nd[sel]) - 0.5) / sel.sum()
    return out


# ---------------------------------------------------------------- bare soil (tillage)
BARE_MONTHS = (4, 5, 9, 10, 11)   # sowing (Apr-May) and stubble / autumn ploughing (Sep-Nov)
BARE_NDVI = 0.25                  # bare earth; dormant brown grass stays above this
BARE_PER_TILE_MONTH = 4
BARE_LOOKBACK_MONTHS = 25   # two seasons, so the gap after the Copernicus ploughing map's last year is covered
BARE_BUDGET_S = 25 * 60


def _search_sorted(bbox, start, end, max_cloud=50):
    body = {"collections": ["sentinel-2-l2a"], "bbox": list(bbox),
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": max_cloud}}, "limit": 200,
            "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}]}
    r = requests.post(STAC, json=body, headers=UA, timeout=120)
    r.raise_for_status()
    return r.json().get("features", [])


def _tile(item):
    p = item["properties"]
    return p.get("grid:code") or p.get("s2:mgrs_tile") or item["id"].split("_")[1]


def bare_soil(region, grid: Grid, workdir):
    """Share of clear Sentinel-2 views (Apr-May, Sep-Nov, last ~25 months) in which the
    ground was bare earth: scene class 5 ("not vegetated") AND NDVI below 0.25, so winter-brown
    grass does not count. Old pasture is never bare; tilled fields are, after ploughing or sowing.
    Computed at ~50 m and averaged to the map grid, so part-tilled pixels get partial values."""
    from rasterio.enums import Resampling
    from .grid import block_mean
    today = dt.date.today()
    fine = grid.finer(2)
    n_clear = np.zeros(fine.shape, np.uint8)
    n_bare = np.zeros(fine.shape, np.uint8)
    # one pass per rank (best scene of every tile-month first), so a time-out still covers every month
    plan = []
    for back in range(BARE_LOOKBACK_MONTHS, -1, -1):
        y, m = today.year, today.month - back
        while m <= 0:
            m, y = m + 12, y - 1
        if m not in BARE_MONTHS:
            continue
        start = dt.date(y, m, 1)
        end = min((dt.date(y + m // 12, m % 12 + 1, 1) - dt.timedelta(days=1)), today)
        if end < start:
            continue
        try:
            items = _search_sorted(region["bbox"], start.isoformat(), end.isoformat())
        except Exception as exc:
            log.warn(f"bare soil: search {y}-{m:02d} failed ({exc.__class__.__name__})")
            continue
        per_tile = {}
        for it in items:
            per_tile.setdefault(_tile(it), []).append(it)
        for tile, its in per_tile.items():
            for rank, it in enumerate(its[:BARE_PER_TILE_MONTH]):
                plan.append((rank, f"{y}-{m:02d}", tile, it))
    plan.sort(key=lambda x: (x[0], x[1], x[2]))
    log.info(f"bare soil: {len(plan)} Sentinel-2 scenes planned "
             f"({len({p[1] for p in plan})} months, {len({p[2] for p in plan})} tiles)")
    used, t0 = 0, time.time()
    for rank, month, tile, it in plan:
        if time.time() - t0 > BARE_BUDGET_S:
            log.warn(f"bare soil: time budget reached after {used} scenes")
            break
        a, p = it["assets"], it["properties"]
        try:
            scl = _read(a["scl"]["href"], fine, Resampling.nearest, np.uint8, 0)
            clear = np.isin(scl, GOOD_SCL)
            if clear.mean() < 0.002:
                continue
            cand = scl == 5
            if cand.any():
                red = _read(a["red"]["href"], fine, Resampling.average, np.float32, 0, overview=2).astype(np.float32)
                nir = _read(a["nir08"]["href"], fine, Resampling.average, np.float32, 0).astype(np.float32)
                if str(p.get("s2:processing_baseline", "0")) >= "04.00" and not p.get("earthsearch:boa_offset_applied", False):
                    red, nir = red - 1000, nir - 1000
                ndvi = (nir - red) / np.maximum(nir + red, 1)
                cand &= (ndvi < BARE_NDVI) & (nir > 0)
        except Exception as exc:
            log.warn(f"bare soil {it['id']}: {exc.__class__.__name__}: {str(exc)[:140]}")
            continue
        n_clear += clear.astype(np.uint8)
        n_bare += cand.astype(np.uint8)
        used += 1
    frac = np.where(n_clear >= 2, n_bare / np.maximum(n_clear, 1), np.nan).astype(np.float32)
    known = np.isfinite(frac)
    frac_c = block_mean(np.nan_to_num(frac), 2)
    cover = block_mean(known.astype(np.float32), 2)
    out = np.where(cover > 0.5, frac_c / np.maximum(cover, 1e-6), np.nan).astype(np.float32)
    obs = block_mean(n_clear.astype(np.float32), 2)
    log.info(f"bare soil: {used} scenes in {time.time() - t0:.0f}s; {np.isfinite(out).mean():.0%} of region "
             f"with 2+ clear views (median {np.median(obs):.0f} views)")
    if np.isfinite(out).mean() < 0.3:
        raise RuntimeError("too few clear views for the bare-soil layer")
    return {"bare_frac": out.astype(np.float16), "bare_views": np.clip(obs, 0, 255).astype(np.uint8),
            "bare_built": np.array(today.isoformat())}
