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
