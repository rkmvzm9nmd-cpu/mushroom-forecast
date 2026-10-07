"""Satellite soil wetness: Copernicus Soil Water Index (SWI), Europe, 1 km, daily.

SWI is a relative index (0 % = driest, 100 % = wettest seen at that spot), derived
from Sentinel-1 radar and MetOp ASCAT. T=10 is the topsoil-weighted version.
It arrives ~2 days late, so it corrects the near-term moisture estimate and the
rain-based model takes over for later forecast days.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from . import log
from .static_layers import cdse_env, cdse_search

# tried in order; the band is the topsoil-weighted index (T=10) or surface soil moisture
SOURCES = [("clms_swi_europe_1km_daily_v2_cog", "swi010", "soil water index"),
           ("clms_swi_europe_1km_daily_v1_cog", "swi010", "soil water index v1"),
           ("clms_ssm_europe_1km_daily_v1_cog", "ssm", "surface soil moisture")]
MAX_AGE_DAYS = 10


def _decode(arr, src):
    a = arr.astype(np.float32)
    nod = src.nodata
    bad = np.zeros(a.shape, bool) if nod is None else (arr == nod)
    scale = (src.scales or [1.0])[0] or 1.0
    offset = (src.offsets or [0.0])[0] or 0.0
    if src.dtypes[0] == "uint8" and scale == 1.0:
        bad |= arr > 200          # 241-255 are flags
        a = a * 0.5               # 0-200 -> 0-100 %
    else:
        a = a * scale + offset
    a[bad | (a < 0) | (a > 100)] = np.nan
    return a


def latest(bbox, lon, lat, today):
    """SWI (%) sampled at lon/lat arrays, plus its date. Raises if unavailable."""
    import rasterio
    from rasterio.windows import from_bounds
    w, s, e, n = bbox
    items, label, band = [], None, None
    for coll, bnd, lab in SOURCES:
        try:
            got = cdse_search(coll, bbox, limit=10,
                              sortby=[{"field": "properties.datetime", "direction": "desc"}])
        except Exception as exc:
            log.warn(f"{lab}: search failed ({exc.__class__.__name__}: {str(exc)[:120]})")
            continue
        got.sort(key=lambda i: i["properties"].get("datetime", ""), reverse=True)
        newest = got[0]["properties"].get("datetime", "")[:10] if got else None
        log.info(f"{lab} ({coll}): newest product {newest}")
        if newest and (dt.date.fromisoformat(today) - dt.date.fromisoformat(newest)).days <= MAX_AGE_DAYS:
            items, label, band = got, lab, bnd
            break
    if not items:
        raise RuntimeError("no soil moisture product from the last 10 days")
    with cdse_env():
        for item in items[:5]:
            asset = item["assets"].get(band) or next(
                (a for k, a in item["assets"].items() if k.startswith(band)), None)
            if asset is None:
                log.warn(f"{label}: no '{band}' asset in {item['id']} (assets: {list(item['assets'])[:12]})")
                continue
            href = asset["href"]
            path = "/vsis3/" + href[len("s3://"):]
            try:
                with rasterio.open(path) as src:
                    win = from_bounds(w - 0.05, s - 0.05, e + 0.05, n + 0.05, src.transform).round_offsets().round_lengths()
                    raw = src.read(1, window=win)
                    tr = src.window_transform(win)
                    vals = _decode(raw, src)
                    raw_stats = {int(k): int(c) for k, c in zip(*np.unique(raw, return_counts=True))} \
                        if raw.dtype == np.uint8 else {"min": float(np.nanmin(raw)), "max": float(np.nanmax(raw))}
            except Exception as exc:
                log.warn(f"{label} {item['id']}: {exc.__class__.__name__}: {str(exc)[:160]}")
                continue
            cover = float(np.isfinite(vals).mean())
            date = item["properties"]["datetime"][:10]
            if cover < 0.5:
                log.warn(f"{label} {date}: only {cover:.0%} valid, trying older")
                continue
            # fill gaps from nearest valid pixel, then bilinear-sample at the target points
            if np.isnan(vals).any():
                _, (iy, ix) = distance_transform_edt(np.isnan(vals), return_indices=True)
                vals = vals[iy, ix]
            col = (lon - tr.c) / tr.a - 0.5
            row = (lat - tr.f) / tr.e - 0.5
            out = map_coordinates(vals, [row, col], order=1, mode="nearest").astype(np.float32)
            log.info(f"{label} {date}: {cover:.0%} valid, regional mean {np.nanmean(out):.0f}% "
                     f"(range {np.nanmin(out):.0f}-{np.nanmax(out):.0f}); raw values {str(raw_stats)[:200]}")
            return out, date
    raise RuntimeError(f"no readable {label} product")
