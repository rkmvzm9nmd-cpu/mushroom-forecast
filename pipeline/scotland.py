"""Scotland-specific habitat layers (all Open Government Licence):

* Woodland types - Scottish Forestry National Forest Inventory (conifer / broadleaf /
  felled), Native Woodland Survey of Scotland (birch, oak, ash... woods) and the
  Caledonian Pinewood Inventory, via NatureScot's WFS.
* Grassland - NatureScot Habitat Map of Scotland (EUNIS habitats): semi-natural acid /
  neutral grassland vs agriculturally improved grassland vs arable.
* Soil pH - James Hutton Institute topsoil median pH (1:250 000 soil map).
"""
from __future__ import annotations

import glob
import os
import re
import zipfile

import numpy as np
import requests

from . import log
from .grid import Grid, block_mean

UA = {"User-Agent": "mushroom-forecast (github.com/rkmvzm9nmd-cpu/mushroom-forecast)"}
NS_WFS = "https://ogc.nature.scot/geoserver/ows"
HUTTON_PH = "https://www.hutton.ac.uk/sites/default/files/files/soils/downloads/Hutton_pH_OpenData.zip"


def _sl():
    from . import static_layers  # late import (static_layers imports this module)
    return static_layers


def _geom_name(layer):
    try:
        r = requests.get(NS_WFS, headers=UA, timeout=60, params={
            "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "DescribeFeatureType", "TYPENAMES": layer})
        m = re.search(r'name="([\w:]+)"[^>]*type="gml:\w*PropertyType"', r.text)
        if m:
            return m.group(1).split(":")[-1]
    except Exception:
        pass
    return "geom"


def _sample_props(layer, grid: Grid, n=3):
    minx, miny, maxx, maxy = grid.bounds_3857
    r = requests.get(NS_WFS, headers=UA, timeout=120, params={
        "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature", "TYPENAMES": layer,
        "OUTPUTFORMAT": "application/json", "SRSNAME": "EPSG:3857", "COUNT": n,
        "BBOX": f"{minx},{miny},{maxx},{maxy},EPSG:3857"})
    r.raise_for_status()
    feats = r.json().get("features", [])
    return [f.get("properties") or {} for f in feats]


def _text(p, keys=None):
    vals = [p.get(k) for k in keys] if keys else list(p.values())
    return " ".join(str(v) for v in vals if isinstance(v, str)).lower()


# ------------------------------------------------------------------ woodland
def woodland(grid: Grid):
    sl = _sl()
    types = sl.FOREST_TYPES
    idx = {t: i + 1 for i, t in enumerate(types)}
    fine = grid.finer(sl.SUB)
    raster = np.zeros(fine.shape, np.uint8)

    def nfi(p):
        t = _text(p)
        if "non woodland" in t or "non-woodland" in t:
            return 0, "non woodland"
        for key, cat in [("felled", None), ("ground prep", None), ("young", None), ("windblow", None),
                         ("failed", None), ("shrub", None), ("mixed mainly conifer", "mixed"),
                         ("mixed mainly broad", "mixed"), ("conifer", "sprucefir"), ("broadleave", "mixedbroad"),
                         ("coppice", "mixedbroad"), ("low density", "otherbroad")]:
            if key in t:
                return (idx[cat] if cat else 0), key
        return 0, "other"

    def nwss(p):
        keys = [k for k in p if "HAB" in k.upper()] or None
        t = _text(p, keys)
        for key, cat in [("birch", "birch"), ("pinewood", "nativepine"), ("scots pine", "nativepine"),
                         ("oak", "oak"), ("beech", "beech"), ("ash", "mixedbroad"), ("hazel", "mixedbroad"),
                         ("wet woodland", "mixedbroad"), ("alder", "mixedbroad"), ("willow", "mixedbroad"),
                         ("mixed deciduous", "mixedbroad"), ("conifer", "sprucefir"), ("plantation", "sprucefir")]:
            if key in t:
                return idx[cat], key
        return 0, "unclassified"

    layers = [("scottishforestry:National_Forest_Inventory_Woodland_Scotland", nfi, "NFI"),
              ("scottishforestry:Native_Woodland_Survey_of_Scotland", nwss, "NWSS"),
              ("scottishforestry:Caledonian_Pinewood_Inventory", lambda p: (idx["nativepine"], "caledonian"), "Pinewoods")]
    got_any = False
    for layer, fn, label in layers:   # later layers override earlier ones where they say more
        try:
            sample = _sample_props(layer, grid)
            if sample:
                log.info(f"{label} fields: {sorted(sample[0])[:20]}; example: "
                         f"{ {k: v for k, v in list(sample[0].items())[:8]} }")
            feats = sl.wfs_features(layer, grid, label, url=NS_WFS)
            part = sl._rasterise_features(feats, grid, fn, label)
            np.copyto(raster, part, where=part > 0)
            got_any = True
        except Exception as exc:
            log.warn(f"{label} unavailable: {exc.__class__.__name__}: {str(exc)[:200]}")
    if not got_any:
        raise RuntimeError("no Scottish woodland layers could be read")
    out = {"ft_" + t: block_mean((raster == i + 1).astype(np.float32), sl.SUB).astype(np.float16)
           for i, t in enumerate(types)}
    log.info("Scottish woodland types: " + ", ".join(f"{k[3:]} {float(v.mean()) * 100:.1f}%" for k, v in out.items()))
    return out


# ------------------------------------------------------------------ grassland (HabMoS)
def _eunis_class(code):
    c = (code or "").strip().upper().split("/")[0].split(" ")[0]
    if c.startswith("I1"):
        return 4, "arable I1"
    if c.startswith("E2.6"):
        return 3, "improved grassland E2.6"
    if c.startswith("E3") or c.startswith("E4"):
        return 2, f"wet/upland grassland {c[:2]}"
    if c.startswith("E1") or c.startswith("E2") or c.startswith("E5") or c.startswith("E7"):
        return 1, f"semi-natural grassland {c[:2]}"
    return 0, f"other {c[:2] or '?'}"


def habitat_map(grid: Grid):
    sl = _sl()
    layer = "habitatsandspecies:habmos"
    sample = _sample_props(layer, grid, n=20)
    if not sample:
        raise RuntimeError("Habitat Map of Scotland returned no features")
    log.info(f"HabMoS fields: {sorted(sample[0])}; example: { {k: v for k, v in list(sample[0].items())[:10]} }")
    pat = re.compile(r"^[A-J]\d")
    field = None
    for k in sample[0]:
        vals = [str(p.get(k) or "") for p in sample]
        if sum(bool(pat.match(v)) for v in vals) >= max(3, len(vals) // 3):
            field = k
            break
    if field is None:
        raise RuntimeError("could not find the EUNIS code field")
    log.info(f"HabMoS: using EUNIS field '{field}'")
    geom = _geom_name(layer)
    cql = f"\"{field}\" LIKE 'E%' OR \"{field}\" LIKE 'I1%'"
    feats = sl.wfs_features(layer, grid, "HabMoS", cql=cql, url=NS_WFS, geom=geom)
    raster = sl._rasterise_features(feats, grid, lambda p: _eunis_class(p.get(field)), "HabMoS")
    out = {"pa_" + name: block_mean((raster == i + 1).astype(np.float32), sl.SUB).astype(np.float16)
           for i, name in enumerate(sl.PASTURE_TYPES)}
    out["pa_tilled"] = block_mean((raster == 4).astype(np.float32), sl.SUB).astype(np.float16)
    log.info("HabMoS grassland: " + ", ".join(f"{k[3:]} {float(v.mean()) * 100:.1f}%" for k, v in out.items()))
    return out


# ------------------------------------------------------------------ soil pH (Hutton)
def _ph_value(v):
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(v))]
    nums = [x for x in nums if 2.5 <= x <= 10]
    return float(np.mean(nums)) if nums else np.nan


def hutton_ph(grid: Grid, workdir):
    import pyogrio
    import shapely
    from rasterio.crs import CRS
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject, transform_bounds
    zpath = os.path.join(workdir, "hutton_ph.zip")
    with requests.get(HUTTON_PH, headers=UA, timeout=600, stream=True) as d:
        d.raise_for_status()
        with open(zpath, "wb") as f:
            for chunk in d.iter_content(1 << 20):
                f.write(chunk)
    out_dir = os.path.join(workdir, "hutton_ph")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out_dir)
    srcs = glob.glob(os.path.join(out_dir, "**", "*.shp"), recursive=True) + \
        glob.glob(os.path.join(out_dir, "**", "*.gpkg"), recursive=True)
    if not srcs:
        raise RuntimeError(f"no vector file in pH zip: {os.listdir(out_dir)[:10]}")
    path = srcs[0]
    info = pyogrio.read_info(path)
    fields = list(info["fields"])
    ph_field = next((f for f in fields if "ph" in f.lower()), None)
    log.info(f"Hutton pH file {os.path.basename(path)}: fields {fields}; using '{ph_field}'; crs {info.get('crs')}")
    if ph_field is None:
        raise RuntimeError("no pH field")
    _, _, geoms, fdata = pyogrio.raw.read(path, columns=[ph_field])
    vals = np.array([_ph_value(v) for v in fdata[0]], dtype=np.float32)
    shapes = [(g, v) for g, v in zip(shapely.from_wkb(geoms), vals) if g is not None and np.isfinite(v)]
    log.info(f"Hutton pH: {len(shapes)} polygons, values {np.nanmin(vals):.1f}-{np.nanmax(vals):.1f}")
    src_crs = CRS.from_user_input(info.get("crs") or "EPSG:27700")
    l, b, r, t = transform_bounds("EPSG:3857", src_crs, *grid.bounds_3857)
    res = 100.0
    W, H = int((r - l) / res) + 2, int((t - b) / res) + 2
    tr = from_origin(l - res, t + res, res, res)
    ras = rasterize(shapes, out_shape=(H, W), transform=tr, fill=np.nan, dtype="float32")
    dst = np.full(grid.shape, np.nan, np.float32)
    reproject(ras, dst, src_transform=tr, src_crs=src_crs, src_nodata=np.nan, dst_transform=grid.transform,
              dst_crs=CRS.from_epsg(3857), dst_nodata=np.nan, resampling=Resampling.bilinear)
    log.info(f"Hutton pH covers {np.isfinite(dst).mean():.0%} of the grid")
    return dst
